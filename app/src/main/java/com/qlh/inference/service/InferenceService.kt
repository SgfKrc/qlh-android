package com.qlh.inference.service

import android.annotation.SuppressLint
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.os.Binder
import android.os.Build
import android.os.IBinder
import android.os.PowerManager
import android.util.Log
import androidx.core.app.NotificationCompat
import com.qlh.inference.MainActivity
import com.qlh.inference.QlhApplication
import com.qlh.inference.R
import com.qlh.inference.model.Gemma4NativeAssets
import com.qlh.inference.status.AndroidRuntimeStatus
import com.qlh.inference.status.BackendStatus
import com.qlh.inference.status.ContextRuntimeStatus
import com.qlh.inference.status.GpuStatus
import com.qlh.inference.status.MultimodalStatus
import com.qlh.inference.status.ModelRuntimeStatus
import com.qlh.inference.system.AndroidDeviceInfoProvider
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.launch

/**
 * 本地推理引擎前台 Service。
 *
 * 生命周期：
 * 1. startService() → onCreate() → 前台通知 + 初始化引擎
 * 2. bindService() → onBind() → 返回 LocalBinder（供 Activity/Repository 调用）
 * 3. unbindService() → onUnbind()
 * 4. stopService() → onDestroy() → 卸载模型 + 释放 WakeLock
 *
 * 调用方通过 [LocalBinder] 获取 [InferenceService] 实例，直接调用推理方法。
 */
class InferenceService : Service() {

    companion object {
        private const val TAG = "InferenceService"

        /** Intent action: 预加载模型 */
        const val ACTION_LOAD_MODEL = "com.qlh.inference.LOAD_MODEL"
        /** @deprecated SAF 模式不再接受外部路径；预加载始终使用设置中选中的模型。 */
        @Deprecated("SAF 模式不再接受路径，预加载始终使用设置中选中的模型")
        const val EXTRA_MODEL_PATH = "model_path"
        /** Intent extra: context size */
        const val EXTRA_CONTEXT_SIZE = "context_size"
    }

    // ---- 公开属性 ----

    /** 推理引擎（Service 停止后为 null） */
    var engine: LocalInferenceEngine? = null
        private set

    /** 模型管理器 */
    lateinit var modelManager: ModelManager
        private set

    /** 当前上下文长度（由 ViewModel 同步设置，加载模型时生效） */
    @Volatile
    var modelContextSize: Int = 2048

    /** 引擎是否就绪 */
    val isReady: Boolean get() = engine?.isLoaded == true

    // ---- 内部状态 ----

    private var wakeLock: PowerManager.WakeLock? = null
    private val binder = LocalBinder()
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)

    // ---- Binder ----

    inner class LocalBinder : Binder() {
        fun getService(): InferenceService = this@InferenceService
    }

    // ================================================================
    // Service 生命周期
    // ================================================================

    override fun onCreate() {
        super.onCreate()
        Log.i(TAG, "Service onCreate")

        modelManager = ModelManager(this)
        engine = LocalInferenceEngine(this)
        QlhApplication.instance.inferenceService = this

        startForegroundNotification()
        acquireWakeLock()
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        Log.i(TAG, "Service onStartCommand: action=${intent?.action}")

        when (intent?.action) {
            ACTION_LOAD_MODEL -> {
                val contextSize = intent.getIntExtra(EXTRA_CONTEXT_SIZE, 2048)

                // 使用 Service scope 预加载当前选中模型；onDestroy() 会通过 scope.cancel() 取消任务。
                // 不再直接走路径接口，避免绕过 SAF fd 生命周期。
                scope.launch {
                    val result = ensureModelLoaded(contextSize)
                    result.onSuccess {
                        Log.i(TAG, "模型预加载完成")
                    }.onFailure { e ->
                        Log.e(TAG, "模型预加载失败: ${e.message}", e)
                    }
                }
            }
        }

        return START_STICKY
    }

    override fun onBind(intent: Intent?): IBinder {
        Log.i(TAG, "Service onBind")
        return binder
    }

    override fun onUnbind(intent: Intent?): Boolean {
        Log.i(TAG, "Service onUnbind")
        return true // 允许 rebind
    }

    override fun onDestroy() {
        Log.i(TAG, "Service onDestroy — 释放资源")

        // 卸载模型
        engine?.let {
            kotlinx.coroutines.runBlocking {
                it.shutdown()
            }
        }
        engine = null

        if (QlhApplication.instance.inferenceService === this) {
            QlhApplication.instance.inferenceService = null
        }
        releaseWakeLock()
        scope.cancel()
        super.onDestroy()
    }

    // ================================================================
    // 公开 API（供 ChatRepository 调用）
    // ================================================================

    /**
     * 确保模型已加载（如未加载则尝试加载默认路径的模型）。
     */
    suspend fun ensureModelLoaded(contextSize: Int = 2048): Result<Unit> {
        val eng = engine ?: return Result.failure(IllegalStateException("Service 未初始化"))

        val selectedUri = modelManager.getSelectedModelUri()
        if (eng.isLoaded && eng.loadedModelSourceUri == selectedUri) {
            return Result.success(Unit)
        }
        if (eng.isLoaded) {
            Log.i(TAG, "模型选择已变化，卸载旧模型: ${eng.loadedModelSourceUri} -> $selectedUri")
            eng.unloadModel()
        }

        if (!modelManager.isModelReady()) {
            return Result.failure(
                IllegalStateException(
                    "模型未就绪。请在「设置 → 模型管理」选择包含 GGUF 的目录，并选中要加载的模型。"
                )
            )
        }

        val handleResult = modelManager.openModelForLlama(preferFd = true)
        if (handleResult.isFailure) {
            return Result.failure(handleResult.exceptionOrNull()!!)
        }

        val handle = handleResult.getOrThrow()
        val fdResult = eng.loadModel(handle, contextSize)
        if (fdResult.isSuccess) {
            return fdResult
        }
        if (handle.mode != ModelManager.STORAGE_MODE_SAF_FD) {
            return fdResult
        }

        // 防御式关闭 SAF fd 句柄：LocalInferenceEngine 失败路径也会关闭，
        // 这里再次 close 是幂等的，避免未来实现变更造成 fd 泄漏。
        try {
            handle.close()
        } catch (_: Exception) {
        }

        Log.w(TAG, "SAF fd 加载失败，尝试复制到内部缓存后加载: ${fdResult.exceptionOrNull()?.message}")
        val fallbackHandle = modelManager.openModelForLlama(preferFd = false).getOrElse {
            return Result.failure(fdResult.exceptionOrNull() ?: it)
        }
        return eng.loadModel(fallbackHandle, contextSize)
    }

    suspend fun unloadModel(): Result<Unit> {
        val eng = engine ?: return Result.failure(IllegalStateException("Service 未初始化"))
        return eng.unloadModel()
    }

    /** Load the fixed Gemma4 model/mmproj pair before accepting local image input. */
    suspend fun ensureMultimodalReady(): Result<Unit> {
        val eng = engine ?: return Result.failure(IllegalStateException("Service 未初始化"))
        val assets = modelManager.inspectGemma4NativeAssets()
        if (!assets.sizeVerified) {
            return Result.failure(IllegalStateException(assets.reason))
        }
        val selected = modelManager.getSelectedModel()
        if (selected?.name?.equals(Gemma4NativeAssets.MAIN_FILENAME, ignoreCase = true) != true) {
            return Result.failure(
                IllegalStateException(
                    "请先选中 Gemma4 主模型 ${Gemma4NativeAssets.MAIN_FILENAME}"
                )
            )
        }
        val loadResult = ensureModelLoaded(modelContextSize)
        if (loadResult.isFailure) return loadResult
        if (eng.multimodalLoaded) return Result.success(Unit)

        val projector = modelManager.openGemma4MmprojForLlama(preferFd = true)
            .getOrElse { return Result.failure(it) }
        val projectorResult = eng.loadMultimodalProjector(projector)
        if (projectorResult.isFailure) return projectorResult
        val capability = eng.getMultimodalCapability().getOrNull().orEmpty()
        return if (capability["vision_supported"] == "true") {
            Result.success(Unit)
        } else {
            Result.failure(
                IllegalStateException(capability["reason"] ?: "MTMD 图像桥接不可用")
            )
        }
    }

    suspend fun generateMultimodal(
        prompt: String,
        imageBytes: List<ByteArray>,
        maxTokens: Int = 1024,
        temperature: Float = 0.7f,
        topP: Float = 0.9f,
    ): Result<String> {
        if (imageBytes.isEmpty() || imageBytes.size > 4) {
            return Result.failure(IllegalArgumentException("图片数量必须在 1-4 张之间"))
        }
        if (imageBytes.any { it.isEmpty() || it.size > 8 * 1024 * 1024 } ||
            imageBytes.sumOf { it.size.toLong() } > 16L * 1024 * 1024
        ) {
            return Result.failure(IllegalArgumentException("图片总大小超过本地多模态限制"))
        }
        val ready = ensureMultimodalReady()
        if (ready.isFailure) return Result.failure(ready.exceptionOrNull()!!)
        return engine?.generateMultimodal(
            prompt, imageBytes, maxTokens, temperature, topP,
        ) ?: Result.failure(IllegalStateException("Service 未初始化"))
    }

    /** Return the intersection of registered Gemma4 assets and compiled JNI MTMD capability. */
    suspend fun getMultimodalStatus(): MultimodalStatus {
        val assets = runCatching { modelManager.inspectGemma4NativeAssets() }
            .getOrElse { error ->
                return MultimodalStatus(
                    reason = "asset scan failed: ${error.message ?: error.javaClass.simpleName}"
                )
            }
        val native = engine?.getMultimodalCapability()?.getOrNull().orEmpty()
        val visionSupported = native["vision_supported"] == "true"
        val ready = visionSupported && assets.sizeVerified
        val reason = when {
            native.isEmpty() -> "本地 JNI 未初始化"
            !visionSupported -> native["reason"] ?: "MTMD 图像桥接未启用"
            !assets.sizeVerified -> assets.reason
            else -> "ready"
        }
        return MultimodalStatus(
            visionSupported = visionSupported,
            assetsPresent = assets.pairPresent,
            assetSizesVerified = assets.sizeVerified,
            ready = ready,
            reason = reason,
        )
    }

    suspend fun getRuntimeStatus(
        inferenceMode: String,
        isLite: Boolean,
    ): AndroidRuntimeStatus {
        val provider = AndroidDeviceInfoProvider(this)
        val eng = engine
        val backendInfo = eng?.getBackendInfo()?.getOrNull().orEmpty()
        val modelInfo = if (eng?.isLoaded == true) eng.getModelInfo().getOrNull().orEmpty() else emptyMap()
        val stats = if (eng?.isLoaded == true) eng.getLastGenerationStats().getOrNull().orEmpty() else emptyMap()
        val selected = try {
            modelManager.getSelectedModel()
        } catch (_: Exception) {
            null
        }
        val nativeResult = eng?.checkNativeRuntime()
        val gpu = provider.getGpuStatus().copy(
            supportsGpuOffload = backendInfo.bool("supports_gpu_offload"),
            backendDevices = backendInfo["backend_devices"].orEmpty(),
        )
        val multimodalStatus = getMultimodalStatus()

        return AndroidRuntimeStatus(
            nativeRuntimeAvailable = nativeResult?.isSuccess == true,
            nativeRuntimeError = nativeResult?.exceptionOrNull()?.message,
            serviceRunning = true,
            inferenceMode = inferenceMode,
            isLite = isLite,
            system = provider.getSystemStatus(),
            memory = provider.getMemoryStatus(),
            storage = provider.getStorageStatus(),
            gpu = gpu,
            backend = BackendStatus(
                engine = modelInfo["backend"] ?: "llama.cpp Android CPU",
                systemInfo = backendInfo["system_info"].orEmpty(),
                supportsMmap = backendInfo.bool("supports_mmap"),
                supportsMlock = backendInfo.bool("supports_mlock"),
                supportsGpuOffload = backendInfo.bool("supports_gpu_offload"),
                supportsRpc = backendInfo.bool("supports_rpc"),
            ),
            multimodal = multimodalStatus,
            model = ModelRuntimeStatus(
                selectedName = selected?.name.orEmpty(),
                selectedSizeBytes = selected?.sizeBytes ?: 0L,
                selectedSource = selected?.source?.name.orEmpty(),
                selectedUri = selected?.uri?.toString().orEmpty(),
                loaded = eng?.isLoaded == true,
                loadedPath = eng?.loadedModelPath.orEmpty(),
                loadedSourceUri = eng?.loadedModelSourceUri.orEmpty(),
                name = modelInfo["name"].orEmpty(),
                backend = modelInfo["backend"].orEmpty(),
                params = modelInfo["n_params"].orEmpty(),
                layers = modelInfo["n_layer"].orEmpty(),
                sizeBytes = modelInfo.long("size_bytes"),
                vocabTokens = modelInfo["vocab_tokens"].orEmpty(),
                embedding = modelInfo["n_embd"].orEmpty(),
                heads = listOf(modelInfo["n_head"], modelInfo["n_head_kv"])
                    .filter { !it.isNullOrBlank() }
                    .joinToString(" / "),
            ),
            context = ContextRuntimeStatus(
                configuredContextSize = modelContextSize,
                modelContextSize = modelInfo["n_ctx"].orEmpty(),
                trainContextSize = modelInfo["n_ctx_train"].orEmpty(),
                batchSize = modelInfo["n_batch"].orEmpty(),
                microBatchSize = modelInfo["n_ubatch"].orEmpty(),
                lastPromptTokens = stats.int("prompt_tokens"),
                lastGeneratedTokens = stats.int("generated_tokens"),
                lastTotalTokens = stats.int("total_tokens"),
                lastElapsedSeconds = stats.double("elapsed_seconds"),
                lastTokensPerSecond = stats.double("tokens_per_second"),
                stopReason = stats["stop_reason"].orEmpty(),
                estimatedKvMemoryMb = stats.double("estimated_memory_mb")
                    .takeIf { it > 0.0 }
                    ?: modelInfo.double("estimated_kv_memory_mb"),
            )
        )
    }

    /**
     * 流式推理。
     *
     * @param prompt 输入文本（用户消息）
     * @param maxTokens 最大生成 token 数
     * @param temperature 温度
     * @param topP top_p
     * @return Flow<String> 逐 token 文本流
     */
    fun generateStream(
        prompt: String,
        maxTokens: Int = 1024,
        temperature: Float = 0.7f,
        topP: Float = 0.9f
    ): Flow<String> {
        val eng = engine
            ?: throw IllegalStateException("推理引擎未初始化")

        if (!eng.isLoaded) {
            throw IllegalStateException("模型未加载，请先调用 ensureModelLoaded()")
        }

        return eng.generateStream(prompt, maxTokens, temperature, topP)
    }

    /**
     * 非流式推理（返回完整结果）。
     */
    suspend fun generate(
        prompt: String,
        maxTokens: Int = 1024,
        temperature: Float = 0.7f,
        topP: Float = 0.9f
    ): Result<String> {
        val eng = engine ?: return Result.failure(IllegalStateException("引擎未初始化"))
        if (!eng.isLoaded) {
            return Result.failure(IllegalStateException("模型未加载"))
        }
        return eng.generate(prompt, maxTokens, temperature, topP)
    }

    // ================================================================
    // 内部 — 前台通知 + WakeLock
    // ================================================================

    @SuppressLint("ForegroundServiceType")
    private fun startForegroundNotification() {
        val pendingIntent = PendingIntent.getActivity(
            this, 0,
            Intent(this, MainActivity::class.java),
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
        )

        val notification = NotificationCompat.Builder(
            this, QlhApplication.NOTIFICATION_CHANNEL_INFERENCE
        )
            .setContentTitle(getString(R.string.notification_inference_running))
            .setContentText("QLH 本地推理引擎运行中 · 全有模式")
            .setSmallIcon(android.R.drawable.ic_media_play)
            .setOngoing(true)
            .setContentIntent(pendingIntent)
            .setPriority(NotificationCompat.PRIORITY_LOW)
            .build()

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.UPSIDE_DOWN_CAKE) {
            startForeground(
                QlhApplication.NOTIFICATION_ID_INFERENCE,
                notification,
                android.content.pm.ServiceInfo.FOREGROUND_SERVICE_TYPE_SPECIAL_USE
            )
        } else {
            startForeground(QlhApplication.NOTIFICATION_ID_INFERENCE, notification)
        }
    }

    @Suppress("DEPRECATION")
    private fun acquireWakeLock() {
        if (wakeLock == null) {
            val powerManager = getSystemService(Context.POWER_SERVICE) as PowerManager
            wakeLock = powerManager.newWakeLock(
                PowerManager.PARTIAL_WAKE_LOCK,
                "QLH:InferenceEngine"
            )
            wakeLock?.acquire(30 * 60 * 1000L) // 30 分钟超时
        }
    }

    private fun releaseWakeLock() {
        wakeLock?.let {
            if (it.isHeld) it.release()
        }
        wakeLock = null
    }
}

private fun Map<String, String>.bool(key: String): Boolean = this[key] == "true"

private fun Map<String, String>.int(key: String): Int = this[key]?.toIntOrNull() ?: 0

private fun Map<String, String>.long(key: String): Long = this[key]?.toLongOrNull() ?: 0L

private fun Map<String, String>.double(key: String): Double = this[key]?.toDoubleOrNull() ?: 0.0
