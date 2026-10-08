package com.qlh.inference.service

import android.annotation.SuppressLint
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.os.Binder
import android.os.Build
import android.os.IBinder
import android.net.wifi.WifiManager
import android.os.Handler
import android.os.Looper
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
import com.qlh.inference.system.WorkerWakeLockPolicy
import com.qlh.inference.system.WorkerWifiLockPolicy
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

    /**
     * ★ 2026-10-08：层段/推理期间同时持有的 **WiFi lock**（延迟保障，不保吞吐）。
     *
     * 实测动机：Y700 局域网 ICMP 延迟 9–79ms（抖动 3–8×），decode 每步一个网络往返 ⇒
     * 息屏/空闲的 WiFi 省电会直接变成每一步的固定税。与 [wakeLock] 同生命周期。
     */
    private var wifiLock: WifiManager.WifiLock? = null

    /** ★ 2026-10-08：WiFi lock 的**空闲释放**调度器（见 [WorkerWifiLockPolicy.IDLE_RELEASE_MS]）。 */
    private val wifiIdleHandler: Handler by lazy { Handler(Looper.getMainLooper()) }
    private val releaseWifiLockRunnable: Runnable by lazy { Runnable { releaseWifiLock() } }

    /**
     * ★ 2026-10-07（DIST-NEXT-5b）：**活动任务数**（本地推理 / 层段前向）。
     *
     * `WorkerWakeLockPolicy.shouldHold()` 以它决定是否持有 WakeLock —— 空闲即释放，
     * 不再用固定 30 分钟硬撑（审计 P1-3 点名的那条）。
     */
    private val activeTasks = java.util.concurrent.atomic.AtomicInteger(0)
    private val binder = LocalBinder()
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
    @Volatile
    private var loadedLayerArtifactSha256: String = ""

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
        // ★ 2026-10-07（DIST-NEXT-5b）：启动时**不**持有 WakeLock（空闲不唤醒）。有任务时由
        //   `beginForegroundWork()` 获取，并按 `RENEWAL_TIMEOUT_MS` 在任务期间续租。
        renewWakeLock()
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
        releaseWifiLock()
        scope.cancel()
        super.onDestroy()
    }

    // ================================================================
    // 公开 API（供 ChatRepository 调用）
    // ================================================================

    /**
     * 确保模型已加载（如未加载则尝试加载默认路径的模型）。
     *
     * @param extractHidden ★ 2026-09-20（层段）：是否开启隐藏态导出。
     *   只有要当**中间层段**时才需要（末段只需 logits）；开启有额外内存/时间成本。
     *   ⚠️ 该选项**只能在创建 context 时**设定 ⇒ 改变它就等于需要重新加载模型。
     */
    suspend fun ensureModelLoaded(
        contextSize: Int = 2048,
        extractHidden: Boolean = false,
    ): Result<Unit> {
        val eng = engine ?: return Result.failure(IllegalStateException("Service 未初始化"))
        val gpuLayers = preferredGpuLayers(eng)

        val selectedUri = modelManager.getSelectedModelUri()
        // ★ 2026-09-20 修 BUG：短路判据原先只看「同一个 URI」。
        //   但「同一模型文件、是否导出隐藏态」是**两种不同的加载方式**，
        //   若视为等价，层段做中间段时会一直取不到 hidden（静默失败）。
        //   故把 loadedExtractHidden 一并纳入判据。
        if (eng.isLoaded && eng.loadedModelSourceUri == selectedUri &&
            eng.loadedExtractHidden == extractHidden &&
            eng.loadedLayerRange.isEmpty() &&
            eng.loadedGpuLayers == gpuLayers
        ) {
            return Result.success(Unit)
        }
        if (eng.isLoaded) {
            Log.i(
                TAG,
                "模型选择或加载方式已变化，卸载旧模型: " +
                    "${eng.loadedModelSourceUri}(hidden=${eng.loadedExtractHidden}) -> " +
                    "$selectedUri(hidden=$extractHidden)",
            )
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
        val fdResult = eng.loadModel(
            handle = handle,
            contextSize = contextSize,
            extractHidden = extractHidden,
            gpuLayers = gpuLayers,
        )
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
        return eng.loadModel(
            handle = fallbackHandle,
            contextSize = contextSize,
            extractHidden = extractHidden,
            gpuLayers = gpuLayers,
        )
    }

    /** Load the exact crop GGUF requested by an Android layer worker offer. */
    suspend fun ensureLayerModelLoaded(
        layerRange: List<Int>,
        contextSize: Int = 2048,
        extractHidden: Boolean = false,
        expectedModelSha256: String = "",
        expectedEmbeddingWidth: Int = 0,
    ): Result<Unit> {
        val eng = engine ?: return Result.failure(IllegalStateException("Service 未初始化"))
        if (layerRange.size != 2 || layerRange[1] <= layerRange[0] || layerRange[0] < 0) {
            return Result.failure(IllegalArgumentException("invalid layer range"))
        }
        val gpuLayers = preferredGpuLayers(eng)
        val expectedDigest = expectedModelSha256.trim().lowercase()
        if (eng.isLoaded && eng.loadedLayerRange == layerRange &&
            eng.loadedExtractHidden == extractHidden && eng.loadedGpuLayers == gpuLayers &&
            expectedDigest.isNotBlank() && loadedLayerArtifactSha256 == expectedDigest
        ) {
            return validateLayerModel(eng, layerRange, expectedEmbeddingWidth)
        }
        if (eng.isLoaded) eng.unloadModel()
        loadedLayerArtifactSha256 = ""

        suspend fun load(preferFd: Boolean): Result<Unit> {
            val artifact = modelManager.openLayerModelForLlama(
                layerRange = layerRange,
                expectedModelSha256 = expectedModelSha256,
                preferFd = preferFd,
            ).getOrElse { return Result.failure(it) }
            val loaded = eng.loadModel(
                handle = artifact.handle,
                contextSize = contextSize,
                extractHidden = extractHidden,
                gpuLayers = gpuLayers,
                layerRange = layerRange,
            )
            if (loaded.isFailure) return loaded
            val validation = validateLayerModel(eng, layerRange, expectedEmbeddingWidth)
            if (validation.isSuccess) {
                loadedLayerArtifactSha256 = artifact.artifactSha256.lowercase()
            }
            return validation
        }

        val first = load(preferFd = true)
        if (first.isSuccess) return first
        Log.w(TAG, "layer artifact fd load failed; retrying cached copy")
        if (eng.isLoaded) eng.unloadModel()
        loadedLayerArtifactSha256 = ""
        return load(preferFd = false)
    }

    private suspend fun validateLayerModel(
        eng: LocalInferenceEngine,
        layerRange: List<Int>,
        expectedEmbeddingWidth: Int,
    ): Result<Unit> {
        val info = eng.layerForwardInfo().getOrElse { return Result.failure(it) }
        val expectedLayers = layerRange[1] - layerRange[0]
        val actualLayers = info["n_layer"]?.toIntOrNull() ?: 0
        if (actualLayers != expectedLayers) {
            return Result.failure(
                IllegalStateException("layer_artifact_mismatch:n_layer=$actualLayers expected=$expectedLayers")
            )
        }
        if (expectedEmbeddingWidth > 0 && info["n_embd"]?.toIntOrNull() != expectedEmbeddingWidth) {
            return Result.failure(IllegalStateException("layer_artifact_mismatch:n_embd"))
        }
        return Result.success(Unit)
    }

    private suspend fun preferredGpuLayers(eng: LocalInferenceEngine): Int {
        val backend = eng.getBackendInfo().getOrNull().orEmpty()
        val devices = backend["backend_devices"].orEmpty()
        return if (backend.bool("supports_gpu_offload") &&
            (devices.contains("(gpu)") || devices.contains("(igpu)"))
        ) -1 else 0
    }

    suspend fun unloadModel(): Result<Unit> {
        val eng = engine ?: return Result.failure(IllegalStateException("Service 未初始化"))
        loadedLayerArtifactSha256 = ""
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
        // ★ DIST-NEXT-5b：多模态推理同样按任务持有 WakeLock。
        beginForegroundWork()
        try {
            return engine?.generateMultimodal(
                prompt, imageBytes, maxTokens, temperature, topP,
            ) ?: Result.failure(IllegalStateException("Service 未初始化"))
        } finally {
            endForegroundWork()
        }
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
        // ★ DIST-NEXT-5b：推理期间持有 WakeLock（按任务续租），结束即释放。
        beginForegroundWork()
        try {
            return eng.generate(prompt, maxTokens, temperature, topP)
        } finally {
            endForegroundWork()
        }
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

    /**
     * ★ 2026-10-07（DIST-NEXT-5b）：标记一个前台任务开始（本地推理入口 / 层段执行器调用）。
     */
    fun beginForegroundWork() {
        if (activeTasks.incrementAndGet() == 1) {
            renewWakeLock()
            renewWifiLock()
        }
    }

    /** ★ 2026-10-07（DIST-NEXT-5b）：标记一个前台任务结束；计数归零即释放 WakeLock。
     *
     *  ★ 2026-10-08：WiFi lock **不在这里释放** —— 它改由 [WorkerWifiLockPolicy.IDLE_RELEASE_MS]
     *  空闲超时释放（单次层段只有几十毫秒，成对开关会让锁在步与步之间全松掉）。 */
    fun endForegroundWork() {
        if (activeTasks.decrementAndGet() <= 0) {
            activeTasks.set(0)
            releaseWakeLock()
        }
    }

    /**
     * 按当前任务数获取 / 续租 / 释放 WakeLock（幂等）。
     *
     * 空闲 ⇒ 释放；有任务 ⇒ 以 `RENEWAL_TIMEOUT_MS` 持有。重复 `acquire` 会累加引用计数，
     * 因此续租前先释放已持有的那一份，保持"单次持有"。
     */
    @Suppress("DEPRECATION")
    private fun renewWakeLock() {
        if (!WorkerWakeLockPolicy.shouldHold(activeTasks.get())) {
            releaseWakeLock()
            return
        }
        val powerManager = getSystemService(Context.POWER_SERVICE) as PowerManager
        val lock = wakeLock ?: powerManager.newWakeLock(
            PowerManager.PARTIAL_WAKE_LOCK,
            "QLH:InferenceEngine",
        ).also { wakeLock = it }
        if (lock.isHeld) lock.release()
        lock.acquire(WorkerWakeLockPolicy.RENEWAL_TIMEOUT_MS)
    }

    private fun releaseWakeLock() {
        wakeLock?.let {
            if (it.isHeld) it.release()
        }
        wakeLock = null
    }

    /**
     * ★ 2026-10-08：WiFi lock 与 WakeLock **成对**获取/释放（同一 `activeTasks` 判据）。
     *
     * 取 `WIFI_MODE_FULL_LOW_LATENCY`（API 29+）而不是 `WIFI_MODE_FULL_HIGH_PERF`：
     * decode 每步是一个"小包往返、延迟敏感、吞吐无所谓"的负载，前者正是为该场景新增的模式；
     * 低版本回退到 `HIGH_PERF`。锁不做引用计数，`acquire/release` 以 `isHeld` 幂等。
     */
    @Suppress("DEPRECATION")
    private fun renewWifiLock() {
        if (!WorkerWifiLockPolicy.shouldHold(activeTasks.get())) {
            releaseWifiLock()
            return
        }
        val wifiManager = applicationContext.getSystemService(Context.WIFI_SERVICE) as? WifiManager
            ?: return
        val lock = wifiLock ?: wifiManager.createWifiLock(
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                WifiManager.WIFI_MODE_FULL_LOW_LATENCY
            } else {
                WifiManager.WIFI_MODE_FULL_HIGH_PERF
            },
            WorkerWifiLockPolicy.LOCK_TAG,
        ).also {
            // 引用计数必须"未 acquire 前"设置；设 false 后 acquire/release 语义即幂等持有。
            it.setReferenceCounted(false)
            wifiLock = it
        }
        if (!lock.isHeld) {
            lock.acquire()
            Log.i(
                TAG,
                "wifi lock acquired tag=${WorkerWifiLockPolicy.LOCK_TAG} " +
                    "mode=${if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) "full_low_latency" else "full_high_perf"} " +
                    "idleReleaseMs=${WorkerWifiLockPolicy.IDLE_RELEASE_MS}",
            )
        }
        // ★ 续租 + 空闲超时释放：连续工作的每一步都会重置窗口，真空闲才放手。
        //   （只在获取/释放时打日志：每步续租不打，避免刷屏 —— 续租只重置定时器，不重复 acquire。）
        wifiIdleHandler.removeCallbacks(releaseWifiLockRunnable)
        wifiIdleHandler.postDelayed(
            releaseWifiLockRunnable,
            WorkerWifiLockPolicy.IDLE_RELEASE_MS,
        )
    }

    private fun releaseWifiLock() {
        if (wifiLock != null) Log.i(TAG, "wifi lock released (idle timeout or service stop)")
        wifiIdleHandler.removeCallbacks(releaseWifiLockRunnable)
        wifiLock?.let {
            if (it.isHeld) it.release()
        }
        wifiLock = null
    }
}

private fun Map<String, String>.bool(key: String): Boolean = this[key] == "true"

private fun Map<String, String>.int(key: String): Int = this[key]?.toIntOrNull() ?: 0

private fun Map<String, String>.long(key: String): Long = this[key]?.toLongOrNull() ?: 0L

private fun Map<String, String>.double(key: String): Double = this[key]?.toDoubleOrNull() ?: 0.0
