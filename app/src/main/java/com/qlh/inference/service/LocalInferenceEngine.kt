package com.qlh.inference.service

import android.content.Context
import android.os.Build
import android.util.Log
import android.system.Os
import android.system.OsConstants
import android.system.StructStat
import com.qlh.inference.status.ReleaseProcessTree
import com.qlh.inference.status.RuntimeProcessObservations
import com.qlh.inference.worker.MultiPositionArgmaxContract
import com.qlh.inference.worker.WorkerPrepareIdentity
import com.qlh.inference.worker.WorkerPrepareException
import com.qlh.inference.worker.GgufLayerWeightReader
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.ensureActive
import java.io.FileInputStream
import com.qlh.inference.status.ReleaseRuntimeProbe
import com.qlh.inference.status.ReleaseRuntimeProbe.ErrorCode
import com.qlh.inference.worker.TaskWorkerSnapshot
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.channels.awaitClose
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.callbackFlow
import kotlinx.coroutines.flow.flowOn
import kotlinx.coroutines.withContext
import java.io.File

internal object RuntimePrepareFailure {
    enum class Stage(val code: String) {
        SUPPLY("prepare_supply_check_failed"), VERIFY("prepare_artifact_verification_failed"),
        OPEN("prepare_artifact_open_failed"), LOAD("prepare_native_load_failed"),
        VALIDATE("prepare_native_validation_failed"), RESOLVE("prepare_receipt_artifact_resolution_failed"),
        MANIFEST("prepare_receipt_manifest_read_failed"), STATE("prepare_receipt_runtime_mismatch"),
        STAMP("prepare_receipt_artifact_changed"), METADATA("prepare_receipt_metadata_invalid"),
        COUNT("prepare_receipt_native_count_mismatch"), BIND("prepare_receipt_bind_failed"),
    }

    /** Never include exception messages, URIs or manifest content in diagnostics. */
    fun diagnostic(stage: Stage, error: Exception): String {
        val locations = error.stackTrace.take(6).joinToString(",") {
            "${it.className}.${it.methodName}:${it.lineNumber}"
        }
        return "worker prepare failed stage=${stage.name} code=${stage.code} exception=${error.javaClass.simpleName} at=$locations"
    }

    fun exception(stage: Stage, error: Exception): WorkerPrepareException =
        (if (error is WorkerPrepareException && stage != Stage.METADATA) error
        else WorkerPrepareException(stage.code, "worker prepare failed at ${stage.name}").also { it.initCause(error) })
}

/**
 * 本地推理引擎 — llama.cpp JNI 封装。
 *
 * 职责：
 * 1. 加载 QLH llama.cpp JNI 桥接库（libqlh_llama_jni.so）
 * 2. 加载 GGUF 模型文件
 * 3. 执行 tokenize → generate → detokenize 推理循环
 * 4. 通过 Flow 逐 token 推送生成结果
 *
 * 线程模型：
 * - 加载/卸载：调用方线程
 * - 推理：Dispatchers.IO 线程池
 * - 回调：Flow collect 线程
 *
 * JNI 约定（libqlh_llama_jni.so 需实现以下 native 方法）：
 * - nativeLoadModel(path: String, nCtx: Int): Long       → 返回模型指针
 * - nativeFreeModel(modelPtr: Long)
 * - nativeGenerate(modelPtr: Long, prompt: String,
 *       maxTokens: Int, temperature: Float, topP: Float,
 *       onToken: (String) -> Unit): String               → 返回完整文本
 */
class LocalInferenceEngine(private val context: Context) {

    // Native model pointers and retained handles must outlive every native call.
    private val modelMutex = Mutex()
    private var closed = false
    private data class ArtifactStamp(val device: Long, val inode: Long, val size: Long,
        val mtimeSeconds: Long, val mtimeNanos: Long, val ctimeSeconds: Long, val ctimeNanos: Long)
    private fun artifactStamp(stat: StructStat): ArtifactStamp {
        require(OsConstants.S_ISREG(stat.st_mode) && stat.st_size > 0)
        return if (Build.VERSION.SDK_INT >= 27) ArtifactStamp(stat.st_dev, stat.st_ino, stat.st_size,
            stat.st_mtim.tv_sec, stat.st_mtim.tv_nsec, stat.st_ctim.tv_sec, stat.st_ctim.tv_nsec)
        else ArtifactStamp(stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime, 0, stat.st_ctime, 0)
    }
    private fun artifactStamp(path: String): ArtifactStamp? = runCatching { artifactStamp(Os.stat(path)) }.getOrNull()
    private var loadedArtifactStamp: ArtifactStamp? = null
    @Volatile private var loadedPrepareIdentity: WorkerPrepareIdentity? = null
    private var loadedPreparedMetadata: ReleaseRuntimeProbe.PreparedMetadata? = null


    private suspend fun <T> withModelContext(action: suspend CoroutineScope.() -> T): T =
        withContext(Dispatchers.IO) {
            modelMutex.withLock {
                check(!closed) { "推理引擎已停止" }
                action()
            }
        }

    companion object {
        private const val TAG = "LocalInference"
        private const val DEFAULT_CONTEXT_SIZE = 2048

        private const val NATIVE_LIBRARY_NAME = "qlh_llama_jni"

        // 推荐的 GGUF 文件名
        const val MODEL_FILENAME = "qwen-1.8b-Q4_K_M.gguf"
        const val MODEL_SHA256_FILENAME = "qwen-1.8b-Q4_K_M.gguf.sha256"

        @Volatile
        private var nativeLoaded = false
    }

    /** 当前模型指针（0 = 未加载） */
    @Volatile
    private var modelPtr: Long = 0

    /** 当前模型路径（用于状态查询） */
    @Volatile
    var loadedModelPath: String = ""
        private set

    /** 当前模型来源 URI，用于判断 UI 选择的模型是否已经切换。 */
    @Volatile
    var loadedModelSourceUri: String = ""
        private set

    /**
     * ★ 2026-09-20（层段）：当前模型加载时是否开启了隐藏态导出。
     *
     * 必须记录 —— 层段做**中间段**（要输出 hidden）时，模型**必须**以
     * `extractHidden = true` 加载。而「同一个模型文件、两次加载方式不同」在
     * 卸载判据里必须被区分开，否则会出现「模型已加载但 hidden 取不到」的静默失败。
     */
    @Volatile
    var loadedExtractHidden: Boolean = false
        private set

    /** Source-model range represented by the loaded GGUF; empty means full model. */
    @Volatile
    var loadedLayerRange: List<Int> = emptyList()
        private set

    /** llama.cpp offload setting used for the loaded model (-1 means all layers). */
    @Volatile
    var loadedGpuLayers: Int = 0
        private set

    /** Whether a verified/loaded MTMD projector is attached to the model. */
    @Volatile
    var multimodalLoaded: Boolean = false
        private set

    /**
     * ★ 2026-10-07（DIST-NEXT-1）：层段前向的取消标志。
     *
     * 层段前向是一次 `llama_decode` 原子调用，Kotlin 的协程取消打断不了它。
     * 取消合同因此把它做成显式标志：native 侧 `llama_set_abort_callback` 读该
     * 标志，在下一个可分割的张量边界退出，JNI 返回 -4（`layer_forward_aborted`）。
     */
    @Volatile
    private var layerForwardAbortRequested = false

    /** 请求中止正在执行的层段前向（幂等；未加载模型时只置标志）。 */
    fun requestLayerForwardAbort() {
        layerForwardAbortRequested = true
        if (nativeLoaded) {
            runCatching { nativeSetLayerForwardAbort(true) }
        }
    }

    /** 清除层段前向的取消标志（每次进入 `layerForward` 前调用；幂等）。 */
    fun clearLayerForwardAbort() {
        layerForwardAbortRequested = false
        if (nativeLoaded) {
            runCatching { nativeSetLayerForwardAbort(false) }
        }
    }

    /** SAF fd 或缓存加载句柄。模型卸载前必须保持 fd 存活。 */
    private var modelOpenHandle: ModelManager.ModelOpenHandle? = null

    /** mmproj fd/cache handle retained until the native projector is released. */
    private var multimodalOpenHandle: ModelManager.ModelOpenHandle? = null

    /** 引擎是否就绪 */
    val isLoaded: Boolean get() = modelPtr != 0L && nativeLoaded

    // ================================================================
    // 原生库加载
    // ================================================================

    /**
     * 加载 QLH llama.cpp JNI 桥接库（首次调用时自动执行）。
     */
    @Synchronized
    fun ensureNativeLibraryLoaded(): Result<Unit> {
        if (nativeLoaded) return Result.success(Unit)
        return try {
            System.loadLibrary(NATIVE_LIBRARY_NAME)
            nativeLoaded = true
            Log.i(TAG, "llama.cpp JNI 桥接库加载成功: lib$NATIVE_LIBRARY_NAME.so")
            Result.success(Unit)
        } catch (e: UnsatisfiedLinkError) {
            Log.e(TAG, "llama.cpp JNI 桥接库加载失败: ${e.message}", e)
            Result.failure(
                IllegalStateException(
                    "当前 APK 未包含可用的本地推理运行时，或设备 ABI 不受支持。\n" +
                        "请安装包含 llama.cpp 本地运行时的 full-local APK，" +
                        "或切换到全无模式连接 PC 主节点。\n" +
                        "设备 ABI: ${Build.SUPPORTED_ABIS.joinToString()}\n" +
                        "缺失库: lib$NATIVE_LIBRARY_NAME.so\n" +
                        "原始错误: ${e.message}"
                )
            )
        }
    }

    fun checkNativeRuntime(): Result<Unit> = ensureNativeLibraryLoaded()

    // ================================================================
    // 模型管理
    // ================================================================

    /**
     * 加载普通文件路径的 GGUF 模型到内存。
     *
     * SAF 模型必须使用 [loadModel(ModelManager.ModelOpenHandle, Int)]，否则 fd 生命周期无法被持有。
     *
     * @param modelPath GGUF 文件绝对路径
     * @param contextSize 上下文长度（默认 2048）
     */
    @Deprecated(
        message = "Use loadModel(ModelOpenHandle) for SAF-aware loading; this overload is only for internal test files.",
        replaceWith = ReplaceWith("loadModel(handle, contextSize)")
    )
    suspend fun loadModel(
        modelPath: String,
        contextSize: Int = DEFAULT_CONTEXT_SIZE,
        extractHidden: Boolean = false,
        gpuLayers: Int = 0,
        layerRange: List<Int>? = null,
    ): Result<Unit> = withModelContext {
        val libResult = ensureNativeLibraryLoaded()
        if (libResult.isFailure) {
            return@withModelContext Result.failure(libResult.exceptionOrNull()!!)
        }

        // 卸载旧模型
        if (modelPtr != 0L) {
            unloadModelInternal()
        }

        val file = File(modelPath)
        if (!file.exists()) {
            return@withModelContext Result.failure(
                IllegalStateException("模型文件不存在: $modelPath")
            )
        }
        if (!file.canRead()) {
            return@withModelContext Result.failure(
                IllegalStateException("模型文件不可读: $modelPath")
            )
        }

        try {
            val beforeLoad = artifactStamp(modelPath)
            val ptr = nativeLoadModel(modelPath, contextSize, extractHidden, gpuLayers)
            if (ptr == 0L) {
                return@withModelContext Result.failure(
                    IllegalStateException("模型加载失败（native 返回空指针）: $modelPath")
                )
            }
            modelPtr = ptr
            loadedArtifactStamp = beforeLoad?.takeIf { it == artifactStamp(modelPath) }
            loadedPrepareIdentity = null
            loadedPreparedMetadata = null
            loadedModelPath = modelPath
            loadedModelSourceUri = ""
            loadedExtractHidden = extractHidden
            loadedLayerRange = layerRange.orEmpty()
            loadedGpuLayers = gpuLayers
            Log.i(TAG, "模型加载成功: $modelPath (context=$contextSize, ptr=$ptr)")
            Result.success(Unit)
        } catch (e: Exception) {
            Log.e(TAG, "模型加载异常: $modelPath", e)
            Result.failure(e)
        }
    }

    /**
     * 从 ModelManager 提供的加载句柄加载模型。
     *
     * SAF fd 模式下 loadPath 形如 /proc/self/fd/<fd>，必须把 handle 持有到卸载模型时。
     */
    suspend fun loadModel(
        handle: ModelManager.ModelOpenHandle,
        contextSize: Int = DEFAULT_CONTEXT_SIZE,
        extractHidden: Boolean = false,
        gpuLayers: Int = 0,
        layerRange: List<Int>? = null,
    ): Result<Unit> = withModelContext {
        val libResult = ensureNativeLibraryLoaded()
        if (libResult.isFailure) {
            handle.close()
            return@withModelContext Result.failure(libResult.exceptionOrNull()!!)
        }

        if (modelPtr != 0L) {
            unloadModelInternal()
        }

        val modelPath = handle.loadPath
        if (!modelPath.startsWith("/proc/self/fd/")) {
            val file = File(modelPath)
            if (!file.exists()) {
                handle.close()
                return@withModelContext Result.failure(
                    IllegalStateException("模型文件不存在: $modelPath")
                )
            }
            if (!file.canRead()) {
                handle.close()
                return@withModelContext Result.failure(
                    IllegalStateException("模型文件不可读: $modelPath")
                )
            }
        }

        try {
            val beforeLoad = artifactStamp(modelPath)
            val ptr = nativeLoadModel(modelPath, contextSize, extractHidden, gpuLayers)
            if (ptr == 0L) {
                handle.close()
                return@withModelContext Result.failure(
                    IllegalStateException("模型加载失败（native 返回空指针）: ${handle.displayName}")
                )
            }
            modelPtr = ptr
            loadedArtifactStamp = beforeLoad?.takeIf { it == artifactStamp(modelPath) }
            loadedPrepareIdentity = null
            loadedPreparedMetadata = null
            loadedModelPath = modelPath
            loadedModelSourceUri = handle.sourceUri.toString()
            loadedExtractHidden = extractHidden
            loadedLayerRange = layerRange.orEmpty()
            loadedGpuLayers = gpuLayers
            modelOpenHandle = handle
            Log.i(
                TAG,
                "模型加载成功: ${handle.displayName} (${handle.mode}, context=$contextSize, ptr=$ptr)"
            )
            Result.success(Unit)
        } catch (e: Exception) {
            handle.close()
            Log.e(TAG, "模型加载异常: ${handle.displayName}", e)
            Result.failure(e)
        }
    }

    /**
     * 卸载当前模型，释放内存。
     */
    suspend fun unloadModel(): Result<Unit> = withModelContext {
        try {
            unloadModelInternal()
            Result.success(Unit)
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    /** Load the user-owned Gemma4 mmproj beside the already loaded text model. */
    suspend fun loadMultimodalProjector(handle: ModelManager.ModelOpenHandle): Result<Unit> =
        withModelContext {
            if (!isLoaded) {
                handle.close()
                return@withModelContext Result.failure(IllegalStateException("模型未加载"))
            }
            try {
                val previousHandle = multimodalOpenHandle
                val loaded = nativeLoadMultimodalProjector(modelPtr, handle.loadPath)
                previousHandle?.close()
                multimodalOpenHandle = null
                if (!loaded) {
                    handle.close()
                    multimodalLoaded = false
                    Result.failure(IllegalStateException("mmproj 加载失败"))
                } else {
                    multimodalOpenHandle = handle
                    multimodalLoaded = true
                    Result.success(Unit)
                }
            } catch (error: Exception) {
                handle.close()
                multimodalLoaded = false
                Result.failure(error)
            }
        }

    /** Run bounded in-memory images through the loaded MTMD projector. */
    suspend fun generateMultimodal(
        prompt: String,
        imageBytes: List<ByteArray>,
        maxTokens: Int = 1024,
        temperature: Float = 0.7f,
        topP: Float = 0.9f,
    ): Result<String> = withModelContext {
        if (!isLoaded || !multimodalLoaded) {
            return@withModelContext Result.failure(IllegalStateException("mmproj 未加载"))
        }
        if (imageBytes.isEmpty()) {
            return@withModelContext Result.failure(IllegalArgumentException("至少需要一张图片"))
        }
        try {
            val markedPrompt = if (prompt.contains("<__media__>")) {
                prompt
            } else {
                "<__media__>\n$prompt"
            }
            Result.success(
                nativeGenerateMultimodal(
                    modelPtr,
                    markedPrompt,
                    imageBytes.toTypedArray(),
                    maxTokens,
                    temperature,
                    topP,
                ) { _ -> },
            )
        } catch (error: Exception) {
            Log.e(TAG, "多模态推理失败", error)
            Result.failure(error)
        }
    }

    private fun unloadModelInternal() {
        if (modelPtr != 0L) {
            try {
                nativeFreeModel(modelPtr)
                Log.i(TAG, "模型已卸载: $loadedModelPath")
            } catch (e: Exception) {
                Log.w(TAG, "模型卸载异常: ${e.message}")
                throw e
            }
            modelPtr = 0
            loadedArtifactStamp = null
            loadedPrepareIdentity = null
            loadedPreparedMetadata = null
            loadedModelPath = ""
            loadedModelSourceUri = ""
            loadedExtractHidden = false
            loadedLayerRange = emptyList()
            loadedGpuLayers = 0
        }
        modelOpenHandle?.close()
        modelOpenHandle = null
        multimodalOpenHandle?.close()
        multimodalOpenHandle = null
        multimodalLoaded = false
    }

    // ================================================================
    // 推理（流式）
    // ================================================================

    /**
     * 执行推理并以 Flow 逐 token 推送结果。
     *
     * @param prompt 输入文本（原始用户消息，不含 chat template）
     * @param maxTokens 最大生成 token 数
     * @param temperature 温度（0 为贪心）
     * @param topP nucleus sampling
     * @return Flow<String> — 每个 emit 为一个增量文本 chunk
     */
    fun generateStream(
        prompt: String,
        maxTokens: Int = 1024,
        temperature: Float = 0.7f,
        topP: Float = 0.9f
    ): Flow<String> = callbackFlow {
        modelMutex.withLock {
            check(!closed) { "推理引擎已停止" }
            if (!isLoaded) {
                throw IllegalStateException("模型未加载，请先调用 loadModel()")
            }

            // nativeGenerate 在 JNI 线程中回调，通过 trySend 发送到 Flow
            nativeGenerate(
                modelPtr, prompt, maxTokens, temperature, topP
            ) { token ->
                trySend(token)
            }
        }
        close()
        awaitClose {}
    }.flowOn(Dispatchers.IO)

    /**
     * 执行推理并返回完整结果（非流式）。
     */
    suspend fun generate(
        prompt: String,
        maxTokens: Int = 1024,
        temperature: Float = 0.7f,
        topP: Float = 0.9f
    ): Result<String> = withModelContext {
        if (!isLoaded) {
            return@withModelContext Result.failure(IllegalStateException("模型未加载"))
        }
        try {
            val fullText = nativeGenerate(
                modelPtr, prompt, maxTokens, temperature, topP
            ) { _ -> /* 非流式 — 忽略逐 token 回调 */ }
            Result.success(fullText)
        } catch (e: Exception) {
            Log.e(TAG, "推理失败", e)
            Result.failure(e)
        }
    }

    // ================================================================
    // 资源管理
    // ================================================================

    /**
     * 获取模型信息（从 GGUF 元数据）。
     */
    suspend fun getModelInfo(): Result<Map<String, String>> = withModelContext {
        if (!isLoaded) {
            return@withModelContext Result.failure(IllegalStateException("模型未加载"))
        }
        try {
            val info = nativeGetModelInfo(modelPtr)
            Result.success(info)
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    suspend fun getBackendInfo(): Result<Map<String, String>> = withModelContext {
        val libResult = ensureNativeLibraryLoaded()
        if (libResult.isFailure) {
            return@withModelContext Result.failure(libResult.exceptionOrNull()!!)
        }
        try {
            Result.success(nativeGetBackendInfo())
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    /** Called only after ModelManager verified this exact artifact and JNI load succeeded. */
    internal suspend fun bindPreparedRuntime(identity: WorkerPrepareIdentity, sourceUri: String,
        artifactName: String, nativeManifest: String): Result<Unit> = withModelContext {
        var stage = RuntimePrepareFailure.Stage.STATE
        try {
            require(isLoaded && loadedModelSourceUri == sourceUri && loadedLayerRange == identity.range)
            require(loadedExtractHidden == (identity.mode != "tail"))
            stage = RuntimePrepareFailure.Stage.STAMP
            require(loadedArtifactStamp != null && loadedArtifactStamp == artifactStamp(loadedModelPath))
            stage = RuntimePrepareFailure.Stage.METADATA
            val metadata = ReleaseRuntimeProbe.PreparedMetadata.parse(identity, nativeManifest, artifactName)
            stage = RuntimePrepareFailure.Stage.COUNT
            val actualCount = nativeGetModelInfo(modelPtr)["n_layer"]?.toIntOrNull()
            Log.i(TAG, "worker prepare receipt native_layers=$actualCount expected_layers=${metadata.decoderLayers}")
            require(actualCount == metadata.decoderLayers)
            loadedPreparedMetadata = metadata
            loadedPrepareIdentity = identity
            Log.i(TAG, "worker prepare receipt bound range=${metadata.range} generation=${identity.generation}")
            Result.success(Unit)
        } catch (cancelled: CancellationException) { throw cancelled }
          catch (error: Exception) {
              Log.e(TAG, RuntimePrepareFailure.diagnostic(stage, error))
              Result.failure(RuntimePrepareFailure.exception(stage, error))
          }
    }

    internal suspend fun deploymentReceipt(identity: WorkerPrepareIdentity, spec: com.google.gson.JsonObject): com.google.gson.JsonObject = modelMutex.withLock {
        check(isLoaded && loadedPrepareIdentity == identity && loadedGpuLayers == 0)
        val info = nativeGetModelInfo(modelPtr)
        check(info["n_ctx"]?.toIntOrNull() == spec["context_size"].asInt)
        check(info["n_batch"]?.toIntOrNull() == spec["batch_size"].asInt)
        val bytes = requireNotNull(info["materialized_bytes"]?.toLongOrNull())
        check(bytes > 0)
        com.google.gson.JsonObject().apply {
            add("load_spec", spec.deepCopy()); addProperty("materialized_bytes", bytes)
            addProperty("memory_scope", "weight_tensors"); add("layer_range", identity.wire()["layer_range"].deepCopy())
            addProperty("artifact_sha256", identity.text("artifact_sha256"))
        }
    }

    internal fun matchesPreparedIdentity(identity: WorkerPrepareIdentity): Boolean = loadedPrepareIdentity == identity

    internal fun loadedPreparedIdentity(): WorkerPrepareIdentity? = loadedPrepareIdentity

    /** Reopen the retained load path, not a selected document or a new inventory result. */
    private fun runtimeReceipt(identity: WorkerPrepareIdentity?, count: Int?, check: () -> Unit): Map<String, Any?> {
        require(identity != null && identity == loadedPrepareIdentity)
        val metadata = requireNotNull(loadedPreparedMetadata)
        val stamp = requireNotNull(loadedArtifactStamp)
        require(metadata.range == loadedLayerRange && metadata.decoderLayers == count)
        require(stamp == artifactStamp(loadedModelPath))
        val path = File(loadedModelPath).canonicalPath
        require(path.startsWith('/') && !path.endsWith(" (deleted)"))
        val scan = FileInputStream(loadedModelPath).use { input ->
            require(artifactStamp(Os.fstat(input.fd)) == stamp)
            GgufLayerWeightReader.scan(input, metadata.decoderLayers)
        }
        require(scan != null && GgufMetadataContract.matches(scan.ggufMetadata,
            metadata.contract["gguf_metadata"], metadata.decoderLayers))
        require(GgufLayerWeightReader.matchesTokenizerSha256(identity.text("tokenizer_sha256"), scan))
        val digest = FileInputStream(loadedModelPath).use { input ->
            require(artifactStamp(Os.fstat(input.fd)) == stamp)
            val digest = ReleaseRuntimeProbe.rehash(input, stamp.size, check)
            require(artifactStamp(Os.fstat(input.fd)) == stamp)
            digest
        }
        check()
        require(stamp == artifactStamp(loadedModelPath) && digest == metadata.sha256)
        val pid = android.os.Process.myPid()
        return linkedMapOf(
            "schema_version" to 1, "source" to "committed_worker_prepare", "engine" to "llama_cpp",
            "pid" to pid, "loader_pid" to pid,
            "operation_id" to identity.text("operation_id"), "config_id" to identity.text("config_id"),
            "generation" to identity.generation, "node_id" to identity.text("node_id"),
            "assignment_id" to identity.text("assignment_id"), "loaded" to true, "full_model_loaded" to false,
            "layer_range" to metadata.range, "source_layer_range" to metadata.range,
            "range_source" to "verified_loaded_native_manifest", "total_layers" to metadata.total,
            "native_layer_count" to count, "captured_at" to ReleaseRuntimeProbe.unixSeconds(),
            "artifact" to linkedMapOf("path" to path, "load_path" to loadedModelPath,
                "sha256" to metadata.sha256, "rehashed_sha256" to digest, "size_bytes" to stamp.size,
                "layer_range" to metadata.range, "decoder_layers" to count,
                "native_manifest_sha256" to metadata.manifestSha256,
                "device" to stamp.device, "inode" to stamp.inode),
        )
    }

    /** One actual JNI/model generation; never call getBackendInfo/getModelInfo under this mutex. */
    internal suspend fun releaseRuntimeProbe(
        readMode: suspend () -> String,
        readWorker: () -> TaskWorkerSnapshot,
        readCommitted: () -> WorkerPrepareIdentity? = { null },
        product: Boolean = false,
    ): ReleaseRuntimeProbe.Snapshot = withContext(Dispatchers.IO) {
        modelMutex.withLock {
            val probeContext = currentCoroutineContext()
            val snapshot = ReleaseRuntimeProbe.capture(readMode, readWorker, readCommitted) {
                if (closed) {
                    ReleaseRuntimeProbe.RuntimeState(
                        ReleaseRuntimeProbe.NativeState(errorCode = ErrorCode.ENGINE_CLOSED),
                    )
                } else {
                    val loaded = modelPtr != 0L
                    val range = loadedLayerRange.toList()
                    val backend = try {
                        if (ensureNativeLibraryLoaded().isFailure) {
                            ReleaseRuntimeProbe.NativeState(errorCode = ErrorCode.NATIVE_LIBRARY_UNAVAILABLE)
                        } else {
                            ReleaseRuntimeProbe.NativeState.fromBackend(nativeGetBackendInfo())
                        }
                    } catch (cancelled: CancellationException) {
                        throw cancelled
                    } catch (_: LinkageError) {
                        ReleaseRuntimeProbe.NativeState(errorCode = ErrorCode.NATIVE_QUERY_FAILED)
                    } catch (_: Exception) {
                        ReleaseRuntimeProbe.NativeState(errorCode = ErrorCode.NATIVE_QUERY_FAILED)
                    }
                    var modelError: ErrorCode? = null
                    val count = if (loaded && nativeLoaded) {
                        try {
                            // Read the retained pointer directly while holding modelMutex.
                            nativeGetModelInfo(modelPtr)["n_layer"]?.toIntOrNull()?.takeIf { it > 0 }
                                .also { if (it == null) modelError = ErrorCode.MODEL_INFO_FAILED }
                        } catch (cancelled: CancellationException) {
                            throw cancelled
                        } catch (_: LinkageError) {
                            modelError = ErrorCode.MODEL_INFO_FAILED
                            null
                        } catch (_: Exception) {
                            modelError = ErrorCode.MODEL_INFO_FAILED
                            null
                        }
                    } else null
                    var receipt: Map<String, Any?>? = null
                    if (product && loaded && range.isNotEmpty()) {
                        val owner = readCommitted()
                        if (owner == null || owner != loadedPrepareIdentity) {
                            modelError = ErrorCode.RUNTIME_RECEIPT_UNAVAILABLE
                        } else try {
                            receipt = runtimeReceipt(owner, count) { probeContext.ensureActive() }
                        } catch (cancelled: CancellationException) { throw cancelled }
                          catch (_: Exception) { modelError = ErrorCode.ARTIFACT_REHASH_FAILED }
                    }
                    val process = if (product) ReleaseProcessTree.captureCurrent { nativeReadTcpState(it) } else RuntimeProcessObservations.capture()
                    val receiptWithIdentity = receipt?.let {
                        it + mapOf("created_at" to process["created_at"], "package_name" to context.packageName)
                    }
                    if (receiptWithIdentity != null && process["created_at"] == null) modelError = ErrorCode.RUNTIME_RECEIPT_INVALID
                    ReleaseRuntimeProbe.RuntimeState(
                        native = backend,
                        model = ReleaseRuntimeProbe.ModelState(loaded, range, count),
                        errorCode = backend.errorCode ?: modelError,
                        process = process,
                        receipt = receiptWithIdentity,
                    )
                }
            }
            snapshot.copy(product = product)
        }
    }

    /** Query the compiled multimodal bridge; text-only builds must report false. */
    suspend fun getMultimodalCapability(): Result<Map<String, String>> = withModelContext {
        val libResult = ensureNativeLibraryLoaded()
        if (libResult.isFailure) {
            return@withModelContext Result.failure(libResult.exceptionOrNull()!!)
        }
        try {
            Result.success(nativeGetMultimodalCapability(modelPtr))
        } catch (e: Throwable) {
            Result.failure(e)
        }
    }

    suspend fun getLastGenerationStats(): Result<Map<String, String>> = withModelContext {
        if (!isLoaded) {
            return@withModelContext Result.success(emptyMap())
        }
        try {
            Result.success(nativeGetLastGenerationStats(modelPtr))
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    // ------------------- 层段（layer_forward）-------------------

    /**
     * 层段前向的输出。`hiddenOut` 仅在请求了「中间段」（[wantHidden]=true）且
     * 本节点在 load 时开启了隐藏态导出时有值。
     */
    data class LayerForwardOutput(
        /** 末位置 argmax token。 */
        val tokenArgmax: Int,
        /** 末位置输出 hidden（本节点为中间段时），否则为 null。 */
        val hiddenOut: FloatArray?,
        val tokenArgmaxByPosition: List<Int>? = null,
        val positions: List<Int>? = null,
        val nVocab: Int? = null,
    ) {
        override fun equals(other: Any?): Boolean = this === other
        override fun hashCode(): Int = System.identityHashCode(this)
    }

    /**
     * 层段前向：把上游 hidden 注入本节点的**裁层 GGUF**，只算自己的层区间。
     *
     * 这是「Android 参与主仓层流水线」的执行入口，对应任务协议 v3 的
     * `layer_forward` stage。**本方法只管算，不管协议** —— 层区间/handoff 的
     * 合法性、模型身份对账由 `worker` 层的 `AndroidFullWorkerStageExecutor` 负责。
     *
     * @param hidden f32 hidden 平铺数组，长度必须恰为 `nTokens * n_embd`
     * @param nTokens 注入的 token 数
     * @param posBase 位置起点（每步可用绝对位置累加）
     * @param wantHidden 本节点是否为**中间段**（需要把输出 hidden 交给下一段）。
     *   若为 true 但模型未以 `extractHidden=true` 加载 ⇒ **失败**（不返回空 hidden）。
     * @param keepHead ★ 中间段通道选择（默认 false，保持向后兼容）：
     *   - `false` —— `extract_hidden`（embeddings 通道）。该通道返回的是
     *     **`output_norm(H)`**，比层接力所需的 hidden **多一次归一化**，与主仓 D→L / L→L
     *     的判据不一致；只应用于历史对照。
     *   - `true` —— **keep-head（nextn）通道**，取末层输出（`output_norm` 之前），与主仓
     *     `scripts/model_tools/keep_head_shim` 同语义。三段链路与跨框架接力必须用这个。
     */
    suspend fun layerForward(
        hidden: FloatArray,
        nTokens: Int,
        posBase: Int,
        wantHidden: Boolean,
        keepHead: Boolean = false,
        // ★ P3 多序列：显式 seq/pos（长度必须等于 nTokens）；都给时走 Seq 版 JNI 入口
        seqIds: IntArray? = null,
        positions: IntArray? = null,
        tokenOutput: String? = null,
    ): Result<LayerForwardOutput> = withModelContext {
        if (!isLoaded) {
            return@withModelContext Result.failure(IllegalStateException("model_not_loaded"))
        }
        // ★ 2026-10-07（DIST-NEXT-1）：取消已经到达 ⇒ 直接失败，不做任何计算。
        //   （标志必须在这里清掉，否则会污染下一次前向。）
        if (layerForwardAbortRequested) {
            clearLayerForwardAbort()
            return@withModelContext Result.failure(IllegalStateException("layer_forward_aborted"))
        }
        if (nTokens <= 0) {
            return@withModelContext Result.failure(IllegalArgumentException("nTokens must be positive"))
        }
        if ((seqIds != null && seqIds.size != nTokens) ||
            (positions != null && positions.size != nTokens)
        ) {
            return@withModelContext Result.failure(
                IllegalArgumentException("seqIds and positions must each have nTokens entries when provided")
            )
        }
        // ★ 2026-10-07（DIST-NEXT-1）：进入 native 前清标志 —— native 侧
        //   `llama_set_abort_callback` 读的就是它。
        clearLayerForwardAbort()
        try {
            if (tokenOutput != null) {
                MultiPositionArgmaxContract.validateRequest(mapOf(
                    "stage_type" to "layer_forward", "token_output" to tokenOutput,
                    "hidden_spec" to mapOf("n_tokens" to nTokens),
                    "root_input" to mapOf("want_hidden" to wantHidden),
                    "positions" to positions?.toList(),
                ) + (seqIds?.let { mapOf("seq_ids" to it.toList()) } ?: emptyMap()))
                check(supportsMultiPositionArgmax()) { "multi_position_argmax_unsupported" }
                val tokens = IntArray(nTokens)
                val nVocab = nativeLayerForwardTokens(modelPtr, hidden, nTokens, seqIds,
                    requireNotNull(positions), tokens)
                check(nVocab != -4) { "layer_forward_aborted" }
                check(nVocab > 0) { "layer_forward_all_positions_failed ($nVocab)" }
                val output = mapOf("token_argmax" to tokens.last(),
                    "token_argmax_by_position" to tokens.toList(), "positions" to positions.toList(),
                    "n_vocab" to nVocab, "n_tokens" to nTokens)
                MultiPositionArgmaxContract.validateResult(output)
                Result.success(LayerForwardOutput(tokens.last(), null, tokens.toList(), positions.toList(), nVocab))
            } else if (wantHidden) {
                val embeddingWidth = estimateEmbeddingWidth(hidden.size, nTokens)
                val out = FloatArray(embeddingWidth * nTokens)
                // keep-head（nextn）通道给的是末层输出（output_norm 之前）—— 与主仓接力
                // 上游/中间段同语义；embeddings 通道给的是 output_norm(H)，多一次归一化。
                // ★ P3 多序列：给了 seqIds/positions 时走显式绑定入口（与主仓同一契约）。
                val token = when {
                    keepHead && (seqIds != null || positions != null) ->
                        nativeLayerForwardHiddenKeepHeadSeq(modelPtr, hidden, nTokens, posBase,
                                                            null, seqIds, positions, out)
                    !keepHead && (seqIds != null || positions != null) ->
                        nativeLayerForwardHiddenSeq(modelPtr, hidden, nTokens, posBase, out,
                                                    seqIds, positions)
                    keepHead ->
                        nativeLayerForwardHiddenKeepHead(modelPtr, hidden, nTokens, posBase, out)
                    else ->
                        nativeLayerForwardHidden(modelPtr, hidden, nTokens, posBase, out)
                }
                when (token) {
                    // -2 = native 侧报告「load 时未开启 extract_hidden_states」。
                    // 如实转成失败：中间段拿不到 hidden 就必须停，不能悄悄降级。
                    -2 -> return@withModelContext Result.failure(
                        IllegalStateException("extract_hidden_not_enabled")
                    )
                    // -3 = keep-head（层输出）通道不可用：该架构没登记 `t_layer_inp[n_layer]` 槽位
                    //（未登记者会在 decode 时 GGML_ASSERT；这里只在拿到 -3 时兜底）。
                    // 同样 fail-closed —— 绝不退回 embeddings 通道（那会多一次归一化）。
                    -3 -> return@withModelContext Result.failure(
                        IllegalStateException("keep_head_layer_out_unavailable")
                    )
                    // -4 = 层段前向被取消标志中止（`llama_set_abort_callback` 触发）。
                    // 这是**取消**而不是执行失败：调用方（worker 客户端）据该码走取消
                    // 终态，绝不回 `stage_error`。
                    -4 -> return@withModelContext Result.failure(
                        IllegalStateException("layer_forward_aborted")
                    )
                    else -> Result.success(LayerForwardOutput(token, if (token >= 0) out else null))
                }
            } else {
                val token = if (seqIds != null || positions != null) {
                    nativeLayerForwardTokenSeq(modelPtr, hidden, nTokens, posBase, seqIds, positions)
                } else {
                    nativeLayerForwardToken(modelPtr, hidden, nTokens, posBase)
                }
                // -4 = 同上：被取消标志中止 ⇒ 按取消（而非执行失败）上报。
                if (token == -4) {
                    return@withModelContext Result.failure(
                        IllegalStateException("layer_forward_aborted")
                    )
                }
                Result.success(LayerForwardOutput(token, null))
            }
        } catch (e: LinkageError) {
            Result.failure(IllegalStateException("multi_position_argmax_unsupported", e))
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    /** A missing symbol in an older .so must never advertise this capability. */
    fun supportsMultiPositionArgmax(): Boolean = nativeLoaded && probeMultiPositionArgmax {
        nativeSupportsMultiPositionArgmax()
    }

    /**
     * 层段能力探测：`layer_forward_supported` / `n_embd` / `n_layer` /
     * `hidden_dtype` / `n_pos_per_embd` / `acceptance` / `can_tail` /
     * `extract_hidden` / `can_middle`。
     *
     * 与 [getBackendInfo] 同一模式 —— 能力由 native 侧**上报**，调用方据此决定
     * 是否把本节点纳入层流水线，而不是靠 `if engine_type` 猜。
     */
    suspend fun layerForwardInfo(): Result<Map<String, String>> = withModelContext {
        if (!isLoaded) {
            return@withModelContext Result.failure(IllegalStateException("model_not_loaded"))
        }
        try {
            Result.success(nativeLayerForwardInfo(modelPtr))
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    /**
     * 由 hidden 平铺长度与 token 数反推宽度（= `n_embd`），用作输出缓冲尺寸。
     * 形状不自洽时抛错 —— 与 native 侧同样的 fail-closed 纪律。
     */
    private fun estimateEmbeddingWidth(hiddenSize: Int, nTokens: Int): Int {
        if (hiddenSize <= 0 || nTokens <= 0 || hiddenSize % nTokens != 0) {
            throw IllegalArgumentException(
                "hidden array size $hiddenSize is not a multiple of nTokens $nTokens"
            )
        }
        return hiddenSize / nTokens
    }

    /**
     * 释放所有资源（模型 + 原生库引用）。
     */
    suspend fun shutdown() {
        withContext(Dispatchers.IO) {
            modelMutex.withLock {
                closed = true
                unloadModelInternal()
            }
        }
    }

    // ================================================================
    // JNI 声明（需在 libqlh_llama_jni.so 中实现）
    // ================================================================

    /**
     * 加载 GGUF 模型。
     * @return 模型指针（> 0 成功，0 失败）
     */
    private external fun nativeLoadModel(
        path: String,
        nCtx: Int,
        extractHidden: Boolean,
        gpuLayers: Int,
    ): Long

    /** 释放模型内存 */
    private external fun nativeFreeModel(modelPtr: Long)

    /** Attach a GGUF mmproj to an already loaded text model. */
    private external fun nativeLoadMultimodalProjector(modelPtr: Long, mmprojPath: String): Boolean

    /**
     * 执行自回归生成。
     * @param modelPtr 模型指针
     * @param prompt 输入文本
     * @param maxTokens 最大新 token 数
     * @param temperature 温度
     * @param topP top_p
     * @param onToken 每生成一个 token 时回调（传入解码后的文本片段）
     * @return 完整生成文本
     */
    private external fun nativeGenerate(
        modelPtr: Long,
        prompt: String,
        maxTokens: Int,
        temperature: Float,
        topP: Float,
        onToken: (String) -> Unit
    ): String

    /** Execute one bounded multimodal prompt with in-memory encoded images. */
    private external fun nativeGenerateMultimodal(
        modelPtr: Long,
        prompt: String,
        imageBytes: Array<ByteArray>,
        maxTokens: Int,
        temperature: Float,
        topP: Float,
        onToken: (String) -> Unit,
    ): String

    /** 获取模型元数据（name, arch, n_params, n_layers 等） */
    private external fun nativeGetModelInfo(modelPtr: Long): Map<String, String>

    /** 获取 llama.cpp / ggml 后端能力。 */
    private external fun nativeGetBackendInfo(): Map<String, String>

    /** Read-only getsockopt on a caller-owned duplicate: 0 = verified non-TCP; 1..11 = TCP_INFO state. */
    private external fun nativeReadTcpState(fd: Int): Int

    /** 获取 Android MTMD 图像/音频桥接能力。 */
    private external fun nativeGetMultimodalCapability(modelPtr: Long): Map<String, String>

    /** 获取最近一次生成的 token 和耗时统计。 */
    private external fun nativeGetLastGenerationStats(modelPtr: Long): Map<String, String>

    // ===================== Koakuma RPC worker（票 11 / P0-3） =====================

    /**
     * 在本进程内启动 ggml RPC server，把本机算力暴露给集群。
     *
     * 对应 PC 侧 `ggml-rpc-server` 的角色，但跑在 App 进程的独立线程里
     * （Android 无法像桌面那样 fork 独立可执行文件）。
     *
     * @param endpoint 监听地址，形如 `"0.0.0.0:50052"`；**切勿暴露到开放网络**
     * @param nThreads 计算线程数；`<= 0` 时使用默认值
     * @param cacheDir 可选本地缓存目录；`null` 表示不缓存
     * @return `true` 表示 server 线程已启动
     */
    private external fun nativeRpcWorkerStart(
        endpoint: String,
        nThreads: Int,
        cacheDir: String?
    ): Boolean

    /** 查询 RPC worker 状态（running / endpoint / n_devices / stop_supported 等）。 */
    private external fun nativeRpcWorkerStatus(): Map<String, String>

    /**
     * 请求停止 RPC worker。
     *
     * **注意**：上游 llama.cpp 未提供优雅停止 API，本方法只做逻辑停止标记；
     * 状态里以 `stop_supported = false` 如实上报该限制。
     */
    private external fun nativeRpcWorkerStop(): Boolean

    /** 探活：把 endpoint 当远端 RPC 设备访问，确认可达并读取其内存信息。 */
    private external fun nativeRpcProbe(endpoint: String): Map<String, String>

    // ------------------- 层段（layer_forward）-------------------

    /** This query is available only in libraries with the all-position entry point. */
    private external fun nativeSupportsMultiPositionArgmax(): Boolean

    /** Returns positive n_vocab only after all rows succeeded; -4 means cancelled. */
    private external fun nativeLayerForwardTokens(
        modelPtr: Long, hidden: FloatArray, nTokens: Int,
        seqIds: IntArray?, positions: IntArray, outTokens: IntArray,
    ): Int

    /**
     * 层段前向：把上游 hidden 注入 embd 批次，从本节点的裁层 GGUF 继续算，
     * 返回**末位置 argmax token**（本节点是层流水线的**末段**时用）。
     *
     * 用于主仓任务协议 v3 的 `layer_forward` stage。前提是加载的是**裁层 GGUF**
     * （只含尾段层）；本节点实际负责的层区间由主仓按 `layer_range` 下发并核对。
     *
     * @param modelPtr `nativeLoadModel` 返回的句柄
     * @param hidden 上游 hidden，**f32**、长度必须恰为 `nTokens * n_embd`
     * @param nTokens 本次注入的 token 数（通常为当前整段序列长度）
     * @param posBase 位置起点（每步可用绝对位置累加）
     * @return 末位置 argmax token；形状不符/解码失败返回 `-1`
     */
    private external fun nativeLayerForwardToken(
        modelPtr: Long,
        hidden: FloatArray,
        nTokens: Int,
        posBase: Int
    ): Int

    private external fun nativeLayerForwardTokenSeq(
        modelPtr: Long,
        hidden: FloatArray,
        nTokens: Int,
        posBase: Int,
        seqIds: IntArray?,
        positions: IntArray?,
    ): Int

    /**
     * 层段前向（**中间段**）：除 argmax 外，额外把末位置的输出 hidden 拷回
     * `outHidden`，供上层交给下一段继续接力。
     *
     * `outHidden` 的长度必须等于 `n_embd`；不符时不写入（仅返回 argmax）。
     */
    private external fun nativeLayerForwardHidden(
        modelPtr: Long,
        hidden: FloatArray,
        nTokens: Int,
        posBase: Int,
        outHidden: FloatArray
    ): Int

    private external fun nativeLayerForwardHiddenSeq(
        modelPtr: Long,
        hidden: FloatArray,
        nTokens: Int,
        posBase: Int,
        outHidden: FloatArray,
        seqIds: IntArray?,
        positions: IntArray?,
    ): Int

    /**
     * 层段前向（**中间段，keep-head 语义**）：把**末层输出（`output_norm` 之前）**拷回
     * `outHidden`，供下一段继续接力。
     *
     * 与 [nativeLayerForwardHidden] 的区别只在 hidden 通道：那个走 `extract_hidden`
     * （embeddings 通道，返回 `output_norm(H)`，多一次归一化）；本方法走补丁导出的
     * nextn 通道（末层输出，norm 之前），与主仓 `scripts/model_tools/keep_head_shim`
     * 同语义 —— 三段链路/跨框架接力必须用这个通道。
     *
     * 返回末位 argmax；`-1` = 形状/参数错；`-3` = nextn 通道不可用（调用方 fail-closed，
     * **不得**退回 embeddings 通道）。
     */
    private external fun nativeLayerForwardHiddenKeepHead(
        modelPtr: Long,
        hidden: FloatArray,
        nTokens: Int,
        posBase: Int,
        outHidden: FloatArray
    ): Int

    /**
     * ★ P3：层段前向（**中间段 + 多序列**，keep-head 语义）—— 显式 `seqIds` / `positions`。
     *
     * 与 [nativeLayerForwardHiddenKeepHead] 同语义，只是序列绑定改为逐 token 显式给出
     * （与主仓 `forward_layers_from_hidden(seq_ids=..., positions=...)` 同一契约）。
     * 三个数组长度都必须等于 `nTokens`（传 `null` 表示不指定）；长度不符 ⇒ `-1`。
     */
    private external fun nativeLayerForwardHiddenKeepHeadSeq(
        modelPtr: Long,
        hidden: FloatArray,
        nTokens: Int,
        posBase: Int,
        nSeqId: IntArray?,
        seqIds: IntArray?,
        positions: IntArray?,
        outHidden: FloatArray
    ): Int

    /**
     * ★ 2026-10-07（DIST-NEXT-1）：设置/清除 native 侧的层段前向取消标志。
     *
     * native 侧把它存成 `std::atomic<bool>`，由 `llama_set_abort_callback` 注册的
     * 回调读取；被中止的 `llama_decode` 使 `nativeLayerForward*` 返回 `-4`。
     * 标志是**进程级**的（一次只有一个层段前向在跑，与 worker 单并发契约一致）。
     */
    suspend fun clearLayerKv(): Result<Unit> = withModelContext {
        if (modelPtr == 0L || !nativeClearLayerKv(modelPtr)) {
            Result.failure(IllegalStateException("stage_kv_cleanup_failed"))
        } else Result.success(Unit)
    }

    private external fun nativeClearLayerKv(modelPtr: Long): Boolean

    private external fun nativeSetLayerForwardAbort(abort: Boolean)

    /**
     * 层段能力探测：`layer_forward_supported` / `n_embd` / `n_layer` /
     * `hidden_dtype` / `n_pos_per_embd` / `acceptance`。
     *
     * 与 [AndroidWorkerCapabilities] 的「能力探测是单一来源」约定一致 ——
     * 上层据此决定是否把本节点纳入层流水线，而不是靠 `if engine_type` 猜。
     */
    private external fun nativeLayerForwardInfo(modelPtr: Long): Map<String, String>

    // ------------------- 公开封装（供 UI / 集群接入层调用） -------------------

    /** 启动本机 RPC worker；已在运行时返回 `false`。 */
    fun startRpcWorker(endpoint: String, nThreads: Int = 0, cacheDir: String? = null): Boolean =
        nativeRpcWorkerStart(endpoint, nThreads, cacheDir)

    /** 当前 RPC worker 状态快照。 */
    fun rpcWorkerStatus(): Map<String, String> = nativeRpcWorkerStatus()

    /** 请求停止 RPC worker（逻辑停止；上游无优雅停止机制）。 */
    fun stopRpcWorker(): Boolean = nativeRpcWorkerStop()

    /** 探测某个 RPC endpoint 是否可达。 */
    fun probeRpcEndpoint(endpoint: String): Map<String, String> = nativeRpcProbe(endpoint)
}

/** Kept injectable so JVM coverage exercises compatibility with old JNI libraries. */
internal fun probeMultiPositionArgmax(probe: () -> Boolean): Boolean = try {
    probe()
} catch (_: LinkageError) {
    false
} catch (_: Exception) {
    false
}
