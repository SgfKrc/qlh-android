package com.qlh.inference.service

import android.content.Context
import android.os.Build
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.channels.awaitClose
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.callbackFlow
import kotlinx.coroutines.flow.flowOn
import kotlinx.coroutines.withContext
import java.io.File

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
    ): Result<Unit> = withContext(Dispatchers.IO) {
        val libResult = ensureNativeLibraryLoaded()
        if (libResult.isFailure) {
            return@withContext Result.failure(libResult.exceptionOrNull()!!)
        }

        // 卸载旧模型
        if (modelPtr != 0L) {
            unloadModelInternal()
        }

        val file = File(modelPath)
        if (!file.exists()) {
            return@withContext Result.failure(
                IllegalStateException("模型文件不存在: $modelPath")
            )
        }
        if (!file.canRead()) {
            return@withContext Result.failure(
                IllegalStateException("模型文件不可读: $modelPath")
            )
        }

        try {
            val ptr = nativeLoadModel(modelPath, contextSize, extractHidden, gpuLayers)
            if (ptr == 0L) {
                return@withContext Result.failure(
                    IllegalStateException("模型加载失败（native 返回空指针）: $modelPath")
                )
            }
            modelPtr = ptr
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
    ): Result<Unit> = withContext(Dispatchers.IO) {
        val libResult = ensureNativeLibraryLoaded()
        if (libResult.isFailure) {
            handle.close()
            return@withContext Result.failure(libResult.exceptionOrNull()!!)
        }

        if (modelPtr != 0L) {
            unloadModelInternal()
        }

        val modelPath = handle.loadPath
        if (!modelPath.startsWith("/proc/self/fd/")) {
            val file = File(modelPath)
            if (!file.exists()) {
                handle.close()
                return@withContext Result.failure(
                    IllegalStateException("模型文件不存在: $modelPath")
                )
            }
            if (!file.canRead()) {
                handle.close()
                return@withContext Result.failure(
                    IllegalStateException("模型文件不可读: $modelPath")
                )
            }
        }

        try {
            val ptr = nativeLoadModel(modelPath, contextSize, extractHidden, gpuLayers)
            if (ptr == 0L) {
                handle.close()
                return@withContext Result.failure(
                    IllegalStateException("模型加载失败（native 返回空指针）: ${handle.displayName}")
                )
            }
            modelPtr = ptr
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
    suspend fun unloadModel(): Result<Unit> = withContext(Dispatchers.IO) {
        try {
            unloadModelInternal()
            Result.success(Unit)
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    /** Load the user-owned Gemma4 mmproj beside the already loaded text model. */
    suspend fun loadMultimodalProjector(handle: ModelManager.ModelOpenHandle): Result<Unit> =
        withContext(Dispatchers.IO) {
            if (!isLoaded) {
                handle.close()
                return@withContext Result.failure(IllegalStateException("模型未加载"))
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
    ): Result<String> = withContext(Dispatchers.IO) {
        if (!isLoaded || !multimodalLoaded) {
            return@withContext Result.failure(IllegalStateException("mmproj 未加载"))
        }
        if (imageBytes.isEmpty()) {
            return@withContext Result.failure(IllegalArgumentException("至少需要一张图片"))
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
            }
            modelPtr = 0
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
        if (!isLoaded) {
            throw IllegalStateException("模型未加载，请先调用 loadModel()")
        }

        // nativeGenerate 在 JNI 线程中回调，通过 trySend 发送到 Flow
        nativeGenerate(
            modelPtr, prompt, maxTokens, temperature, topP
        ) { token ->
            trySend(token)
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
    ): Result<String> = withContext(Dispatchers.IO) {
        if (!isLoaded) {
            return@withContext Result.failure(IllegalStateException("模型未加载"))
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
    suspend fun getModelInfo(): Result<Map<String, String>> = withContext(Dispatchers.IO) {
        if (!isLoaded) {
            return@withContext Result.failure(IllegalStateException("模型未加载"))
        }
        try {
            val info = nativeGetModelInfo(modelPtr)
            Result.success(info)
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    suspend fun getBackendInfo(): Result<Map<String, String>> = withContext(Dispatchers.IO) {
        val libResult = ensureNativeLibraryLoaded()
        if (libResult.isFailure) {
            return@withContext Result.failure(libResult.exceptionOrNull()!!)
        }
        try {
            Result.success(nativeGetBackendInfo())
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    /** Query the compiled multimodal bridge; text-only builds must report false. */
    suspend fun getMultimodalCapability(): Result<Map<String, String>> = withContext(Dispatchers.IO) {
        val libResult = ensureNativeLibraryLoaded()
        if (libResult.isFailure) {
            return@withContext Result.failure(libResult.exceptionOrNull()!!)
        }
        try {
            Result.success(nativeGetMultimodalCapability(modelPtr))
        } catch (e: Throwable) {
            Result.failure(e)
        }
    }

    suspend fun getLastGenerationStats(): Result<Map<String, String>> = withContext(Dispatchers.IO) {
        if (!isLoaded) {
            return@withContext Result.success(emptyMap())
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
     */
    suspend fun layerForward(
        hidden: FloatArray,
        nTokens: Int,
        posBase: Int,
        wantHidden: Boolean,
    ): Result<LayerForwardOutput> = withContext(Dispatchers.IO) {
        if (!isLoaded) {
            return@withContext Result.failure(IllegalStateException("model_not_loaded"))
        }
        if (nTokens <= 0) {
            return@withContext Result.failure(IllegalArgumentException("nTokens must be positive"))
        }
        try {
            if (wantHidden) {
                val out = FloatArray(estimateEmbeddingWidth(hidden.size, nTokens))
                val token = nativeLayerForwardHidden(modelPtr, hidden, nTokens, posBase, out)
                when (token) {
                    // -2 = native 侧报告「load 时未开启 extract_hidden_states」。
                    // 如实转成失败：中间段拿不到 hidden 就必须停，不能悄悄降级。
                    -2 -> return@withContext Result.failure(
                        IllegalStateException("extract_hidden_not_enabled")
                    )
                    else -> Result.success(LayerForwardOutput(token, if (token >= 0) out else null))
                }
            } else {
                val token = nativeLayerForwardToken(modelPtr, hidden, nTokens, posBase)
                Result.success(LayerForwardOutput(token, null))
            }
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    /**
     * 层段能力探测：`layer_forward_supported` / `n_embd` / `n_layer` /
     * `hidden_dtype` / `n_pos_per_embd` / `acceptance` / `can_tail` /
     * `extract_hidden` / `can_middle`。
     *
     * 与 [getBackendInfo] 同一模式 —— 能力由 native 侧**上报**，调用方据此决定
     * 是否把本节点纳入层流水线，而不是靠 `if engine_type` 猜。
     */
    suspend fun layerForwardInfo(): Result<Map<String, String>> = withContext(Dispatchers.IO) {
        if (!isLoaded) {
            return@withContext Result.failure(IllegalStateException("model_not_loaded"))
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
            unloadModelInternal()
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
