package com.qlh.inference.worker

import kotlin.math.roundToInt

/** Stable worker-side failures; TaskWorkerClient maps these to protocol error codes. */
class AndroidFullWorkerStageException(
    val code: String,
    message: String,
) : IllegalStateException(message)

/**
 * 一次层段前向的输入（由 `layer_forward` stage_offer 解析而来）。
 *
 * @param layerRange 本节点负责的**源模型**层区间 `[start, end)`
 * @param handoffAt 上游交接点（hidden 从源模型第 `handoffAt` 层之后跨边界）
 * @param hidden f32 hidden 平铺数组，长度必须为 `nTokens * nEmbd`
 * @param nTokens 注入的 token 数
 * @param nEmbd hidden 宽度（与模型 `n_embd` 对账）
 * @param posBase 位置起点
 * @param wantHidden 本节点是否**中间段**（需要把输出 hidden 交给下一段）
 */
data class LayerForwardRequest(
    val layerRange: List<Int>,
    val handoffAt: Int,
    val hidden: FloatArray,
    val nTokens: Int,
    val nEmbd: Int,
    val posBase: Int,
    val wantHidden: Boolean,
) {
    override fun equals(other: Any?): Boolean = this === other
    override fun hashCode(): Int = System.identityHashCode(this)
}

/** 层段前向结果：`hiddenOut` 仅在 `wantHidden=true` 且成功时有值。 */
data class LayerForwardResult(
    val tokenArgmax: Int,
    val hiddenOut: FloatArray?,
)

/**
 * Text-only Full Worker executor shared by the foreground service and JVM fake
 * worker tests. It never accepts a path or a model downloaded by the coordinator.
 */
class AndroidFullWorkerStageExecutor(
    private val expectedModelIdentity: () -> Map<String, Any?>?,
    private val ensureModelLoaded: suspend (contextSize: Int) -> Result<Unit>,
    private val generate: suspend (
        prompt: String,
        maxTokens: Int,
        temperature: Float,
        topP: Float,
    ) -> Result<String>,
    /**
     * ★ 2026-09-20（层段）：`layer_forward` stage 的执行器。
     *
     * 与 [generate] 同一模式 —— executor 本身不懂 llama.cpp，只做协议↔执行器的
     * 翻译与校验；真正的注入/前向由 native 侧完成。
     * 默认实现**明确失败**（fail-closed），避免未接线时被当成「支持层段」。
     */
    private val layerForward: suspend (LayerForwardRequest) -> Result<LayerForwardResult> = {
        Result.failure(IllegalStateException("layer forward executor is not wired"))
    },
    /**
     * ★ 2026-09-20：保证模型已按 `layer_forward` 的需要加载
     * （中间段需 `extract_hidden_states`）。默认委托 [ensureModelLoaded]。
     */
    private val ensureModelLoadedForLayer: suspend (contextSize: Int) -> Result<Unit> = ensureModelLoaded,
) : TaskWorkerStageHandler {
    override suspend fun execute(offer: TaskWorkerEnvelope): TaskWorkerStageExecution {
        if (offer.messageType != TaskWorkerProtocol.STAGE_OFFER) {
            throw AndroidFullWorkerStageException(
                "invalid_stage_message",
                "Android Full Worker accepts stage_offer only",
            )
        }
        val payload = offer.payload
        val stageType = payload["stage_type"] as? String
        if (!AndroidWorkerCapabilities.isSupportedStageType(stageType)) {
            throw AndroidFullWorkerStageException(
                "unsupported_stage_type",
                "Android Full Worker accepts only supported stage types " +
                    "(got [${stageType ?: "null"}]; " +
                    "supported: [${AndroidWorkerCapabilities.describeSupportedStageTypes()}])",
            )
        }
        val advertised = expectedModelIdentity()
        val requested = payload["model_identity"] as? Map<*, *>
        if (advertised == null || requested == null || !sameIdentity(advertised, requested)) {
            throw AndroidFullWorkerStageException(
                "model_identity_mismatch",
                "Stage model identity does not match the Android worker",
            )
        }
        // ★ 2026-09-20：按 stage 类型分派。两种 stage 的输入形态完全不同
        //   （整模型要 prompt；层段要 hidden + 层区间），不能共用一条路径。
        return when (stageType) {
            "layer_forward" -> executeLayerForward(payload, advertised)
            else -> executeFullInference(payload, advertised)
        }
    }

    private suspend fun executeFullInference(
        payload: Map<String, Any?>,
        advertised: Map<String, Any?>,
    ): TaskWorkerStageExecution {
        val rootInput = payload["root_input"] as? Map<*, *>
            ?: throw AndroidFullWorkerStageException("invalid_stage_input", "root_input must be an object")
        val prompt = extractPrompt(rootInput)
        val maxTokens = boundedInt(rootInput["max_new_tokens"], 1024, 1, 4096)
        val temperature = boundedFloat(rootInput["temperature"], 0.7f, 0.0f, 2.0f)
        val topP = boundedFloat(rootInput["top_p"], 0.9f, 0.0f, 1.0f)
        val contextSize = boundedInt(rootInput["context_size"], 2048, 256, 32768)

        ensureModelLoaded(contextSize).getOrElse { error ->
            throw AndroidFullWorkerStageException(
                "model_not_ready",
                error.message ?: "Android inference model is not ready",
            )
        }
        val content = generate(prompt, maxTokens, temperature, topP).getOrElse { error ->
            throw AndroidFullWorkerStageException(
                "worker_execution_failed",
                error.message ?: "Android inference failed",
            )
        }
        return TaskWorkerStageExecution(
            output = mapOf("content" to content),
            metadata = mapOf(
                // v2 text results allow `model`; provider/node provenance is
                // persisted by the coordinator's attempt journal.
                "model" to (advertised["model_id"] as? String).orEmpty(),
            ),
        )
    }

    /**
     * `layer_forward`：把上游 hidden 注入本节点的**裁层 GGUF**，只算自己的层区间。
     *
     * ## 与整模型路径的关键区别
     * * 输入不是 prompt，而是 **hidden 本体 + 层区间 + 交接点**；
     * * 输出不是文本，而是 **末位置 argmax**（末段）或 **hidden**（中间段）；
     * * 层区间必须与**实际加载的工件**自洽 —— 本 executor 只做**形状/范围**校验，
     *   真正的「裁层 GGUF 是否匹配 `layer_range`」由 native 上报的 `n_layer` 对账。
     *
     * ## fail-closed
     * 缺字段、形状不符、层区间非法、hidden 摘要不符、执行器未接线 —— 一律
     * **明确抛错**，不做静默降级（尤其**不得**把层段请求当整模型跑）。
     */
    private suspend fun executeLayerForward(
        payload: Map<String, Any?>,
        advertised: Map<String, Any?>,
    ): TaskWorkerStageExecution {
        val layerRange = (payload["layer_range"] as? List<*>)
            ?.mapNotNull { (it as? Number)?.toInt() }
            ?: throw AndroidFullWorkerStageException("invalid_layer_range", "layer_range is required")
        if (layerRange.size != 2 || layerRange[1] <= layerRange[0] || layerRange[0] < 0) {
            throw AndroidFullWorkerStageException(
                "invalid_layer_range",
                "layer_range must be [start, end) with end > start >= 0",
            )
        }
        val handoffAt = (payload["handoff_at"] as? Number)?.toInt()
            ?: throw AndroidFullWorkerStageException("invalid_handoff", "handoff_at is required")
        if (handoffAt < 0) {
            throw AndroidFullWorkerStageException("invalid_handoff", "handoff_at must be >= 0")
        }
        val spec = payload["hidden_spec"] as? Map<*, *>
            ?: throw AndroidFullWorkerStageException("invalid_hidden_spec", "hidden_spec is required")
        val nTokens = (spec["n_tokens"] as? Number)?.toInt() ?: 0
        val nEmbd = (spec["n_embd"] as? Number)?.toInt() ?: 0
        val dtype = spec["dtype"] as? String
        if (nTokens < 1 || nEmbd < 1) {
            throw AndroidFullWorkerStageException(
                "invalid_hidden_spec",
                "hidden_spec.n_tokens/n_embd must be positive",
            )
        }
        if (dtype != "float32") {
            // 仅支持 f32：跨框架接力以 f32 为基线（见主仓 relay 合同）。
            throw AndroidFullWorkerStageException(
                "unsupported_hidden_dtype",
                "Android layer_forward accepts float32 hidden (got [${dtype ?: "null"}])",
            )
        }
        val rootInput = payload["root_input"] as? Map<*, *>
            ?: throw AndroidFullWorkerStageException("invalid_stage_input", "root_input must be an object")
        val hidden = readHiddenPayload(rootInput, nTokens, nEmbd)
            ?: throw AndroidFullWorkerStageException(
                "invalid_hidden_payload",
                "root_input must carry hidden_f32 (base64) of length n_tokens * n_embd",
            )
        // 摘要对账：协议层的 hidden_sha256 由 native/上层计算；这里只做**存在性**
        // 与**长度**校验 —— 真正的逐字节校验在 relay 传输层完成，避免 Kotlin 侧
        // 重算一遍大数组摘要。
        val declaredSha = (payload["hidden_sha256"] as? String).orEmpty()
        if (declaredSha.length != 64) {
            throw AndroidFullWorkerStageException(
                "invalid_hidden_digest",
                "hidden_sha256 must be a 64-char hex digest",
            )
        }

        val contextSize = boundedInt(rootInput["context_size"], 2048, 256, 32768)
        // 中间段（要产出 hidden）必须由 load 时开 extract_hidden_states；
        // 末段只需 argmax，不开也能跑 ⇒ 是否中间段由调用方显式声明。
        val wantHidden = (rootInput["want_hidden"] as? Boolean) ?: false
        val loader = if (wantHidden) ensureModelLoadedForLayer else ensureModelLoaded
        loader(contextSize).getOrElse { error ->
            throw AndroidFullWorkerStageException(
                "model_not_ready",
                error.message ?: "Android inference model is not ready",
            )
        }

        val posBase = boundedInt(rootInput["pos_base"], 0, 0, Int.MAX_VALUE - 1)
        val result = layerForward(
            LayerForwardRequest(
                layerRange = layerRange,
                handoffAt = handoffAt,
                hidden = hidden,
                nTokens = nTokens,
                nEmbd = nEmbd,
                posBase = posBase,
                wantHidden = wantHidden,
            ),
        ).getOrElse { error ->
            throw AndroidFullWorkerStageException(
                "worker_execution_failed",
                error.message ?: "Android layer forward failed",
            )
        }
        if (result.tokenArgmax < 0) {
            throw AndroidFullWorkerStageException(
                "layer_forward_failed",
                "runner returned no token (${result.tokenArgmax})",
            )
        }

        val output = LinkedHashMap<String, Any?>()
        output["token_argmax"] = result.tokenArgmax
        output["layer_range"] = layerRange
        output["n_tokens"] = nTokens
        result.hiddenOut?.let { out ->
            output["hidden_out_sha256"] = sha256Hex(out)
        }
        return TaskWorkerStageExecution(
            output = output,
            metadata = mapOf(
                "model" to (advertised["model_id"] as? String).orEmpty(),
                "stage" to "layer_forward",
                "handoff_at" to handoffAt,
                "tail" to (!wantHidden).toString(),
            ),
        )
    }

    /** 从 `root_input` 读 hidden：支持 `hidden_f32`（base64）或 `hidden_f32_b64`。 */
    private fun readHiddenPayload(rootInput: Map<*, *>, nTokens: Int, nEmbd: Int): FloatArray? {
        val encoded = (rootInput["hidden_f32"] as? String)
            ?: (rootInput["hidden_f32_b64"] as? String)
            ?: return null
        val bytes = try {
            java.util.Base64.getDecoder().decode(encoded)
        } catch (_: IllegalArgumentException) {
            return null
        }
        val expected = nTokens.toLong() * nEmbd.toLong() * 4L
        if (bytes.size.toLong() != expected) return null
        val out = FloatArray(nTokens * nEmbd)
        val buf = java.nio.ByteBuffer.wrap(bytes).order(java.nio.ByteOrder.LITTLE_ENDIAN)
        for (i in out.indices) {
            out[i] = buf.float
        }
        return out
    }

    private fun sha256Hex(data: FloatArray): String {
        val buf = java.nio.ByteBuffer.allocate(data.size * 4)
            .order(java.nio.ByteOrder.LITTLE_ENDIAN)
        for (v in data) buf.putFloat(v)
        val digest = java.security.MessageDigest.getInstance("SHA-256").digest(buf.array())
        return digest.joinToString("") { "%02x".format(it) }
    }

    private fun extractPrompt(rootInput: Map<*, *>): String {
        val direct = rootInput["prompt"] as? String
        if (!direct.isNullOrBlank()) return direct.take(MAX_PROMPT_CHARS)
        val message = rootInput["message"] as? String
        if (!message.isNullOrBlank()) return message.take(MAX_PROMPT_CHARS)
        val messages = rootInput["messages"] as? List<*>
        if (messages != null) {
            val joined = messages.mapNotNull { item ->
                val entry = item as? Map<*, *> ?: return@mapNotNull null
                val content = entry["content"] as? String ?: return@mapNotNull null
                val role = (entry["role"] as? String).orEmpty().ifBlank { "user" }
                "$role: $content"
            }.joinToString("\n")
            if (joined.isNotBlank()) return joined.take(MAX_PROMPT_CHARS)
        }
        throw AndroidFullWorkerStageException(
            "invalid_stage_input",
            "root_input must contain prompt, message, or messages",
        )
    }

    private fun sameIdentity(expected: Map<String, Any?>, requested: Map<*, *>): Boolean =
        listOf("model_id", "engine", "format", "revision", "sha256").all { key ->
            expected[key] == requested[key]
        }

    private fun boundedInt(value: Any?, fallback: Int, minimum: Int, maximum: Int): Int {
        val number = (value as? Number)?.toDouble() ?: return fallback
        if (!number.isFinite()) return fallback
        return number.roundToInt().coerceIn(minimum, maximum)
    }

    private fun boundedFloat(value: Any?, fallback: Float, minimum: Float, maximum: Float): Float {
        val number = (value as? Number)?.toFloat() ?: return fallback
        return if (number.isFinite()) number.coerceIn(minimum, maximum) else fallback
    }

    companion object {
        private const val MAX_PROMPT_CHARS = 64 * 1024
    }
}
