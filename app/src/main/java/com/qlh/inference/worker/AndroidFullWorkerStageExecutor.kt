package com.qlh.inference.worker

import kotlin.math.pow
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
    /**
     * ★ 2026-09-23：**中间段取 hidden 的通道**（来自 stage_offer 的 `middle_channel`）。
     *
     * `"keep_head_layer_out"` ⇒ 走 keep-head 通道（**末层输出**，`output_norm` 之前，
     * 层段接力所需的形态）；其余（含缺省 `"extract_hidden"`）⇒ 旧通道 `output_norm(H)`。
     */
    val middleChannel: String = "extract_hidden",
    /**
     * ★ 2026-09-23（A12）：**多序列显式位置**（来自 stage_offer 的 `seq_ids` / `positions`）。
     *
     * 长度必须等于 [nTokens]（协议层已校验）；`null` = 单序列旧行为。多序列交错推进时必须给，
     * 否则 llama.cpp 会按隐式位置递增报 "tokens … have inconsistent sequence positions"。
     */
    val seqIds: IntArray? = null,
    val positions: IntArray? = null,
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
    private val ensureModelLoadedForLayer: suspend (
        layerRange: List<Int>,
        contextSize: Int,
        embeddingWidth: Int,
        wantHidden: Boolean,
        modelSha256: String,
    ) -> Result<Unit> = { _, contextSize, _, _, _ -> ensureModelLoaded(contextSize) },
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
        if (dtype != "float32" && dtype != "float16") {
            // ★ 2026-09-23（对称性缺口 A13）：与主仓协议对齐 —— `float32` / `float16` 都接受；
            //   f16 在本段就地转 f32 再喂 JNI（native 侧只吃 f32）。
            throw AndroidFullWorkerStageException(
                "unsupported_hidden_dtype",
                "Android layer_forward accepts float32 or float16 hidden (got [${dtype ?: "null"}])",
            )
        }
        val rootInput = payload["root_input"] as? Map<*, *>
            ?: throw AndroidFullWorkerStageException("invalid_stage_input", "root_input must be an object")
        val hidden = readHiddenPayload(rootInput, nTokens, nEmbd, dtype ?: "float32")
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
        // ★ 2026-09-23：中间段取 hidden 的通道（缺省 = 旧的 `extract_hidden`）。
        //   协议层已做值域校验，这里再兜一次 fail-closed（执行器也可能被直接调用）。
        val middleChannel = (payload["middle_channel"] as? String) ?: "extract_hidden"
        if (middleChannel != "extract_hidden" && middleChannel != "keep_head_layer_out") {
            throw AndroidFullWorkerStageException(
                "unsupported_middle_channel",
                "middle_channel must be extract_hidden or keep_head_layer_out (got [$middleChannel])",
            )
        }
        // ★ 2026-09-23（A12）：多序列显式位置（可选）。协议层已校验形状与取值，这里**再兜一次**
        //   fail-closed —— 存在但非法时抛错，绝不静默退化成单序列（那会让多序列链路悄悄算错）。
        val seqIds = optionalIntList(payload, "seq_ids", nTokens)
        val positions = optionalIntList(payload, "positions", nTokens)
        val loaderResult = if (wantHidden || layerRange.isNotEmpty()) {
            ensureModelLoadedForLayer(
                layerRange,
                contextSize,
                nEmbd,
                wantHidden,
                (advertised["sha256"] as? String).orEmpty(),
            )
        } else {
            ensureModelLoaded(contextSize)
        }
        loaderResult.getOrElse { error ->
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
                middleChannel = middleChannel,
                seqIds = seqIds,
                positions = positions,
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
                // ★ 2026-09-23：回记实际生效的中间段通道，便于对账（缺省即 extract_hidden）。
                "middle_channel" to middleChannel,
                "handoff_at" to handoffAt,
                "tail" to (!wantHidden).toString(),
            ),
        )
    }

    /** 从 `root_input` 读 hidden：支持 `hidden_f32`（base64）或 `hidden_f32_b64`。 */
    /**
     * 解析**可选**的整数数组字段（`seq_ids` / `positions`）。
     *
     * ⚠️ fail-closed：字段**存在但形状/取值非法**时抛错，绝不静默退化成单序列 ——
     * 否则多序列链路会「看起来在跑、其实算错」。字段不存在 ⇒ 返回 `null`（单序列旧行为）。
     */
    private fun optionalIntList(payload: Map<String, Any?>, field: String,
                                expectedSize: Int): IntArray? {
        if (!payload.containsKey(field)) return null
        val list = payload[field] as? List<*>
            ?: throw AndroidFullWorkerStageException(
                "invalid_$field", "$field must be a list of length $expectedSize")
        if (list.size != expectedSize) {
            throw AndroidFullWorkerStageException(
                "invalid_$field", "$field must have $expectedSize entries (got ${list.size})")
        }
        val out = IntArray(list.size)
        for (i in list.indices) {
            val number = (list[i] as? Number)?.toInt()
            if (number == null || number < 0) {
                throw AndroidFullWorkerStageException(
                    "invalid_$field", "$field entries must be non-negative integers")
            }
            out[i] = number
        }
        return out
    }

    /**
     * 解析 `root_input` 里的 hidden（**f32 与 f16 都支持**，little-endian + base64）。
     *
     * ★ 2026-09-23（A13 对称性缺口）：主仓协议允许 `hidden_spec.dtype` 为 `float32` **或**
     * `float16`，而这里原先只解析 `hidden_f32` ⇒ 会造成「协商通过、执行必失败」的隐性不对称。
     * 现在按 `dtype` 取对应字段（`hidden_f32` / `hidden_f16`，各带 `_b64` 别名），
     * f16 就在本段转成 f32（native 侧只吃 f32）。
     */
    private fun readHiddenPayload(
        rootInput: Map<*, *>,
        nTokens: Int,
        nEmbd: Int,
        dtype: String,
    ): FloatArray? {
        val elementBytes = when (dtype) {
            "float32" -> 4
            "float16" -> 2
            else -> return null
        }
        val suffix = if (dtype == "float16") "f16" else "f32"
        val encoded = (rootInput["hidden_$suffix"] as? String)
            ?: (rootInput["hidden_${suffix}_b64"] as? String)
            ?: return null
        val bytes = try {
            java.util.Base64.getDecoder().decode(encoded)
        } catch (_: IllegalArgumentException) {
            return null
        }
        val expected = nTokens.toLong() * nEmbd.toLong() * elementBytes.toLong()
        if (bytes.size.toLong() != expected) return null
        val out = FloatArray(nTokens * nEmbd)
        val buf = java.nio.ByteBuffer.wrap(bytes).order(java.nio.ByteOrder.LITTLE_ENDIAN)
        if (elementBytes == 4) {
            for (i in out.indices) {
                out[i] = buf.float
            }
        } else {
            for (i in out.indices) {
                out[i] = halfToFloat(buf.short.toInt())
            }
        }
        return out
    }

    /**
     * IEEE-754 **半精度 → float**（`hidden_f16` 的 wire 口径：LE 的 16 位二进制）。
     *
     * 覆盖三种形态：规格化、非规格化（含 ±0）、Inf/NaN —— 后者在数值链路里本应被上游挡住，
     * 但解析层不能把它们变成垃圾值。实现用 `pow` 而不是位拼装，避免非规格化的精度损失。
     */
    private fun halfToFloat(bits: Int): Float {
        val sign = if ((bits shr 15) and 0x1 == 1) -1.0f else 1.0f
        val exponent = (bits shr 10) and 0x1F
        val mantissa = bits and 0x3FF
        return when {
            exponent == 0 -> sign * mantissa.toFloat() * 2f.pow(-24)
            exponent == 0x1F && mantissa == 0 -> sign * Float.POSITIVE_INFINITY
            exponent == 0x1F -> Float.NaN
            else -> sign * (1.0f + mantissa.toFloat() / 1024f) * 2f.pow(exponent - 15)
        }
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
