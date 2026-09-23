package com.qlh.inference.worker

import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Test

/**
 * ★ 2026-09-20：**能力声明 ⇔ 执行实现** 的一致性断言。
 *
 * ## 为什么需要这个测试
 * `AndroidWorkerCapabilities.SUPPORTED_STAGE_TYPES` 是**能力探测的单一来源** ——
 * 它会随 `hello` 上报给 coordinator，coordinator 据此决定派不派这类 stage。
 * 如果清单里写了一个执行器根本不认的 stage，就会出现
 * 「**声明支持、执行时 unsupported_stage_type**」的**能力撒谎**。
 *
 * 故本测试遍历清单里**每一个** stage 类型，断言
 * `AndroidFullWorkerStageExecutor` **确实能按该类型分派**（而不是抛
 * `unsupported_stage_type`）。这样将来往清单里加东西时，忘了同步执行器会**直接挂测试**。
 */
class AndroidWorkerCapabilitiesStageParityTest {
    private val model = mapOf(
        "model_id" to "qwen_1_8b",
        "engine" to "llama_cpp",
        "format" to "gguf",
        "revision" to "local-v1",
        "sha256" to "a".repeat(64),
    )

    /** 各 stage 类型的最小合法 `root_input`（只为让执行器走到分派之后）。 */
    private val rootInputByStage = mapOf(
        "full_inference" to mapOf(
            "prompt" to "hello",
            "context_size" to 2048,
        ),
        // 层段：1 token × 4 维的 f32 hidden（base64），保持极小。
        "layer_forward" to mapOf(
            "hidden_f32" to floatArrayToBase64(FloatArray(4) { it.toFloat() }),
            "context_size" to 2048,
            "want_hidden" to false,
        ),
    )

    private fun offer(
        stageType: String,
        version: Int = TaskWorkerProtocol.VERSION,
        // ★ 2026-09-23：可选注入 `middle_channel`（缺省 = 不发该字段 ⇒ 旧行为）。
        middleChannel: String? = null,
    ): TaskWorkerEnvelope =
        TaskWorkerProtocol.buildStageOffer(
            identity = TaskWorkerAttemptIdentity(
                workflowId = "wf_parity_stage_01",
                stageId = "stage_1",
                attemptId = "att_parity_stage_01",
                leaseId = "lease_parity_stage_01",
                leaseEpoch = 1,
            ),
            requestId = "request_parity_stage_01",
            stageType = stageType,
            providerId = "remote_android_worker_01",
            leaseExpiresAtMs = 2_000,
            rootInput = rootInputByStage.getValue(stageType),
            dependencies = emptyMap(),
            modelIdentity = model,
            messageId = "msg_parity_stage01",
            sentAtMs = 1_000,
            stageFields = if (stageType == "layer_forward") {
                mapOf(
                    "layer_range" to listOf(4, 8),
                    "handoff_at" to 4,
                    "hidden_sha256" to "b".repeat(64),
                    "hidden_spec" to mapOf(
                        "n_tokens" to 1,
                        "n_embd" to 4,
                        "dtype" to "float32",
                    ),
                ) + (middleChannel?.let { mapOf("middle_channel" to it) } ?: emptyMap())
            } else {
                emptyMap()
            },
        ).copy(version = version)

    /** 构造一个**已接线**的执行器：两条路径都成功，便于观察分派去向。 */
    private fun wiredExecutor(
        onGenerate: (String) -> Result<String> = { Result.success("ok") },
        onLayer: (LayerForwardRequest) -> Result<LayerForwardResult> =
            { Result.success(LayerForwardResult(tokenArgmax = 7, hiddenOut = null)) },
    ) = AndroidFullWorkerStageExecutor(
        expectedModelIdentity = { model },
        ensureModelLoaded = { Result.success(Unit) },
        generate = { prompt, _, _, _ -> onGenerate(prompt) },
        layerForward = onLayer,
    )

    @Test
    fun `every advertised stage type is actually dispatchable by the executor`() = runBlocking {
        for (stageType in AndroidWorkerCapabilities.SUPPORTED_STAGE_TYPES) {
            val executor = wiredExecutor()
            try {
                executor.execute(offer(stageType))
            } catch (e: AndroidFullWorkerStageException) {
                fail(
                    "SUPPORTED_STAGE_TYPES advertises [$stageType] but the executor " +
                        "cannot dispatch it (${e.code}): ${e.message}",
                )
            }
        }
    }

    @Test
    fun `layer_forward is routed to the layer executor and not to generate`() = runBlocking {
        var generateCalls = 0
        var layerCalls = 0
        val executor = wiredExecutor(
            onGenerate = {
                generateCalls++
                Result.success("should not be called")
            },
            onLayer = {
                layerCalls++
                Result.success(LayerForwardResult(tokenArgmax = 42, hiddenOut = null))
            },
        )
        val result = executor.execute(offer("layer_forward"))
        assertEquals(0, generateCalls)
        assertEquals(1, layerCalls)
        assertEquals(42, result.output["token_argmax"])
        assertEquals(listOf(4, 8), result.output["layer_range"])
        assertEquals("layer_forward", result.metadata["stage"])
    }

    @Test
    fun `layer_forward fails closed when the executor is not wired`() = runBlocking {
        // 默认构造（不传 layerForward）必须**明确失败**，不能假装成功。
        val executor = AndroidFullWorkerStageExecutor(
            expectedModelIdentity = { model },
            ensureModelLoaded = { Result.success(Unit) },
            generate = { _, _, _, _ -> Result.success("ok") },
        )
        try {
            executor.execute(offer("layer_forward"))
            fail("unwired layer_forward executor must fail closed")
        } catch (e: AndroidFullWorkerStageException) {
            assertEquals("worker_execution_failed", e.code)
            assertTrue(e.message.orEmpty().contains("not wired"))
        }
    }

    @Test
    fun `layer_forward shape mismatch is rejected before reaching the executor`() = runBlocking {
        var layerCalls = 0
        val executor = wiredExecutor(
            onLayer = {
                layerCalls++
                Result.success(LayerForwardResult(tokenArgmax = 1, hiddenOut = null))
            },
        )
        // hidden_spec 要求 1×4，但实际给了 8 个 float ⇒ 必须被拒，且**不能**调用执行器。
        val bad = offer("layer_forward").let { env ->
            val payload = env.payload + mapOf(
                "root_input" to (env.payload["root_input"] as Map<*, *>)
                    .entries.associate { it.key.toString() to it.value }
                    .plus("hidden_f32" to floatArrayToBase64(FloatArray(8))),
            )
            env.copy(payload = payload)
        }
        try {
            executor.execute(bad)
            fail("mismatched hidden shape must be rejected")
        } catch (e: AndroidFullWorkerStageException) {
            assertEquals("invalid_hidden_payload", e.code)
            assertEquals(0, layerCalls)
        }
    }

    @Test
    fun `layer_forward passes middle_channel through to the executor`() = runBlocking {
        // ★ 2026-09-23：`middle_channel=keep_head_layer_out` 必须一路透传到 `LayerForwardRequest`
        //   —— 否则中间段仍走 `extract_hidden`（= `output_norm(H)`，多一次归一化）。
        var seen: LayerForwardRequest? = null
        val executor = wiredExecutor(
            onLayer = { req ->
                seen = req
                Result.success(LayerForwardResult(tokenArgmax = 5, hiddenOut = null))
            },
        )
        val result = executor.execute(offer("layer_forward", middleChannel = "keep_head_layer_out"))
        assertEquals("keep_head_layer_out", seen?.middleChannel)
        // 实际生效的通道要回记在 metadata 里，便于对账。
        assertEquals("keep_head_layer_out", result.metadata["middle_channel"])
    }

    @Test
    fun `layer_forward defaults to extract_hidden when middle_channel is absent`() = runBlocking {
        var seen: LayerForwardRequest? = null
        val executor = wiredExecutor(
            onLayer = { req ->
                seen = req
                Result.success(LayerForwardResult(tokenArgmax = 6, hiddenOut = null))
            },
        )
        val result = executor.execute(offer("layer_forward"))
        assertEquals("extract_hidden", seen?.middleChannel)
        assertEquals("extract_hidden", result.metadata["middle_channel"])
    }

    @Test
    fun `unknown middle_channel is rejected by the protocol when building the offer`() {
        // 协议层在**构造** stage_offer 时就会校验值域 ⇒ 非法通道根本发不出去。
        try {
            offer("layer_forward", middleChannel = "bogus_channel")
            fail("unknown middle_channel must be rejected while building the offer")
        } catch (e: TaskWorkerProtocolException) {
            assertTrue(e.message.orEmpty().contains("middle_channel"))
        }
    }

    @Test
    fun `layer_forward executor also rejects an unknown middle_channel`() = runBlocking {
        // 执行器也可能被**直接**调用（不经协议层）⇒ 兜底校验同样要 fail-closed。
        var layerCalls = 0
        val executor = wiredExecutor(
            onLayer = {
                layerCalls++
                Result.success(LayerForwardResult(tokenArgmax = 1, hiddenOut = null))
            },
        )
        val env = offer("layer_forward")
        val tampered = env.copy(payload = env.payload + ("middle_channel" to "bogus_channel"))
        try {
            executor.execute(tampered)
            fail("unknown middle_channel must fail closed")
        } catch (e: AndroidFullWorkerStageException) {
            assertEquals("unsupported_middle_channel", e.code)
        }
        assertEquals(0, layerCalls)
    }

    private fun floatArrayToBase64(values: FloatArray): String {
        val buf = java.nio.ByteBuffer.allocate(values.size * 4)
            .order(java.nio.ByteOrder.LITTLE_ENDIAN)
        for (v in values) buf.putFloat(v)
        return java.util.Base64.getEncoder().encodeToString(buf.array())
    }
}
