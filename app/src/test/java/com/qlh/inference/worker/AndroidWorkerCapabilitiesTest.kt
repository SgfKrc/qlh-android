package com.qlh.inference.worker

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class AndroidWorkerCapabilitiesTest {
    @Test
    fun `resource gate defaults closed and withholds model identity`() {
        val capabilities = AndroidWorkerCapabilities.build(modelId = "qwen_1_8b")
        assertEquals(1, capabilities["max_concurrency"])
        assertTrue((capabilities["models"] as List<*>).isEmpty())
        assertEquals(
            mapOf("admitted" to false, "reason_code" to "resource_gate_not_confirmed"),
            capabilities["resource_gate"],
        )
        val hello = TaskWorkerProtocol.buildHello(
            nodeId = "android_worker_01",
            capabilities = capabilities,
            messageId = "msg_capabilities_android01",
            sentAtMs = 1_700_000_000_000,
        )
        TaskWorkerProtocol.validate(hello)
    }

    @Test
    fun `admitted gate advertises one exact llama model`() {
        val capabilities = AndroidWorkerCapabilities.build(
            modelId = "qwen_1_8b",
            modelFormat = "gguf",
            modelRevision = "local-v1",
            modelSha256 = "A".repeat(64),
            resourceAdmitted = true,
        )
        val models = capabilities["models"] as List<*>
        assertEquals(1, models.size)
        assertEquals(
            mapOf(
                "model_id" to "qwen_1_8b",
                "engine" to "llama_cpp",
                "format" to "gguf",
                "revision" to "local-v1",
                "sha256" to "a".repeat(64),
            ),
            models.single(),
        )
        val hello = TaskWorkerProtocol.buildHello(
            nodeId = "android_worker_01",
            capabilities = capabilities,
            messageId = "msg_capabilities_android02",
            sentAtMs = 1_700_000_000_000,
        )
        TaskWorkerProtocol.validate(hello)
    }

    @Test
    fun `middle_channel and n_pos_per_embd are advertised when native reports them`() {
        // ★ 2026-09-23：native `layerForwardInfo()` 上报 ⇒ worker capabilities 里要能看到
        //   中间段通道与 M-RoPE 位置分量数；缺省不写这两个键（向后兼容）。
        val advertised = AndroidWorkerCapabilities.build(
            modelId = "qwen3_5_9b",
            resourceAdmitted = true,
            middleChannel = "keep_head_layer_out",
            nPosPerEmbd = 4,
        )
        assertEquals("keep_head_layer_out", advertised["middle_channel"])
        assertEquals(4, (advertised["n_pos_per_embd"] as? Number)?.toInt())
        TaskWorkerProtocol.validate(
            TaskWorkerProtocol.buildHello(
                nodeId = "android_worker_01",
                capabilities = advertised,
                messageId = "msg_capabilities_channel_01",
                sentAtMs = 1_700_000_000_000,
            ),
        )

        // 不传 ⇒ 两个键都不出现（旧行为不变）。
        val plain = AndroidWorkerCapabilities.build(modelId = "qwen3_5_9b")
        assertTrue(!plain.containsKey("middle_channel"))
        assertTrue(!plain.containsKey("n_pos_per_embd"))
    }
}
