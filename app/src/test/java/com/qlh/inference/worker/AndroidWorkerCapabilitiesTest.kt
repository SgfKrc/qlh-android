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
}
