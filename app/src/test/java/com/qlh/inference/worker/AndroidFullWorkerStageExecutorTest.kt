package com.qlh.inference.worker

import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class AndroidFullWorkerStageExecutorTest {
    private val model = mapOf(
        "model_id" to "qwen_1_8b",
        "engine" to "llama_cpp",
        "format" to "gguf",
        "revision" to "local-v1",
        "sha256" to "a".repeat(64),
    )

    @Test
    fun `fake Android executor runs bounded prompt and returns path free output`() = runBlocking {
        var loadedContext = 0
        var receivedPrompt = ""
        val executor = AndroidFullWorkerStageExecutor(
            expectedModelIdentity = { model },
            ensureModelLoaded = { context ->
                loadedContext = context
                Result.success(Unit)
            },
            generate = { prompt, maxTokens, temperature, topP ->
                receivedPrompt = prompt
                assertEquals(64, maxTokens)
                assertEquals(0.2f, temperature)
                assertEquals(0.8f, topP)
                Result.success("android fake result")
            },
        )
        val offer = TaskWorkerProtocol.buildStageOffer(
            identity = TaskWorkerAttemptIdentity(
                workflowId = "wf_android_exec1",
                stageId = "stage_1",
                attemptId = "att_android_exec1",
                leaseId = "lease_android_exec1",
                leaseEpoch = 1,
            ),
            requestId = "request_android_exec1",
            stageType = "full_inference",
            providerId = "remote_android_worker_01",
            leaseExpiresAtMs = 2_000,
            rootInput = mapOf(
                "messages" to listOf(
                    mapOf("role" to "user", "content" to "hello Android"),
                ),
                "max_new_tokens" to 64,
                "temperature" to 0.2,
                "top_p" to 0.8,
                "context_size" to 4096,
            ),
            dependencies = emptyMap(),
            modelIdentity = model,
            messageId = "msg_android_exec01",
            sentAtMs = 1_000,
        )

        val result = executor.execute(offer)

        assertEquals(4096, loadedContext)
        assertEquals("user: hello Android", receivedPrompt)
        assertEquals(mapOf("content" to "android fake result"), result.output)
        assertEquals(mapOf("model" to "qwen_1_8b"), result.metadata)
    }

    @Test
    fun `fake Android executor rejects model mismatch before loading`() = runBlocking {
        var loadCalled = false
        val executor = AndroidFullWorkerStageExecutor(
            expectedModelIdentity = { model },
            ensureModelLoaded = {
                loadCalled = true
                Result.success(Unit)
            },
            generate = { _, _, _, _ -> Result.success("must not run") },
        )
        val offer = TaskWorkerProtocol.buildStageOffer(
            identity = TaskWorkerAttemptIdentity(
                workflowId = "wf_android_exec2",
                stageId = "stage_1",
                attemptId = "att_android_exec2",
                leaseId = "lease_android_exec2",
                leaseEpoch = 1,
            ),
            requestId = "request_android_exec2",
            stageType = "full_inference",
            providerId = "remote_android_worker_01",
            leaseExpiresAtMs = 2_000,
            rootInput = mapOf("message" to "hello"),
            dependencies = emptyMap(),
            modelIdentity = model + ("revision" to "other"),
            messageId = "msg_android_exec02",
            sentAtMs = 1_000,
        )

        try {
            executor.execute(offer)
            assertTrue("model mismatch must fail", false)
        } catch (error: AndroidFullWorkerStageException) {
            assertEquals("model_identity_mismatch", error.code)
        }
        assertTrue(!loadCalled)
    }
}
