package com.qlh.inference.worker

import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.Base64
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicInteger

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

    @Test
    fun `DIST-NEXT-1 executor reports in-flight state and forwards abort`() = runBlocking {
        val aborts = AtomicInteger(0)
        val started = CompletableDeferred<Unit>()
        val release = AtomicBoolean(false)
        val executor = AndroidFullWorkerStageExecutor(
            expectedModelIdentity = { model },
            ensureModelLoaded = { Result.success(Unit) },
            generate = { _, _, _, _ ->
                started.complete(Unit)
                // 模拟不可被协程取消打断的原子执行（真实形态是 native 层段前向）。
                val deadline = System.currentTimeMillis() + 5_000L
                while (!release.get() && System.currentTimeMillis() < deadline) Thread.sleep(5)
                Result.success("late result")
            },
            requestExecutionAbort = {
                aborts.incrementAndGet()
                release.set(true)
            },
        )
        val identity = TaskWorkerAttemptIdentity(
            workflowId = "wf_android_exec3",
            stageId = "stage_1",
            attemptId = "att_android_exec3",
            leaseId = "lease_android_exec3",
            leaseEpoch = 1,
        )
        val offer = TaskWorkerProtocol.buildStageOffer(
            identity = identity,
            requestId = "request_android_exec3",
            stageType = "full_inference",
            providerId = "remote_android_worker_01",
            leaseExpiresAtMs = System.currentTimeMillis() + 30_000L,
            rootInput = mapOf("message" to "hello", "max_new_tokens" to 8),
            dependencies = emptyMap(),
            modelIdentity = model,
            messageId = "msg_android_exec03",
            sentAtMs = System.currentTimeMillis(),
        )

        val job = launch(Dispatchers.Default) { executor.execute(offer) }
        started.await()
        assertTrue(executor.isExecutionInFlight(identity))
        // 身份不符 ⇒ 不得转发中止（否则会打断别的 attempt）。
        executor.requestCancel(identity.copy(attemptId = "att_android_other"))
        assertEquals(0, aborts.get())
        executor.requestCancel(identity)
        assertEquals(1, aborts.get())
        job.join()
        assertFalse(executor.isExecutionInFlight(identity))
    }

    @Test
    fun `DIST-NEXT-2 layer output within budget still returns hidden`() = runBlocking {
        val nEmbd = 2048
        val executor = layerExecutor(nEmbd)
        val result = executor.execute(layerForwardOffer(nEmbd))

        assertEquals(7, result.output["token_argmax"])
        assertTrue(result.output.containsKey("hidden_out_f32"))
        assertTrue(result.output.containsKey("hidden_out_sha256"))
    }

    @Test
    fun `DIST-NEXT-2 oversized layer output is rejected before encoding`() = runBlocking {
        // 1_600_000 个 f32 ⇒ ≈ 8.53 MB base64 > 帧预算（8 MiB − 256 KiB）
        val nEmbd = 1_600_000
        val executor = layerExecutor(nEmbd)

        try {
            executor.execute(layerForwardOffer(nEmbd))
            assertTrue("超预算的层段输出必须在编码前被拒", false)
        } catch (error: AndroidFullWorkerStageException) {
            assertEquals("stage_result_too_large", error.code)
        }
    }

    private fun layerExecutor(nEmbd: Int) = AndroidFullWorkerStageExecutor(
        expectedModelIdentity = { model },
        ensureModelLoaded = { Result.success(Unit) },
        generate = { _, _, _, _ -> Result.success("unused") },
        layerForward = { _ ->
            Result.success(LayerForwardResult(tokenArgmax = 7, hiddenOut = FloatArray(nEmbd)))
        },
        ensureModelLoadedForLayer = { _, _, _, _, _ -> Result.success(Unit) },
        resolveLayerModelIdentity = { _, _ -> model },
    )

    private fun layerForwardOffer(nEmbd: Int): TaskWorkerEnvelope {
        val inputBytes = ByteBuffer.allocate(nEmbd * 4).order(ByteOrder.LITTLE_ENDIAN)
        repeat(nEmbd) { inputBytes.putFloat(0f) }
        return TaskWorkerProtocol.buildStageOffer(
            identity = TaskWorkerAttemptIdentity(
                workflowId = "wf_android_frame01",
                stageId = "stage_1",
                attemptId = "att_android_frame01",
                leaseId = "lease_android_frame01",
                leaseEpoch = 1,
            ),
            requestId = "request_android_frame01",
            stageType = "layer_forward",
            providerId = "remote_android_worker_01",
            leaseExpiresAtMs = System.currentTimeMillis() + 30_000L,
            rootInput = mapOf(
                "hidden_f32" to Base64.getEncoder().encodeToString(inputBytes.array()),
                "context_size" to 2048,
                "pos_base" to 0,
                "want_hidden" to true,
            ),
            dependencies = emptyMap(),
            modelIdentity = model,
            messageId = "msg_android_frame01",
            sentAtMs = System.currentTimeMillis(),
            stageFields = mapOf(
                "layer_range" to listOf(2, 4),
                "handoff_at" to 4,
                "hidden_sha256" to "a".repeat(64),
                "hidden_spec" to mapOf(
                    "n_tokens" to 1,
                    "n_embd" to nEmbd,
                    "dtype" to "float32",
                ),
                "middle_channel" to "keep_head_layer_out",
            ),
        )
    }
}
