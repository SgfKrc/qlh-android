package com.qlh.inference.worker

import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.isActive
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withContext
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import java.util.concurrent.CopyOnWriteArrayList
import java.util.concurrent.LinkedBlockingQueue
import java.util.concurrent.TimeUnit

/**
 * ★ 2026-10-07（DIST-NEXT-2b）：Android 侧的**分片输入装配**（client 级）。
 *
 * 顺序：coordinator 先发 `stage_chunk`，再发带 `hidden_ref` 的 `stage_offer`；
 * client 在 accept 之前装配并回填 `hidden_f32`，执行器因此无需感知分片。
 */
class TaskWorkerChunkedInputContractTest {
    private val model = mapOf(
        "model_id" to "qwen_1_8b",
        "engine" to "llama_cpp",
        "format" to "gguf",
        "revision" to "local_v1",
        "sha256" to "a".repeat(64),
    )
    private val identity = TaskWorkerAttemptIdentity(
        workflowId = "wf_chunkedinputa1",
        stageId = "candidate_a",
        attemptId = "att_chunkedinputa1",
        leaseId = "lease_chunkedinputa1",
        leaseEpoch = 1,
    )
    private val offeredProviderId = "remote_android_worker_01"

    @Test
    fun `chunked hidden is assembled before the stage is accepted`() = runBlocking {
        val raw = ByteArray(5000) { (it % 253).toByte() }
        val chunks = listOf(
            raw.copyOfRange(0, 2000),
            raw.copyOfRange(2000, 4000),
            raw.copyOfRange(4000, raw.size),
        )
        val transport = RecordingTransport()
        val handler = CapturingHandler()
        val client = TaskWorkerClient(
            host = "127.0.0.1",
            port = 18831,
            nodeId = "android_worker_01",
            capabilities = { workerCapabilities() },
            transportFactory = { _, _ -> transport },
            stageHandler = handler,
        )
        try {
            client.start()
            awaitTrue { transport.messageTypes().contains(TaskWorkerProtocol.HELLO) }
            transport.push(TaskWorkerProtocol.buildHelloAck(
                coordinatorNodeId = "master_12345678",
                accepted = true,
                selectedVersion = TaskWorkerProtocol.VERSION,
                reasonCode = "",
                messageId = "msg_ack_chunkedinput1",
                sentAtMs = 1_700_000_000_000L,
            ))
            awaitTrue { client.snapshot.value.connection == TaskWorkerConnectionState.READY }

            // 分片先到（乱序：2、0、1）
            for (index in listOf(2, 0, 1)) {
                transport.push(chunkEnvelope(chunks[index], index))
            }
            transport.push(offerEnvelope(raw))

            awaitTrue { handler.received.isCompleted }
            val delivered = handler.received.await()
            val deliveredRoot = delivered.payload["root_input"] as Map<*, *>
            val assembled = java.util.Base64.getDecoder()
                .decode(deliveredRoot["hidden_f32"] as String)
            assertArrayEquals(raw, assembled)
            assertFalse(deliveredRoot.containsKey("hidden_ref"))
            assertEquals(true, deliveredRoot["want_hidden"])
            assertEquals(0, transport.messageTypes().count {
                it == TaskWorkerProtocol.STAGE_ERROR
            })
            assertTrue(transport.messageTypes().contains(TaskWorkerProtocol.STAGE_ACCEPT))
        } finally {
            client.stopAndJoin()
        }
    }

    @Test
    fun `incomplete chunks are refused at accept with a named reason`() = runBlocking {
        val raw = ByteArray(4000) { 9 }
        val transport = RecordingTransport()
        val handler = CapturingHandler()
        val client = TaskWorkerClient(
            host = "127.0.0.1",
            port = 18832,
            nodeId = "android_worker_01",
            capabilities = { workerCapabilities() },
            transportFactory = { _, _ -> transport },
            stageHandler = handler,
        )
        try {
            client.start()
            awaitTrue { transport.messageTypes().contains(TaskWorkerProtocol.HELLO) }
            transport.push(TaskWorkerProtocol.buildHelloAck(
                coordinatorNodeId = "master_12345678",
                accepted = true,
                selectedVersion = TaskWorkerProtocol.VERSION,
                reasonCode = "",
                messageId = "msg_ack_chunkedinput2",
                sentAtMs = 1_700_000_000_000L,
            ))
            awaitTrue { client.snapshot.value.connection == TaskWorkerConnectionState.READY }

            // 只发第 0 片（共 2 片）⇒ offer 到达时装配不齐
            transport.push(chunkEnvelope(raw.copyOfRange(0, 2000), 0, chunkCount = 2))
            transport.push(offerEnvelope(raw, chunkCount = 2))

            awaitTrue { transport.messageTypes().contains(TaskWorkerProtocol.STAGE_ACCEPT) }
            val accept = transport.envelopes(TaskWorkerProtocol.STAGE_ACCEPT).first()
            assertEquals(false, accept.payload["accepted"])
            assertEquals("incomplete_chunks", accept.payload["reason_code"])
            assertFalse("装配不齐时不得执行", handler.received.isCompleted)
        } finally {
            client.stopAndJoin()
        }
    }

    private fun workerCapabilities(): Map<String, Any?> = mapOf(
        "stage_types" to listOf("layer_forward"),
        "engines" to listOf("llama_cpp"),
        "models" to listOf(model),
        "max_concurrency" to 1,
        "stage_chunked_input" to true,
    )

    private fun chunkEnvelope(
        chunk: ByteArray,
        index: Int,
        chunkCount: Int = 3,
    ): TaskWorkerEnvelope = TaskWorkerProtocol.build(
        messageType = TaskWorkerProtocol.STAGE_CHUNK,
        payload = identity.asPayload() + mapOf(
            "provider_id" to offeredProviderId,
            "chunk_index" to index,
            "chunk_count" to chunkCount,
            "payload_b64" to java.util.Base64.getEncoder().encodeToString(chunk),
            "payload_sha256" to StageChunkAssembler.sha256Hex(chunk),
            "total_bytes" to chunk.size,
        ),
        messageId = "msg_chunk_inbound_$index",
        sentAtMs = System.currentTimeMillis(),
    )

    private fun offerEnvelope(raw: ByteArray, chunkCount: Int = 3): TaskWorkerEnvelope =
        TaskWorkerProtocol.buildStageOffer(
            identity = identity,
            requestId = "request_chunkedinputa1",
            stageType = "layer_forward",
            providerId = offeredProviderId,
            leaseExpiresAtMs = System.currentTimeMillis() + 30_000L,
            rootInput = mapOf(
                "hidden_ref" to mapOf(
                    "chunk_count" to chunkCount,
                    "total_bytes" to raw.size,
                    "payload_sha256" to StageChunkAssembler.sha256Hex(raw),
                ),
                "context_size" to 2048,
                "pos_base" to 0,
                "want_hidden" to true,
            ),
            dependencies = emptyMap(),
            modelIdentity = model,
            messageId = "msg_offer_chunkedinput1",
            sentAtMs = System.currentTimeMillis(),
            stageFields = mapOf(
                "layer_range" to listOf(2, 4),
                "handoff_at" to 4,
                "hidden_sha256" to StageChunkAssembler.sha256Hex(raw),
                "hidden_spec" to mapOf(
                    "n_tokens" to 1, "n_embd" to (raw.size / 4), "dtype" to "float32",
                ),
                "middle_channel" to "keep_head_layer_out",
            ),
        )

    private fun awaitTrue(timeoutMs: Long = 5_000L, predicate: () -> Boolean) {
        val deadline = System.currentTimeMillis() + timeoutMs
        while (System.currentTimeMillis() < deadline) {
            if (predicate()) return
            Thread.sleep(5)
        }
        assertTrue("condition was not met within ${timeoutMs}ms", predicate())
    }

    private class CapturingHandler : TaskWorkerStageHandler {
        val received = CompletableDeferred<TaskWorkerEnvelope>()

        override suspend fun execute(offer: TaskWorkerEnvelope): TaskWorkerStageExecution {
            received.complete(offer)
            return TaskWorkerStageExecution(output = mapOf("token_argmax" to 1))
        }
    }

    private class RecordingTransport : TaskWorkerTransport {
        val sent = CopyOnWriteArrayList<TaskWorkerEnvelope>()
        private val inbound = LinkedBlockingQueue<TaskWorkerEnvelope>()

        override suspend fun send(envelope: TaskWorkerEnvelope) {
            sent.add(envelope)
        }

        override suspend fun receive(): TaskWorkerEnvelope? = withContext(Dispatchers.IO) {
            while (coroutineContext.isActive) {
                val item = inbound.poll(50L, TimeUnit.MILLISECONDS)
                if (item != null) return@withContext item
            }
            null
        }

        override suspend fun close() {}

        fun push(envelope: TaskWorkerEnvelope) {
            inbound.put(envelope)
        }

        fun messageTypes(): List<String> = sent.map { it.messageType }

        fun envelopes(type: String): List<TaskWorkerEnvelope> =
            sent.filter { it.messageType == type }
    }
}
