package com.qlh.inference.worker

import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.isActive
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withContext
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import java.util.concurrent.CopyOnWriteArrayList
import java.util.concurrent.LinkedBlockingQueue
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean

/**
 * ★ 2026-10-07（DIST-NEXT-1）：取消合同。
 *
 * 层段 worker 收到 `stage_cancel` 后必须**立即**回 `stage_cancelled`，并如实报告
 * 执行状态：native 层段前向是一次原子调用，取消到达时它可能仍在飞
 * （`execution_in_flight`）；执行真正停止后再补一条 `execution_stopped`。
 * 取消一律走 `stage_cancelled` —— 不得编码成 `stage_error`。
 */
class TaskWorkerCancelContractTest {
    private val model = mapOf(
        "model_id" to "qwen_1_8b",
        "engine" to "llama_cpp",
        "format" to "gguf",
        "revision" to "local_v1",
        "sha256" to "a".repeat(64),
    )
    private val identity = TaskWorkerAttemptIdentity(
        workflowId = "wf_cancelcontract1",
        stageId = "candidate_a",
        attemptId = "att_cancelcontract1",
        leaseId = "lease_cancelcontract1",
        leaseEpoch = 1,
    )
    private val offeredProviderId = "remote_android_worker_01"

    @Test
    fun `layer worker acknowledges cancel immediately and upgrades to stopped`() = runBlocking {
        val transport = RecordingTransport()
        val handler = AtomicLayerHandler()
        val client = TaskWorkerClient(
            host = "127.0.0.1",
            port = 18822,
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
                messageId = "msg_ack_cancelcontract1",
                sentAtMs = 1_700_000_000_000L,
            ))
            awaitTrue {
                client.snapshot.value.connection == TaskWorkerConnectionState.READY
            }
            transport.push(offerEnvelope())
            awaitTrue { handler.started.isCompleted }

            transport.push(cancelEnvelope())

            // ① ACK 与「执行是否已停止」解耦：必须立刻到，且状态如实是 in-flight。
            awaitTrue { transport.messageTypes().contains(TaskWorkerProtocol.STAGE_CANCELLED) }
            val acknowledgement = transport.envelope(TaskWorkerProtocol.STAGE_CANCELLED)
            assertEquals(TaskWorkerProtocol.EXECUTION_IN_FLIGHT, acknowledgement.payload["execution_state"])
            assertEquals(identity.attemptId, acknowledgement.payload["attempt_id"])
            assertEquals(offeredProviderId, acknowledgement.payload["provider_id"])
            assertTrue(handler.cancelRequested)
            assertTrue("ACK 到达时执行仍应在飞", handler.inFlight)
            assertFalse(transport.messageTypes().contains(TaskWorkerProtocol.STAGE_RESULT))
            assertFalse(transport.messageTypes().contains(TaskWorkerProtocol.STAGE_ERROR))

            // ② 执行真正停止后，把 ACK 升级成终态；迟到的结果不得回传。
            handler.release.set(true)
            awaitTrue {
                transport.envelopes(TaskWorkerProtocol.STAGE_CANCELLED).size >= 2
            }
            val terminal = transport.envelopes(TaskWorkerProtocol.STAGE_CANCELLED).last()
            assertEquals(TaskWorkerProtocol.EXECUTION_STOPPED, terminal.payload["execution_state"])
            assertFalse(transport.messageTypes().contains(TaskWorkerProtocol.STAGE_RESULT))
            assertFalse(transport.messageTypes().contains(TaskWorkerProtocol.STAGE_ERROR))
            awaitTrue {
                client.snapshot.value.activeAttempt.state == TaskWorkerAttemptState.CANCELLED
            }
        } finally {
            handler.release.set(true)
            client.stopAndJoin()
        }
    }

    @Test
    fun `local cancel reports stage_cancelled with the offered provider id`() = runBlocking {
        val transport = RecordingTransport()
        val handler = AtomicLayerHandler()
        val client = TaskWorkerClient(
            host = "127.0.0.1",
            port = 18823,
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
                messageId = "msg_ack_cancelcontract2",
                sentAtMs = 1_700_000_000_000L,
            ))
            awaitTrue {
                client.snapshot.value.connection == TaskWorkerConnectionState.READY
            }
            transport.push(offerEnvelope())
            awaitTrue { handler.started.isCompleted }

            // ★ worker 侧主动取消（Service ACTION_CANCEL / 用户在设备上停止）：
            //   回程必须是 `stage_cancelled`，且 provider_id 用 offer 里的主节点口径。
            assertTrue(client.cancelActive("user_cancelled"))

            awaitTrue { transport.messageTypes().contains(TaskWorkerProtocol.STAGE_CANCELLED) }
            val acknowledgement = transport.envelope(TaskWorkerProtocol.STAGE_CANCELLED)
            assertEquals("user_cancelled", acknowledgement.payload["reason_code"])
            assertEquals(offeredProviderId, acknowledgement.payload["provider_id"])
            assertEquals(TaskWorkerProtocol.EXECUTION_IN_FLIGHT, acknowledgement.payload["execution_state"])
            assertFalse(transport.messageTypes().contains(TaskWorkerProtocol.STAGE_ERROR))

            handler.release.set(true)
            awaitTrue {
                transport.envelopes(TaskWorkerProtocol.STAGE_CANCELLED).size >= 2
            }
            assertEquals(
                TaskWorkerProtocol.EXECUTION_STOPPED,
                transport.envelopes(TaskWorkerProtocol.STAGE_CANCELLED).last().payload["execution_state"],
            )
        } finally {
            handler.release.set(true)
            client.stopAndJoin()
        }
    }

    @Test
    fun `cancel releases the attempt slot so the next offer is accepted`() = runBlocking {
        val transport = RecordingTransport()
        val handler = AtomicLayerHandler()
        val client = TaskWorkerClient(
            host = "127.0.0.1",
            port = 18824,
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
                messageId = "msg_ack_cancelcontract3",
                sentAtMs = 1_700_000_000_000L,
            ))
            awaitTrue {
                client.snapshot.value.connection == TaskWorkerConnectionState.READY
            }
            transport.push(offerEnvelope())
            awaitTrue { handler.started.isCompleted }

            transport.push(cancelEnvelope())
            awaitTrue { transport.messageTypes().contains(TaskWorkerProtocol.STAGE_CANCELLED) }

            // ★ 2026-10-07（真机 P0）回归：执行协程被 `executionJob.cancel()` 取消后，
            //   终态 ACK 与 `machine.cancelled()` 恰恰在那条（已取消的）协程上 ⇒ 收敛必须
            //   仍然发生，否则 attempt 永远停在 `CANCELLING`。
            handler.release.set(true)
            awaitTrue {
                client.snapshot.value.activeAttempt.state == TaskWorkerAttemptState.CANCELLED
            }
            awaitTrue {
                transport.envelopes(TaskWorkerProtocol.STAGE_CANCELLED).last()
                    .payload["execution_state"] == TaskWorkerProtocol.EXECUTION_STOPPED
            }

            // 槽位立即可复用：下一个 attempt（不同 attempt_id / lease）必须被接受，
            // 而不是 `worker_busy_or_lease_invalid`（现场症状正是后者持续到 lease 过期）。
            transport.push(secondOfferEnvelope())
            awaitTrue {
                transport.envelopes(TaskWorkerProtocol.STAGE_ACCEPT).any {
                    it.payload["attempt_id"] == secondIdentity.attemptId
                }
            }
            val accept = transport.envelopes(TaskWorkerProtocol.STAGE_ACCEPT)
                .first { it.payload["attempt_id"] == secondIdentity.attemptId }
            assertEquals(true, accept.payload["accepted"])
            assertFalse(
                "取消之后的 offer 不得再被判 busy",
                accept.payload["reason_code"] == "worker_busy_or_lease_invalid",
            )
        } finally {
            handler.release.set(true)
            client.stopAndJoin()
        }
    }

    private fun workerCapabilities(): Map<String, Any?> = mapOf(
        "stage_types" to listOf("full_inference"),
        "engines" to listOf("llama_cpp"),
        "models" to listOf(model),
        "max_concurrency" to 1,
    )

    private fun offerEnvelope(): TaskWorkerEnvelope = TaskWorkerProtocol.buildStageOffer(
        identity = identity,
        requestId = "request_cancelcontract1",
        stageType = "full_inference",
        providerId = offeredProviderId,
        leaseExpiresAtMs = System.currentTimeMillis() + 30_000L,
        rootInput = mapOf(
            "message" to "hello",
            "max_new_tokens" to 8,
            "context_size" to 2048,
        ),
        dependencies = emptyMap(),
        modelIdentity = model,
        messageId = "msg_offer_cancelcontract1",
        sentAtMs = System.currentTimeMillis(),
    )

    private fun cancelEnvelope(): TaskWorkerEnvelope = TaskWorkerProtocol.buildStageCancel(
        identity = identity,
        reasonCode = "coordinator_cancelled",
        messageId = "msg_cancel_cancelcontract1",
        sentAtMs = System.currentTimeMillis(),
    )

    /** 取消之后的下一个 attempt（不同 attempt_id / lease）：用于验证槽位已释放。 */
    private val secondIdentity = TaskWorkerAttemptIdentity(
        workflowId = "wf_cancelcontract2",
        stageId = "candidate_b",
        attemptId = "att_cancelcontract2",
        leaseId = "lease_cancelcontract2",
        leaseEpoch = 1,
    )

    private fun secondOfferEnvelope(): TaskWorkerEnvelope = TaskWorkerProtocol.buildStageOffer(
        identity = secondIdentity,
        requestId = "request_cancelcontract2",
        stageType = "full_inference",
        providerId = offeredProviderId,
        leaseExpiresAtMs = System.currentTimeMillis() + 30_000L,
        rootInput = mapOf(
            "message" to "hello again",
            "max_new_tokens" to 8,
            "context_size" to 2048,
        ),
        dependencies = emptyMap(),
        modelIdentity = model,
        messageId = "msg_offer_cancelcontract2",
        sentAtMs = System.currentTimeMillis(),
    )

    private fun awaitTrue(timeoutMs: Long = 5_000L, predicate: () -> Boolean) {
        val deadline = System.currentTimeMillis() + timeoutMs
        while (System.currentTimeMillis() < deadline) {
            if (predicate()) return
            Thread.sleep(5)
        }
        assertTrue("condition was not met within ${timeoutMs}ms", predicate())
    }

    private class RecordingTransport : TaskWorkerTransport {
        val sent = CopyOnWriteArrayList<TaskWorkerEnvelope>()
        private val inbound = LinkedBlockingQueue<TaskWorkerEnvelope>()

        /**
         * 真机的 `SocketTaskWorkerTransport.send` 走 `withContext(Dispatchers.IO)` ——
         * **在已被取消的协程里调用会立刻抛 `CancellationException`**。测试必须先具备这个语义，
         * 否则「取消后终态 ACK 与槽位释放正在一条被取消的协程上」这个真机缺陷在测试里不可见。
         */
        override suspend fun send(envelope: TaskWorkerEnvelope) {
            currentCoroutineContext().ensureActive()
            sent.add(envelope)
        }

        /** 可取消的等待：`loopJob.cancelAndJoin()` 是服务停止路径，必须能返回。 */
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

        fun envelope(type: String): TaskWorkerEnvelope = envelopes(type).first()
    }

    /**
     * 模拟「不可被协程取消打断的原子层段前向」：只在测试显式放行后返回。
     *
     * 真实 native 调用的语义正是如此 —— `executionJob.cancel()` 不会打断它，
     * 取消只能靠 `requestCancel` 设置的标志在下一个可分割边界生效。
     */
    private class AtomicLayerHandler : TaskWorkerStageHandler {
        val started = CompletableDeferred<Unit>()
        val release = AtomicBoolean(false)

        @Volatile
        var inFlight: Boolean = false

        @Volatile
        var cancelRequested: Boolean = false

        override suspend fun execute(offer: TaskWorkerEnvelope): TaskWorkerStageExecution {
            inFlight = true
            started.complete(Unit)
            try {
                val deadline = System.currentTimeMillis() + 5_000L
                while (!release.get() && System.currentTimeMillis() < deadline) {
                    Thread.sleep(5)
                }
            } finally {
                // ★ 2026-10-07（真机 P0 同源）：`inFlight` 必须在**所有**出口归位 ——
                //   `Thread.sleep` 被中断时若跳过赋值，就等价于「执行永远不会被观察到已停止」，
                //   取消收敛会一直等到 30 秒上限。真机的
                //   `AndroidFullWorkerStageExecutor` 是在 `finally` 里清 `runningIdentity`，
                //   测试 fake 必须同语义。
                inFlight = false
            }
            return TaskWorkerStageExecution(output = mapOf("content" to "late result"))
        }

        override fun requestCancel(identity: TaskWorkerAttemptIdentity) {
            cancelRequested = true
        }

        override fun isExecutionInFlight(identity: TaskWorkerAttemptIdentity): Boolean = inFlight
    }
}
