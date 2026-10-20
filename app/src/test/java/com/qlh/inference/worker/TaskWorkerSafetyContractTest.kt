package com.qlh.inference.worker

import kotlinx.coroutines.CancellableContinuation
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Delay
import kotlinx.coroutines.InternalCoroutinesApi
import kotlinx.coroutines.NonCancellable
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.async
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withContext
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import java.util.PriorityQueue
import kotlin.coroutines.CoroutineContext
import kotlin.coroutines.resume

/** COMM-MERGE-01: deterministic lifecycle tests; no network, wall-clock sleep or extra test dependency. */
class TaskWorkerSafetyContractTest {
    private val identity = TaskWorkerAttemptIdentity(
        "wf_safety0001", "stage_1", "att_safety0001", "lease_safety0001", 1,
    )
    private val model = mapOf("model_id" to "test_model", "engine" to "llama_cpp",
        "format" to "gguf", "revision" to "local_v1", "sha256" to "a".repeat(64))

    @Test
    fun `lease expiring during accept write never launches the executor`() {
        val rig = Rig()
        val releaseWrite = CompletableDeferred<Unit>()
        rig.transport.beforeSend = {
            if (it.messageType == TaskWorkerProtocol.STAGE_ACCEPT && it.payload["accepted"] == true) {
                releaseWrite.await()
            }
        }
        try {
            rig.start()
            rig.push(offer(rig, expires = rig.now + 100))
            assertEquals(TaskWorkerAttemptState.OFFERED, rig.machine.snapshot().activeAttempt.state)
            rig.dispatcher.advanceBy(101)
            releaseWrite.complete(Unit)
            rig.dispatcher.runCurrent()
            assertEquals(0, rig.handler.executions)
            val error = rig.transport.messages(TaskWorkerProtocol.STAGE_ERROR).single()
            assertEquals("worker_lease_expired_before_execution", error.payload["error_code"])
            TaskWorkerProtocol.validate(error)
            assertTrue(rig.transport.messages(TaskWorkerProtocol.STAGE_RESULT).isEmpty())
        } finally { releaseWrite.complete(Unit); rig.close() }
    }

    @Test
    fun `cancellation crossing the result check suppresses success and converges`() {
        val rig = Rig()
        var cancelledDuringHash = false
        rig.handler.output = {
            object : AbstractMap<String, Any?>() {
                override val entries: Set<Map.Entry<String, Any?>>
                    get() {
                        if (!cancelledDuringHash) {
                            cancelledDuringHash = true
                            assertEquals(identity, rig.machine.requestCancel(rig.now, identity))
                        }
                        return mapOf<String, Any?>("content" to "late result").entries
                    }
            }
        }
        try {
            rig.start(); rig.push(offer(rig))
            assertTrue(cancelledDuringHash)
            assertEquals(1, rig.handler.executions)
            assertTrue(rig.transport.messages(TaskWorkerProtocol.STAGE_RESULT).isEmpty())
            assertTrue(rig.transport.messages(TaskWorkerProtocol.STAGE_ERROR).isEmpty())
            assertEquals(TaskWorkerAttemptState.CANCELLED, rig.client.snapshot.value.activeAttempt.state)
            assertEquals(listOf(TaskWorkerProtocol.EXECUTION_STOPPED), rig.cancelStates())
        } finally { rig.close() }
    }

    @Test
    fun `thirty second timeout retains physical slot even after lease expiry and later converges`() {
        val rig = Rig()
        val releaseNative = CompletableDeferred<Unit>()
        rig.handler.release = releaseNative
        val replacement = identity.copy(attemptId = "att_safety0002", leaseId = "lease_safety0002")
        try {
            rig.start(); rig.push(offer(rig, expires = rig.now + 10_000))
            rig.push(cancel(rig))
            assertTrue(rig.handler.isExecutionInFlight(identity))
            assertEquals(listOf(TaskWorkerProtocol.EXECUTION_IN_FLIGHT), rig.cancelStates())
            rig.dispatcher.advanceBy(TaskWorkerClient.CANCEL_STOP_WAIT_MS + 1)
            assertEquals(TaskWorkerAttemptState.CANCELLING, rig.client.snapshot.value.activeAttempt.state)
            assertEquals(listOf(TaskWorkerProtocol.EXECUTION_IN_FLIGHT), rig.cancelStates())
            rig.push(offer(rig, replacement))
            val refused = rig.transport.messages(TaskWorkerProtocol.STAGE_ACCEPT).last()
            assertEquals(false, refused.payload["accepted"])
            assertEquals("worker_execution_in_flight", refused.payload["reason_code"])
            assertEquals(1, rig.handler.executions)
            // Renew also visits the lease-expiry transition; LOST must not hide physical ownership.
            rig.push(TaskWorkerProtocol.buildLeaseRenew(identity, rig.now + 100_000,
                "msg_renew_safety01", rig.now))
            rig.push(offer(rig, replacement))
            assertEquals("worker_execution_in_flight",
                rig.transport.messages(TaskWorkerProtocol.STAGE_ACCEPT).last().payload["reason_code"])
            assertEquals(1, rig.handler.executions)
            releaseNative.complete(Unit)
            rig.dispatcher.runCurrent()
            rig.dispatcher.advanceBy(TaskWorkerClient.CANCEL_STOP_POLL_MS)
            assertFalse(rig.handler.isExecutionInFlight(identity))
            assertTrue(rig.transport.messages(TaskWorkerProtocol.STAGE_RESULT).isEmpty())
            assertEquals(listOf(TaskWorkerProtocol.EXECUTION_IN_FLIGHT, TaskWorkerProtocol.EXECUTION_STOPPED),
                rig.cancelStates())
            rig.push(offer(rig, replacement))
            assertEquals(true, rig.transport.messages(TaskWorkerProtocol.STAGE_ACCEPT).last().payload["accepted"])
            assertEquals(2, rig.handler.executions)
        } finally { releaseNative.complete(Unit); rig.close() }
    }

    @Test
    fun `timeout within a live lease sends stopped only after native actually exits`() {
        val rig = Rig()
        val releaseNative = CompletableDeferred<Unit>()
        rig.handler.release = releaseNative
        try {
            rig.start(); rig.push(offer(rig)); rig.push(cancel(rig))
            rig.dispatcher.advanceBy(TaskWorkerClient.CANCEL_STOP_WAIT_MS + 1)
            assertEquals(listOf(TaskWorkerProtocol.EXECUTION_IN_FLIGHT), rig.cancelStates())
            assertEquals(TaskWorkerAttemptState.CANCELLING, rig.client.snapshot.value.activeAttempt.state)
            releaseNative.complete(Unit)
            rig.dispatcher.runCurrent()
            rig.dispatcher.advanceBy(TaskWorkerClient.CANCEL_STOP_POLL_MS)
            assertEquals(listOf(TaskWorkerProtocol.EXECUTION_IN_FLIGHT, TaskWorkerProtocol.EXECUTION_STOPPED),
                rig.cancelStates())
            assertEquals(TaskWorkerAttemptState.CANCELLED, rig.client.snapshot.value.activeAttempt.state)
        } finally { releaseNative.complete(Unit); rig.close() }
    }

    @Test
    fun `duplicate successful offer is refused before and after reconnect`() {
        val rig = Rig()
        try {
            rig.start()
            val original = offer(rig)
            rig.push(original)
            assertEquals(1, rig.transport.messages(TaskWorkerProtocol.STAGE_RESULT).size)
            rig.push(original)
            assertEquals("worker_attempt_already_admitted",
                rig.transport.messages(TaskWorkerProtocol.STAGE_ACCEPT).last().payload["reason_code"])
            assertEquals(1, rig.handler.executions)
            val second = RecordingTransport()
            rig.transports += second
            second.push(helloAck(rig))
            rig.transport.push(null)
            rig.dispatcher.runCurrent()
            rig.dispatcher.advanceBy(1_000)
            assertEquals(TaskWorkerConnectionState.READY, rig.client.snapshot.value.connection)
            second.push(original)
            rig.dispatcher.runCurrent()
            val refused = second.messages(TaskWorkerProtocol.STAGE_ACCEPT).single()
            assertEquals(false, refused.payload["accepted"])
            assertEquals(false, refused.payload["retryable"])
            assertEquals("worker_attempt_already_admitted", refused.payload["reason_code"])
            assertEquals(1, rig.handler.executions)
            assertTrue(second.messages(TaskWorkerProtocol.STAGE_RESULT).isEmpty())
        } finally { rig.close() }
    }

    @Test
    fun `cancel ACK writes are serialized and a new lease epoch is not deduplicated away`() {
        val rig = Rig()
        val releaseAck = CompletableDeferred<Unit>()
        var blocked = false
        rig.transport.beforeSend = {
            if (!blocked && it.messageType == TaskWorkerProtocol.STAGE_CANCELLED) {
                blocked = true
                releaseAck.await()
            }
        }
        val firstRelease = CompletableDeferred<Unit>()
        rig.handler.release = firstRelease
        try {
            rig.start(); rig.push(offer(rig)); rig.push(cancel(rig))
            assertTrue(blocked)
            firstRelease.complete(Unit)
            rig.dispatcher.runCurrent()
            assertTrue(rig.transport.messages(TaskWorkerProtocol.STAGE_CANCELLED).isEmpty())
            releaseAck.complete(Unit)
            rig.dispatcher.runCurrent()
            rig.dispatcher.advanceBy(TaskWorkerClient.CANCEL_STOP_POLL_MS)
            assertEquals(listOf(TaskWorkerProtocol.EXECUTION_IN_FLIGHT, TaskWorkerProtocol.EXECUTION_STOPPED),
                rig.cancelStates())
            val nextEpoch = identity.copy(leaseEpoch = 2)
            val secondRelease = CompletableDeferred<Unit>()
            rig.handler.release = secondRelease
            rig.push(offer(rig, nextEpoch)); rig.push(cancel(rig, nextEpoch))
            secondRelease.complete(Unit)
            rig.dispatcher.runCurrent()
            rig.dispatcher.advanceBy(TaskWorkerClient.CANCEL_STOP_POLL_MS)
            val states = rig.transport.messages(TaskWorkerProtocol.STAGE_CANCELLED)
                .filter { it.payload["lease_epoch"] == 2 }.map { it.payload["execution_state"] }
            assertEquals(listOf(TaskWorkerProtocol.EXECUTION_IN_FLIGHT, TaskWorkerProtocol.EXECUTION_STOPPED), states)
            assertEquals(2, rig.handler.executions)
        } finally { releaseAck.complete(Unit); rig.handler.release?.complete(Unit); rig.close() }
    }

    @Test
    fun `next stage starts when result is visible before its send returns`() {
        assertNextStageDuringTerminalWrite(failFirst = false)
    }

    @Test
    fun `next stage starts when error is visible before its send returns`() {
        assertNextStageDuringTerminalWrite(failFirst = true)
    }

    private fun assertNextStageDuringTerminalWrite(failFirst: Boolean) {
        val rig = Rig()
        val releaseWrite = CompletableDeferred<Unit>()
        val next = identity.copy(attemptId = "att_safety_next01", leaseId = "lease_safety_next01")
        val terminal = if (failFirst) TaskWorkerProtocol.STAGE_ERROR else TaskWorkerProtocol.STAGE_RESULT
        rig.transport.afterSend = { envelope ->
            if (envelope.messageType == terminal && envelope.payload["attempt_id"] == identity.attemptId) {
                // The peer has the complete response while the local send is still suspended.
                releaseWrite.await()
            }
        }
        rig.handler.output = {
            if (failFirst) throw IllegalStateException("injected executor failure")
            mapOf("content" to "first result")
        }
        try {
            rig.start(); rig.push(offer(rig))
            assertEquals(1, rig.transport.messages(terminal).size)
            assertFalse("the terminal writer is deliberately still suspended", releaseWrite.isCompleted)
            assertFalse(rig.handler.isExecutionInFlight(identity))
            rig.handler.output = { mapOf("content" to "next result") }
            rig.push(offer(rig, next).copy(messageId = "msg_offer_next_safety01"))
            val accepted = rig.transport.messages(TaskWorkerProtocol.STAGE_ACCEPT).last()
            assertEquals("the peer may submit its next stage immediately after terminal delivery",
                true, accepted.payload["accepted"])
            assertEquals(next.attemptId, accepted.payload["attempt_id"])
            assertEquals(2, rig.handler.executions)
            assertTrue(rig.transport.messages(TaskWorkerProtocol.STAGE_RESULT)
                .any { it.payload["attempt_id"] == next.attemptId })
            assertEquals(next, rig.client.snapshot.value.activeAttempt.identity)
            releaseWrite.complete(Unit)
            rig.dispatcher.runCurrent()
            assertEquals("old receipt completion must not change the successor", next,
                rig.client.snapshot.value.activeAttempt.identity)
        } finally { releaseWrite.complete(Unit); rig.dispatcher.runCurrent(); rig.close() }
    }

    @Test
    fun `old terminal write failure cannot cancel the next running stage`() {
        val rig = Rig()
        val releaseWrite = CompletableDeferred<Unit>()
        val releaseNext = CompletableDeferred<Unit>()
        val next = identity.copy(attemptId = "att_safety_next02", leaseId = "lease_safety_next02")
        rig.transport.afterSend = { envelope ->
            if (envelope.messageType == TaskWorkerProtocol.STAGE_RESULT &&
                envelope.payload["attempt_id"] == identity.attemptId) {
                releaseWrite.await()
                throw java.io.IOException("old response send failed after peer delivery")
            }
        }
        try {
            rig.start(); rig.push(offer(rig))
            assertEquals(1, rig.transport.messages(TaskWorkerProtocol.STAGE_RESULT).size)
            rig.handler.release = releaseNext
            rig.push(offer(rig, next).copy(messageId = "msg_offer_next_safety02"))
            assertEquals(true, rig.transport.messages(TaskWorkerProtocol.STAGE_ACCEPT).last().payload["accepted"])
            assertTrue(rig.handler.isExecutionInFlight(next))
            releaseWrite.complete(Unit)
            rig.dispatcher.runCurrent()
            assertEquals(next, rig.client.snapshot.value.activeAttempt.identity)
            assertEquals(TaskWorkerAttemptState.RUNNING, rig.client.snapshot.value.activeAttempt.state)
            assertTrue(rig.handler.isExecutionInFlight(next))
            releaseNext.complete(Unit)
            rig.dispatcher.runCurrent()
            assertEquals(TaskWorkerAttemptState.SUCCEEDED, rig.client.snapshot.value.activeAttempt.state)
            assertEquals(2, rig.handler.executions)
            assertEquals(1, rig.transport.messages(TaskWorkerProtocol.STAGE_RESULT)
                .count { it.payload["attempt_id"] == next.attemptId })
        } finally { releaseWrite.complete(Unit); releaseNext.complete(Unit); rig.dispatcher.runCurrent(); rig.close() }
    }

    private fun offer(rig: Rig, attempt: TaskWorkerAttemptIdentity = identity, expires: Long = rig.now + 120_000) =
        TaskWorkerProtocol.buildStageOffer(attempt, "request_safety01", "full_inference", "remote_android_test",
            expires, mapOf("message" to "hello", "max_new_tokens" to 8), emptyMap(), model,
            "msg_offer_safety01", rig.now)

    private fun cancel(rig: Rig, attempt: TaskWorkerAttemptIdentity = identity) =
        TaskWorkerProtocol.buildStageCancel(attempt, "coordinator_cancelled", "msg_cancel_safety01", rig.now)

    private fun helloAck(rig: Rig) = TaskWorkerProtocol.buildHelloAck("master_12345678", true,
        TaskWorkerProtocol.VERSION, "", "msg_hello_safety01", rig.now)

    private inner class Rig {
        val dispatcher = VirtualTimeDispatcher()
        val now: Long get() = 1_000L + dispatcher.nowMs
        val transport = RecordingTransport()
        val transports = mutableListOf(transport)
        var connections = 0
        val handler = ControlledHandler()
        val machine = TaskWorkerStateMachine()
        val client = TaskWorkerClient("127.0.0.1", 18828, "android_test",
            capabilities = { mapOf("stage_types" to listOf("full_inference"), "engines" to listOf("llama_cpp"),
                "models" to listOf(model), "max_concurrency" to 1) },
            transportFactory = { _, _ -> transports[connections++] }, stageHandler = handler,
            scope = CoroutineScope(SupervisorJob() + dispatcher), clockMs = { now }, machine = machine,
            monotonicMs = { dispatcher.nowMs })

        fun start() {
            client.start(); dispatcher.runCurrent()
            transport.push(helloAck(this)); dispatcher.runCurrent()
            assertEquals(TaskWorkerConnectionState.READY, client.snapshot.value.connection)
        }
        fun push(message: TaskWorkerEnvelope) { transport.push(message); dispatcher.runCurrent() }
        fun cancelStates() = transport.messages(TaskWorkerProtocol.STAGE_CANCELLED).map { it.payload["execution_state"] }
        fun close() {
            handler.release?.complete(Unit)
            dispatcher.runCurrent()
            runBlocking {
                val stopped = async(dispatcher) { client.stopAndJoin() }
                dispatcher.runCurrent()
                assertTrue("retirement must join without advancing virtual time", stopped.isCompleted)
                stopped.await()
            }
        }
    }

    private class RecordingTransport : TaskWorkerTransport {
        private val inbound = Channel<TaskWorkerEnvelope?>(Channel.UNLIMITED)
        private val sent = mutableListOf<TaskWorkerEnvelope>()
        var beforeSend: suspend (TaskWorkerEnvelope) -> Unit = {}
        var afterSend: suspend (TaskWorkerEnvelope) -> Unit = {}
        override suspend fun send(envelope: TaskWorkerEnvelope) {
            currentCoroutineContext().ensureActive()
            beforeSend(envelope)
            sent += envelope
            afterSend(envelope)
        }
        override suspend fun receive(): TaskWorkerEnvelope? = inbound.receive()
        override suspend fun close() { inbound.close() }
        fun push(envelope: TaskWorkerEnvelope?) { check(inbound.trySend(envelope).isSuccess) }
        fun messages(type: String) = sent.filter { it.messageType == type }
    }

    private class ControlledHandler : TaskWorkerStageHandler {
        var executions = 0
        var release: CompletableDeferred<Unit>? = null
        var output: () -> Map<String, Any?> = { mapOf("content" to "result") }
        private var running: TaskWorkerAttemptIdentity? = null
        override suspend fun execute(offer: TaskWorkerEnvelope): TaskWorkerStageExecution {
            executions++
            running = TaskWorkerAttemptIdentity(offer.payload["workflow_id"] as String,
                offer.payload["stage_id"] as String, offer.payload["attempt_id"] as String,
                offer.payload["lease_id"] as String, (offer.payload["lease_epoch"] as Number).toInt())
            try {
                release?.let { withContext(NonCancellable) { it.await() } }
                return TaskWorkerStageExecution(output())
            } finally { running = null }
        }
        override fun isExecutionInFlight(identity: TaskWorkerAttemptIdentity) = running == identity
    }

    @OptIn(InternalCoroutinesApi::class)
    private class VirtualTimeDispatcher : CoroutineDispatcher(), Delay {
        private data class Scheduled(val at: Long, val order: Long, val action: Runnable)
        private val pending = PriorityQueue(compareBy<Scheduled> { it.at }.thenBy { it.order })
        private var sequence = 0L
        var nowMs = 0L
            private set
        override fun dispatch(context: CoroutineContext, block: Runnable) { schedule(nowMs, block) }
        override fun scheduleResumeAfterDelay(timeMillis: Long, continuation: CancellableContinuation<Unit>) {
            schedule(nowMs + timeMillis, Runnable { if (continuation.isActive) continuation.resume(Unit) })
        }
        private fun schedule(at: Long, block: Runnable) { pending += Scheduled(at, sequence++, block) }
        fun runCurrent() = advanceBy(0)
        fun advanceBy(milliseconds: Long) {
            val target = nowMs + milliseconds
            var count = 0
            while (pending.peek()?.at?.let { it <= target } == true) {
                check(++count < 100_000) { "virtual scheduler failed to quiesce" }
                val task = pending.remove()
                nowMs = task.at
                task.action.run()
            }
            nowMs = target
        }
    }
}
