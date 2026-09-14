package com.qlh.inference.worker

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class TaskWorkerClientStateTest {
    private val identity = TaskWorkerAttemptIdentity(
        workflowId = "wf_12345678",
        stageId = "stage_1",
        attemptId = "att_12345678",
        leaseId = "lease_12345678",
        leaseEpoch = 1,
    )

    @Test
    fun `hello handshake reaches ready and clears reconnect state`() {
        val machine = TaskWorkerStateMachine()

        assertEquals(TaskWorkerConnectionState.CONNECTING, machine.start().connection)
        assertEquals(TaskWorkerConnectionState.HELLO_SENT, machine.onConnected().connection)
        val ready = machine.onHelloAck(true, "", "", nowMs = 1_000L)

        assertEquals(TaskWorkerConnectionState.READY, ready.connection)
        assertEquals(0, ready.reconnectAttempt)
        assertEquals(0L, ready.nextRetryAtMs)
        assertNull(ready.lastErrorCode)
    }

    @Test
    fun `rejected hello uses bounded backoff and retry gate`() {
        val machine = TaskWorkerStateMachine(baseBackoffMs = 1_000L, maxBackoffMs = 4_000L)
        machine.start()
        machine.onConnected()

        val rejected = machine.onHelloAck(false, "node_not_admitted", "rejected", nowMs = 5_000L)
        assertEquals(TaskWorkerConnectionState.BACKING_OFF, rejected.connection)
        assertEquals(6_000L, rejected.nextRetryAtMs)
        assertEquals("node_not_admitted", rejected.lastErrorCode)
        assertEquals(TaskWorkerConnectionState.BACKING_OFF, machine.retryIfDue(5_999L).connection)
        assertEquals(TaskWorkerConnectionState.CONNECTING, machine.retryIfDue(6_000L).connection)
    }

    @Test
    fun `disconnect fences active attempt and rejects its late result`() {
        val machine = readyMachine()
        assertTrue(machine.offer(identity, leaseExpiresAtMs = 20_000L, nowMs = 1_000L))
        assertTrue(machine.markRunning(identity, nowMs = 1_001L))

        val disconnected = machine.onDisconnected(2_000L, "transport_disconnected", "eof")

        assertEquals(TaskWorkerConnectionState.BACKING_OFF, disconnected.connection)
        assertEquals(TaskWorkerAttemptState.LOST, disconnected.activeAttempt.state)
        assertFalse(machine.complete(identity, "a".repeat(64), nowMs = 2_001L))
        assertEquals(TaskWorkerAttemptState.LOST, machine.snapshot().activeAttempt.state)
    }

    @Test
    fun `single concurrency rejects overlapping offer`() {
        val machine = readyMachine()
        val other = identity.copy(attemptId = "att_abcdefgh", leaseId = "lease_abcdefgh")

        assertTrue(machine.offer(identity, leaseExpiresAtMs = 20_000L, nowMs = 1_000L))
        assertFalse(machine.offer(other, leaseExpiresAtMs = 20_000L, nowMs = 1_000L))
        assertEquals(identity, machine.snapshot().activeAttempt.identity)
    }

    @Test
    fun `lease renewal extends result fence`() {
        val machine = readyMachine()
        machine.offer(identity, leaseExpiresAtMs = 2_000L, nowMs = 1_000L)
        machine.markRunning(identity, nowMs = 1_001L)

        assertTrue(machine.renew(identity, leaseExpiresAtMs = 5_000L, nowMs = 1_500L))
        assertFalse(machine.renew(identity, leaseExpiresAtMs = 1_400L, nowMs = 1_500L))
        assertTrue(machine.complete(identity, "b".repeat(64), nowMs = 4_999L))
        assertEquals(TaskWorkerAttemptState.SUCCEEDED, machine.snapshot().activeAttempt.state)
        assertEquals("b".repeat(64), machine.snapshot().activeAttempt.outputSha256)
    }

    @Test
    fun `cancel requires current live attempt and terminal acknowledgement`() {
        val machine = readyMachine()
        assertNull(machine.requestCancel(nowMs = 1_000L))
        machine.offer(identity, leaseExpiresAtMs = 5_000L, nowMs = 1_000L)
        machine.markRunning(identity, nowMs = 1_001L)

        assertNotNull(machine.requestCancel(nowMs = 1_100L))
        assertEquals(TaskWorkerAttemptState.CANCELLING, machine.snapshot().activeAttempt.state)
        assertTrue(machine.cancelled(identity))
        assertEquals(TaskWorkerAttemptState.CANCELLED, machine.snapshot().activeAttempt.state)
        assertFalse(machine.cancelled(identity))
    }

    @Test
    fun `expired offer and stale identity are fail closed`() {
        val machine = readyMachine()
        assertFalse(machine.offer(identity, leaseExpiresAtMs = 999L, nowMs = 1_000L))
        assertTrue(machine.offer(identity, leaseExpiresAtMs = 2_000L, nowMs = 1_000L))
        val stale = identity.copy(leaseEpoch = 2)
        assertFalse(machine.markRunning(stale, nowMs = 1_001L))
        assertFalse(machine.fail(stale, "stale_lease", retryable = true))
    }

    @Test
    fun `expired active attempt becomes lost and allows replacement`() {
        val machine = readyMachine()
        assertTrue(machine.offer(identity, leaseExpiresAtMs = 2_000L, nowMs = 1_000L))
        assertTrue(machine.markRunning(identity, nowMs = 1_001L))

        val replacement = identity.copy(attemptId = "att_abcdefgh", leaseId = "lease_abcdefgh")
        assertTrue(machine.offer(replacement, leaseExpiresAtMs = 4_000L, nowMs = 2_000L))
        assertEquals(replacement, machine.snapshot().activeAttempt.identity)
        assertEquals(TaskWorkerAttemptState.OFFERED, machine.snapshot().activeAttempt.state)
    }

    @Test
    fun `stopping service fences active attempt`() {
        val machine = readyMachine()
        machine.offer(identity, leaseExpiresAtMs = 5_000L, nowMs = 1_000L)
        machine.markRunning(identity, nowMs = 1_001L)

        val stopped = machine.stop()

        assertEquals(TaskWorkerConnectionState.STOPPED, stopped.connection)
        assertEquals(TaskWorkerAttemptState.LOST, stopped.activeAttempt.state)
    }

    @Test
    fun `stale cancel cannot change current attempt`() {
        val machine = readyMachine()
        val other = identity.copy(attemptId = "att_abcdefgh", leaseId = "lease_abcdefgh")
        machine.offer(identity, leaseExpiresAtMs = 5_000L, nowMs = 1_000L)
        machine.markRunning(identity, nowMs = 1_001L)

        assertNull(machine.requestCancel(nowMs = 1_100L, expectedIdentity = other))
        assertEquals(TaskWorkerAttemptState.RUNNING, machine.snapshot().activeAttempt.state)
    }

    private fun readyMachine(): TaskWorkerStateMachine = TaskWorkerStateMachine().also {
        it.start()
        it.onConnected()
        it.onHelloAck(true, "", "", nowMs = 0L)
    }
}
