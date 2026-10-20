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

    @Test
    fun `DIST-NEXT-4 heartbeat interval matches the master threshold source`() {
        // 主仓 `scheduler_types.ANDROID_TASK_WORKER_HEARTBEAT_INTERVAL_SECONDS = 15.0`；
        // 主仓据它推导容忍上限（WORKER_HEARTBEAT_MAX_AGE = 120s）与控制面 health 超时
        // （30s）。两侧不同值会让设备心跳被判过期（历史实测：45s 心跳 vs 10s 上限）。
        assertEquals(15_000L, TaskWorkerClient.HEARTBEAT_INTERVAL_MS)
    }

    @Test
    fun `consecutive failures stop at a named limit and retain the root cause`() {
        val machine = TaskWorkerStateMachine(baseBackoffMs = 1_000, maxBackoffMs = 2_000, maxConsecutiveFailures = 4)
        machine.start()
        val delays = mutableListOf<Long>()
        var now = 0L
        repeat(4) {
            val failed = machine.onDisconnected(now, "transport_connect_failed", "refused")
            if (it < 3) {
                delays += failed.nextRetryAtMs - now
                now = failed.nextRetryAtMs
                machine.retryIfDue(now)
            }
        }
        val failed = machine.snapshot()
        assertEquals(listOf(1_000L, 2_000L, 2_000L), delays)
        assertEquals(TaskWorkerConnectionState.RETRY_EXHAUSTED, failed.connection)
        assertEquals(4, failed.reconnectAttempt)
        assertEquals(0L, failed.nextRetryAtMs)
        assertEquals("transport_connect_failed", failed.lastErrorCode)
        assertEquals("network", failed.failureKind)
        assertEquals(failed, machine.retryIfDue(Long.MAX_VALUE))
        assertEquals(failed, machine.onDisconnected(now, "transport_loop_stopped", "finished"))
    }

    @Test
    fun `accepted registration resets the consecutive failure budget`() {
        val machine = TaskWorkerStateMachine(baseBackoffMs = 1, maxConsecutiveFailures = 2)
        machine.start()
        machine.onDisconnected(0, "transport_connect_failed", "refused")
        machine.retryIfDue(1)
        machine.onConnected()
        machine.onHelloAck(true, "", "", 2)
        val next = machine.onDisconnected(3, "heartbeat_timeout", "no ack")
        assertEquals(TaskWorkerConnectionState.BACKING_OFF, next.connection)
        assertEquals(1, next.reconnectAttempt)
    }

    @Test
    fun `registration and identity failures stay distinct from network errors`() {
        assertEquals("identity", TaskWorkerSnapshot(lastErrorCode = "registration_rejected").failureKind)
        assertEquals("identity", TaskWorkerSnapshot(lastErrorCode = "master_identity_unconfirmed").failureKind)
        assertEquals("identity", TaskWorkerSnapshot(lastErrorCode = "auth_failed").failureKind)
        assertEquals("network", TaskWorkerSnapshot(lastErrorCode = "heartbeat_timeout").failureKind)
        assertEquals("network", TaskWorkerSnapshot(lastErrorCode = "transport_timeout").failureKind)
        assertEquals("protocol", TaskWorkerSnapshot(lastErrorCode = "invalid_frame").failureKind)
    }

    @Test
    fun `completion requires running and cancellation wins atomically`() {
        val machine = readyMachine()
        assertTrue(machine.offer(identity, 20_000L, 1_000L))
        assertFalse(machine.complete(identity, "a".repeat(64), 1_001L))
        assertTrue(machine.markRunning(identity, 1_002L))
        assertEquals(identity, machine.requestCancel(1_003L, identity))
        assertFalse(machine.complete(identity, "a".repeat(64), 1_004L))
        assertFalse(machine.fail(identity, "worker_execution_failed", true))
        assertEquals(TaskWorkerAttemptState.CANCELLING, machine.snapshot().activeAttempt.state)
    }

    @Test
    fun `successful attempt cannot execute twice during its lease even after reconnect`() {
        val machine = readyMachine()
        assertTrue(machine.offer(identity, 20_000L, 1_000L))
        assertTrue(machine.markRunning(identity, 1_001L))
        assertTrue(machine.complete(identity, "a".repeat(64), 1_002L))
        assertEquals("worker_attempt_already_admitted", machine.admitOffer(identity, 20_000L, 1_003L))
        machine.onDisconnected(1_004L, "transport_disconnected", "eof")
        machine.retryIfDue(3_000L)
        machine.onConnected()
        machine.onHelloAck(true, "", "", 3_001L)
        assertEquals("worker_attempt_already_admitted", machine.admitOffer(identity, 20_000L, 3_002L))
    }

    @Test
    fun `attempt history never evicts an unexpired lease to admit more work`() {
        val machine = TaskWorkerStateMachine(maxAttemptHistory = 2).also {
            it.start(); it.onConnected(); it.onHelloAck(true, "", "", 0L)
        }
        val second = identity.copy(attemptId = "att_second01")
        val third = identity.copy(attemptId = "att_third001")
        for (attempt in listOf(identity, second)) {
            assertTrue(machine.offer(attempt, 20_000L, 1_000L))
            assertTrue(machine.markRunning(attempt, 1_001L))
            assertTrue(machine.complete(attempt, "a".repeat(64), 1_002L))
        }
        assertEquals("worker_attempt_history_full", machine.admitOffer(third, 30_000L, 1_003L))
        assertEquals("worker_attempt_already_admitted", machine.admitOffer(identity, 20_000L, 1_004L))
        assertNull(machine.admitOffer(third, 30_000L, 20_000L))
    }

    @Test
    fun `renewal retains the largest lease deadline in attempt history`() {
        val machine = readyMachine()
        assertTrue(machine.offer(identity, 2_000L, 1_000L))
        assertTrue(machine.markRunning(identity, 1_001L))
        assertTrue(machine.renew(identity, 20_000L, 1_100L))
        assertTrue(machine.renew(identity, 5_000L, 1_200L))
        assertTrue(machine.complete(identity, "a".repeat(64), 1_300L))
        assertEquals("worker_attempt_already_admitted", machine.admitOffer(identity, 20_000L, 6_000L))
        val nextEpoch = identity.copy(leaseEpoch = identity.leaseEpoch + 1)
        assertTrue(machine.offer(nextEpoch, 20_000L, 6_001L))
    }

    private fun readyMachine(): TaskWorkerStateMachine = TaskWorkerStateMachine().also {
        it.start()
        it.onConnected()
        it.onHelloAck(true, "", "", nowMs = 0L)
    }
}
