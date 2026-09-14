package com.qlh.inference.network

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class AndroidPresenceStateMachineTest {
    @Test
    fun `registration creates fenced online lease and heartbeat refreshes it`() {
        val machine = AndroidPresenceStateMachine(baseBackoffMs = 1000L, maxBackoffMs = 4000L)
        machine.start(1000L)
        assertTrue(machine.beginRegistration(1000L))
        machine.onRegistered(
            generation = 3L,
            leaseId = "lease-3",
            leaseExpiresAtMs = 121_000L,
            heartbeatIntervalSeconds = 45,
            nowMs = 1000L,
            serverTimeMs = 1000L,
        )
        assertEquals(AndroidPresenceState.ONLINE, machine.snapshot().state)
        assertFalse(machine.heartbeatDue(10_000L))
        assertTrue(machine.heartbeatDue(46_000L))
        machine.onHeartbeatSuccess(3L, "lease-3", 166_000L, 45, 46_000L, 46_000L)
        assertEquals(166_000L, machine.snapshot().leaseExpiresAtMs)
    }

    @Test
    fun `stale lease response cannot replace current generation`() {
        val machine = AndroidPresenceStateMachine()
        machine.start(0L)
        machine.onRegistered(2L, "new", 120_000L, 45, 0L, 0L)
        val before = machine.snapshot()
        machine.onHeartbeatSuccess(1L, "old", 180_000L, 45, 10_000L, 10_000L)
        assertEquals(before.leaseId, machine.snapshot().leaseId)
        assertEquals(before.leaseExpiresAtMs, machine.snapshot().leaseExpiresAtMs)
    }

    @Test
    fun `network failures use bounded backoff and lease errors re-register`() {
        val machine = AndroidPresenceStateMachine(baseBackoffMs = 1000L, maxBackoffMs = 4000L)
        machine.start(0L)
        machine.onFailure("timeout", "unreachable", 0L)
        assertEquals(AndroidPresenceState.BACKING_OFF, machine.snapshot().state)
        assertEquals(1000L, machine.snapshot().nextRetryAtMs)
        machine.retryIfDue(1000L)
        assertEquals(AndroidPresenceState.REGISTERING, machine.snapshot().state)
        machine.onFailure("stale_lease", "expired", 1000L)
        assertEquals(AndroidPresenceState.REGISTERING, machine.snapshot().state)
        assertEquals(0L, machine.snapshot().generation)
        assertEquals("", machine.snapshot().leaseId)
    }
}
