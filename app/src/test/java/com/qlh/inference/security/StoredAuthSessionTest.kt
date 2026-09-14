package com.qlh.inference.security

import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class StoredAuthSessionTest {
    private fun validSession() = StoredAuthSession(
        accessToken = "a".repeat(32),
        sessionId = "sess_123",
        expiresAt = "2026-08-18T00:00:00Z",
        userId = "user_123",
        username = "owner",
        role = "owner",
    )

    @Test
    fun `accepts a complete local session record`() {
        assertTrue(validSession().isValid())
    }

    @Test
    fun `rejects incomplete token and unrecognized role`() {
        assertFalse(validSession().copy(accessToken = "short").isValid())
        assertFalse(validSession().copy(role = "superuser").isValid())
        assertFalse(validSession().copy(sessionId = "").isValid())
    }
}
