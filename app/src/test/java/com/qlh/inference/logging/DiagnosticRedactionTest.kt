package com.qlh.inference.logging

import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class DiagnosticRedactionTest {
    @Test
    fun `redacts bearer and named credentials while preserving ordinary log text`() {
        val redacted = redactDiagnosticText(
            """request Authorization: Bearer abcDEF_123456789
               access_token="token-secret-value"
               recovery_code=code-secret-value
               model loaded successfully
            """.trimIndent(),
        )

        assertFalse(redacted.contains("abcDEF_123456789"))
        assertFalse(redacted.contains("token-secret-value"))
        assertFalse(redacted.contains("code-secret-value"))
        assertTrue(redacted.contains("[REDACTED]"))
        assertTrue(redacted.contains("model loaded successfully"))
    }
}
