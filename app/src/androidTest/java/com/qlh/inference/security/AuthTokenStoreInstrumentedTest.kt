package com.qlh.inference.security

import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Test

class AuthTokenStoreInstrumentedTest {
    @Test
    fun storesSessionEncryptedWithAndroidKeystoreAndClearsIt() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val store = AuthTokenStore(context)
        store.clear()
        val token = "android-auth-token-${"a".repeat(32)}"
        val session = StoredAuthSession(
            accessToken = token,
            sessionId = "sess_android",
            expiresAt = "2026-08-18T00:00:00Z",
            userId = "user_android",
            username = "owner",
            role = "owner",
        )

        store.save(session)
        assertEquals(session, store.read())
        val raw = context.getSharedPreferences("qlh_auth_session", 0).all.values.joinToString()
        assertFalse(raw.contains(token))

        store.clear()
        assertNull(store.read())
    }
}
