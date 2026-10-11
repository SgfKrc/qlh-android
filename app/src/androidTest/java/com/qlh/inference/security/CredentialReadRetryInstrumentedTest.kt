package com.qlh.inference.security

import android.content.Context
import android.util.Base64
import androidx.test.platform.app.InstrumentationRegistry
import java.security.KeyStore
import java.security.KeyStoreException
import java.util.UUID
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class CredentialReadRetryInstrumentedTest {
    private val context: Context
        get() = InstrumentationRegistry.getInstrumentation().targetContext

    private val testId = UUID.randomUUID().toString()
    private val clusterPrefs = "qlh_test_cluster_$testId"
    private val clusterAlias = "com.qlh.inference.test.cluster.$testId"
    private val authPrefs = "qlh_test_auth_$testId"
    private val authAlias = "com.qlh.inference.test.auth.$testId"

    @After
    fun cleanOnlyTestCredentials() {
        ClusterCredentialStore(context, clusterPrefs, clusterAlias).clear()
        AuthTokenStore(context, authPrefs, authAlias).clear()
        context.deleteSharedPreferences(clusterPrefs)
        context.deleteSharedPreferences(authPrefs)
    }

    @Test
    fun transientClusterKeystoreFailureAndMissingKeyDoNotEraseRecord() {
        val secret = "test-cluster-secret-${"a".repeat(32)}"
        val store = ClusterCredentialStore(context, clusterPrefs, clusterAlias)
        store.save(secret)
        val original = context.getSharedPreferences(clusterPrefs, 0).all.toMap()

        val temporaryFailure = ClusterCredentialStore(context, clusterPrefs, clusterAlias) {
            throw KeyStoreException("temporary Keystore failure")
        }
        assertNull(temporaryFailure.read())
        assertEquals(original, context.getSharedPreferences(clusterPrefs, 0).all)
        assertTrue(keyStore().containsAlias(clusterAlias))

        val missingKey = ClusterCredentialStore(context, clusterPrefs, clusterAlias) { null }
        assertNull(missingKey.read())
        assertEquals(original, context.getSharedPreferences(clusterPrefs, 0).all)
        assertEquals(secret, store.read())
    }

    @Test
    fun clusterTamperingDiscardsOnlyRecordAndLeavesKeyUsable() {
        val store = ClusterCredentialStore(context, clusterPrefs, clusterAlias)
        store.save("test-cluster-secret-${"b".repeat(32)}")
        val preferences = context.getSharedPreferences(clusterPrefs, 0)
        val ciphertext = Base64.decode(
            preferences.getString(ClusterCredentialStore.KEY_CIPHERTEXT, null),
            Base64.NO_WRAP,
        )
        ciphertext[0] = (ciphertext[0].toInt() xor 1).toByte()
        preferences.edit().putString(
            ClusterCredentialStore.KEY_CIPHERTEXT,
            Base64.encodeToString(ciphertext, Base64.NO_WRAP),
        ).commit()

        assertNull(store.read())
        assertTrue(preferences.all.isEmpty())
        assertTrue(keyStore().containsAlias(clusterAlias))
        store.save("replacement-secret")
        assertEquals("replacement-secret", store.read())
    }

    @Test
    fun clusterReadNeverCreatesAReplacementForMissingKey() {
        val store = ClusterCredentialStore(context, clusterPrefs, clusterAlias)
        store.save("test-cluster-secret-${"c".repeat(32)}")
        val original = context.getSharedPreferences(clusterPrefs, 0).all.toMap()
        keyStore().deleteEntry(clusterAlias)

        assertNull(store.read())
        assertEquals(original, context.getSharedPreferences(clusterPrefs, 0).all)
        assertFalse(keyStore().containsAlias(clusterAlias))
    }

    @Test
    fun transientAuthKeystoreFailureRetainsSessionForRetry() {
        val store = AuthTokenStore(context, authPrefs, authAlias)
        val session = StoredAuthSession(
            accessToken = "android-auth-token-${"d".repeat(32)}",
            sessionId = "sess_test",
            expiresAt = "2026-12-31T00:00:00Z",
            userId = "test_user",
            username = "test_owner",
            role = "owner",
        )
        store.save(session)
        val original = context.getSharedPreferences(authPrefs, 0).all.toMap()

        val temporaryFailure = AuthTokenStore(context, authPrefs, authAlias) {
            throw KeyStoreException("temporary Keystore failure")
        }
        assertNull(temporaryFailure.read())
        assertEquals(original, context.getSharedPreferences(authPrefs, 0).all)
        assertTrue(keyStore().containsAlias(authAlias))
        assertEquals(session, store.read())
    }

    @Test
    fun authTamperingDiscardsOnlyRecordAndReadDoesNotRecreateMissingKey() {
        val store = AuthTokenStore(context, authPrefs, authAlias)
        val session = StoredAuthSession(
            accessToken = "android-auth-token-${"e".repeat(32)}",
            sessionId = "sess_test",
            expiresAt = "2026-12-31T00:00:00Z",
            userId = "test_user",
            username = "test_owner",
            role = "owner",
        )
        store.save(session)
        val preferences = context.getSharedPreferences(authPrefs, 0)
        val ciphertext = Base64.decode(
            preferences.getString(AuthTokenStore.KEY_CIPHERTEXT, null),
            Base64.NO_WRAP,
        )
        ciphertext[0] = (ciphertext[0].toInt() xor 1).toByte()
        preferences.edit().putString(
            AuthTokenStore.KEY_CIPHERTEXT,
            Base64.encodeToString(ciphertext, Base64.NO_WRAP),
        ).commit()

        assertNull(store.read())
        assertTrue(preferences.all.isEmpty())
        assertTrue(keyStore().containsAlias(authAlias))
        store.save(session)
        val encrypted = preferences.all.toMap()
        keyStore().deleteEntry(authAlias)

        assertNull(store.read())
        assertEquals(encrypted, preferences.all)
        assertFalse(keyStore().containsAlias(authAlias))
    }

    private fun keyStore(): KeyStore = KeyStore.getInstance("AndroidKeyStore").apply { load(null) }
}
