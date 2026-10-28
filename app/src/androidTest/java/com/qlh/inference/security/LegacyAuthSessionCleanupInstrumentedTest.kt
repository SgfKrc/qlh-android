package com.qlh.inference.security

import android.security.keystore.KeyGenParameterSpec
import android.security.keystore.KeyProperties
import androidx.test.platform.app.InstrumentationRegistry
import java.security.KeyStore
import javax.crypto.KeyGenerator
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class LegacyAuthSessionCleanupInstrumentedTest {
    @Test
    fun removesLegacySessionPreferencesAndKeystoreKey() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val storeName = "qlh_auth_session"
        val keyAlias = "com.qlh.inference.auth.session.v1"
        context.getSharedPreferences(storeName, 0).edit()
            .putString("session_ciphertext", "legacy-ciphertext")
            .putString("session_iv", "legacy-iv")
            .commit()

        val keyGenerator = KeyGenerator.getInstance(KeyProperties.KEY_ALGORITHM_AES, "AndroidKeyStore")
        keyGenerator.init(
            KeyGenParameterSpec.Builder(
                keyAlias,
                KeyProperties.PURPOSE_ENCRYPT or KeyProperties.PURPOSE_DECRYPT,
            )
                .setBlockModes(KeyProperties.BLOCK_MODE_GCM)
                .setEncryptionPaddings(KeyProperties.ENCRYPTION_PADDING_NONE)
                .setKeySize(256)
                .build(),
        )
        keyGenerator.generateKey()

        LegacyAuthSessionCleanup.clear(context)

        assertTrue(context.getSharedPreferences(storeName, 0).all.isEmpty())
        assertFalse(KeyStore.getInstance("AndroidKeyStore").apply { load(null) }.containsAlias(keyAlias))
    }
}
