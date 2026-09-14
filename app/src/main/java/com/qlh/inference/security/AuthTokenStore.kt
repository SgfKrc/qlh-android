package com.qlh.inference.security

import android.content.Context
import android.security.keystore.KeyGenParameterSpec
import android.security.keystore.KeyProperties
import android.util.Base64
import com.google.gson.Gson
import java.security.KeyStore
import javax.crypto.Cipher
import javax.crypto.KeyGenerator
import javax.crypto.SecretKey
import javax.crypto.spec.GCMParameterSpec

/**
 * Keeps the local control-plane bearer token encrypted with a key that never
 * leaves AndroidKeyStore. Session metadata is encrypted together with it so a
 * backup of app preferences cannot reveal account identity or expiry details.
 */
interface AuthSessionStore {
    fun read(): StoredAuthSession?
    fun save(session: StoredAuthSession)
    fun clear()
}

class AuthTokenStore(context: Context) : AuthSessionStore {
    companion object {
        private const val STORE_NAME = "qlh_auth_session"
        private const val KEY_CIPHERTEXT = "session_ciphertext"
        private const val KEY_IV = "session_iv"
        private const val KEY_ALIAS = "com.qlh.inference.auth.session.v1"
        private const val KEYSTORE = "AndroidKeyStore"
        private const val TRANSFORMATION = "AES/GCM/NoPadding"
        private const val GCM_TAG_BITS = 128
    }

    private val preferences = context.getSharedPreferences(STORE_NAME, Context.MODE_PRIVATE)
    private val gson = Gson()

    @Synchronized
    override fun read(): StoredAuthSession? {
        val ciphertext = preferences.getString(KEY_CIPHERTEXT, null) ?: return null
        val iv = preferences.getString(KEY_IV, null) ?: run {
            clear()
            return null
        }
        return try {
            val cipher = Cipher.getInstance(TRANSFORMATION)
            cipher.init(
                Cipher.DECRYPT_MODE,
                getOrCreateKey(),
                GCMParameterSpec(GCM_TAG_BITS, decode(iv)),
            )
            val record = gson.fromJson(
                String(cipher.doFinal(decode(ciphertext)), Charsets.UTF_8),
                StoredAuthSession::class.java,
            )
            record.takeIf { it.isValid() } ?: run {
                clear()
                null
            }
        } catch (_: Exception) {
            // Key invalidation, tampering and malformed storage all fail closed.
            clear()
            null
        }
    }

    @Synchronized
    override fun save(session: StoredAuthSession) {
        require(session.isValid()) { "invalid auth session" }
        val cipher = Cipher.getInstance(TRANSFORMATION)
        cipher.init(Cipher.ENCRYPT_MODE, getOrCreateKey())
        val encrypted = cipher.doFinal(gson.toJson(session).toByteArray(Charsets.UTF_8))
        check(preferences.edit()
            .putString(KEY_CIPHERTEXT, encode(encrypted))
            .putString(KEY_IV, encode(cipher.iv))
            .commit()) { "unable to persist encrypted auth session" }
    }

    @Synchronized
    override fun clear() {
        preferences.edit().remove(KEY_CIPHERTEXT).remove(KEY_IV).commit()
        runCatching {
            KeyStore.getInstance(KEYSTORE).apply { load(null) }.deleteEntry(KEY_ALIAS)
        }
    }

    private fun getOrCreateKey(): SecretKey {
        val keyStore = KeyStore.getInstance(KEYSTORE).apply { load(null) }
        val existing = keyStore.getKey(KEY_ALIAS, null) as? SecretKey
        if (existing != null) return existing

        val keyGenerator = KeyGenerator.getInstance(KeyProperties.KEY_ALGORITHM_AES, KEYSTORE)
        keyGenerator.init(
            KeyGenParameterSpec.Builder(
                KEY_ALIAS,
                KeyProperties.PURPOSE_ENCRYPT or KeyProperties.PURPOSE_DECRYPT,
            )
                .setBlockModes(KeyProperties.BLOCK_MODE_GCM)
                .setEncryptionPaddings(KeyProperties.ENCRYPTION_PADDING_NONE)
                .setKeySize(256)
                .build(),
        )
        return keyGenerator.generateKey()
    }

    private fun encode(value: ByteArray): String = Base64.encodeToString(value, Base64.NO_WRAP)

    private fun decode(value: String): ByteArray = Base64.decode(value, Base64.NO_WRAP)
}

data class StoredAuthSession(
    val accessToken: String = "",
    val sessionId: String = "",
    val expiresAt: String = "",
    val userId: String = "",
    val username: String = "",
    val displayName: String? = null,
    val role: String = "",
) {
    fun isValid(): Boolean =
        accessToken.matches(Regex("[A-Za-z0-9_-]{16,4096}")) &&
            sessionId.isNotBlank() &&
            expiresAt.isNotBlank() &&
            userId.isNotBlank() &&
            username.isNotBlank() &&
            role in setOf("owner", "admin", "member")
}
