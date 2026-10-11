package com.qlh.inference.security

import android.content.Context
import android.security.keystore.KeyGenParameterSpec
import android.security.keystore.KeyProperties
import android.util.Base64
import com.google.gson.Gson
import com.google.gson.JsonParseException
import java.security.KeyStore
import javax.crypto.AEADBadTagException
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

class AuthTokenStore internal constructor(
    context: Context,
    private val storeName: String,
    private val keyAlias: String,
    private val readKey: (String) -> SecretKey?,
) : AuthSessionStore {
    constructor(context: Context) : this(context, STORE_NAME, KEY_ALIAS, ::existingKey)

    internal constructor(context: Context, storeName: String, keyAlias: String) :
        this(context, storeName, keyAlias, ::existingKey)

    companion object {
        private const val STORE_NAME = "qlh_auth_session"
        internal const val KEY_CIPHERTEXT = "session_ciphertext"
        internal const val KEY_IV = "session_iv"
        private const val KEY_ALIAS = "com.qlh.inference.auth.session.v1"
        private const val KEYSTORE = "AndroidKeyStore"
        private const val TRANSFORMATION = "AES/GCM/NoPadding"
        private const val GCM_TAG_BITS = 128
        private const val GCM_IV_BYTES = 12

        private fun existingKey(alias: String): SecretKey? =
            (KeyStore.getInstance(KEYSTORE).apply { load(null) }.getKey(alias, null) as? SecretKey)
    }

    private val preferences = context.getSharedPreferences(storeName, Context.MODE_PRIVATE)
    private val gson = Gson()

    @Synchronized
    override fun read(): StoredAuthSession? {
        val (ciphertext, iv) = try {
            preferences.getString(KEY_CIPHERTEXT, null) to preferences.getString(KEY_IV, null)
        } catch (_: ClassCastException) {
            discardRecord()
            return null
        } catch (_: Exception) {
            return null
        }
        if (ciphertext == null && iv == null) return null
        if (ciphertext == null || iv == null) {
            discardRecord()
            return null
        }
        val ivBytes = decodeStored(iv)
        val encrypted = decodeStored(ciphertext)
        if (ivBytes?.size != GCM_IV_BYTES || encrypted == null || encrypted.size < GCM_TAG_BITS / 8) {
            discardRecord()
            return null
        }
        return try {
            val key = readKey(keyAlias) ?: return null
            val cipher = Cipher.getInstance(TRANSFORMATION)
            cipher.init(
                Cipher.DECRYPT_MODE,
                key,
                GCMParameterSpec(GCM_TAG_BITS, ivBytes),
            )
            val record = gson.fromJson(
                String(cipher.doFinal(encrypted), Charsets.UTF_8),
                StoredAuthSession::class.java,
            )
            record?.takeIf { it.isValid() } ?: run {
                discardRecord()
                null
            }
        } catch (_: AEADBadTagException) {
            discardRecord()
            null
        } catch (_: JsonParseException) {
            discardRecord()
            null
        } catch (_: Exception) {
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
        discardRecord()
        runCatching {
            KeyStore.getInstance(KEYSTORE).apply { load(null) }.deleteEntry(keyAlias)
        }
    }

    private fun discardRecord() {
        preferences.edit().remove(KEY_CIPHERTEXT).remove(KEY_IV).commit()
    }

    private fun getOrCreateKey(): SecretKey {
        val keyStore = KeyStore.getInstance(KEYSTORE).apply { load(null) }
        val existing = keyStore.getKey(keyAlias, null) as? SecretKey
        if (existing != null) return existing

        val keyGenerator = KeyGenerator.getInstance(KeyProperties.KEY_ALGORITHM_AES, KEYSTORE)
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
        return keyGenerator.generateKey()
    }

    private fun encode(value: ByteArray): String = Base64.encodeToString(value, Base64.NO_WRAP)

    private fun decode(value: String): ByteArray = Base64.decode(value, Base64.NO_WRAP)

    private fun decodeStored(value: String): ByteArray? {
        val bytes = try {
            decode(value)
        } catch (_: IllegalArgumentException) {
            return null
        }
        return bytes.takeIf { encode(it) == value }
    }
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
