package com.qlh.inference.security

import android.content.Context
import android.security.keystore.KeyGenParameterSpec
import android.security.keystore.KeyProperties
import android.util.Base64
import java.security.KeyStore
import javax.crypto.Cipher
import javax.crypto.KeyGenerator
import javax.crypto.SecretKey
import javax.crypto.spec.GCMParameterSpec

/** Persistence boundary for the long-lived cluster credential. */
interface ClusterCredentialStorage {
    fun read(): String?
    fun save(clusterSecret: String)
    fun clear()
}

/**
 * Stores the cluster secret encrypted with a non-exportable AndroidKeyStore key.
 *
 * SharedPreferences contains only the AES-GCM ciphertext and IV. Key invalidation,
 * malformed storage and tampering all fail closed and remove the unusable record.
 */
class ClusterCredentialStore(context: Context) : ClusterCredentialStorage {
    companion object {
        internal const val STORE_NAME = "qlh_cluster_credential"
        internal const val KEY_CIPHERTEXT = "cluster_secret_ciphertext"
        internal const val KEY_IV = "cluster_secret_iv"

        private const val KEY_ALIAS = "com.qlh.inference.cluster.credential.v1"
        private const val KEYSTORE = "AndroidKeyStore"
        private const val TRANSFORMATION = "AES/GCM/NoPadding"
        private const val GCM_TAG_BITS = 128
        private val AAD = "qlh-cluster-credential-v1".toByteArray(Charsets.UTF_8)
        private val STORE_LOCK = Any()
    }

    private val preferences = context.applicationContext.getSharedPreferences(
        STORE_NAME,
        Context.MODE_PRIVATE,
    )

    override fun read(): String? = synchronized(STORE_LOCK) {
        val ciphertext = preferences.getString(KEY_CIPHERTEXT, null)
        val iv = preferences.getString(KEY_IV, null)
        if (ciphertext == null && iv == null) return@synchronized null
        if (ciphertext == null || iv == null) {
            clear()
            return@synchronized null
        }
        try {
            val cipher = Cipher.getInstance(TRANSFORMATION)
            cipher.init(
                Cipher.DECRYPT_MODE,
                getOrCreateKey(),
                GCMParameterSpec(GCM_TAG_BITS, decode(iv)),
            )
            cipher.updateAAD(AAD)
            String(cipher.doFinal(decode(ciphertext)), Charsets.UTF_8)
                .takeIf { it.isNotBlank() }
                ?: run {
                    clear()
                    null
                }
        } catch (_: Exception) {
            clear()
            null
        }
    }

    override fun save(clusterSecret: String) = synchronized(STORE_LOCK) {
        require(clusterSecret.isNotBlank()) { "cluster secret must not be blank" }
        val cipher = Cipher.getInstance(TRANSFORMATION)
        cipher.init(Cipher.ENCRYPT_MODE, getOrCreateKey())
        cipher.updateAAD(AAD)
        val encrypted = cipher.doFinal(clusterSecret.toByteArray(Charsets.UTF_8))
        check(
            preferences.edit()
                .putString(KEY_CIPHERTEXT, encode(encrypted))
                .putString(KEY_IV, encode(cipher.iv))
                .commit(),
        ) { "unable to persist encrypted cluster credential" }
    }

    override fun clear() {
        synchronized(STORE_LOCK) {
            preferences.edit().remove(KEY_CIPHERTEXT).remove(KEY_IV).commit()
            runCatching {
                KeyStore.getInstance(KEYSTORE).apply { load(null) }.deleteEntry(KEY_ALIAS)
            }
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
