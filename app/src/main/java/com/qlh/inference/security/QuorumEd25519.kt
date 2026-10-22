package com.qlh.inference.security

import android.content.Context
import android.security.keystore.KeyGenParameterSpec
import android.security.keystore.KeyProperties
import android.util.Base64 as AndroidBase64
import java.security.KeyStore
import java.security.SecureRandom
import javax.crypto.Cipher
import javax.crypto.KeyGenerator
import javax.crypto.SecretKey
import javax.crypto.spec.GCMParameterSpec
import org.bouncycastle.math.ec.rfc8032.Ed25519

internal object QuorumEd25519 {
    fun generate(): Pair<ByteArray, ByteArray> {
        val seed = ByteArray(32)
        val publicKey = ByteArray(32)
        try {
            Ed25519.generatePrivateKey(SecureRandom(), seed)
            Ed25519.generatePublicKey(seed, 0, publicKey, 0)
            return seed to publicKey
        } catch (error: Exception) {
            seed.fill(0)
            publicKey.fill(0)
            throw error
        }
    }

    fun sign(seed: ByteArray, message: ByteArray): ByteArray {
        require(seed.size == 32)
        val publicKey = ByteArray(32)
        Ed25519.generatePublicKey(seed, 0, publicKey, 0)
        return ByteArray(64).also { signature ->
            Ed25519.sign(seed, 0, publicKey, 0, message, 0, message.size, signature, 0)
            publicKey.fill(0)
        }
    }

    fun verify(publicKey: ByteArray, message: ByteArray, signature: ByteArray): Boolean {
        if (publicKey.size != 32 || signature.size != 64) return false
        return runCatching { Ed25519.verify(signature, 0, publicKey, 0, message, 0, message.size) }
            .getOrDefault(false)
    }

    fun matches(seed: ByteArray, publicKey: ByteArray): Boolean {
        val challenge = "qlh-quorum-voter-key-check-v1".toByteArray(Charsets.US_ASCII)
        return verify(publicKey, challenge, sign(seed, challenge))
    }

    fun encode(bytes: ByteArray): String = java.util.Base64.getUrlEncoder().withoutPadding().encodeToString(bytes)

    fun decode(value: String, expectedSize: Int): ByteArray {
        val decoded = java.util.Base64.getUrlDecoder().decode(value)
        require(decoded.size == expectedSize && encode(decoded) == value)
        return decoded
    }

}

internal data class QuorumVoterKeyMaterial(val seed: ByteArray, val publicKey: ByteArray)

/** Stores the exportable Ed25519 seed only as AndroidKeyStore AES-GCM ciphertext. */
internal class QuorumVoterKeyStore(
    context: Context,
    private val storeName: String = STORE,
    private val keyAlias: String = KEY_ALIAS,
) {
    private val preferences = context.applicationContext.getSharedPreferences(storeName, Context.MODE_PRIVATE)

    @Synchronized
    fun readOrCreate(): QuorumVoterKeyMaterial {
        val ciphertext = preferences.getString(KEY_CIPHERTEXT, null)
        val iv = preferences.getString(KEY_IV, null)
        val publicKey = preferences.getString(KEY_PUBLIC, null)
        if (ciphertext != null || iv != null || publicKey != null) {
            check(ciphertext != null && iv != null && publicKey != null) { "quorum voter key record is incomplete" }
            val seed = Cipher.getInstance("AES/GCM/NoPadding").run {
                init(Cipher.DECRYPT_MODE, getExistingKey() ?: error("quorum voter wrapping key is unavailable"),
                    GCMParameterSpec(128, AndroidBase64.decode(iv, AndroidBase64.NO_WRAP)))
                updateAAD(AAD)
                doFinal(AndroidBase64.decode(ciphertext, AndroidBase64.NO_WRAP))
            }
            try {
                val decodedPublic = QuorumEd25519.decode(publicKey, 32)
                check(seed.size == 32 && QuorumEd25519.matches(seed, decodedPublic)) {
                    "quorum voter key identity mismatch"
                }
                return QuorumVoterKeyMaterial(seed, decodedPublic)
            } catch (error: Exception) {
                seed.fill(0)
                throw error
            }
        }
        val (seed, publicKeyBytes) = QuorumEd25519.generate()
        try {
            val cipher = Cipher.getInstance("AES/GCM/NoPadding")
            cipher.init(Cipher.ENCRYPT_MODE, getOrCreateKey())
            cipher.updateAAD(AAD)
            val encrypted = cipher.doFinal(seed)
            try {
                check(preferences.edit()
                    .putString(KEY_CIPHERTEXT, AndroidBase64.encodeToString(encrypted, AndroidBase64.NO_WRAP))
                    .putString(KEY_IV, AndroidBase64.encodeToString(cipher.iv, AndroidBase64.NO_WRAP))
                    .putString(KEY_PUBLIC, QuorumEd25519.encode(publicKeyBytes))
                    .commit()) { "unable to persist quorum voter key" }
            } finally {
                encrypted.fill(0)
            }
            return QuorumVoterKeyMaterial(seed, publicKeyBytes)
        } catch (error: Exception) {
            seed.fill(0)
            throw error
        }
    }

    @Synchronized
    fun readPublicKey(): String? = preferences.getString(KEY_PUBLIC, null)

    @Synchronized
    fun clear() {
        preferences.edit().clear().commit()
        runCatching { KeyStore.getInstance(KEYSTORE).apply { load(null) }.deleteEntry(keyAlias) }
    }

    private fun getExistingKey(): SecretKey? = KeyStore.getInstance(KEYSTORE).run {
        load(null)
        getKey(keyAlias, null) as? SecretKey
    }

    private fun getOrCreateKey(): SecretKey {
        getExistingKey()?.let { return it }
        val generator = KeyGenerator.getInstance(KeyProperties.KEY_ALGORITHM_AES, KEYSTORE)
        generator.init(KeyGenParameterSpec.Builder(keyAlias,
            KeyProperties.PURPOSE_ENCRYPT or KeyProperties.PURPOSE_DECRYPT)
            .setBlockModes(KeyProperties.BLOCK_MODE_GCM)
            .setEncryptionPaddings(KeyProperties.ENCRYPTION_PADDING_NONE)
            .setKeySize(256)
            .build())
        return generator.generateKey()
    }

    companion object {
        private const val STORE = "qlh_quorum_voter_key"
        private const val KEYSTORE = "AndroidKeyStore"
        private const val KEY_ALIAS = "com.qlh.inference.quorum.voter.v1"
        private const val KEY_CIPHERTEXT = "seed_ciphertext"
        private const val KEY_IV = "seed_iv"
        private const val KEY_PUBLIC = "public_key"
        private val AAD = "qlh-quorum-voter-seed-v1".toByteArray(Charsets.UTF_8)
    }
}
