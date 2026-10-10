package com.qlh.inference.security

import com.google.gson.JsonParser
import com.google.gson.annotations.SerializedName
import java.io.IOException
import java.security.KeyPairGenerator
import java.security.PrivateKey
import java.security.interfaces.RSAPrivateKey
import java.security.interfaces.RSAPublicKey
import java.security.spec.MGF1ParameterSpec
import java.util.Base64
import javax.crypto.Cipher
import javax.crypto.spec.GCMParameterSpec
import javax.crypto.spec.OAEPParameterSpec
import javax.crypto.spec.PSource
import javax.crypto.spec.SecretKeySpec

data class BootstrapCredentialAad(
    @SerializedName("cluster_id")
    val clusterId: String = "",
    @SerializedName("node_id")
    val nodeId: String = "",
    @SerializedName("request_nonce")
    val requestNonce: String = "",
    @SerializedName("secret_epoch")
    val secretEpoch: Int = 0,
    @SerializedName("issued_at")
    val issuedAt: Long = 0L,
    @SerializedName("expires_at")
    val expiresAt: Long = 0L,
)

data class BootstrapCredentialEnvelope(
    val schema: String = "",
    @SerializedName("key_algorithm")
    val keyAlgorithm: String = "",
    @SerializedName("content_algorithm")
    val contentAlgorithm: String = "",
    val aad: BootstrapCredentialAad = BootstrapCredentialAad(),
    @SerializedName("wrapped_key")
    val wrappedKey: String = "",
    val nonce: String = "",
    val ciphertext: String = "",
)

data class BootstrapCredentialRequestFields(
    val publicKey: String,
    val requestNonce: String,
    val requestedAt: Long,
)

class BootstrapCredentialException(
    val errorCode: String,
) : IOException("Bootstrap credential rejected [$errorCode]")

/**
 * One-shot in-memory key exchange for the first-connect bootstrap credential.
 *
 * The RSA private key is never serialized and is discarded before the envelope
 * is decrypted. A failed response therefore cannot be retried with the same
 * key/nonce pair; callers must start a fresh bootstrap exchange.
 */
class BootstrapCredentialExchange private constructor(
    private val expectedNodeId: String,
    val requestFields: BootstrapCredentialRequestFields,
    private val clock: () -> Long,
    private var privateKey: PrivateKey?,
) {
    companion object {
        const val SCHEMA = "qlh.cluster.bootstrap-credential.v1"
        const val KEY_ALGORITHM = "RSA-OAEP-SHA256"
        const val CONTENT_ALGORITHM = "AES-256-GCM"

        private const val RSA_KEY_BITS = 2048
        private const val REQUEST_NONCE_BYTES = 18
        private const val AES_KEY_BYTES = 32
        private const val GCM_NONCE_BYTES = 12
        private const val GCM_TAG_BITS = 128
        private const val MAX_CLOCK_SKEW_SECONDS = 300L
        private const val MAX_CREDENTIAL_TTL_SECONDS = 300L
        private const val MAX_CIPHERTEXT_BYTES = 8192
        private val BASE64URL_PATTERN = Regex("[A-Za-z0-9_-]+")

        internal fun create(
            nodeId: String,
            clock: () -> Long = { System.currentTimeMillis() / 1000L },
        ): BootstrapCredentialExchange {
            validateAscii("node_id", nodeId)
            val keyPair = KeyPairGenerator.getInstance("RSA").apply {
                initialize(RSA_KEY_BITS)
            }.generateKeyPair()
            val publicKey = keyPair.public as? RSAPublicKey
                ?: reject("public_key_invalid")
            if (publicKey.modulus.bitLength() != RSA_KEY_BITS) {
                reject("public_key_size_invalid")
            }
            val requestNonce = ByteArray(REQUEST_NONCE_BYTES).also {
                java.security.SecureRandom().nextBytes(it)
            }
            return BootstrapCredentialExchange(
                expectedNodeId = nodeId,
                requestFields = BootstrapCredentialRequestFields(
                    publicKey = encodeBase64Url(publicKey.encoded),
                    requestNonce = encodeBase64Url(requestNonce),
                    requestedAt = clock(),
                ),
                clock = clock,
                privateKey = keyPair.private,
            )
        }

        private fun encodeBase64Url(value: ByteArray): String =
            Base64.getUrlEncoder().withoutPadding().encodeToString(value)

        private fun decodeBase64Url(value: String, code: String, maxBytes: Int): ByteArray {
            if (value.isEmpty() || value.contains('=') || !BASE64URL_PATTERN.matches(value)) {
                reject(code)
            }
            val decoded = try {
                Base64.getUrlDecoder().decode(value)
            } catch (_: IllegalArgumentException) {
                reject(code)
            }
            if (decoded.size > maxBytes || encodeBase64Url(decoded) != value) {
                reject(code)
            }
            return decoded
        }

        private fun validateAscii(field: String, value: String) {
            if (value.isEmpty() || value.length > 512 || value.any { it.code > 0x7f }) {
                reject("${field}_invalid")
            }
        }

        private fun reject(code: String): Nothing = throw BootstrapCredentialException(code)
    }

    fun decrypt(
        envelope: BootstrapCredentialEnvelope,
        clusterId: String,
        nodeId: String,
        secretEpoch: Int,
    ): String {
        val exchangePrivateKey = synchronized(this) {
            privateKey?.also { privateKey = null } ?: reject("exchange_already_consumed")
        }

        if (envelope.schema != SCHEMA) reject("schema_invalid")
        if (envelope.keyAlgorithm != KEY_ALGORITHM) reject("key_algorithm_invalid")
        if (envelope.contentAlgorithm != CONTENT_ALGORITHM) reject("content_algorithm_invalid")
        validateAscii("cluster_id", clusterId)
        validateAscii("node_id", nodeId)
        if (nodeId != expectedNodeId) reject("node_id_mismatch")
        if (secretEpoch <= 0) reject("secret_epoch_invalid")

        val aad = envelope.aad
        validateAscii("cluster_id", aad.clusterId)
        validateAscii("node_id", aad.nodeId)
        if (aad.clusterId != clusterId) reject("cluster_id_mismatch")
        if (aad.nodeId != nodeId || aad.nodeId != expectedNodeId) reject("node_id_mismatch")
        if (aad.requestNonce != requestFields.requestNonce) reject("request_nonce_mismatch")
        if (aad.secretEpoch != secretEpoch) reject("secret_epoch_mismatch")

        val now = clock()
        if (aad.issuedAt <= 0L || aad.expiresAt <= aad.issuedAt) reject("validity_invalid")
        if (aad.issuedAt > now + MAX_CLOCK_SKEW_SECONDS) reject("credential_not_yet_valid")
        if (aad.expiresAt <= now) reject("credential_expired")
        if (aad.expiresAt - aad.issuedAt > MAX_CREDENTIAL_TTL_SECONDS) {
            reject("validity_invalid")
        }

        val wrappedKey = decodeBase64Url(envelope.wrappedKey, "wrapped_key_invalid", 512)
        val rsaPrivateKey = exchangePrivateKey as? RSAPrivateKey
            ?: reject("private_key_invalid")
        val expectedWrappedSize = (rsaPrivateKey.modulus.bitLength() + 7) / 8
        if (wrappedKey.size != expectedWrappedSize) reject("wrapped_key_invalid")
        val aesKey = try {
            val cipher = Cipher.getInstance("RSA/ECB/OAEPWithSHA-256AndMGF1Padding")
            cipher.init(
                Cipher.DECRYPT_MODE,
                exchangePrivateKey,
                OAEPParameterSpec(
                    "SHA-256",
                    "MGF1",
                    MGF1ParameterSpec.SHA256,
                    PSource.PSpecified.DEFAULT,
                ),
            )
            cipher.doFinal(wrappedKey)
        } catch (_: Exception) {
            reject("wrapped_key_invalid")
        }
        if (aesKey.size != AES_KEY_BYTES) {
            aesKey.fill(0)
            reject("wrapped_key_invalid")
        }

        val plaintext = try {
            val nonce = decodeBase64Url(envelope.nonce, "nonce_invalid", 64)
            if (nonce.size != GCM_NONCE_BYTES) reject("nonce_invalid")
            val ciphertext = decodeBase64Url(
                envelope.ciphertext,
                "ciphertext_invalid",
                MAX_CIPHERTEXT_BYTES,
            )
            if (ciphertext.size < GCM_TAG_BITS / 8) reject("ciphertext_invalid")
            val cipher = Cipher.getInstance("AES/GCM/NoPadding")
            cipher.init(
                Cipher.DECRYPT_MODE,
                SecretKeySpec(aesKey, "AES"),
                GCMParameterSpec(GCM_TAG_BITS, nonce),
            )
            cipher.updateAAD(canonicalAad(aad).toByteArray(Charsets.UTF_8))
            cipher.doFinal(ciphertext)
        } catch (error: BootstrapCredentialException) {
            throw error
        } catch (_: Exception) {
            reject("ciphertext_invalid")
        } finally {
            aesKey.fill(0)
        }

        return try {
            val root = JsonParser.parseString(plaintext.toString(Charsets.UTF_8)).asJsonObject
            if (root.keySet() != setOf("cluster_secret")) reject("plaintext_invalid")
            val secret = root.get("cluster_secret")?.takeUnless { it.isJsonNull }?.asString.orEmpty()
            if (secret.length < 16 || secret.length > 4096 || secret.any { it.code < 0x20 }) {
                reject("cluster_secret_invalid")
            }
            secret
        } catch (error: BootstrapCredentialException) {
            throw error
        } catch (_: Exception) {
            reject("plaintext_invalid")
        } finally {
            plaintext.fill(0)
        }
    }

    private fun canonicalAad(aad: BootstrapCredentialAad): String = buildString {
        append("{\"cluster_id\":")
        appendJsonAsciiString(aad.clusterId)
        append(",\"expires_at\":${aad.expiresAt}")
        append(",\"issued_at\":${aad.issuedAt}")
        append(",\"node_id\":")
        appendJsonAsciiString(aad.nodeId)
        append(",\"request_nonce\":")
        appendJsonAsciiString(aad.requestNonce)
        append(",\"secret_epoch\":${aad.secretEpoch}}")
    }

    private fun StringBuilder.appendJsonAsciiString(value: String) {
        append('"')
        for (character in value) {
            when (character) {
                '"' -> append("\\\"")
                '\\' -> append("\\\\")
                '\b' -> append("\\b")
                '\u000c' -> append("\\f")
                '\n' -> append("\\n")
                '\r' -> append("\\r")
                '\t' -> append("\\t")
                else -> when {
                    character.code < 0x20 -> append("\\u%04x".format(character.code))
                    character.code <= 0x7f -> append(character)
                    else -> reject("aad_non_ascii")
                }
            }
        }
        append('"')
    }
}
