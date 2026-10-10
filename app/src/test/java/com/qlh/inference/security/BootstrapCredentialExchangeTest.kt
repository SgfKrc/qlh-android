package com.qlh.inference.security

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Test
import java.security.KeyFactory
import java.security.interfaces.RSAPublicKey
import java.security.spec.MGF1ParameterSpec
import java.security.spec.X509EncodedKeySpec
import java.util.Base64
import javax.crypto.Cipher
import javax.crypto.spec.GCMParameterSpec
import javax.crypto.spec.OAEPParameterSpec
import javax.crypto.spec.PSource
import javax.crypto.spec.SecretKeySpec

class BootstrapCredentialExchangeTest {
    companion object {
        private const val NOW = 1_800_000_000L
        private const val NODE_ID = "android-node-1"
        private const val CLUSTER_ID = "cluster-1"
        private const val SECRET_EPOCH = 7
        private const val CLUSTER_SECRET = "cluster-secret-that-must-never-appear-in-errors"
    }

    @Test
    fun `request uses unpadded base64url RSA 2048 SPKI and 18 byte nonce`() {
        val exchange = newExchange()
        val fields = exchange.requestFields

        assertFalse(fields.publicKey.contains('='))
        assertFalse(fields.requestNonce.contains('='))
        assertEquals(18, decode(fields.requestNonce).size)
        assertEquals(NOW, fields.requestedAt)

        val publicKey = KeyFactory.getInstance("RSA")
            .generatePublic(X509EncodedKeySpec(decode(fields.publicKey))) as RSAPublicKey
        assertEquals(2048, publicKey.modulus.bitLength())
    }

    @Test
    fun `decrypts Python compatible RSA OAEP SHA256 and AES GCM envelope`() {
        val exchange = newExchange()
        val envelope = envelopeFor(exchange)

        assertEquals(
            CLUSTER_SECRET,
            exchange.decrypt(envelope, CLUSTER_ID, NODE_ID, SECRET_EPOCH),
        )
    }

    @Test
    fun `rejects envelope metadata and response binding mismatches before install`() {
        assertRejected("schema_invalid") { exchange, envelope ->
            exchange.decrypt(envelope.copy(schema = "other"), CLUSTER_ID, NODE_ID, SECRET_EPOCH)
        }
        assertRejected("key_algorithm_invalid") { exchange, envelope ->
            exchange.decrypt(envelope.copy(keyAlgorithm = "RSA-OAEP-SHA1"), CLUSTER_ID, NODE_ID, SECRET_EPOCH)
        }
        assertRejected("content_algorithm_invalid") { exchange, envelope ->
            exchange.decrypt(envelope.copy(contentAlgorithm = "AES-128-GCM"), CLUSTER_ID, NODE_ID, SECRET_EPOCH)
        }
        assertRejected("request_nonce_mismatch") { exchange, envelope ->
            exchange.decrypt(
                envelope.copy(aad = envelope.aad.copy(requestNonce = "wrong-nonce")),
                CLUSTER_ID,
                NODE_ID,
                SECRET_EPOCH,
            )
        }
        assertRejected("cluster_id_mismatch") { exchange, envelope ->
            exchange.decrypt(envelope, "different-cluster", NODE_ID, SECRET_EPOCH)
        }
        assertRejected("node_id_mismatch") { exchange, envelope ->
            exchange.decrypt(envelope, CLUSTER_ID, "different-node", SECRET_EPOCH)
        }
        assertRejected("secret_epoch_mismatch") { exchange, envelope ->
            exchange.decrypt(envelope, CLUSTER_ID, NODE_ID, SECRET_EPOCH + 1)
        }
    }

    @Test
    fun `rejects expired future and pre request credentials`() {
        assertRejected("credential_expired") { exchange, _ ->
            val aad = aadFor(exchange).copy(issuedAt = NOW - 120, expiresAt = NOW - 1)
            exchange.decrypt(envelopeFor(exchange, aad = aad), CLUSTER_ID, NODE_ID, SECRET_EPOCH)
        }
        assertRejected("credential_not_yet_valid") { exchange, _ ->
            val aad = aadFor(exchange).copy(issuedAt = NOW + 301, expiresAt = NOW + 360)
            exchange.decrypt(envelopeFor(exchange, aad = aad), CLUSTER_ID, NODE_ID, SECRET_EPOCH)
        }
        assertRejected("validity_invalid") { exchange, _ ->
            val aad = aadFor(exchange).copy(issuedAt = NOW, expiresAt = NOW + 301)
            exchange.decrypt(envelopeFor(exchange, aad = aad), CLUSTER_ID, NODE_ID, SECRET_EPOCH)
        }
    }

    @Test
    fun `rejects tampering without exposing plaintext or response material`() {
        val exchange = newExchange()
        val envelope = envelopeFor(exchange)
        val replacement = if (envelope.ciphertext.last() == 'A') 'B' else 'A'
        val tampered = envelope.copy(
            ciphertext = envelope.ciphertext.dropLast(1) + replacement,
        )

        val error = expectRejected("ciphertext_invalid") {
            exchange.decrypt(tampered, CLUSTER_ID, NODE_ID, SECRET_EPOCH)
        }
        assertFalse(error.message.orEmpty().contains(CLUSTER_SECRET))
        assertFalse(error.message.orEmpty().contains(envelope.ciphertext))
    }

    @Test
    fun `exchange is consumed after one decrypt attempt`() {
        val exchange = newExchange()
        val envelope = envelopeFor(exchange)
        assertEquals(CLUSTER_SECRET, exchange.decrypt(envelope, CLUSTER_ID, NODE_ID, SECRET_EPOCH))

        expectRejected("exchange_already_consumed") {
            exchange.decrypt(envelope, CLUSTER_ID, NODE_ID, SECRET_EPOCH)
        }
    }

    private fun newExchange(): BootstrapCredentialExchange =
        BootstrapCredentialExchange.create(NODE_ID) { NOW }

    private fun aadFor(exchange: BootstrapCredentialExchange): BootstrapCredentialAad =
        BootstrapCredentialAad(
            clusterId = CLUSTER_ID,
            nodeId = NODE_ID,
            requestNonce = exchange.requestFields.requestNonce,
            secretEpoch = SECRET_EPOCH,
            issuedAt = NOW,
            expiresAt = NOW + 60,
        )

    private fun envelopeFor(
        exchange: BootstrapCredentialExchange,
        aad: BootstrapCredentialAad = aadFor(exchange),
        secret: String = CLUSTER_SECRET,
    ): BootstrapCredentialEnvelope {
        val publicKey = KeyFactory.getInstance("RSA").generatePublic(
            X509EncodedKeySpec(decode(exchange.requestFields.publicKey)),
        )
        val aesKey = ByteArray(32) { (it + 1).toByte() }
        val nonce = ByteArray(12) { (it + 17).toByte() }

        val wrappingCipher = Cipher.getInstance("RSA/ECB/OAEPWithSHA-256AndMGF1Padding")
        wrappingCipher.init(
            Cipher.ENCRYPT_MODE,
            publicKey,
            OAEPParameterSpec(
                "SHA-256",
                "MGF1",
                MGF1ParameterSpec.SHA256,
                PSource.PSpecified.DEFAULT,
            ),
        )
        val wrappedKey = wrappingCipher.doFinal(aesKey)

        val contentCipher = Cipher.getInstance("AES/GCM/NoPadding")
        contentCipher.init(
            Cipher.ENCRYPT_MODE,
            SecretKeySpec(aesKey, "AES"),
            GCMParameterSpec(128, nonce),
        )
        contentCipher.updateAAD(canonicalAad(aad).toByteArray(Charsets.UTF_8))
        val plaintext = "{\"cluster_secret\":\"$secret\"}".toByteArray(Charsets.UTF_8)
        val ciphertext = contentCipher.doFinal(plaintext)
        aesKey.fill(0)
        plaintext.fill(0)

        return BootstrapCredentialEnvelope(
            schema = BootstrapCredentialExchange.SCHEMA,
            keyAlgorithm = BootstrapCredentialExchange.KEY_ALGORITHM,
            contentAlgorithm = BootstrapCredentialExchange.CONTENT_ALGORITHM,
            aad = aad,
            wrappedKey = encode(wrappedKey),
            nonce = encode(nonce),
            ciphertext = encode(ciphertext),
        )
    }

    private fun canonicalAad(aad: BootstrapCredentialAad): String =
        "{\"cluster_id\":\"${aad.clusterId}\"," +
            "\"expires_at\":${aad.expiresAt}," +
            "\"issued_at\":${aad.issuedAt}," +
            "\"node_id\":\"${aad.nodeId}\"," +
            "\"request_nonce\":\"${aad.requestNonce}\"," +
            "\"secret_epoch\":${aad.secretEpoch}}"

    private fun assertRejected(
        code: String,
        action: (BootstrapCredentialExchange, BootstrapCredentialEnvelope) -> Unit,
    ) {
        val exchange = newExchange()
        val envelope = envelopeFor(exchange)
        expectRejected(code) { action(exchange, envelope) }
    }

    private fun expectRejected(code: String, action: () -> Unit): BootstrapCredentialException {
        try {
            action()
            fail("expected BootstrapCredentialException [$code]")
        } catch (error: BootstrapCredentialException) {
            assertEquals(code, error.errorCode)
            return error
        }
        throw AssertionError("unreachable")
    }

    private fun encode(value: ByteArray): String =
        Base64.getUrlEncoder().withoutPadding().encodeToString(value)

    private fun decode(value: String): ByteArray = Base64.getUrlDecoder().decode(value)
}
