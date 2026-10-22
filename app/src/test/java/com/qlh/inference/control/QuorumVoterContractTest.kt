package com.qlh.inference.control

import com.google.gson.JsonParser
import com.qlh.inference.security.QuorumEd25519
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class QuorumVoterContractTest {
    private val seed = ByteArray(32) { 1 }
    private val publicKey = QuorumEd25519.decode("iojj3XQJ8ZX9UtstPLpdcspnCb8dlBIb83SIAbQPb1w", 32)

    private fun certificate(watermark: String = ""): AndroidQuorumCertificate {
        val suffix = if (watermark.isEmpty()) "" else ",\"commit_watermark\":[{\"workflow_id\":\"workflow-1\",\"last_sequence\":9,\"checkpoint_digest\":\"${"a".repeat(64)}\"}]"
        return AndroidQuorumCertificate.parse(JsonParser.parseString(
            """{"schema_version":"qlh.cluster.control.v1","document_type":"quorum_certificate","cluster_id":"cluster-test","voter_set_epoch":4,"term":7,"leader_id":"node-0","lease_id":"lease-7","expires_at_ms":2000,"signatures":[]$suffix}""",
        ).asJsonObject)
    }

    @Test
    fun `canonical payload and signature match Python golden vectors`() {
        val plain = certificate()
        assertEquals(
            "{\"cluster_id\":\"cluster-test\",\"document_type\":\"quorum_certificate\"," +
                "\"expires_at_ms\":2000,\"leader_id\":\"node-0\",\"lease_id\":\"lease-7\"," +
                "\"schema_version\":\"qlh.cluster.control.v1\",\"term\":7,\"voter_set_epoch\":4}",
            String(plain.signingBytes(), Charsets.UTF_8),
        )
        val signature = QuorumEd25519.sign(seed, plain.signingBytes())
        assertEquals(
            "oMEs7AUuSwFa-Ohy3Dz2-0qfhIcPSDxfAByDU-_zCy71gxmxE8uGrl76iPIpv8Q7r4q-EKc4lRaf-uiync0kDw",
            QuorumEd25519.encode(signature),
        )
        assertTrue(QuorumEd25519.verify(publicKey, plain.signingBytes(), signature))
        assertTrue(QuorumEd25519.matches(seed, publicKey))
    }

    @Test
    fun generatedEd25519KeySignsAndVerifies() {
        val (generatedSeed, generatedPublicKey) = QuorumEd25519.generate()
        try {
            val message = "generated-key-check".toByteArray(Charsets.US_ASCII)
            val signature = QuorumEd25519.sign(generatedSeed, message)
            assertTrue(QuorumEd25519.verify(generatedPublicKey, message, signature))
            assertFalse(QuorumEd25519.verify(generatedPublicKey, byteArrayOf(0), signature))
        } finally {
            generatedSeed.fill(0)
        }
    }

    @Test
    fun `canonical payload preserves watermark field and ordering`() {
        val withWatermark = certificate("present")
        assertEquals(
            "{\"cluster_id\":\"cluster-test\",\"commit_watermark\":[{" +
                "\"checkpoint_digest\":\"${"a".repeat(64)}\",\"last_sequence\":9,\"workflow_id\":\"workflow-1\"}]," +
                "\"document_type\":\"quorum_certificate\",\"expires_at_ms\":2000,\"leader_id\":\"node-0\"," +
                "\"lease_id\":\"lease-7\",\"schema_version\":\"qlh.cluster.control.v1\",\"term\":7,\"voter_set_epoch\":4}",
            String(withWatermark.signingBytes(), Charsets.UTF_8),
        )
        assertEquals(
            "GhVvX7rXIWjv464ZIdUF3HHX7LEUV5MLr03Rc1DI-v9I-nhpeuSJtGF8Gd1_9YmooPEWvZ1wbdjtWdkfNn12Ag",
            QuorumEd25519.encode(QuorumEd25519.sign(seed, withWatermark.signingBytes())),
        )
    }

    @Test
    fun `voter set rejects unknown fields and duplicate identities`() {
        val valid = """{"schema_version":"qlh.cluster.control.v1","document_type":"voter_set","cluster_id":"cluster-test","voter_set_epoch":4,"voters":[{"voter_id":"node-0","public_key":"iojj3XQJ8ZX9UtstPLpdcspnCb8dlBIb83SIAbQPb1w"}]}"""
        assertEquals(1, AndroidVoterSet.parse(JsonParser.parseString(valid).asJsonObject).quorumSize)
        assertFalse(QuorumEd25519.verify(publicKey, byteArrayOf(1), ByteArray(64)))
        try {
            AndroidVoterSet.parse(JsonParser.parseString(valid.dropLast(1) + ",\"extra\":true}").asJsonObject)
            throw AssertionError("unknown voter-set field accepted")
        } catch (_: IllegalArgumentException) {
            assertTrue(true)
        }
        try {
            AndroidVoterSet.parse(JsonParser.parseString(valid.replace("]}", ",{\"voter_id\":\"node-0\",\"public_key\":\"iojj3XQJ8ZX9UtstPLpdcspnCb8dlBIb83SIAbQPb1w\"}]}")).asJsonObject)
            throw AssertionError("duplicate voter identity accepted")
        } catch (_: IllegalArgumentException) {
            assertTrue(true)
        }
    }

    @Test
    fun `base64url decoder rejects padding and noncanonical data`() {
        try {
            QuorumEd25519.decode("iojj3XQJ8ZX9UtstPLpdcspnCb8dlBIb83SIAbQPb1w=", 32)
            throw AssertionError("padded key accepted")
        } catch (_: IllegalArgumentException) {
            assertTrue(true)
        }
        try {
            QuorumEd25519.decode("not base64", 32)
            throw AssertionError("invalid key accepted")
        } catch (_: IllegalArgumentException) {
            assertTrue(true)
        }
    }
}
