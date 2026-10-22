package com.qlh.inference.control

import android.content.Context
import android.os.Build
import androidx.core.content.ContextCompat
import androidx.test.platform.app.InstrumentationRegistry
import com.google.gson.JsonObject
import com.google.gson.JsonParser
import com.qlh.inference.data.AndroidQuorumVoter
import com.qlh.inference.data.QuorumVoterLedger
import com.qlh.inference.data.SettingsDataStore
import com.qlh.inference.control.QuorumVoterException
import com.qlh.inference.network.QuorumVoterRpcServer
import com.qlh.inference.service.QuorumVoterService
import com.qlh.inference.security.QuorumEd25519
import com.qlh.inference.security.QuorumVoterKeyStore
import org.junit.Assume.assumeTrue
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import kotlinx.coroutines.runBlocking
import java.net.HttpURLConnection
import java.net.Socket
import java.net.URL
import java.util.Base64
import java.util.UUID

class QuorumVoterInstrumentedTest {
    private val context: Context
        get() = InstrumentationRegistry.getInstrumentation().targetContext
    private val databaseName = "quorum-test-${UUID.randomUUID()}.sqlite3"

    @Before
    fun clearDatabase() {
        context.deleteDatabase(databaseName)
    }

    @After
    fun removeDatabase() {
        context.deleteDatabase(databaseName)
    }

    @Test
    fun ledgerPersistsPreparedVoteAndCertificateAcrossReopen() {
        val keys = (0..2).map { QuorumEd25519.generate() }
        val voterSet = AndroidVoterSet(
            clusterId = "android-voter-test",
            epoch = 3,
            voters = keys.mapIndexed { index, pair ->
                val id = "voter-$index"
                id to AndroidVoterIdentity(id, pair.second)
            }.toMap(),
        )
        val ledger = QuorumVoterLedger(context, "voter-0", voterSet, databaseName)
        assertEquals(1L, ledger.reserveTerm("voter-0", 1_000))
        ledger.prepare("voter-0", 1, 1_000)
        val unsigned = AndroidQuorumCertificate.parse(JsonObject().apply {
            addProperty("schema_version", "qlh.cluster.control.v1")
            addProperty("document_type", "quorum_certificate")
            addProperty("cluster_id", voterSet.clusterId)
            addProperty("voter_set_epoch", voterSet.epoch)
            addProperty("term", 1)
            addProperty("leader_id", "voter-0")
            addProperty("lease_id", "lease-1")
            addProperty("expires_at_ms", 2_000)
            add("signatures", com.google.gson.JsonArray())
        })
        val localSignature = ledger.signVote(unsigned, keys[0].first, 1_000)
        val remoteSignature = AndroidVoterSignature(
            "voter-1",
            QuorumEd25519.encode(QuorumEd25519.sign(keys[1].first, unsigned.signingBytes())),
        )
        val signed = unsigned.withSignature(localSignature).withSignature(remoteSignature)
        ledger.commitCertificate(signed, 1_000)
        ledger.close()

        val reopened = QuorumVoterLedger(context, "voter-0", voterSet, databaseName)
        val snapshot = reopened.snapshot()
        assertEquals(1, snapshot["active_term"].asInt)
        assertEquals("voter-0", snapshot["active_leader"].asString)
        assertEquals(signed.digest(), snapshot["active_certificate_digest"].asString)
        assertEquals(2L, reopened.reserveTerm("voter-2", 2_001))
        reopened.close()
    }

    @Test
    fun rpcServerEnforcesSharedSecretAndServesVoterRoute() {
        val key = QuorumEd25519.generate()
        val voterSet = AndroidVoterSet(
            clusterId = "rpc-test",
            epoch = 1,
            voters = mapOf("voter-0" to AndroidVoterIdentity("voter-0", key.second)),
        )
        val ledger = QuorumVoterLedger(context, "voter-0", voterSet, databaseName)
        val server = QuorumVoterRpcServer(
            AndroidQuorumVoter("voter-0", voterSet, key.first, ledger),
            "secret-test",
            port = 0,
        )
        try {
            server.start()
            val denied = post(server.boundPort, "wrong-secret")
            assertTrue(denied.startsWith("HTTP/1.1 403"))
            val accepted = post(server.boundPort, "secret-test")
            assertTrue(accepted.startsWith("HTTP/1.1 200"))
            assertTrue(accepted.contains("\"voter_id\":\"voter-0\""))
            assertTrue(accepted.contains("\"active_term\":0"))
        } finally {
            server.close()
            ledger.close()
        }
    }

    @Test
    fun keystoreWrappedSigningKeySurvivesStoreRecreation() {
        assumeTrue(Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU)
        val suffix = UUID.randomUUID().toString().replace("-", "")
        val store = QuorumVoterKeyStore(
            context,
            storeName = "quorum-key-test-$suffix",
            keyAlias = "com.qlh.inference.quorum.test.$suffix",
        )
        try {
            val first = store.readOrCreate()
            val publicKey = QuorumEd25519.encode(first.publicKey)
            first.seed.fill(0)
            assertEquals(publicKey, store.readPublicKey())

            val reopened = QuorumVoterKeyStore(
                context,
                storeName = "quorum-key-test-$suffix",
                keyAlias = "com.qlh.inference.quorum.test.$suffix",
            ).readOrCreate()
            try {
                assertEquals(publicKey, QuorumEd25519.encode(reopened.publicKey))
                val message = "keystore-reopen".toByteArray(Charsets.US_ASCII)
                assertTrue(QuorumEd25519.verify(reopened.publicKey, message,
                    QuorumEd25519.sign(reopened.seed, message)))
            } finally {
                reopened.seed.fill(0)
            }
        } finally {
            store.clear()
        }
    }

    @Test
    fun watermarkedProposalFailsClosedWithoutJournalReplica() {
        val keys = (0..2).map { QuorumEd25519.generate() }
        val voterSet = AndroidVoterSet(
            clusterId = "watermark-test",
            epoch = 1,
            voters = keys.mapIndexed { index, pair ->
                val id = "voter-$index"
                id to AndroidVoterIdentity(id, pair.second)
            }.toMap(),
        )
        val ledger = QuorumVoterLedger(context, "voter-0", voterSet, databaseName)
        try {
            ledger.reserveTerm("voter-0", 1_000)
            ledger.prepare("voter-0", 1, 1_000)
            val certificate = AndroidQuorumCertificate.parse(JsonObject().apply {
                addProperty("schema_version", "qlh.cluster.control.v1")
                addProperty("document_type", "quorum_certificate")
                addProperty("cluster_id", voterSet.clusterId)
                addProperty("voter_set_epoch", voterSet.epoch)
                addProperty("term", 1)
                addProperty("leader_id", "voter-0")
                addProperty("lease_id", "lease-1")
                addProperty("expires_at_ms", 2_000)
                add("signatures", com.google.gson.JsonArray())
                add("commit_watermark", com.google.gson.JsonArray().apply {
                    add(JsonObject().apply {
                        addProperty("workflow_id", "workflow-1")
                        addProperty("last_sequence", 1)
                        addProperty("checkpoint_digest", "a".repeat(64))
                    })
                })
            })
            try {
                ledger.signVote(certificate, keys[0].first, 1_000)
                throw AssertionError("watermarked proposal accepted without local journal evidence")
            } catch (error: QuorumVoterException) {
                assertEquals("quorum_watermark_unavailable", error.code)
            }
        } finally {
            keys.forEach { it.first.fill(0) }
            ledger.close()
        }
    }

    @Test
    fun installPhysicalEpochTwoVoterConfiguration() {
        val encoded = InstrumentationRegistry.getArguments().getString("voter_set_b64")
        assumeTrue(!encoded.isNullOrBlank())
        val voterSet = String(Base64.getUrlDecoder().decode(encoded), Charsets.UTF_8)
        val settings = SettingsDataStore(context)
        assertEquals("qlh-default", runBlocking { settings.getClusterId() })
        runBlocking { settings.saveQuorumVoterConfig("y700-voter-r2", voterSet, true) }
        ContextCompat.startForegroundService(context, QuorumVoterService.startIntent(context))
        assertTrue(runBlocking { settings.isQuorumVoterEnabled() })
    }

    private fun post(port: Int, secret: String): String {
        val body = """{"op":"snapshot"}""".toByteArray(Charsets.UTF_8)
        return Socket("127.0.0.1", port).use { socket ->
            socket.soTimeout = 3_000
            val request = "POST /api/cluster/quorum/voter HTTP/1.1\r\n" +
                "Host: 127.0.0.1\r\n" +
                "Content-Type: application/json\r\n" +
                "Content-Length: ${body.size}\r\n" +
                "X-QLH-Quorum-Secret: $secret\r\n\r\n"
            socket.getOutputStream().apply {
                write(request.toByteArray(Charsets.US_ASCII))
                write(body)
                flush()
            }
            socket.getInputStream().readBytes().toString(Charsets.UTF_8)
        }
    }
}
