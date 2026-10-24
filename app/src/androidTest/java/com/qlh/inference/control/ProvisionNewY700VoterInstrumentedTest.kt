package com.qlh.inference.control

import android.content.Context
import androidx.core.content.ContextCompat
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.google.gson.JsonObject
import com.google.gson.JsonParser
import com.qlh.inference.data.SettingsDataStore
import com.qlh.inference.security.QuorumEd25519
import com.qlh.inference.security.QuorumVoterKeyStore
import com.qlh.inference.service.QuorumVoterService
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.json.JSONObject
import java.io.BufferedReader
import java.io.InputStreamReader
import java.net.InetSocketAddress
import java.net.Socket

@RunWith(AndroidJUnit4::class)
class ProvisionNewY700VoterInstrumentedTest {
    private val context: Context
        get() = InstrumentationRegistry.getInstrumentation().targetContext

    @Test
    fun reportsPersistedY700VoterPublicKey() {
        val publicKey = requireNotNull(QuorumVoterKeyStore(context).readPublicKey())
        android.util.Log.i("Y700Provision", "public_key=$publicKey")
    }

    @Test
    fun provisionsNewVoterIdentityAndStartsRpc() = runBlocking {
        val arguments = InstrumentationRegistry.getArguments()
        val port = requireNotNull(arguments.getString("credential_port")).toInt()
        val token = requireNotNull(arguments.getString("credential_token"))
        val material = QuorumVoterKeyStore(context).readOrCreate()
        val publicKey = try {
            QuorumEd25519.encode(material.publicKey)
        } finally {
            material.seed.fill(0)
        }
        val response = exchangeProvisioning(port, token, publicKey)
        val secret = response.getString("secret")
        val clusterId = response.getString("cluster_id")
        val clusterSecretEpoch = response.getInt("cluster_secret_epoch")
        val voterId = response.getString("voter_id")
        val voterSetJson = response.getJSONObject("voter_set").toString()
        val voterSet = AndroidVoterSet.parse(JsonParser.parseString(voterSetJson).asJsonObject)
        assertTrue(secret.length >= 32)
        assertEquals("qlh-default", clusterId)
        assertEquals(4, voterSet.epoch)
        assertEquals(clusterId, voterSet.clusterId)
        assertEquals("y700-voter-r4", voterId)
        assertEquals(publicKey, voterSet.voters.getValue(voterId).publicKey.let(QuorumEd25519::encode))

        val settings = SettingsDataStore(context)
        assertTrue(settings.saveBootstrapConfig(
            serverHost = response.getString("master_api_host"),
            serverPort = response.getInt("master_api_port"),
            masterTcpHost = response.getString("master_tcp_host"),
            masterTcpPort = response.getInt("master_tcp_port"),
            clusterId = clusterId,
            clusterSecret = secret,
            clusterSecretEpoch = clusterSecretEpoch,
            nodeId = settings.getOrCreateAndroidNodeId(),
            modelManifestUrl = "",
        ))
        settings.saveQuorumVoterConfig(voterId, voterSetJson, enabled = true)
        ContextCompat.startForegroundService(context, QuorumVoterService.startIntent(context))

        val deadline = System.nanoTime() + 20_000_000_000L
        while (System.nanoTime() < deadline) {
            val listening = runCatching {
                Socket().use { it.connect(InetSocketAddress("127.0.0.1", 8765), 300) }
            }.isSuccess
            if (listening) return@runBlocking
            Thread.sleep(250)
        }
        throw AssertionError("Y700 voter RPC listener did not start")
    }

    private fun exchangeProvisioning(port: Int, token: String, publicKey: String): JSONObject =
        Socket().use { socket ->
            socket.connect(InetSocketAddress("127.0.0.1", port), 3_000)
            socket.soTimeout = 10_000
            val request = JsonObject().apply {
                addProperty("token", token)
                addProperty("public_key", publicKey)
            }
            socket.getOutputStream().write((request.toString() + "\n").toByteArray(Charsets.UTF_8))
            val response = BufferedReader(InputStreamReader(socket.getInputStream(), Charsets.UTF_8)).readLine()
            JSONObject(requireNotNull(response))
        }
}
