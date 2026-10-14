package com.qlh.inference.network

import com.google.gson.JsonParser
import kotlinx.coroutines.*
import org.junit.Assert.*
import org.junit.Test
import java.net.DatagramPacket
import java.net.DatagramSocket
import java.net.InetSocketAddress
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit

class MasterDiscoveryTest {
    private val nonce = "a".repeat(32)
    private fun reply(nonce: String = this.nonce, host: String = "100.64.0.1") =
        """{"protocol":"qlh-master-discovery-v1","nonce":"$nonce","master_api_host":"$host","master_api_port":8000,"master_tcp_port":8888,"node_id":"master-1"}"""
    private fun parse(text: String) = parseMasterDiscoveryHint(text.toByteArray(Charsets.UTF_8), nonce)

    @Test fun `discovery validates exact schema nonce protocol and primitive endpoint types`() {
        assertEquals(MasterDiscoveryHint("100.64.0.1", 8000, 8888, "master-1"), parse(reply()))
        assertNull(parse(reply("b".repeat(32))))
        assertNull(parse(reply().replace("qlh-master-discovery-v1", "other")))
        assertNull(parse(reply().replace("8000", "\"8000\"")))
        for (port in listOf("0", "65536", "8000.0", "8e3", "-1", "null", "true")) {
            assertNull("accepted port $port", parse(reply().replace("8000", port)))
        }
        assertNull(parse(reply().dropLast(1) + ",\"extra\":1}"))
        assertNull(parse(reply().replace("\"node_id\":\"master-1\"", "\"node_id\":{}")))
        assertNull(parse(reply().replace("\"node_id\":\"master-1\"", "\"node_id\":\"master-1\",\"nonce\":\"$nonce\"")))
        assertNull(parse(reply().replace("\"protocol\"", "protocol")))
        assertNull(parse(reply() + "{}"))
    }

    @Test fun `discovery rejects paths URL controls malformed hosts and length overruns`() {
        for (host in listOf("http://100.64.0.1", "100.64.0.1/path", "user@host", "host:8000", "bad host",
            "", " 100.64.0.1", "1.2.3.999", "1.2.3", "001.2.3.4", "-host", "host-", "0.0.0.0", "a".repeat(254))) {
            assertNull("accepted host $host", parse(reply(host = host)))
        }
        assertNotNull(parse(reply(host = "master.example.local")))
        assertNotNull(parse(reply(host = "fd7a:115c:a1e0::1")))
        assertNull(parse(reply().replace("master-1", "x".repeat(129))))
        assertNull(parse(reply() + " ".repeat(MASTER_DISCOVERY_MAX_BYTES)))
        assertNull(parseMasterDiscoveryHint(byteArrayOf(0xc3.toByte(), 0x28), nonce))
    }

    @Test fun `localhost discovery sends fresh nonce and accepts only matching reply`() = runBlocking {
        DatagramSocket(0).use { server ->
            val failure = CompletableDeferred<Throwable?>()
            val worker = Thread {
                try {
                    server.soTimeout = 1500
                    val packet = DatagramPacket(ByteArray(2048), 2048)
                    server.receive(packet)
                    val request = JsonParser.parseString(packet.data.copyOf(packet.length).toString(Charsets.UTF_8)).asJsonObject
                    assertEquals(setOf("protocol", "nonce"), request.keySet())
                    assertEquals(MASTER_DISCOVERY_PROTOCOL, request["protocol"].asString)
                    val freshNonce = request["nonce"].asString
                    assertTrue(Regex("[0-9a-f]{32}").matches(freshNonce))
                    assertNotEquals(nonce, freshNonce)
                    for (payload in listOf(reply("f".repeat(32)), reply(freshNonce))) {
                        val bytes = payload.toByteArray(Charsets.UTF_8)
                        server.send(DatagramPacket(bytes, bytes.size, packet.socketAddress))
                    }
                    failure.complete(null)
                } catch (error: Throwable) { failure.complete(error) }
            }.apply { isDaemon = true; start() }
            val hints = MasterDiscovery(listOf(InetSocketAddress("127.0.0.1", server.localPort)), 200).discover()
            failure.await()?.let { throw it }
            worker.join(1000)
            assertEquals(listOf(MasterDiscoveryHint("100.64.0.1", 8000, 8888, "master-1")), hints)
        }
    }

    @Test fun `discovery cancellation stops receive and releases the ephemeral socket`() = runBlocking {
        DatagramSocket(0).use { server ->
            val entered = CountDownLatch(1)
            var clientPort = 0
            val worker = Thread {
                server.soTimeout = 2000
                val packet = DatagramPacket(ByteArray(2048), 2048)
                server.receive(packet)
                clientPort = packet.port
                entered.countDown()
            }.apply { isDaemon = true; start() }
            val discovery = async { MasterDiscovery(listOf(InetSocketAddress("127.0.0.1", server.localPort)), 3000).discover() }
            assertTrue(withContext(Dispatchers.IO) { entered.await(2, TimeUnit.SECONDS) })
            discovery.cancel()
            withTimeout(700) { discovery.join() }
            assertTrue(discovery.isCancelled)
            assertTrue(clientPort > 0)
            DatagramSocket(clientPort).use { assertEquals(clientPort, it.localPort) }
            worker.join(1000)
        }
    }
}
