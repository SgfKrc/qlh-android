package com.qlh.inference.network

import com.google.gson.JsonObject
import com.google.gson.JsonParser
import com.google.gson.JsonPrimitive
import com.google.gson.Strictness
import com.google.gson.stream.JsonReader
import com.google.gson.stream.JsonToken
import java.io.StringReader
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.withContext
import java.net.DatagramPacket
import java.net.DatagramSocket
import java.net.InetAddress
import java.net.InetSocketAddress
import java.net.NetworkInterface
import java.net.SocketTimeoutException
import java.security.SecureRandom

/** Untrusted hints. Callers must verify /api/bootstrap/info and complete encrypted first-connect. */
data class MasterDiscoveryHint(
    val masterApiHost: String,
    val masterApiPort: Int,
    val masterTcpPort: Int,
    val nodeId: String,
)

internal const val MASTER_DISCOVERY_PROTOCOL = "qlh-master-discovery-v1"
internal const val MASTER_DISCOVERY_MAX_BYTES = 1024

internal fun validDiscoveryHost(host: String): Boolean {
    if (host.isEmpty() || host.length > 253 || host != host.trim()) return false
    if (host.contains(':')) {
        return host.length <= 45 && Regex("[0-9A-Fa-f:]+" ).matches(host) &&
            runCatching { InetAddress.getByName(host) is java.net.Inet6Address }.getOrDefault(false)
    }
    if (Regex("[0-9.]+").matches(host)) {
        val parts = host.split('.')
        return parts.size == 4 && parts.all { part ->
            part.isNotEmpty() && part.length <= 3 && part.toIntOrNull() in 0..255 &&
                (part.length == 1 || !part.startsWith('0'))
        } && host != "0.0.0.0" && host != "255.255.255.255"
    }
    return host.split('.').all { label ->
        label.length in 1..63 && Regex("[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?").matches(label)
    }
}

internal fun parseMasterDiscoveryHint(bytes: ByteArray, nonce: String): MasterDiscoveryHint? = runCatching {
    require(bytes.size in 1..MASTER_DISCOVERY_MAX_BYTES)
    // Reject malformed UTF-8 rather than replacing bytes in an endpoint.
    val text = Charsets.UTF_8.newDecoder().decode(java.nio.ByteBuffer.wrap(bytes)).toString()
    val json = JsonObject()
    JsonReader(StringReader(text)).use { reader ->
        reader.strictness = Strictness.STRICT
        reader.beginObject()
        while (reader.hasNext()) {
            val key = reader.nextName()
            require(!json.has(key))
            val value = when (reader.peek()) {
                JsonToken.STRING -> JsonPrimitive(reader.nextString())
                JsonToken.NUMBER -> JsonParser.parseString(reader.nextString())
                else -> error("discovery_field_invalid")
            }
            json.add(key, value)
        }
        reader.endObject()
        require(reader.peek() == JsonToken.END_DOCUMENT)
    }
    require(json.keySet() == setOf("protocol", "nonce", "master_api_host", "master_api_port", "master_tcp_port", "node_id"))
    fun string(key: String): String {
        val value = json[key].asJsonPrimitive
        require(value.isString)
        return value.asString
    }
    fun port(key: String): Int {
        val value = json[key].asJsonPrimitive
        require(value.isNumber && Regex("[0-9]{1,5}").matches(value.toString()))
        return value.asInt.also { require(it in 1..65535) }
    }
    require(Regex("[0-9a-f]{32}").matches(nonce) && string("nonce") == nonce)
    require(string("protocol") == MASTER_DISCOVERY_PROTOCOL)
    val host = string("master_api_host")
    require(validDiscoveryHost(host))
    val node = string("node_id")
    require(Regex("[A-Za-z0-9_.:-]{1,128}").matches(node))
    MasterDiscoveryHint(host, port("master_api_port"), port("master_tcp_port"), node)
}.getOrNull()

/** Sends multicast/broadcast and receives unicast replies; no multicast group or Wi-Fi lock needed. */
class MasterDiscovery(
    private val targets: List<InetSocketAddress>? = null,
    private val timeoutMillis: Long = 3000,
) {
    suspend fun discover(): List<MasterDiscoveryHint> = withContext(Dispatchers.IO) {
        require(timeoutMillis in 1..10_000)
        currentCoroutineContext().ensureActive()
        val nonce = ByteArray(16).also { SecureRandom().nextBytes(it) }
            .joinToString("") { "%02x".format(it.toInt() and 255) }
        val payload = JsonObject().apply {
            addProperty("protocol", MASTER_DISCOVERY_PROTOCOL)
            addProperty("nonce", nonce)
        }.toString().toByteArray(Charsets.UTF_8)
        val result = linkedSetOf<MasterDiscoveryHint>()
        val deadline = System.nanoTime() + timeoutMillis * 1_000_000
        DatagramSocket().use { socket ->
            socket.broadcast = true
            val destinations = targets ?: defaultTargets()
            destinations.forEach { target ->
                currentCoroutineContext().ensureActive()
                runCatching { socket.send(DatagramPacket(payload, payload.size, target)) }
            }
            while (System.nanoTime() < deadline && result.size < 8) {
                currentCoroutineContext().ensureActive()
                val remaining = ((deadline - System.nanoTime()) / 1_000_000).coerceAtLeast(1)
                socket.soTimeout = remaining.coerceAtMost(100).toInt()
                // One extra byte distinguishes truncated/oversize datagrams from valid packets.
                val packet = DatagramPacket(ByteArray(MASTER_DISCOVERY_MAX_BYTES + 1), MASTER_DISCOVERY_MAX_BYTES + 1)
                try {
                    socket.receive(packet)
                    parseMasterDiscoveryHint(packet.data.copyOf(packet.length), nonce)?.let(result::add)
                } catch (_: SocketTimeoutException) {
                    // Cancellation is checked at most 100 ms after a blocked receive.
                }
            }
        }
        currentCoroutineContext().ensureActive()
        result.toList()
    }

    private fun defaultTargets(): List<InetSocketAddress> {
        val hosts = linkedSetOf("239.255.81.76", "255.255.255.255")
        runCatching {
            val interfaces = NetworkInterface.getNetworkInterfaces()
            while (interfaces != null && interfaces.hasMoreElements()) {
                val network = interfaces.nextElement()
                if (network.isUp && !network.isLoopback) network.interfaceAddresses.forEach {
                    it.broadcast?.hostAddress?.let(hosts::add)
                }
            }
        }
        return hosts.map { InetSocketAddress(it, 29876) }
    }
}
