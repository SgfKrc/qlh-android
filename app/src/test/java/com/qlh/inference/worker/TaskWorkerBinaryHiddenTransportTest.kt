package com.qlh.inference.worker

import com.google.gson.JsonObject
import com.google.gson.JsonParser
import java.io.ByteArrayInputStream
import java.io.ByteArrayOutputStream
import java.io.DataInputStream
import java.io.DataOutputStream
import java.net.Socket
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.Base64
import kotlinx.coroutines.runBlocking
import org.junit.Assert.*
import org.junit.Test

/** PC wire fixtures with socket streams; no device or network dependencies. */
class TaskWorkerBinaryHiddenTransportTest {
    private val identity = TaskWorkerAttemptIdentity("wf_12345678", "stage_1", "att_12345678", "lease_12345678", 1)
    private val raw = ByteBuffer.allocate(16).order(ByteOrder.LITTLE_ENDIAN)
        .putInt(0x3f800000).putInt(0x80000000.toInt()).putInt(0x7fc01234).putInt(0x00000001).array()
    private val b64 get() = Base64.getEncoder().encodeToString(raw)

    @Test
    fun `self advertisement and accepted legacy hello keep replies JSON`() = runBlocking {
        for (flag in listOf<Boolean?>(null, false)) {
            val socket = MemorySocket(jsonFrame(ack(flag)))
            val transport = SocketTaskWorkerTransport(socket)
            transport.setBinaryHiddenEnabled(true)
            transport.send(result(raw))
            transport.receive()
            transport.send(result(raw))
            val input = DataInputStream(ByteArrayInputStream(socket.written.toByteArray()))
            repeat(2) {
                val frame = readMeta(input)
                assertEquals("json", frame.get("format").asString)
                assertEquals(b64, frame.getAsJsonObject("data").getAsJsonObject("payload")
                    .getAsJsonObject("output").get("hidden_out_f32").asString)
                assertFalse(frame.has("binary_bytes"))
            }
            assertEquals(-1, input.read())
        }
    }

    @Test
    fun `master ACK enables return raw with big endian lengths and unchanged float bits`() = runBlocking {
        val socket = MemorySocket(jsonFrame(ack(true)))
        val transport = SocketTaskWorkerTransport(socket)
        transport.setBinaryHiddenEnabled(true)
        transport.receive()
        transport.send(result(raw))
        val input = DataInputStream(ByteArrayInputStream(socket.written.toByteArray()))
        val meta = readMeta(input)
        assertEquals("task_worker", meta.get("type").asString)
        assertEquals("binary", meta.get("format").asString)
        assertEquals(raw.size, meta.get("binary_bytes").asInt)
        assertFalse(meta.getAsJsonObject("data").getAsJsonObject("payload")
            .getAsJsonObject("output").has("hidden_out_f32"))
        assertArrayEquals(byteArrayOf(0, 0, 0, 16), ByteArray(4).also(input::readFully))
        assertArrayEquals(raw, ByteArray(raw.size).also(input::readFully))
        assertEquals(-1, input.read())
    }

    @Test
    fun `refresh and new transport require a fresh master ACK`() = runBlocking {
        val socket = MemorySocket(jsonFrame(ack(true)))
        val transport = SocketTaskWorkerTransport(socket)
        transport.setBinaryHiddenEnabled(true)
        transport.receive()
        transport.send(hello())
        transport.send(result(raw))
        val input = DataInputStream(ByteArrayInputStream(socket.written.toByteArray()))
        assertEquals("json", readMeta(input).get("format").asString)
        assertEquals("json", readMeta(input).get("format").asString)
        val replacement = MemorySocket(jsonFrame(ack(true)))
        val replacementTransport = SocketTaskWorkerTransport(replacement)
        replacementTransport.receive()
        replacementTransport.send(result(raw))
        assertEquals("json", readMeta(DataInputStream(ByteArrayInputStream(replacement.written.toByteArray())))
            .get("format").asString)
    }

    @Test
    fun `restored binary larger than JSON limit works in both directions`() = runBlocking {
        val large = ByteArray(6_400_000) { (it % 251).toByte() }
        val result = result(large)
        assertTrue(TaskWorkerProtocol.canonicalBytes(result).size > TaskWorkerProtocol.MAX_MESSAGE_BYTES)
        val socket = MemorySocket(jsonFrame(ack(true)) + binaryFrame(result, large))
        val transport = SocketTaskWorkerTransport(socket)
        transport.setBinaryHiddenEnabled(true)
        transport.receive()
        assertTrue("restored output must retain canonical bytes",
            TaskWorkerProtocol.canonicalBytes(result).contentEquals(TaskWorkerProtocol.canonicalBytes(transport.receive()!!)))
        transport.send(result)
        val input = DataInputStream(ByteArrayInputStream(socket.written.toByteArray()))
        assertEquals(large.size, readMeta(input).get("binary_bytes").asInt)
        assertArrayEquals(large, ByteArray(input.readInt()).also(input::readFully))
    }

    @Test
    fun `large binary layer offer retains its input digest for the Android tail`() = runBlocking {
        val body = ByteArray(6_400_000) { (it % 251).toByte() }
        val offer = TaskWorkerProtocol.buildStageOffer(
            identity, "request_binary_tail", "layer_forward", "remote_android_12345678", 10000,
            mapOf("hidden_f32" to Base64.getEncoder().encodeToString(body), "want_hidden" to false),
            emptyMap(), mapOf("model_id" to "tail24_32", "engine" to "llama_cpp", "format" to "gguf",
                "revision" to "local", "sha256" to "a".repeat(64)), "msg_offer_12345678", 1000,
            mapOf("layer_range" to listOf(24, 32), "handoff_at" to 24,
                "hidden_sha256" to StageChunkAssembler.sha256Hex(body),
                "hidden_spec" to mapOf("n_tokens" to 400, "n_embd" to 4000, "dtype" to "float32")),
        )
        assertTrue(TaskWorkerProtocol.canonicalBytes(offer).size > TaskWorkerProtocol.MAX_MESSAGE_BYTES)
        val transport = negotiatedTransport(binaryFrame(offer, body))
        val received = transport.receive()!!
        assertEquals(offer.payload["input_sha256"], received.payload["input_sha256"])
        val root = received.payload["root_input"] as Map<*, *>
        assertArrayEquals(body, Base64.getDecoder().decode(root["hidden_f32"] as String))
        assertEquals(false, root["want_hidden"])
    }

    @Test
    fun `binary ACK is invalid on rejected hello or selected v2`() {
        val rejected = ack(true).copy(payload = ack(true).payload + mapOf(
            "accepted" to false, "selected_version" to 0, "reason_code" to "not_admitted"))
        val v2 = ack(true).copy(version = 2, payload = ack(true).payload + ("selected_version" to 2))
        for (invalid in listOf(rejected, v2)) {
            try { TaskWorkerProtocol.encode(invalid); fail("invalid binary ACK was accepted") }
            catch (error: TaskWorkerProtocolException) { assertEquals("invalid_negotiation_result", error.code) }
        }
    }

    @Test
    fun `binary input chunks restore before protocol validation without losing the next frame`() = runBlocking {
        val chunk = chunk(TaskWorkerProtocol.STAGE_CHUNK)
        val transport = negotiatedTransport(binaryFrame(chunk, raw) + jsonFrame(ack(null)))
        assertArrayEquals(TaskWorkerProtocol.canonicalBytes(chunk), TaskWorkerProtocol.canonicalBytes(transport.receive()!!))
        assertEquals(TaskWorkerProtocol.HELLO_ACK, transport.receive()!!.messageType)
        assertNull(transport.receive())
    }

    @Test
    fun `return chunk uses PC stage_result_chunk contract`() = runBlocking {
        val socket = MemorySocket(jsonFrame(ack(true)))
        val transport = SocketTaskWorkerTransport(socket)
        transport.setBinaryHiddenEnabled(true)
        transport.receive()
        transport.send(chunk(TaskWorkerProtocol.STAGE_RESULT_CHUNK))
        val input = DataInputStream(ByteArrayInputStream(socket.written.toByteArray()))
        val meta = readMeta(input)
        assertEquals("stage_result_chunk", meta.getAsJsonObject("data").get("message_type").asString)
        assertFalse(meta.getAsJsonObject("data").getAsJsonObject("payload").has("payload_b64"))
        assertArrayEquals(raw, ByteArray(input.readInt()).also(input::readFully))
    }

    @Test
    fun `unadvertised binary input is rejected`() = runBlocking {
        expectInvalid { SocketTaskWorkerTransport(MemorySocket(binaryFrame(result(raw), raw))).receive() }
    }

    @Test
    fun `malformed metadata and conflicting payloads fail closed`() = runBlocking {
        val changes: List<(JsonObject) -> Unit> = listOf(
            { it.remove("binary_bytes") }, { it.addProperty("binary_bytes", "16") },
            { it.addProperty("binary_bytes", true) }, { it.addProperty("binary_bytes", 16.5) },
            { it.addProperty("binary_bytes", -1) }, { it.addProperty("binary_bytes", 0) },
            { it.addProperty("binary_bytes", Long.MAX_VALUE) }, { it.addProperty("type", "heartbeat_ack") },
            { it.add("data", JsonParser.parseString("[]")) },
            { it.getAsJsonObject("data").add("payload", JsonParser.parseString("[]")) },
            { it.getAsJsonObject("data").addProperty("message_type", "hello_ack") },
            { it.getAsJsonObject("data").getAsJsonObject("payload").add("output", JsonParser.parseString("[]")) },
            { it.getAsJsonObject("data").getAsJsonObject("payload").getAsJsonObject("output").addProperty("hidden_out_f32", b64) },
            { it.getAsJsonObject("data").getAsJsonObject("payload").getAsJsonObject("output").add("hidden_out_ref", JsonObject()) },
        )
        for (change in changes) {
            val transport = negotiatedTransport(binaryFrame(result(raw), raw, change = change))
            expectInvalid { transport.receive() }
        }
    }

    @Test
    fun `mismatched empty truncated or partial lengths are invalid frames`() = runBlocking {
        val good = binaryFrame(result(raw), raw)
        val variants = listOf(binaryFrame(result(raw), raw, bodyLength = 15),
            binaryFrame(result(raw), raw, bodyLength = 0), binaryFrame(result(raw), raw, bodyLength = -1),
            good.copyOf(good.size - 1), good.copyOf(good.size - raw.size),
            good.copyOf(good.size - raw.size - 1), good.copyOf(good.size - raw.size - 4),
            good.copyOf(2), good.copyOf(4), byteArrayOf(0, 0, 0, 0), byteArrayOf(-1, -1, -1, -1))
        for (wire in variants) {
            val transport = negotiatedTransport(wire)
            expectInvalid { transport.receive() }
        }
    }

    @Test
    fun `metadata plus body must fit the combined wire budget`() = runBlocking {
        val body = ByteArray(3600)
        val wire = binaryFrame(result(body), body)
        assertTrue(wire.size > 4096)
        val transport = negotiatedTransport(wire, maxFrameBytes = 4096)
        expectInvalid { transport.receive() }
        val socket = MemorySocket(jsonFrame(ack(true)))
        val sending = SocketTaskWorkerTransport(socket, maxFrameBytes = 4096)
        sending.setBinaryHiddenEnabled(true)
        sending.receive()
        try { sending.send(result(body)); fail("oversized frame accepted") }
        catch (error: TaskWorkerProtocolException) { assertEquals("message_too_large", error.code) }
        assertEquals(0, socket.written.size())
    }

    @Test
    fun `incoming binary requires the master ACK on this connection`() = runBlocking {
        val frame = binaryFrame(result(raw), raw)
        val pending = SocketTaskWorkerTransport(MemorySocket(frame))
        pending.setBinaryHiddenEnabled(true)
        expectInvalid { pending.receive() }
        for (flag in listOf<Boolean?>(null, false)) {
            val legacy = SocketTaskWorkerTransport(MemorySocket(jsonFrame(ack(flag)) + frame))
            legacy.setBinaryHiddenEnabled(true)
            legacy.receive()
            expectInvalid { legacy.receive() }
        }
        val refreshed = negotiatedTransport(frame)
        refreshed.send(hello())
        expectInvalid { refreshed.receive() }
    }

    @Test
    fun `raw eight MiB fits the shared binary limit in both directions`() = runBlocking {
        assertEquals(8 * 1024 * 1024 + 1024, TaskWorkerProtocol.MAX_BINARY_FRAME_BYTES)
        val body = ByteArray(TaskWorkerProtocol.MAX_MESSAGE_BYTES)
        val envelope = result(body)
        val frame = binaryFrame(envelope, body)
        assertTrue(frame.size - 4 <= TaskWorkerProtocol.MAX_BINARY_FRAME_BYTES)
        val socket = MemorySocket(jsonFrame(ack(true)) + frame)
        val transport = SocketTaskWorkerTransport(socket)
        transport.setBinaryHiddenEnabled(true)
        transport.receive()
        val restored = transport.receive()!!
        val output = restored.payload["output"] as Map<*, *>
        assertArrayEquals(body, Base64.getDecoder().decode(output["hidden_out_f32"] as String))
        transport.send(envelope)
        val input = DataInputStream(ByteArrayInputStream(socket.written.toByteArray()))
        assertEquals(body.size, readMeta(input).get("binary_bytes").asInt)
        assertArrayEquals(body, ByteArray(input.readInt()).also(input::readFully))
        val oversized = binaryFrame(envelope, body, change = { it.addProperty("padding", "x".repeat(1024)) })
        expectInvalid { negotiatedTransport(oversized).receive() }
    }

    @Test
    fun `capability and ACK values must be boolean`() {
        for (value in listOf<Any?>("true", 1, null, emptyList<Any>())) {
            try {
                TaskWorkerProtocol.buildHello("android_12345678", capabilities() + ("binary_hidden" to value), "msg_hello_12345678", 1000)
                fail("non-boolean capability accepted")
            } catch (error: TaskWorkerProtocolException) { assertEquals("invalid_boolean", error.code) }
            try {
                TaskWorkerProtocol.encode(ack(null).copy(payload = ack(null).payload + ("binary_hidden" to value)))
                fail("non-boolean ACK accepted")
            } catch (error: TaskWorkerProtocolException) { assertEquals("invalid_boolean", error.code) }
        }
    }

    private suspend fun negotiatedTransport(wire: ByteArray,
            maxFrameBytes: Int = TaskWorkerProtocol.MAX_BINARY_FRAME_BYTES): SocketTaskWorkerTransport {
        val transport = SocketTaskWorkerTransport(MemorySocket(jsonFrame(ack(true)) + wire), maxFrameBytes)
        transport.setBinaryHiddenEnabled(true)
        assertEquals(TaskWorkerProtocol.HELLO_ACK, transport.receive()!!.messageType)
        return transport
    }

    private fun result(body: ByteArray) = TaskWorkerProtocol.buildStageResult(identity, "remote_android_12345678",
        mapOf("hidden_out_f32" to Base64.getEncoder().encodeToString(body),
            "hidden_out_sha256" to StageChunkAssembler.sha256Hex(body), "token_argmax" to 7),
        emptyMap(), "msg_result_12345678", 1000)
    private fun chunk(type: String) = TaskWorkerProtocol.build(type, identity.asPayload() + mapOf(
        "provider_id" to "remote_android_12345678", "chunk_index" to 0, "chunk_count" to 1,
        "total_bytes" to raw.size, "payload_b64" to b64, "payload_sha256" to StageChunkAssembler.sha256Hex(raw)),
        "msg_chunk_12345678", 1000)
    private fun capabilities() = mapOf("stage_types" to listOf("layer_forward"), "engines" to listOf("llama_cpp"),
        "models" to emptyList<Any>(), "max_concurrency" to 1, "binary_hidden" to true)
    private fun hello() = TaskWorkerProtocol.buildHello("android_12345678", capabilities(), "msg_hello_12345678", 1000)
    private fun ack(binary: Boolean?) = TaskWorkerProtocol.buildHelloAck("master_12345678", true, 3, "", "msg_ack_12345678", 1000)
        .let { if (binary == null) it else it.copy(payload = it.payload + ("binary_hidden" to binary)) }
    private fun jsonFrame(envelope: TaskWorkerEnvelope) = pack(
        "{\"type\":\"task_worker\",\"format\":\"json\",\"data\":${TaskWorkerProtocol.canonicalBytes(envelope).toString(Charsets.UTF_8)}}".toByteArray())
    private fun binaryFrame(envelope: TaskWorkerEnvelope, body: ByteArray, bodyLength: Int = body.size,
                            change: (JsonObject) -> Unit = {}): ByteArray {
        val inner = JsonParser.parseString(TaskWorkerProtocol.canonicalBytes(envelope).toString(Charsets.UTF_8)).asJsonObject
        val payload = inner.getAsJsonObject("payload")
        when (envelope.messageType) {
            TaskWorkerProtocol.STAGE_OFFER -> payload.getAsJsonObject("root_input").remove("hidden_f32")
            TaskWorkerProtocol.STAGE_RESULT -> payload.getAsJsonObject("output").remove("hidden_out_f32")
            else -> payload.remove("payload_b64")
        }
        val meta = JsonObject().apply {
            addProperty("type", "task_worker")
            addProperty("format", "binary")
            addProperty("binary_bytes", body.size)
            add("data", inner)
        }
        change(meta)
        return ByteArrayOutputStream().apply {
            write(pack(meta.toString().toByteArray(Charsets.UTF_8)))
            DataOutputStream(this).apply { writeInt(bodyLength); write(body) }
        }.toByteArray()
    }
    private fun pack(body: ByteArray) = ByteArrayOutputStream().apply {
        DataOutputStream(this).apply { writeInt(body.size); write(body) }
    }.toByteArray()
    private fun readMeta(input: DataInputStream) = JsonParser.parseString(
        ByteArray(input.readInt()).also(input::readFully).toString(Charsets.UTF_8)).asJsonObject
    private suspend fun expectInvalid(block: suspend () -> Any?) {
        try { block(); fail("malformed frame accepted") }
        catch (error: TaskWorkerProtocolException) { assertEquals("invalid_frame", error.code) }
    }
    private class MemorySocket(bytes: ByteArray) : Socket() {
        val written = ByteArrayOutputStream()
        private val incoming = ByteArrayInputStream(bytes)
        override fun getInputStream() = incoming
        override fun getOutputStream() = written
    }
}
