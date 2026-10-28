package com.qlh.inference.worker

import com.google.gson.JsonObject
import com.google.gson.JsonParser
import kotlinx.coroutines.runBlocking
import org.junit.Assert.*
import org.junit.Test
import java.io.DataInputStream
import java.io.DataOutputStream
import java.net.ServerSocket
import java.net.Socket
import kotlin.concurrent.thread

class WorkerPrepareTransportTest {
    @Test fun `model prepare roundtrips with exact wire fields`() = runBlocking {
        val data = wire()
        val server = ServerSocket(0); var failure: Throwable? = null
        val peer = thread {
            try { server.accept().use { socket ->
                val bytes = JsonObject().apply { addProperty("type", "model_prepare"); addProperty("format", "json"); add("data", data) }.toString().toByteArray()
                DataOutputStream(socket.getOutputStream()).apply { writeInt(bytes.size); write(bytes); flush() }
                val input = DataInputStream(socket.getInputStream()); val reply = ByteArray(input.readInt()); input.readFully(reply)
                val outer = JsonParser.parseString(reply.toString(Charsets.UTF_8)).asJsonObject
                assertEquals("model_prepare", outer["type"].asString)
                val ack = outer["data"].asJsonObject
                assertEquals(setOf("action", "phase", "identity", "status", "loaded"), ack.keySet())
                assertEquals(data["identity"], ack["identity"])
                assertEquals(1_791_800_000_000_000_013L, ack["identity"].asJsonObject["generation"].asLong)
            } } catch (error: Throwable) { failure = error }
        }
        try {
            val transport = SocketTaskWorkerTransport(Socket("127.0.0.1", server.localPort))
            val received = transport.receiveEvent() as TaskWorkerInboundEvent.ModelPrepare
            val request = WorkerPrepareRequest.parse(received.data)
            assertEquals(1_791_800_000_000_000_013L, request.identity.generation)
            transport.sendModelPrepare(request.ack("prepared", true)); transport.close()
        } finally { server.close(); peer.join(5000) }
        assertNull(failure)
    }

    @Test fun `coordinator error frame surfaces stable rejection code`() = runBlocking {
        val server = ServerSocket(0)
        val peer = thread { server.accept().use { socket ->
            val bytes = JsonObject().apply {
                addProperty("type", "error")
                addProperty("format", "json")
                add("data", JsonObject().apply {
                    addProperty("status", "rejected")
                    addProperty("error_code", "node_unavailable")
                })
            }.toString().toByteArray()
            DataOutputStream(socket.getOutputStream()).apply { writeInt(bytes.size); write(bytes); flush() }
        } }
        val transport = SocketTaskWorkerTransport(Socket("127.0.0.1", server.localPort))
        try {
            try {
                transport.receiveEvent()
                fail("coordinator error frame must be surfaced")
            } catch (error: TaskWorkerProtocolException) {
                assertEquals("node_unavailable", error.code)
                assertEquals("data.error_code", error.field)
            }
        } finally {
            transport.close()
            server.close()
            peer.join(5000)
        }
    }

    companion object {
        fun wire(): JsonObject = JsonParser.parseString("""{
            "action":"prepare","context_size":4096,"middle_channel":"keep_head_layer_out",
            "identity":{"operation_id":"supply_11111111111111111111111111111111","config_id":"config_1",
            "generation":1791800000000000013,"assignment_id":"assignment_1","node_id":"android_test",
            "model_id":"master_hf","model_sha256":"${"a".repeat(64)}",
            "stage_model_identity":{"model_id":"tail.gguf","engine":"llama_cpp","format":"gguf","revision":"local","sha256":"${"c".repeat(64)}"},
            "artifact_sha256":"${"c".repeat(64)}","manifest_sha256":"","source_model_id":"source_gguf",
            "source_model_sha256":"${"b".repeat(64)}","tokenizer_sha256":"${"d".repeat(64)}",
            "layer_range":[2,4],"total_layers":4,"hidden_size":1,"segment_mode":"tail"}}
        """).asJsonObject
    }
}
