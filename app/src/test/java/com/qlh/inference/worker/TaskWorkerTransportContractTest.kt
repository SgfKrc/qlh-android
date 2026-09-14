package com.qlh.inference.worker

import com.google.gson.Gson
import com.google.gson.JsonObject
import com.google.gson.JsonParser
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.BufferedInputStream
import java.io.BufferedOutputStream
import java.io.DataInputStream
import java.io.DataOutputStream
import java.net.ServerSocket
import java.nio.charset.StandardCharsets
import kotlin.concurrent.thread

class TaskWorkerTransportContractTest {
    private val gson = Gson()

    @Test
    fun `socket transport round trips PC task worker framing`() = runBlocking {
        val server = ServerSocket(0)
        var serverFailure: Throwable? = null
        val serverThread = thread(start = true, name = "task-worker-contract") {
            try {
                server.accept().use { socket ->
                    val input = DataInputStream(BufferedInputStream(socket.getInputStream()))
                    val output = DataOutputStream(BufferedOutputStream(socket.getOutputStream()))
                    val frame = readFrame(input)
                    assertEquals(setOf("type", "format", "data"), frame.keySet())
                    assertEquals("task_worker", frame.get("type").asString)
                    assertEquals("json", frame.get("format").asString)
                    val hello = TaskWorkerProtocol.decode(
                        gson.toJson(frame.get("data")).toByteArray(StandardCharsets.UTF_8),
                    )
                    assertEquals(TaskWorkerProtocol.HELLO, hello.messageType)
                    assertEquals(TaskWorkerProtocol.ANDROID_WORKER_KIND, hello.payload["worker_kind"])
                    writeFrame(
                        output,
                        TaskWorkerProtocol.buildHelloAck(
                            coordinatorNodeId = "master_12345678",
                            accepted = true,
                            selectedVersion = TaskWorkerProtocol.VERSION,
                            reasonCode = "",
                            messageId = "msg_ack_12345678",
                            sentAtMs = 1_700_000_000_001L,
                        ),
                    )
                    val resultFrame = readFrame(input)
                    val result = TaskWorkerProtocol.decode(
                        gson.toJson(resultFrame.get("data")).toByteArray(StandardCharsets.UTF_8),
                    )
                    assertEquals(TaskWorkerProtocol.STAGE_RESULT, result.messageType)
                    assertEquals(
                        "跨平台 UTF-8",
                        (result.payload["output"] as Map<*, *>)["text"],
                    )
                }
            } catch (error: Throwable) {
                serverFailure = error
            }
        }

        try {
            val transport = SocketTaskWorkerTransportFactory(
                connectTimeoutMs = 2_000,
                readTimeoutMs = 2_000,
            ).connect("127.0.0.1", server.localPort)
            transport.send(
                TaskWorkerProtocol.buildHello(
                    nodeId = "android_12345678",
                    capabilities = mapOf(
                        "stage_types" to listOf("full_inference"),
                        "engines" to listOf("llama_cpp"),
                        "models" to emptyList<Map<String, Any?>>(),
                        "max_concurrency" to 1,
                    ),
                    messageId = "msg_hello_12345678",
                    sentAtMs = 1_700_000_000_000L,
                ),
            )
            val ack = transport.receive()
            assertEquals(TaskWorkerProtocol.HELLO_ACK, ack?.messageType)
            assertEquals(true, ack?.payload?.get("accepted"))
            transport.send(
                TaskWorkerProtocol.buildStageResult(
                    identity = TaskWorkerAttemptIdentity(
                        workflowId = "wf_12345678",
                        stageId = "stage_1",
                        attemptId = "att_12345678",
                        leaseId = "lease_12345678",
                        leaseEpoch = 1,
                    ),
                    providerId = "android_12345678",
                    output = mapOf("text" to "跨平台 UTF-8"),
                    metadata = emptyMap(),
                    messageId = "msg_result_12345678",
                    sentAtMs = 1_700_000_000_002L,
                ),
            )
            transport.close()
        } finally {
            server.close()
        }
        serverThread.join(3_000)
        assertFalse(serverThread.isAlive)
        assertNull(serverFailure)
    }

    @Test
    fun `socket transport rejects non task worker outer frame`() = runBlocking {
        val server = ServerSocket(0)
        val serverThread = thread(start = true, name = "task-worker-invalid-contract") {
            server.accept().use { socket ->
                val output = DataOutputStream(BufferedOutputStream(socket.getOutputStream()))
                val bytes = "{\"type\":\"register\",\"format\":\"json\",\"data\":{}}"
                    .toByteArray(StandardCharsets.UTF_8)
                output.writeInt(bytes.size)
                output.write(bytes)
                output.flush()
            }
        }
        try {
            val transport = SocketTaskWorkerTransportFactory(
                connectTimeoutMs = 2_000,
                readTimeoutMs = 2_000,
            ).connect("127.0.0.1", server.localPort)
            try {
                transport.receive()
                assertTrue("invalid outer frame must fail", false)
            } catch (error: TaskWorkerProtocolException) {
                assertEquals("invalid_frame", error.code)
                assertEquals("message", error.field)
            } finally {
                transport.close()
            }
        } finally {
            server.close()
        }
        serverThread.join(3_000)
        assertFalse(serverThread.isAlive)
    }

    private fun readFrame(input: DataInputStream): JsonObject {
        val size = input.readInt()
        require(size > 0)
        val bytes = ByteArray(size)
        input.readFully(bytes)
        val root = JsonParser.parseString(String(bytes, StandardCharsets.UTF_8))
        require(root.isJsonObject)
        return root.asJsonObject
    }

    private fun writeFrame(output: DataOutputStream, envelope: TaskWorkerEnvelope) {
        val frame = JsonObject().apply {
            addProperty("type", "task_worker")
            addProperty("format", "json")
            add("data", JsonParser.parseString(
                TaskWorkerProtocol.encode(envelope).toString(StandardCharsets.UTF_8),
            ))
        }
        val bytes = gson.toJson(frame).toByteArray(StandardCharsets.UTF_8)
        output.writeInt(bytes.size)
        output.write(bytes)
        output.flush()
    }
}
