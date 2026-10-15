package com.qlh.inference.worker

import com.google.gson.JsonObject
import com.google.gson.JsonParser
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.delay
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withTimeout
import org.junit.Assert.*
import org.junit.Test
import java.io.DataInputStream
import java.io.DataOutputStream
import java.io.IOException
import java.net.InetSocketAddress
import java.net.ServerSocket
import java.net.Socket
import java.util.Locale
import java.util.concurrent.CopyOnWriteArrayList
import java.util.concurrent.atomic.AtomicInteger
import javax.crypto.Mac
import javax.crypto.spec.SecretKeySpec
import kotlin.concurrent.thread

/** Actual loopback framing, authenticated registration and re-registration; no scheduler or persistent store. */
class TaskWorkerReconnectContractTest {
    @Test
    fun `killed registered connection becomes offline and re-registers without restarting process`() = runBlocking {
        Master().use { master ->
            val client = client(master)
            try {
                client.start()
                awaitReady(client, master, 1)
                master.killConnection()
                await { client.snapshot.value.connection == TaskWorkerConnectionState.BACKING_OFF }
                assertTrue(client.isRunning)
                assertFalse(client.isConnectionLive)
                assertFalse(client.snapshot.value.connectionAlive)
                assertFalse(canReuseTaskWorker(client.isRunning, client.isConnectionLive, true, true))
                awaitReady(client, master, 2)
                assertEquals(listOf("online", "offline", "online"), master.transitions.take(3))
                assertEquals(2, master.registrations.get())
                assertEquals(0, client.snapshot.value.reconnectAttempt)
            } finally { client.stopAndJoin() }
            assertFalse(client.isRunning)
            assertFalse(client.snapshot.value.connectionAlive)
            master.assertHealthy()
        }
    }

    @Test
    fun `missing heartbeat acks time out even while other control traffic keeps arriving`() = runBlocking {
        Master().use { master ->
            val client = client(master, heartbeatTimeoutMs = 240)
            try {
                client.start()
                awaitReady(client, master, 1)
                master.ackHeartbeats = false
                master.sendOtherTraffic = true
                await { client.snapshot.value.lastErrorCode == "heartbeat_timeout" }
                assertEquals(TaskWorkerConnectionState.BACKING_OFF, client.snapshot.value.connection)
                assertFalse(client.isConnectionLive)
                assertFalse(client.snapshot.value.connectionAlive)
                assertTrue("other traffic must not count as heartbeat ACK", master.otherTraffic.get() >= 2)
                master.ackHeartbeats = true
                awaitReady(client, master, 2)
                assertEquals(listOf("online", "offline", "online"), master.transitions.take(3))
                assertNull(client.snapshot.value.lastErrorCode)
            } finally { client.stopAndJoin() }
            master.assertHealthy()
        }
    }

    @Test
    fun `socket read timeout closes stale connection and performs registration again`() = runBlocking {
        Master().use { master ->
            val client = client(master, readTimeoutMs = 180, heartbeatTimeoutMs = 2_000)
            try {
                client.start()
                awaitReady(client, master, 1)
                master.ackHeartbeats = false
                await { client.snapshot.value.lastErrorCode == "transport_timeout" }
                assertFalse(client.isConnectionLive)
                master.ackHeartbeats = true
                awaitReady(client, master, 2)
                assertEquals(listOf("online", "offline", "online"), master.transitions.take(3))
            } finally { client.stopAndJoin() }
            master.assertHealthy()
        }
    }

    @Test
    fun `control plane restart retries refusal then restores online on the same endpoint`() = runBlocking {
        Master().use { master ->
            val client = client(master)
            try {
                client.start()
                awaitReady(client, master, 1)
                val port = master.port
                master.pause()
                await { client.snapshot.value.lastErrorCode == "transport_connect_failed" }
                assertTrue(client.isRunning)
                assertFalse(client.isConnectionLive)
                assertTrue(client.snapshot.value.reconnectAttempt >= 2)
                master.resume()
                assertEquals(port, master.port)
                awaitReady(client, master, 2)
                assertEquals(listOf("online", "offline", "online"), master.transitions.take(3))
                assertEquals(2, master.registrations.get())
            } finally { client.stopAndJoin() }
            master.assertHealthy()
        }
    }

    @Test
    fun `persistent identity rejection has finite attempts and cannot claim a live connection`() = runBlocking {
        Master().use { master ->
            master.rejectRegistration = true
            val client = client(master, maxFailures = 3)
            try {
                client.start()
                await { client.snapshot.value.connection == TaskWorkerConnectionState.RETRY_EXHAUSTED && !client.isRunning }
                assertEquals(3, master.registrations.get())
                assertEquals(3, client.snapshot.value.reconnectAttempt)
                assertEquals("auth_failed", client.snapshot.value.lastErrorCode)
                assertEquals("identity", client.snapshot.value.failureKind)
                assertEquals(0L, client.snapshot.value.nextRetryAtMs)
                assertFalse(client.snapshot.value.connectionAlive)
                assertFalse(canReuseTaskWorker(client.isRunning, client.isConnectionLive, true, true))
                delay(200)
                assertEquals("retry limit must end the loop", 3, master.registrations.get())
                // Explicit START at service level creates a new client, rather than reusing the failed owner.
                master.rejectRegistration = false
                val replacement = client(master)
                try {
                    replacement.start()
                    awaitReady(replacement, master, 1)
                    assertEquals(4, master.registrations.get())
                } finally { replacement.stopAndJoin() }
            } finally { client.stopAndJoin() }
            master.assertHealthy()
        }
    }

    @Test
    fun `stopping during register immediately closes the unpublished socket`() = runBlocking {
        ServerSocket(0).use { server ->
            val registerReceived = CompletableDeferred<Unit>()
            val socketClosed = CompletableDeferred<Boolean>()
            val peer = thread(name = "worker-register-cancel-peer") {
                try {
                    server.accept().use { socket ->
                        socket.soTimeout = 3_000
                        val input = DataInputStream(socket.getInputStream())
                        val size = input.readInt()
                        input.readFully(ByteArray(size))
                        registerReceived.complete(Unit)
                        socketClosed.complete(input.read() == -1)
                    }
                } catch (error: Throwable) {
                    registerReceived.completeExceptionally(error)
                    socketClosed.completeExceptionally(error)
                }
            }
            val client = TaskWorkerClient("127.0.0.1", server.localPort, "android_worker_01",
                capabilities = { error("hello must not start while REGISTER is pending") },
                transportFactory = SocketTaskWorkerTransportFactory(readTimeoutMs = 45_000,
                    registration = TaskWorkerRegistration("android_worker_01", "isolated-contract-secret",
                        hostname = "jvm", networkType = "loopback", deviceInfo = emptyMap())))
            try {
                client.start()
                withTimeout(3_000) { registerReceived.await() }
                assertTrue(client.isRunning)
                assertFalse(client.isConnectionLive)
                withTimeout(2_000) { client.stopAndJoin() }
                assertTrue(withTimeout(2_000) { socketClosed.await() })
                assertFalse(client.snapshot.value.processRunning)
                assertEquals(TaskWorkerConnectionState.STOPPED, client.snapshot.value.connection)
            } finally {
                client.stopAndJoin()
                server.close()
                peer.join(3_000)
                assertFalse(peer.isAlive)
            }
        }
    }

    private fun client(master: Master, readTimeoutMs: Int = 2_000, heartbeatTimeoutMs: Long = 1_000,
                       maxFailures: Int = 8): TaskWorkerClient {
        val registration = TaskWorkerRegistration("android_worker_01", "isolated-contract-secret", 7,
            "jvm-test", "loopback", emptyMap())
        return TaskWorkerClient(
            host = "127.0.0.1", port = master.port, nodeId = registration.nodeId,
            registration = registration,
            capabilities = { mapOf("stage_types" to listOf("layer_forward"), "engines" to listOf("llama_cpp"),
                "models" to emptyList<Map<String, Any?>>(), "max_concurrency" to 1) },
            transportFactory = SocketTaskWorkerTransportFactory(300, readTimeoutMs, registration),
            machine = TaskWorkerStateMachine(baseBackoffMs = 80, maxBackoffMs = 160,
                maxConsecutiveFailures = maxFailures),
            heartbeatIntervalMs = 20,
            heartbeatTimeoutMs = heartbeatTimeoutMs,
        )
    }

    private suspend fun awaitReady(client: TaskWorkerClient, master: Master, count: Int) {
        try {
            await { master.hellos.get() >= count && client.isConnectionLive && client.snapshot.value.connectionAlive }
        } catch (error: Exception) {
            throw AssertionError("worker=${client.snapshot.value} registrations=${master.registrations.get()} " +
                "hellos=${master.hellos.get()} transitions=${master.transitions}", error)
        }
        assertTrue(client.snapshot.value.processRunning)
        assertEquals(TaskWorkerConnectionState.READY, client.snapshot.value.connection)
    }

    private suspend fun await(condition: () -> Boolean) = withTimeout(5_000) {
        while (!condition()) delay(5)
    }

    private class Master : AutoCloseable {
        val transitions = CopyOnWriteArrayList<String>()
        val registrations = AtomicInteger()
        val hellos = AtomicInteger()
        val otherTraffic = AtomicInteger()
        private val failures = CopyOnWriteArrayList<Throwable>()
        private val sockets = CopyOnWriteArrayList<Socket>()
        private val handlers = CopyOnWriteArrayList<Thread>()
        private var listener: ServerSocket? = null
        private var acceptThread: Thread? = null
        var port = 0; private set
        @Volatile var ackHeartbeats = true
        @Volatile var sendOtherTraffic = false
        @Volatile var rejectRegistration = false

        init { resume() }

        fun resume() {
            val server = ServerSocket().apply {
                reuseAddress = true
                bind(InetSocketAddress("127.0.0.1", port))
            }
            listener = server
            port = server.localPort
            acceptThread = thread(name = "worker-reconnect-master") {
                try {
                    while (!server.isClosed) {
                        val socket = server.accept()
                        sockets += socket
                        handlers += thread(name = "worker-reconnect-master-session") { session(socket) }
                    }
                } catch (error: IOException) {
                    if (!server.isClosed) failures += error
                }
            }
        }

        private fun session(socket: Socket) {
            var online = false
            try {
                socket.use {
                    socket.soTimeout = 3_000
                    val input = DataInputStream(socket.getInputStream())
                    val output = DataOutputStream(socket.getOutputStream())
                    val register = read(input)
                    assertEquals("register", register["type"].asString)
                    val data = register.getAsJsonObject("data")
                    val auth = data.getAsJsonObject("auth")
                    assertEquals(7, auth["auth_secret_epoch"].asInt)
                    val signed = String.format(Locale.US, "v2\n%s\n%.6f\n%s\n%d", data["client_id"].asString,
                        auth["auth_timestamp"].asDouble, auth["auth_nonce"].asString, auth["auth_secret_epoch"].asInt)
                    val mac = Mac.getInstance("HmacSHA256").apply {
                        init(SecretKeySpec("isolated-contract-secret".toByteArray(), "HmacSHA256"))
                    }
                    assertEquals(mac.doFinal(signed.toByteArray()).joinToString("") { "%02x".format(it) },
                        auth["auth_signature"].asString)
                    registrations.incrementAndGet()
                    if (rejectRegistration) {
                        write(output, """{"type":"register","format":"json","data":{"status":"rejected","error_code":"auth_failed","reason":"test identity rejected"}}""")
                        return
                    }
                    write(output, """{"type":"register","format":"json","data":{"status":"registered"}}""")
                    val hello = read(input)
                    assertEquals(TaskWorkerProtocol.HELLO, hello.getAsJsonObject("data")["message_type"].asString)
                    val count = hellos.incrementAndGet()
                    val ack = TaskWorkerProtocol.buildHelloAck("master_contract_01", true, TaskWorkerProtocol.VERSION,
                        "", "msg_ack_reconnect_$count", System.currentTimeMillis())
                    val frame = JsonObject().apply {
                        addProperty("type", "task_worker"); addProperty("format", "json")
                        add("data", JsonParser.parseString(TaskWorkerProtocol.encode(ack).toString(Charsets.UTF_8)))
                    }
                    transitions += "online"
                    online = true
                    write(output, frame.toString())
                    while (true) {
                        val message = read(input)
                        if (message["type"].asString == "heartbeat") {
                            if (ackHeartbeats) write(output, """{"type":"heartbeat_ack","format":"json","data":{}}""")
                            else if (sendOtherTraffic) {
                                write(output, """{"type":"node_list_sync","format":"json","data":{}}""")
                                otherTraffic.incrementAndGet()
                            }
                        }
                    }
                }
            } catch (_: IOException) {
                // Socket kill, watchdog retirement and shutdown all deliberately terminate reads.
            } catch (error: Throwable) { failures += error }
            finally {
                if (online) transitions += "offline"
                sockets.remove(socket)
            }
        }

        fun killConnection() { sockets.forEach { it.close() } }

        fun pause() {
            listener?.close()
            acceptThread?.join(2_000)
            assertFalse(acceptThread?.isAlive == true)
            killConnection()
            handlers.forEach { it.join(2_000); assertFalse(it.isAlive) }
            handlers.clear()
        }

        override fun close() { pause(); assertHealthy() }
        fun assertHealthy() { failures.firstOrNull()?.let { throw AssertionError("master fixture failed", it) } }
        private fun read(input: DataInputStream): JsonObject {
            val size = input.readInt()
            require(size in 1..1_048_576)
            return JsonParser.parseString(ByteArray(size).also(input::readFully).toString(Charsets.UTF_8)).asJsonObject
        }
        private fun write(output: DataOutputStream, text: String) {
            val bytes = text.toByteArray(Charsets.UTF_8)
            output.writeInt(bytes.size); output.write(bytes); output.flush()
        }
    }
}
