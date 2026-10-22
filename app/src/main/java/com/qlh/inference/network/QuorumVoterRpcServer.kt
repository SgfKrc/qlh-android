package com.qlh.inference.network

import com.google.gson.Gson
import com.google.gson.JsonObject
import com.qlh.inference.data.AndroidQuorumVoter
import java.io.BufferedInputStream
import java.io.BufferedOutputStream
import java.io.ByteArrayOutputStream
import java.net.InetAddress
import java.net.ServerSocket
import java.net.Socket
import java.nio.charset.StandardCharsets
import java.security.MessageDigest
import java.util.concurrent.ArrayBlockingQueue
import java.util.concurrent.ThreadPoolExecutor
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean

internal class QuorumVoterRpcServer(
    private val voter: AndroidQuorumVoter,
    private val secret: String,
    private val port: Int = DEFAULT_PORT,
    private val clock: () -> Long = System::currentTimeMillis,
) : AutoCloseable {
    private val active = AtomicBoolean(false)
    private val workers = ThreadPoolExecutor(
        2, 4, 30, TimeUnit.SECONDS, ArrayBlockingQueue(32),
        { runnable -> Thread(runnable, "quorum-voter-rpc").apply { isDaemon = true } },
        ThreadPoolExecutor.AbortPolicy(),
    )
    @Volatile private var listener: ServerSocket? = null
    private var acceptThread: Thread? = null
    @Volatile var boundPort: Int = port
        private set

    @Synchronized
    fun start() {
        if (!active.compareAndSet(false, true)) return
        val socket = ServerSocket().apply {
            reuseAddress = true
            bind(java.net.InetSocketAddress(InetAddress.getByName("0.0.0.0"), port), 16)
        }
        listener = socket
        boundPort = socket.localPort
        acceptThread = Thread({ accept(socket) }, "quorum-voter-listener").apply {
            isDaemon = true
            start()
        }
    }

    private fun accept(socket: ServerSocket) {
        while (active.get()) {
            try {
                val client = socket.accept().apply { soTimeout = SOCKET_TIMEOUT_MS }
                try {
                    workers.execute { client.use(::serve) }
                } catch (_: java.util.concurrent.RejectedExecutionException) {
                    client.close()
                }
            } catch (_: java.net.SocketException) {
                if (active.get()) close()
            } catch (_: Exception) {
                if (active.get()) close()
            }
        }
    }

    private fun serve(socket: Socket) {
        val input = BufferedInputStream(socket.getInputStream())
        val output = BufferedOutputStream(socket.getOutputStream())
        try {
            val requestLine = readLine(input, MAX_HEADER_BYTES) ?: return
            val request = requestLine.split(' ')
            if (request.size != 3 || request[2] != "HTTP/1.1") {
                respond(output, 400, error("quorum_unavailable")); return
            }
            var consumed = requestLine.length
            val headers = linkedMapOf<String, String>()
            while (true) {
                val line = readLine(input, MAX_HEADER_BYTES - consumed) ?: run {
                    respond(output, 400, error("quorum_unavailable")); return
                }
                consumed += line.length + 2
                if (consumed > MAX_HEADER_BYTES) {
                    respond(output, 413, error("quorum_unavailable")); return
                }
                if (line.isEmpty()) break
                val delimiter = line.indexOf(':')
                if (delimiter <= 0) {
                    respond(output, 400, error("quorum_unavailable")); return
                }
                headers[line.substring(0, delimiter).trim().lowercase()] = line.substring(delimiter + 1).trim()
            }
            if (request[0] != "POST" || request[1] != RPC_PATH) {
                respond(output, 404, error("quorum_unavailable")); return
            }
            if (!MessageDigest.isEqual(
                    secret.toByteArray(StandardCharsets.UTF_8),
                    headers[SECRET_HEADER.lowercase()].orEmpty().toByteArray(StandardCharsets.UTF_8),
                )) {
                respond(output, 403, error("quorum_auth_failed")); return
            }
            val length = headers["content-length"]?.toIntOrNull()
            if (length == null || length < 0 || length > MAX_BODY_BYTES || headers.containsKey("transfer-encoding")) {
                respond(output, 413, error("quorum_unavailable")); return
            }
            val body = input.readNBytes(length)
            if (body.size != length) {
                respond(output, 400, error("quorum_unavailable")); return
            }
            val payload = runCatching { Gson().fromJson(body.toString(Charsets.UTF_8), JsonObject::class.java) }
                .getOrNull()
            if (payload == null) {
                respond(output, 400, error("quorum_unavailable")); return
            }
            respond(output, 200, voter.handle(payload, clock()))
        } catch (_: Exception) {
            runCatching { respond(output, 400, error("quorum_unavailable")) }
        }
    }

    private fun readLine(input: BufferedInputStream, limit: Int): String? {
        if (limit <= 0) return null
        val buffer = ByteArrayOutputStream()
        while (buffer.size() <= limit) {
            val value = input.read()
            if (value < 0) return if (buffer.size() == 0) null else throw IllegalArgumentException("truncated HTTP line")
            if (value == '\n'.code) {
                val bytes = buffer.toByteArray()
                val size = if (bytes.lastOrNull() == '\r'.code.toByte()) bytes.size - 1 else bytes.size
                return bytes.copyOf(size).toString(StandardCharsets.US_ASCII)
            }
            buffer.write(value)
        }
        throw IllegalArgumentException("HTTP line too long")
    }

    private fun respond(output: BufferedOutputStream, status: Int, payload: JsonObject) {
        val body = Gson().toJson(payload).toByteArray(StandardCharsets.UTF_8)
        val reason = when (status) {
            200 -> "OK"
            400 -> "Bad Request"
            403 -> "Forbidden"
            404 -> "Not Found"
            else -> "Payload Too Large"
        }
        val headers = "HTTP/1.1 $status $reason\r\n" +
            "Content-Type: application/json; charset=utf-8\r\n" +
            "Content-Length: ${body.size}\r\nConnection: close\r\n\r\n"
        output.write(headers.toByteArray(StandardCharsets.US_ASCII))
        output.write(body)
        output.flush()
    }

    private fun error(code: String) = JsonObject().apply {
        addProperty("ok", false)
        addProperty("code", code)
    }

    @Synchronized
    override fun close() {
        if (!active.compareAndSet(true, false)) return
        runCatching { listener?.close() }
        listener = null
        acceptThread?.join(1_000)
        acceptThread = null
        workers.shutdownNow()
    }

    companion object {
        const val DEFAULT_PORT = 8765
        private const val RPC_PATH = "/api/cluster/quorum/voter"
        private const val SECRET_HEADER = "X-QLH-Quorum-Secret"
        private const val MAX_HEADER_BYTES = 16 * 1024
        private const val MAX_BODY_BYTES = 1024 * 1024
        private const val SOCKET_TIMEOUT_MS = 3_000
    }
}
