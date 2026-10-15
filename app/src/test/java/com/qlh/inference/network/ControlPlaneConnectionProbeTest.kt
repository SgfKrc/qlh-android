package com.qlh.inference.network

import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.runBlocking
import org.junit.Assert.*
import org.junit.Test
import java.net.ServerSocket
import java.io.IOException
import kotlin.concurrent.thread

class ControlPlaneConnectionProbeTest {
    @Test
    fun `reachable control TCP is independent from unavailable HTTP API`() = runBlocking {
        ServerSocket(0).use { control ->
            val result = probeControlPlaneConnection("127.0.0.1", control.localPort,
                bootstrapInfo = { Result.failure(IOException("connection refused")) })
            assertTrue(result.controlReachable)
            assertEquals(ControlPlaneProbeState.BOOTSTRAP_UNAVAILABLE, result.state)
            assertEquals("bootstrap_api_unreachable", result.code)
            assertFalse(result.message.contains("已连接"))
        }
    }

    @Test
    fun `TCP refusal is a network error and never calls bootstrap`() = runBlocking {
        val port = ServerSocket(0).use { it.localPort }
        var called = false
        val result = probeControlPlaneConnection("127.0.0.1", port, bootstrapInfo = {
            called = true
            Result.success(MasterBootstrapInfo(true))
        })
        assertFalse(called)
        assertFalse(result.controlReachable)
        assertEquals(ControlPlaneProbeState.NETWORK_ERROR, result.state)
        assertEquals("control_plane_unreachable", result.code)
    }

    @Test
    fun `bootstrap master false is an identity error despite reachable TCP`() = runBlocking {
        val result = probeWithHttp("""{"is_master":false}""")
        assertTrue(result.controlReachable)
        assertEquals(ControlPlaneProbeState.IDENTITY_ERROR, result.state)
        assertEquals("master_identity_unconfirmed", result.code)
    }

    @Test
    fun `verified master uses bootstrap info and never cluster status`() = runBlocking {
        val result = probeWithHttp("""{"is_master":true}""")
        assertEquals(ControlPlaneProbeState.VERIFIED, result.state)
        assertEquals("control_plane_reachable", result.code)
        assertFalse(result.message.contains("已连接"))
    }

    @Test
    fun `bootstrap authorization failure is separate from network refusal`() = runBlocking {
        val result = probeWithHttp("""{"detail":{"code":"master_identity_unconfirmed"}}""", 403)
        assertEquals(ControlPlaneProbeState.IDENTITY_ERROR, result.state)
        assertTrue(result.controlReachable)
    }

    @Test
    fun `malformed bootstrap cannot pass identity verification`() = runBlocking {
        val result = probeWithHttp("""{"is_master":"true"}""")
        assertEquals(ControlPlaneProbeState.PROTOCOL_ERROR, result.state)
        assertEquals("bootstrap_info_invalid", result.code)
    }

    @Test
    fun `cancellation is propagated instead of being displayed as a network error`() {
        ServerSocket(0).use { control ->
            assertThrows(CancellationException::class.java) {
                runBlocking {
                    probeControlPlaneConnection("127.0.0.1", control.localPort,
                        bootstrapInfo = { throw CancellationException("selection changed") })
                }
            }
        }
    }

    private suspend fun probeWithHttp(body: String, status: Int = 200): ControlPlaneProbeResult {
        ServerSocket(0).use { control ->
            ServerSocket(0).use { api ->
                var failure: Throwable? = null
                val server = thread(name = "connection-probe-http") {
                    try {
                        api.accept().use { socket ->
                            socket.soTimeout = 3_000
                            val request = socket.getInputStream().bufferedReader()
                            assertEquals("GET /api/bootstrap/info HTTP/1.1", request.readLine())
                            while (request.readLine().isNotEmpty()) { }
                            val bytes = body.toByteArray(Charsets.UTF_8)
                            socket.getOutputStream().apply {
                                write("HTTP/1.1 $status Test\r\nContent-Type: application/json\r\nContent-Length: ${bytes.size}\r\nConnection: close\r\n\r\n".toByteArray())
                                write(bytes)
                                flush()
                            }
                        }
                    } catch (error: Throwable) { failure = error }
                }
                try {
                    return probeControlPlaneConnection("127.0.0.1", control.localPort,
                        bootstrapInfo = { ApiClient("http://127.0.0.1:${api.localPort}", callTimeoutSeconds = 3).getBootstrapInfo() })
                } finally {
                    api.close()
                    server.join(4_000)
                    assertFalse(server.isAlive)
                    failure?.let { throw AssertionError("HTTP stub failed", it) }
                }
            }
        }
    }
}
