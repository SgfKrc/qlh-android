package com.qlh.inference.network

import com.google.gson.Gson
import com.google.gson.GsonBuilder
import com.google.gson.ToNumberPolicy
import com.google.gson.JsonParser
import com.qlh.inference.security.AuthSessionStore
import com.qlh.inference.security.StoredAuthSession
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.async
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import java.io.BufferedInputStream
import java.io.IOException
import java.net.ServerSocket
import java.net.Socket
import java.security.KeyFactory
import java.security.spec.MGF1ParameterSpec
import java.security.spec.X509EncodedKeySpec
import java.util.Base64
import javax.crypto.Cipher
import javax.crypto.spec.GCMParameterSpec
import javax.crypto.spec.OAEPParameterSpec
import javax.crypto.spec.PSource
import javax.crypto.spec.SecretKeySpec

/**
 * ApiClient HTTP 契约测试（JVM，本地 ServerSocket 桩，无需设备/网络/新依赖）。
 *
 * 验证：请求方法/路径/JSON body、响应解析（数组与包装两种格式）、
 * 非 2xx fail-closed（Result.failure 而非抛异常）、404 删除语义、网络不可达
 * 返回 Result.failure 而不是崩溃。
 */
class ApiClientContractTest {

    private lateinit var server: HttpStub
    private lateinit var client: ApiClient

    @Before
    fun setUp() {
        server = HttpStub()
        server.start()
        client = ApiClient(baseUrl = "http://127.0.0.1:${server.port}")
    }

    @After
    fun tearDown() {
        server.stop()
    }

    private fun route(path: String, handler: (Request, (Int, String) -> Unit) -> Unit) {
        server.routes[path] = handler
    }

    private class Request(
        val method: String,
        val path: String,
        val body: String,
        val headers: Map<String, String> = emptyMap(),
    )

    private class FakeAuthStore(initial: StoredAuthSession? = null) : AuthSessionStore {
        var session: StoredAuthSession? = initial
        var clearCount: Int = 0
        var failNextRead: Boolean = false

        override fun read(): StoredAuthSession? {
            if (failNextRead) {
                failNextRead = false
                return null
            }
            return session
        }
        override fun save(session: StoredAuthSession) {
            this.session = session
        }
        override fun clear() {
            clearCount += 1
            session = null
        }
    }

    // T14：完整 JSON 解析比较（Gson 自动还原 \u003d 转义）
    private fun parseJson(body: String): com.google.gson.JsonObject =
        JsonParser.parseString(body).asJsonObject

    /** 迷你 HTTP 桩：解析请求行/头/体，回调后写固定响应。 */
    private class HttpStub {
        private val serverSocket = ServerSocket(0)
        val port: Int get() = serverSocket.localPort
        val routes = mutableMapOf<String, (Request, (Int, String) -> Unit) -> Unit>()
        private var running = true
        private val thread = Thread {
            while (running) {
                val socket = try {
                    serverSocket.accept()
                } catch (_: IOException) {
                    break
                }
                Thread {
                    try {
                        handle(socket)
                    } catch (_: IOException) {
                        // 客户端主动断开等，忽略
                    } finally {
                        try {
                            socket.close()
                        } catch (_: IOException) {
                        }
                    }
                }.start()
            }
        }

        fun start() = thread.start()

        fun stop() {
            running = false
            try {
                serverSocket.close()
            } catch (_: IOException) {
            }
        }

        private fun handle(socket: Socket) {
            val input = BufferedInputStream(socket.getInputStream())
            fun readLine(): String? {
                val buf = java.io.ByteArrayOutputStream()
                while (true) {
                    val b = input.read()
                    if (b < 0) return null
                    if (b == '\n'.code) break
                    if (b != '\r'.code) buf.write(b)
                }
                return buf.toString(Charsets.UTF_8.name())
            }

            val requestLine = readLine() ?: return
            val parts = requestLine.split(" ")
            if (parts.size < 2) return
            val method = parts[0]
            val path = parts[1]

            var contentLength = 0
            val headers = mutableMapOf<String, String>()
            while (true) {
                val line = readLine() ?: break
                if (line.isEmpty()) break
                val lower = line.lowercase()
                if (lower.startsWith("content-length:")) {
                    contentLength = line.substringAfter(':').trim().toIntOrNull() ?: 0
                }
                val separator = line.indexOf(':')
                if (separator > 0) {
                    headers[line.substring(0, separator).trim()] = line.substring(separator + 1).trim()
                }
            }
            val body = if (contentLength > 0) {
                val bytes = ByteArray(contentLength)
                var off = 0
                while (off < contentLength) {
                    val n = input.read(bytes, off, contentLength - off)
                    if (n < 0) break
                    off += n
                }
                bytes.toString(Charsets.UTF_8)
            } else {
                ""
            }

            val handler = routes[path]
            val (code, responseBody) = if (handler != null) {
                var code = 200
                var response = "{}"
                handler(Request(method, path, body, headers)) { c, b ->
                    code = c
                    response = b
                }
                code to response
            } else {
                // 未知路径：记录但不匹配时返回 404
                404 to """{"error":"not found"}"""
            }

            val bytes = responseBody.toByteArray(Charsets.UTF_8)
            val out = socket.getOutputStream()
            out.write(
                ("HTTP/1.1 $code OK\r\n" +
                    "Content-Type: application/json\r\n" +
                    "Content-Length: ${bytes.size}\r\n" +
                    "Connection: close\r\n" +
                    "\r\n").toByteArray(Charsets.UTF_8)
            )
            out.write(bytes)
            out.flush()
        }
    }

    private fun storedSession(): StoredAuthSession = StoredAuthSession(
        accessToken = "token_" + "a".repeat(24),
        sessionId = "session-1",
        expiresAt = "2030-01-01T00:00:00.000Z",
        userId = "user-1",
        username = "owner",
        displayName = "Owner",
        role = "owner",
    )

    private fun bootstrapResponseFor(request: Request, clusterSecret: String): String {
        val body = parseJson(request.body)
        val nodeId = body.get("node_id").asString
        val requestNonce = body.get("credential_request_nonce").asString
        val requestedAt = body.get("credential_requested_at").asLong
        val publicKeyDer = Base64.getUrlDecoder().decode(
            body.get("credential_public_key").asString,
        )
        val publicKey = KeyFactory.getInstance("RSA").generatePublic(
            X509EncodedKeySpec(publicKeyDer),
        )
        val issuedAt = System.currentTimeMillis() / 1000L
        assertTrue(kotlin.math.abs(issuedAt - requestedAt) <= 5L)
        val expiresAt = issuedAt + 60L
        val epoch = 4
        val clusterId = "c1"
        val aad = linkedMapOf<String, Any>(
            "cluster_id" to clusterId,
            "node_id" to nodeId,
            "request_nonce" to requestNonce,
            "secret_epoch" to epoch,
            "issued_at" to issuedAt,
            "expires_at" to expiresAt,
        )
        val canonicalAad = "{\"cluster_id\":\"$clusterId\"," +
            "\"expires_at\":$expiresAt," +
            "\"issued_at\":$issuedAt," +
            "\"node_id\":\"$nodeId\"," +
            "\"request_nonce\":\"$requestNonce\"," +
            "\"secret_epoch\":$epoch}"

        val aesKey = ByteArray(32) { (it + 3).toByte() }
        val nonce = ByteArray(12) { (it + 11).toByte() }
        val wrappingCipher = Cipher.getInstance("RSA/ECB/OAEPWithSHA-256AndMGF1Padding")
        wrappingCipher.init(
            Cipher.ENCRYPT_MODE,
            publicKey,
            OAEPParameterSpec(
                "SHA-256",
                "MGF1",
                MGF1ParameterSpec.SHA256,
                PSource.PSpecified.DEFAULT,
            ),
        )
        val wrappedKey = wrappingCipher.doFinal(aesKey)

        val contentCipher = Cipher.getInstance("AES/GCM/NoPadding")
        contentCipher.init(
            Cipher.ENCRYPT_MODE,
            SecretKeySpec(aesKey, "AES"),
            GCMParameterSpec(128, nonce),
        )
        contentCipher.updateAAD(canonicalAad.toByteArray(Charsets.UTF_8))
        val plaintext = Gson().toJson(mapOf("cluster_secret" to clusterSecret))
            .toByteArray(Charsets.UTF_8)
        val ciphertext = contentCipher.doFinal(plaintext)
        aesKey.fill(0)
        plaintext.fill(0)

        val encoder = Base64.getUrlEncoder().withoutPadding()
        return Gson().toJson(
            linkedMapOf(
                "status" to "ok",
                "cluster" to linkedMapOf(
                    "cluster_id" to clusterId,
                    "master_api_host" to "100.1.2.3",
                    "master_api_port" to 8000,
                    "master_tcp_host" to "100.1.2.3",
                    "master_tcp_port" to 8888,
                    "cluster_secret_epoch" to epoch,
                ),
                "node" to linkedMapOf(
                    "node_id" to nodeId,
                    "role" to "client",
                    "node_type" to "android",
                ),
                "android" to emptyMap<String, Any>(),
                "credential_envelope" to linkedMapOf(
                    "schema" to "qlh.cluster.bootstrap-credential.v1",
                    "key_algorithm" to "RSA-OAEP-SHA256",
                    "content_algorithm" to "AES-256-GCM",
                    "aad" to aad,
                    "wrapped_key" to encoder.encodeToString(wrappedKey),
                    "nonce" to encoder.encodeToString(nonce),
                    "ciphertext" to encoder.encodeToString(ciphertext),
                ),
            ),
        )
    }

    @Test
    fun `auth capability is public and supports gateway shape`() {
        var seen: Request? = null
        route("/api/auth/capability") { req, reply ->
            seen = req
            reply(
                200,
                """{"required":true,"enforced":true,"mode":"local_totp","bootstrap_available":true}""",
            )
        }
        val result = runBlocking { client.getAuthCapability() }
        assertTrue(result.isSuccess)
        assertEquals("GET", seen?.method)
        assertEquals("/api/auth/capability", seen?.path)
        assertTrue(seen?.headers?.keys?.none { it.equals("Authorization", ignoreCase = true) } == true)
        assertTrue(result.getOrNull()!!.required)
        assertTrue(result.getOrNull()!!.enforced)
        assertEquals("local_totp", result.getOrNull()!!.mode)
    }

    @Test
    fun `login saves validated session without sending stale bearer`() {
        val store = FakeAuthStore(storedSession())
        client = ApiClient(baseUrl = "http://127.0.0.1:${server.port}", authStore = store)
        var seen: Request? = null
        route("/api/auth/login") { req, reply ->
            seen = req
            reply(
                200,
                """{"access_token":"new_token_${"b".repeat(24)}","token_type":"Bearer","session_id":"session-2","expires_at":"2030-01-02T00:00:00.000Z","user":{"user_id":"user-2","username":"alice","display_name":"Alice","role":"member"}}""",
            )
        }

        val result = runBlocking { client.login("alice", code = "123456") }
        assertTrue(result.isSuccess)
        assertEquals("alice", result.getOrNull()?.username)
        assertEquals("session-2", store.session?.sessionId)
        assertTrue(seen?.headers?.keys?.none { it.equals("Authorization", ignoreCase = true) } == true)
        assertEquals("123456", parseJson(seen!!.body).get("code").asString)
    }

    @Test
    fun `authenticated requests receive bearer from session store`() {
        val store = FakeAuthStore(storedSession())
        client = ApiClient(baseUrl = "http://127.0.0.1:${server.port}", authStore = store)
        var seen: Request? = null
        route("/api/auth/session") { req, reply ->
            seen = req
            reply(200, """{"session_id":"session-1","expires_at":"2030-01-01T00:00:00.000Z","user":{"user_id":"user-1","username":"owner","role":"owner"}}""")
        }

        val result = runBlocking { client.getAuthSession() }
        assertTrue(result.isSuccess)
        assertEquals("Bearer ${storedSession().accessToken}", seen?.headers?.entries?.firstOrNull {
            it.key.equals("Authorization", ignoreCase = true)
        }?.value)
    }

    @Test
    fun `unauthorized response clears the local session`() {
        val store = FakeAuthStore(storedSession())
        client = ApiClient(baseUrl = "http://127.0.0.1:${server.port}", authStore = store)
        route("/api/auth/session") { _, reply -> reply(401, """{"error":"expired"}""") }

        val result = runBlocking { client.getAuthSession() }
        assertTrue(result.isFailure)
        assertEquals(1, store.clearCount)
        assertEquals(null, store.session)
    }

    @Test
    fun `unauthorized response without readable bearer preserves local session`() {
        val store = FakeAuthStore(storedSession()).apply { failNextRead = true }
        client = ApiClient(baseUrl = "http://127.0.0.1:${server.port}", authStore = store)
        var hadAuthorization = true
        route("/api/auth/session") { request, reply ->
            hadAuthorization = request.headers.keys.any { it.equals("Authorization", ignoreCase = true) }
            reply(401, """{"error":"missing_bearer"}""")
        }

        assertTrue(runBlocking { client.getAuthSession() }.isFailure)
        assertFalse(hadAuthorization)
        assertEquals(0, store.clearCount)
        assertEquals(storedSession(), store.session)
        assertTrue(store.read()?.accessToken?.isNotBlank() == true)
    }

    @Test
    fun `stale unauthorized response does not clear a newer session`() {
        val store = FakeAuthStore(storedSession())
        client = ApiClient(baseUrl = "http://127.0.0.1:${server.port}", authStore = store)
        val newerSession = storedSession().copy(accessToken = "token_${"z".repeat(24)}")
        route("/api/auth/session") { _, reply ->
            store.save(newerSession)
            reply(401, """{"error":"expired"}""")
        }

        assertTrue(runBlocking { client.getAuthSession() }.isFailure)
        assertEquals(0, store.clearCount)
        assertEquals(newerSession, store.session)
    }

    @Test
    fun `logout clears local session when server is unavailable`() {
        val store = FakeAuthStore(storedSession())
        client = ApiClient(baseUrl = "http://127.0.0.1:${server.port}", authStore = store)
        route("/api/auth/logout") { _, reply -> reply(503, """{"error":"offline"}""") }

        val result = runBlocking { client.logout() }
        assertTrue(result.isFailure)
        assertEquals(null, store.session)
        assertTrue(store.clearCount >= 1)
    }

    // ---- AND-CTRL-05 前置：管理摘要 / 审计 / 二次确认 ----

    @Test
    fun `manage summary is authenticated and projects manager matrix`() {
        val store = FakeAuthStore(storedSession())
        client = ApiClient(baseUrl = "http://127.0.0.1:${server.port}", authStore = store)
        var seen: Request? = null
        route("/api/auth/manage/summary") { req, reply ->
            seen = req
            reply(
                200,
                """{"role":"owner","policy_version":"and-ctrl-05-v1","audit_available":true,"confirm_ttl_seconds":120,
                    "actions":{"user_manage":{"allowed":true,"confirm_required":true,"audited":true,"description":"成员管理"},
                               "tailnet_bind":{"allowed":true,"confirm_required":true,"audited":true,"description":"绑定撤销"},
                               "review_admin":{"allowed":true,"confirm_required":false,"audited":false,"description":"审批域"}},
                    "counts":{"users":3,"bindings":2},"review_admin_auth_pending":true}""",
            )
        }

        val result = runBlocking { client.fetchManageSummary() }
        assertTrue(result.isSuccess)
        assertEquals("GET", seen?.method)
        assertEquals("owner", result.getOrNull()?.role)
        assertEquals(true, result.getOrNull()?.actions?.get("user_manage")?.allowed)
        assertEquals(true, result.getOrNull()?.actions?.get("user_manage")?.confirmRequired)
        assertEquals(false, result.getOrNull()?.actions?.get("review_admin")?.confirmRequired)
        assertEquals(true, result.getOrNull()?.reviewAdminAuthPending)
        // 管理端点必须带 Bearer
        assertNotNull(seen?.headers?.entries?.firstOrNull { it.key.equals("Authorization", ignoreCase = true) })
    }

    @Test
    fun `manage audit is bounded client side and parses redacted events`() {
        val store = FakeAuthStore(storedSession())
        client = ApiClient(baseUrl = "http://127.0.0.1:${server.port}", authStore = store)
        var seen: Request? = null
        val auditResponseJson = """{"events":[
            {"event_id":"e1","event_type":"user_revoked","outcome":"success","reason_code":null,
             "actor_user_id":"user-1","user_id":"user-2","subject_id":"user-2","created_at":"2026-08-23T00:00:00.000Z"},
            {"event_id":"e2","event_type":"tailscale_binding_revoked","outcome":"success","reason_code":null,
             "actor_user_id":"user-1","user_id":"user-2","subject_id":"binding-1","created_at":"2026-08-23T00:01:00.000Z"}]}"""
        route("/api/auth/manage/audit?limit=5") { req, reply ->
            seen = req
            reply(
                200,
                auditResponseJson,
            )
        }

        val result = runBlocking { client.fetchManageAudit(limit = 5) }
        assertTrue(result.isSuccess)
        assertEquals("/api/auth/manage/audit?limit=5", seen?.path)
        val events = result.getOrNull()?.events.orEmpty()
        assertEquals(2, events.size)
        assertEquals("user_revoked", events[0].eventType)
        assertEquals("binding-1", events[1].subjectId)
        // 脱敏契约：原始响应体不得含 details/token 类键（data class 会静默丢弃，
        // 必须对原始 JSON 断言才能防回归）
        val raw = JsonParser.parseString(auditResponseJson).asJsonObject.getAsJsonArray("events")
        raw.forEach { element ->
            val event = element.asJsonObject
            assertFalse("脱敏契约：审计事件不得含 details", event.has("details"))
            assertFalse("脱敏契约：审计事件不得含 token", event.has("token"))
        }
    }

    @Test
    fun `manage confirm issues token then revoke carries it in header`() {
        val store = FakeAuthStore(storedSession())
        client = ApiClient(baseUrl = "http://127.0.0.1:${server.port}", authStore = store)
        val issued = mutableListOf<Pair<String, String>>()
        route("/api/auth/manage/confirm") { req, reply ->
            val body = parseJson(req.body)
            issued += body.get("action").asString to body.get("target_id").asString
            reply(200, """{"confirm_token":"ct_1234567890abcdef","expires_at":"2030-01-01T00:00:00.000Z","action":"user_manage","target_id":"user-9"}""")
        }
        var revokeSeen: Request? = null
        route("/api/auth/users/user-9") { req, reply ->
            revokeSeen = req
            reply(200, """{"status":"revoked","user":{"user_id":"user-9","username":"member"}}""")
        }

        val confirmation = runBlocking { client.requestManageConfirm("user_manage", "user-9") }
        assertTrue(confirmation.isSuccess)
        assertEquals("user_manage" to "user-9", issued.firstOrNull())
        val token = confirmation.getOrNull()?.confirmToken.orEmpty()
        assertFalse(token.isBlank())

        val revoke = runBlocking { client.revokeUser("user-9", expectedVersion = 1, confirmToken = token) }
        assertTrue(revoke.isSuccess)
        assertEquals("DELETE", revokeSeen?.method)
        assertEquals("ct_1234567890abcdef", revokeSeen?.headers?.entries?.firstOrNull {
            it.key.equals("X-QLH-Confirm-Token", ignoreCase = true)
        }?.value)
        // revoke 请求体必须携带 expected_version
        assertEquals(1, parseJson(revokeSeen!!.body).get("expected_version").asInt)
    }

    @Test
    fun `managed users and per-user bindings stay authenticated and bounded`() {
        val store = FakeAuthStore(storedSession())
        client = ApiClient(baseUrl = "http://127.0.0.1:${server.port}", authStore = store)
        var usersSeen: Request? = null
        var bindingsSeen: Request? = null
        route("/api/users") { req, reply ->
            usersSeen = req
            reply(200, """{"users":[{"user_id":"user-2","username":"member","role":"member","status":"active","aggregate_version":3,"secret":"drop"}]}""")
        }
        route("/api/auth/users/user-2/tailscale") { req, reply ->
            bindingsSeen = req
            reply(200, """{"bindings":[{"binding_id":"binding-2","user_id":"user-2","tailnet_id":"tailnet-a","tailscale_user_id":"ts-user","node_id":"node-a","state":"active","credential_ref":"drop"}]}""")
        }

        val users = runBlocking { client.fetchManagedUsers() }
        val bindings = runBlocking { client.fetchUserTailscaleBindings("user-2") }
        assertTrue(users.isSuccess)
        assertTrue(bindings.isSuccess)
        assertEquals("member", users.getOrNull()?.users?.single()?.username)
        assertEquals(3, users.getOrNull()?.users?.single()?.aggregateVersion)
        assertEquals("binding-2", bindings.getOrNull()?.bindings?.single()?.bindingId)
        assertEquals("GET", usersSeen?.method)
        assertEquals("GET", bindingsSeen?.method)
        assertNotNull(usersSeen?.headers?.entries?.firstOrNull { it.key.equals("Authorization", ignoreCase = true) })
        assertNotNull(bindingsSeen?.headers?.entries?.firstOrNull { it.key.equals("Authorization", ignoreCase = true) })
    }

    @Test
    fun `manage confirm JSON encodes target and rejects blank target locally`() {
        val store = FakeAuthStore(storedSession())
        client = ApiClient(baseUrl = "http://127.0.0.1:${server.port}", authStore = store)
        var seen: Request? = null
        route("/api/auth/manage/confirm") { req, reply ->
            seen = req
            reply(200, """{"confirm_token":"ct_safe_123456789","expires_at":"2030-01-01T00:00:00.000Z","action":"user_manage","target_id":"user-\\\"9"}""")
        }
        val encoded = runBlocking { client.requestManageConfirm("user_manage", "user-\"9") }
        assertTrue(encoded.isSuccess)
        assertEquals("user-\"9", parseJson(seen!!.body).get("target_id").asString)

        var hitServer = false
        route("/api/auth/manage/confirm") { _, reply -> hitServer = true; reply(200, "{}") }
        val blank = runBlocking { client.requestManageConfirm("user_manage", "") }
        assertTrue(blank.isFailure)
        assertTrue(!hitServer)
    }

    @Test
    fun `revoking own tailscale binding may omit confirm token while cross-user revoke fails without it`() {
        val store = FakeAuthStore(storedSession())
        client = ApiClient(baseUrl = "http://127.0.0.1:${server.port}", authStore = store)
        var seen: Request? = null
        route("/api/auth/tailscale/bindings/binding-5/revoke") { req, reply ->
            seen = req
            reply(200, """{"status":"revoked","binding":{"binding_id":"binding-5","state":"revoked"}}""")
        }
        // 自撤销（无 token 头）
        val own = runBlocking { client.revokeTailscaleBinding("binding-5") }
        assertTrue(own.isSuccess)
        assertEquals("POST", seen?.method)
        assertTrue(seen?.headers?.keys?.none { it.equals("X-QLH-Confirm-Token", ignoreCase = true) } == true)
        // 跨用户管理（带头）
        val crossSeen = arrayOf<Request?>(null)
        route("/api/auth/tailscale/bindings/binding-6/revoke") { req, reply ->
            crossSeen[0] = req
            reply(200, """{"status":"revoked","binding":{"binding_id":"binding-6","state":"revoked"}}""")
        }
        val cross = runBlocking { client.revokeTailscaleBinding("binding-6", confirmToken = "ct_cross") }
        assertTrue(cross.isSuccess)
        assertEquals("ct_cross", crossSeen[0]?.headers?.entries?.firstOrNull {
            it.key.equals("X-QLH-Confirm-Token", ignoreCase = true)
        }?.value)
    }

    @Test
    fun `revoking user without confirm token fails without hitting network`() {
        val store = FakeAuthStore(storedSession())
        client = ApiClient(baseUrl = "http://127.0.0.1:${server.port}", authStore = store)
        var hitServer = false
        route("/api/auth/users/user-9") { _, reply -> hitServer = true; reply(200, "{}") }

        val result = runBlocking { client.revokeUser("user-9", expectedVersion = 1, confirmToken = "") }
        assertTrue(result.isFailure)
        assertFalse(hitServer)
    }

    @Test
    fun `manage confirm rejects invalid token response`() {
        val store = FakeAuthStore(storedSession())
        client = ApiClient(baseUrl = "http://127.0.0.1:${server.port}", authStore = store)
        route("/api/auth/manage/confirm") { _, reply ->
            reply(200, """{"expires_at":"2030-01-01T00:00:00.000Z","action":"user_manage"}""")
        }

        val result = runBlocking { client.requestManageConfirm("user_manage", "user-9") }
        assertTrue(result.isFailure)
    }

    @Test
    fun `manage audit clamps limit to server bound`() {
        val store = FakeAuthStore(storedSession())
        client = ApiClient(baseUrl = "http://127.0.0.1:${server.port}", authStore = store)
        var seen: Request? = null
        route("/api/auth/manage/audit?limit=200") { req, reply ->
            seen = req
            reply(200, """{"events":[]}""")
        }
        val result = runBlocking { client.fetchManageAudit(limit = 500) }
        assertTrue(result.isSuccess)
        assertEquals("/api/auth/manage/audit?limit=200", seen?.path)
    }

    private fun productChatRepository(
        saved: MutableList<com.qlh.inference.data.MessageEntity>,
        localEngine: () -> com.qlh.inference.service.InferenceService?,
    ): ChatRepository {
        val sessions = object : com.qlh.inference.data.SessionDao {
            override fun getAllSessions(): kotlinx.coroutines.flow.Flow<List<com.qlh.inference.data.SessionEntity>> =
                kotlinx.coroutines.flow.flowOf(emptyList())
            override suspend fun getById(sessionId: Long): com.qlh.inference.data.SessionEntity? = error("unexpected session lookup")
            override suspend fun insert(session: com.qlh.inference.data.SessionEntity): Long = error("unexpected session insert")
            override suspend fun update(session: com.qlh.inference.data.SessionEntity) = error("unexpected session update")
            override suspend fun updateTitle(sessionId: Long, title: String, updatedAt: Long) = error("unexpected title update")
            override suspend fun updateMessageCount(sessionId: Long, count: Int, updatedAt: Long) = Unit
            override suspend fun deleteById(sessionId: Long) = error("unexpected session delete")
        }
        val messages = object : com.qlh.inference.data.MessageDao {
            override fun getMessagesBySession(sessionId: Long): kotlinx.coroutines.flow.Flow<List<com.qlh.inference.data.MessageEntity>> =
                kotlinx.coroutines.flow.flowOf(saved.filter { it.sessionId == sessionId })
            override suspend fun getRecentMessages(sessionId: Long, limit: Int): List<com.qlh.inference.data.MessageEntity> = error("unexpected recent messages")
            override suspend fun insert(message: com.qlh.inference.data.MessageEntity): Long {
                saved.add(message)
                return saved.size.toLong()
            }
            override suspend fun deleteBySession(sessionId: Long) = error("unexpected message delete")
            override suspend fun deleteById(messageId: Long) = error("unexpected message delete")
            override suspend fun getCount(sessionId: Long): Int = saved.count { it.sessionId == sessionId }
        }
        return ChatRepository(sessions, messages, { client }, localEngine)
    }

    @Test
    fun `product chat repository requires distribution and preserves response model identity`() {
        var seen: Request? = null
        route("/api/chat") { req, reply ->
            seen = req
            reply(200, """{"content":"2","metrics":{"distributed_used":true,"fallback":false,"model_operation_id":"supply_response","model_config_id":"plan_response","model_generation":7}}""")
        }
        val saved = mutableListOf<com.qlh.inference.data.MessageEntity>()
        val repository = productChatRepository(saved) { error("distributed chat cannot use local engine") }
        val result = runBlocking { repository.sendMessage(1, "What is 1+1?", skipUserSave = true) }
        assertTrue(result.isSuccess)
        assertEquals("2", result.getOrNull())
        val request = parseJson(seen!!.body)
        assertEquals("distributed_required", request.get("routing_preference").asString)
        assertFalse(request.get("enable_thinking").asBoolean)
        assertFalse(request.get("show_thinking").asBoolean)
        assertEquals(1, saved.size)
        val metrics = parseJson(saved.single().metrics!!)
        assertEquals("supply_response", metrics.get("model_operation_id").asString)
        assertEquals("plan_response", metrics.get("model_config_id").asString)
        assertEquals(7, metrics.get("model_generation").asLong)
    }

    @Test
    fun `chat preserves native generation longs and fractional metrics`() = runBlocking {
        for (generation in listOf(1791912357956122300L, 1791909775815263900L)) {
            route("/api/chat") { req, reply ->
                assertEquals("POST", req.method)
                reply(200, """{"content":"2","metrics":{"model_generation":$generation,"tokens_per_second":12.75,"completion_tokens":32}}""")
            }
            val response = client.chat(ChatRequest("1+1=?", routingPreference = "distributed_required")).getOrThrow()
            val metrics = response.metrics!!
            assertTrue("native generation must stay a Long", metrics["model_generation"] is Long)
            assertEquals(generation, metrics["model_generation"])
            assertTrue("fractional TPS must stay a Double", metrics["tokens_per_second"] is Double)
            assertEquals(12.75, metrics["tokens_per_second"] as Double, 0.0)
            assertEquals(32L, metrics["completion_tokens"])
        }
    }

    @Test
    fun `product chat repository persists exact native generation decimal and fractional TPS`() {
        val saved = mutableListOf<com.qlh.inference.data.MessageEntity>()
        val repository = productChatRepository(saved) { error("distributed chat cannot use local engine") }
        for (generation in listOf(1791912357956122300L, 1791909775815263900L)) {
            route("/api/chat") { req, reply ->
                assertEquals("distributed_required", parseJson(req.body)["routing_preference"].asString)
                reply(200, """{"content":"2","metrics":{"distributed_used":true,"fallback":false,"model_operation_id":"supply_response","model_config_id":"plan_response","model_generation":$generation,"tokens_per_second":12.75}}""")
            }
            val result = runBlocking { repository.sendMessage(1, "What is 1+1?", skipUserSave = true) }
            assertEquals("2", result.getOrThrow())
            val assistant = saved.last()
            assertEquals("assistant", assistant.role)
            val metrics = parseJson(assistant.metrics!!)
            val persistedGeneration = metrics["model_generation"].asJsonPrimitive
            assertTrue("generation must remain a JSON number", persistedGeneration.isNumber)
            assertEquals(generation.toString(), persistedGeneration.toString())
            assertEquals(generation, persistedGeneration.asLong)
            assertEquals("supply_response", metrics["model_operation_id"].asString)
            assertEquals("plan_response", metrics["model_config_id"].asString)
            assertTrue(metrics["tokens_per_second"].asJsonPrimitive.isNumber)
            assertEquals(12.75, metrics["tokens_per_second"].asDouble, 0.0)
        }
        assertEquals(2, saved.size)
    }

    @Test
    fun `chat respects an explicitly injected Gson number strategy`() = runBlocking {
        val generation = 1791912357956122300L
        client = ApiClient(
            baseUrl = "http://127.0.0.1:${server.port}",
            gson = GsonBuilder().setObjectToNumberStrategy(ToNumberPolicy.BIG_DECIMAL).create(),
        )
        route("/api/chat") { _, reply ->
            reply(200, """{"content":"2","metrics":{"model_generation":$generation,"tokens_per_second":12.75}}""")
        }
        val metrics = client.chat(ChatRequest("1+1=?")).getOrThrow().metrics!!
        assertEquals(java.math.BigDecimal(generation.toString()), metrics["model_generation"])
        assertEquals(java.math.BigDecimal("12.75"), metrics["tokens_per_second"])
    }

    @Test
    fun `product chat repository propagates distribution refusal without local fallback`() {
        var seen: Request? = null
        route("/api/chat") { req, reply ->
            seen = req
            reply(409, """{"detail":{"code":"distributed_unavailable","message":"No ready stage plan"}}""")
        }
        val saved = mutableListOf<com.qlh.inference.data.MessageEntity>()
        val repository = productChatRepository(saved) { error("refused distributed chat cannot load local model") }
        val result = runBlocking { repository.sendMessage(1, "What is 1+1?", skipUserSave = true) }
        assertTrue(result.isFailure)
        assertEquals("distributed_required", parseJson(seen!!.body).get("routing_preference").asString)
        assertTrue(saved.isEmpty())
    }

    // ---- chat ----

    @Test
    fun `chat posts json body with message and parses response`() {
        var seen: Request? = null
        route("/api/chat") { req, reply ->
            seen = req
            reply(200, """{"content":"hello back"}""")
        }

        val result = runBlocking { client.chat(ChatRequest(message = "hello")) }
        assertTrue(result.isSuccess)
        assertEquals("hello back", result.getOrNull()?.content)
        assertEquals("POST", seen?.method)
        assertEquals("/api/chat", seen?.path)
        val json = parseJson(seen!!.body)
        assertEquals("hello", json.get("message").asString)
        assertEquals(1024, json.get("max_new_tokens").asInt)
        assertEquals(0.7f, json.get("temperature").asFloat, 1e-6f)
        assertEquals(false, json.get("show_thinking").asBoolean)
        assertEquals(false, json.get("enable_thinking").asBoolean)
        assertEquals("distributed_preferred", json.get("routing_preference").asString)
        assertTrue("默认请求不应带 allow_external", !json.has("allow_external"))
    }

    @Test
    fun `chat serializes remote image data urls using PC field name`() {
        var seen: Request? = null
        route("/api/chat") { req, reply ->
            seen = req
            reply(200, """{"content":"vision reply"}""")
        }

        val image = "data:image/png;base64,iVBORw0KGgo="
        val result = runBlocking {
            client.chat(
                ChatRequest(
                    message = "describe",
                    imageDataUrls = listOf(image),
                    allowExternal = true,
                    preferExternal = true,
                )
            )
        }

        assertTrue(result.isSuccess)
        // Gson 转义 '=' 为 \u003d，parseJson 还原后完整比较数组
        val json = parseJson(seen!!.body)
        val images = json.getAsJsonArray("image_data_urls")
        assertEquals(1, images.size())
        assertEquals(image, images[0].asString)
        assertEquals(true, json.get("allow_external").asBoolean)
        assertEquals(true, json.get("prefer_external").asBoolean)
    }

    @Test
    fun `chat non-2xx returns failure not exception`() {
        route("/api/chat") { _, reply -> reply(500, """{"error":"boom"}""") }
        val result = runBlocking { client.chat(ChatRequest(message = "x")) }
        assertTrue(result.isFailure)
        assertTrue(result.exceptionOrNull() is ApiClientHttpException)
        assertTrue(result.exceptionOrNull()!!.message!!.contains("500"))
    }

    // ---- sessions ----

    @Test
    fun `getSessions parses plain array format`() {
        route("/api/sessions") { _, reply ->
            reply(200, """[{"id":"s1","title":"T1","message_count":2}]""")
        }
        val sessions = runBlocking { client.getSessions() }.getOrNull()!!
        assertEquals(1, sessions.size)
        assertEquals("s1", sessions[0].id)
        assertEquals("T1", sessions[0].title)
        assertEquals(2, sessions[0].messageCount)
    }

    @Test
    fun `getSessions parses wrapped object format`() {
        route("/api/sessions") { _, reply -> reply(200, """{"sessions":[{"id":"w1"}]}""") }
        val sessions = runBlocking { client.getSessions() }.getOrNull()!!
        assertEquals(1, sessions.size)
        assertEquals("w1", sessions[0].id)
    }

    @Test
    fun `createSession posts title and parses session`() {
        var seen: Request? = null
        route("/api/sessions") { req, reply ->
            seen = req
            reply(200, """{"id":"new-1","title":"我的对话"}""")
        }

        val result = runBlocking { client.createSession(title = "我的对话") }
        assertTrue("createSession failed: ${result.exceptionOrNull()}", result.isSuccess)
        val session = result.getOrNull()!!
        assertEquals("new-1", session.id)
        assertEquals("我的对话", session.title)
        assertEquals("POST", seen?.method)
        assertTrue(seen!!.body.contains("\"title\":\"我的对话\""))
    }

    @Test
    fun `deleteSession uses DELETE and treats 404 as success`() {
        var seen: String? = null
        route("/api/sessions/del-1") { req, reply ->
            seen = req.method
            reply(404, "")
        }
        val result = runBlocking { client.deleteSession("del-1") }
        assertTrue(result.isSuccess)
        assertEquals("DELETE", seen)
    }

    @Test
    fun `deleteSession other errors return failure`() {
        route("/api/sessions/del-2") { _, reply -> reply(500, "") }
        val result = runBlocking { client.deleteSession("del-2") }
        assertTrue(result.isFailure)
    }

    @Test
    fun `getSessionMessages parses array and rejects non-2xx`() {
        route("/api/sessions/s1/messages") { _, reply ->
            reply(200, """[{"role":"user","content":"hi"}]""")
        }
        val messages = runBlocking { client.getSessionMessages("s1") }.getOrNull()!!
        assertEquals(1, messages.size)
        assertEquals("user", messages[0].role)

        route("/api/sessions/s1/messages") { _, reply -> reply(403, "") }
        assertTrue(runBlocking { client.getSessionMessages("s1") }.isFailure)
    }

    // ---- cluster / connection ----

    @Test
    fun `getClusterStatus parses current node map and task projection`() {
        route("/api/cluster/status") { _, reply ->
            reply(200, """
                {
                  "running": true,
                  "run_mode": "distributed",
                  "nodes_ready": true,
                  "nodes": {
                    "master": {"node_id":"master","role":"master","state":"online","hostname":"main","network_type":"tailscale","task_count":2,"is_available":true},
                    "worker-1": {"node_id":"worker-1","role":"client","node_type":"android","state":"busy","hostname":"phone","network_type":"wifi","error_count":1}
                  },
                  "current_task": {"task_id":"task-7","state":"running","elapsed":12.8}
                }
            """.trimIndent())
        }
        val status = runBlocking { client.getClusterStatus() }.getOrNull()!!
        assertTrue(status.running)
        assertEquals("distributed", status.runMode)
        assertTrue(status.nodesReady)
        assertEquals("main", status.nodes["master"]?.hostname)
        assertEquals("busy", status.nodes["worker-1"]?.state)
        assertEquals(1, status.nodes["worker-1"]?.errorCount)
        assertEquals("task-7", status.currentTask?.taskId)
        assertEquals(12.8, status.currentTask?.elapsed ?: 0.0, 0.001)
    }

    @Test
    fun `getClusterStatus parses pipeline capacity projection and node participation`() {
        route("/api/cluster/status") { _, reply ->
            reply(200, """
                {
                  "running": true,
                  "run_mode": "distributed",
                  "nodes_ready": true,
                  "nodes": {
                    "master": {"node_id":"master","role":"master","state":"online","is_available":true,"pipeline_participating":true,"pipeline_exclusion_reason":""},
                    "android-1": {"node_id":"android-1","role":"client","node_type":"android","state":"online","is_available":true,"pipeline_participating":false,"pipeline_exclusion_reason":"capacity_plan_control_only"}
                  },
                  "pipeline_capacity": {
                    "status":"rejected",
                    "admitted":false,
                    "reason_code":"pipeline_distributed_workers_unavailable",
                    "participating_node_count":0,
                    "control_only_nodes":["android-1"],
                    "worker_count":0,
                    "prepared_node_count":0,
                    "ready_node_count":0,
                    "require_distributed":true
                  }
                }
            """.trimIndent())
        }

        val status = runBlocking { client.getClusterStatus() }.getOrNull()!!

        assertEquals(false, status.pipelineCapacity?.admitted)
        assertEquals(
            "pipeline_distributed_workers_unavailable",
            status.pipelineCapacity?.reasonCode,
        )
        assertEquals(listOf("android-1"), status.pipelineCapacity?.controlOnlyNodes)
        assertEquals(true, status.pipelineCapacity?.requireDistributed)
        assertEquals(true, status.nodes["master"]?.pipelineParticipating)
        assertEquals(false, status.nodes["android-1"]?.pipelineParticipating)
        assertEquals(
            "capacity_plan_control_only",
            status.nodes["android-1"]?.pipelineExclusionReason,
        )
    }

    @Test
    fun `testConnection returns success boolean on any http response`() {
        route("/api/cluster/status") { _, reply -> reply(200, "{}") }
        assertEquals(true, runBlocking { client.testConnection() }.getOrNull())

        route("/api/cluster/status") { _, reply -> reply(503, "") }
        assertEquals(false, runBlocking { client.testConnection() }.getOrNull())
    }

    @Test
    fun `registerAndroidNode posts to android register endpoint`() {
        var seen: Request? = null
        route("/api/cluster/android/register") { req, reply ->
            seen = req
            reply(200, """{"node_id":"n1","status":"registered"}""")
        }
        val response = runBlocking { client.registerAndroidNode(
            RegisterNodeRequest(nodeId = "n1", hostname = "node-1", address = "100.1.2.3")
        ) }.getOrNull()!!
        assertEquals("n1", response.nodeId)
        assertEquals("POST", seen?.method)
        assertEquals("/api/cluster/android/register", seen?.path)
        assertTrue("body was: ${seen!!.body}", seen!!.body.contains("\"node_id\":\"n1\""))
    }

    @Test
    fun `heartbeatAndroidNode posts only the current lease`() {
        var seen: Request? = null
        route("/api/cluster/android/heartbeat") { req, reply ->
            seen = req
            reply(200, """{"status":"heartbeat","node_id":"n1","presence_generation":4,"presence_lease_id":"lease-4","lease_expires_at_ms":1700000120000,"server_time_ms":1700000000000,"heartbeat_interval_seconds":45}""")
        }
        val response = runBlocking {
            client.heartbeatAndroidNode(AndroidPresenceHeartbeatRequest("n1", 4L, "lease-4"))
        }.getOrNull()!!
        assertEquals("heartbeat", response.status)
        assertEquals(4L, response.presenceGeneration)
        assertEquals("POST", seen?.method)
        assertEquals("/api/cluster/android/heartbeat", seen?.path)
        assertTrue(seen!!.body.contains("\"presence_lease_id\":\"lease-4\""))
        assertTrue(!seen!!.body.contains("device_info"))
    }

    @Test
    fun `firstConnectBootstrap posts to bootstrap endpoint`() {
        var seen: Request? = null
        val clusterSecret = "bootstrap-secret-never-sent-in-plaintext"
        route("/api/bootstrap/first-connect") { req, reply ->
            seen = req
            reply(200, bootstrapResponseFor(req, clusterSecret))
        }
        val result = runBlocking { client.firstConnectBootstrap(
            BootstrapRequest(nodeId = "n1", hostname = "100.1.2.3")
        ) }.getOrNull()!!
        assertEquals("c1", result.response.cluster.clusterId)
        assertEquals(4, result.response.cluster.clusterSecretEpoch)
        assertEquals(clusterSecret, result.clusterSecret)
        assertEquals("/api/bootstrap/first-connect", seen?.path)
        val requestBody = parseJson(seen!!.body)
        assertEquals(18, Base64.getUrlDecoder().decode(
            requestBody.get("credential_request_nonce").asString,
        ).size)
        assertTrue(requestBody.get("credential_requested_at").asLong > 0L)
        assertFalse(requestBody.get("credential_public_key").asString.contains('='))
        assertFalse(seen!!.body.contains(clusterSecret))
    }

    @Test
    fun `firstConnectBootstrap failure never exposes response body`() {
        val leakedMaterial = "server-body-secret-that-must-not-reach-exception"
        route("/api/bootstrap/first-connect") { _, reply ->
            reply(403, """{"cluster_secret":"$leakedMaterial"}""")
        }

        val error = runBlocking {
            client.firstConnectBootstrap(BootstrapRequest(nodeId = "n1"))
        }.exceptionOrNull()!!
        assertEquals("Bootstrap failed: HTTP 403", error.message)
        assertFalse(error.message.orEmpty().contains(leakedMaterial))
    }

    @Test
    fun firstConnectBootstrap_rejectsPlaintextClusterSecret() {
        val leakedMaterial = "legacy-plaintext-cluster-secret"
        route("/api/bootstrap/first-connect") { _, reply ->
            reply(
                200,
                """{"status":"ok","cluster":{"cluster_id":"c1","cluster_secret":"$leakedMaterial"},"node":{"node_id":"n1"}}""",
            )
        }

        val error = runBlocking {
            client.firstConnectBootstrap(BootstrapRequest(nodeId = "n1"))
        }.exceptionOrNull() as com.qlh.inference.security.BootstrapCredentialException
        assertEquals("plaintext_secret_rejected", error.errorCode)
        assertFalse(error.message.orEmpty().contains(leakedMaterial))
    }

    @Test
    fun `gguf catalog parses verified models and drops incomplete entries`() {
        route("/api/models/gguf") { req, reply ->
            assertEquals("GET", req.method)
            reply(
                200,
                """{"models":[{"filename":"qwen.gguf","size_bytes":1024,"sha256":"${"a".repeat(64)}","download_url":"/api/models/download/qwen.gguf"},{"filename":"broken.gguf","size_bytes":0,"sha256":""}],"exists":true,"count":2}""",
            )
        }
        val models = runBlocking { client.getGgufModels() }.getOrThrow()
        assertEquals(1, models.size)
        assertEquals("qwen.gguf", models[0].filename)
        assertEquals(1024L, models[0].sizeBytes)
    }

    @Test
    fun `model fleet reads path free registry runtime assets and verified gguf`() {
        route("/api/models/current") { req, reply ->
            assertEquals("GET", req.method)
            reply(
                200,
                """{"loaded":true,"pipeline_prepared":false,"model_id":"qwen-1_8b","model_name":"Qwen 1.8B","engine":"llama_cpp","quant_type":"Q4_K_M","model_path":"C:/must-not-be-projected"}""",
            )
        }
        route("/api/models") { req, reply ->
            assertEquals("GET", req.method)
            reply(
                200,
                """{"active_model_id":"qwen-1_8b","models":[{"model_id":"qwen-1_8b","name":"Qwen 1.8B","model_type":"both","is_available":true,"available_formats":["gguf","safetensors"],"has_safetensors":true,"has_gguf":true,"model_path":"C:/must-not-be-projected"},{"model_id":"missing","name":"Missing","is_available":false}]}""",
            )
        }
        route("/api/models/local-assets") { req, reply ->
            assertEquals("GET", req.method)
            reply(
                200,
                """{"assets":[{"model_id":"qwen-1_8b","name":"Qwen 1.8B","model_type":"both","available_formats":["gguf","safetensors"],"total_bytes":2048,"integrity":"manifest_verified","model_path":"C:/must-not-be-projected"}],"summary":{"total":1,"total_bytes":2048}}""",
            )
        }
        route("/api/models/gguf") { req, reply ->
            assertEquals("GET", req.method)
            reply(
                200,
                """{"models":[{"filename":"qwen.gguf","size_bytes":1024,"sha256":"${"b".repeat(64)}","download_url":"/api/models/download/qwen.gguf"}]}""",
            )
        }

        val fleet = runBlocking { client.getModelFleetData() }.getOrThrow()
        assertTrue(fleet.current.loaded)
        assertEquals("qwen-1_8b", fleet.current.modelId)
        assertEquals(2, fleet.registry.models.size)
        assertEquals(1, fleet.localAssets.summary.total)
        assertEquals(1, fleet.verifiedGguf.size)
        assertEquals("qwen.gguf", fleet.verifiedGguf.single().filename)
    }

    @Test
    fun `current model endpoint keeps prepared pipeline separate from full model`() {
        route("/api/models/current") { request, reply ->
            assertEquals("GET", request.method)
            reply(200, """{"loaded":false,"pipeline_prepared":true,"segment_materialized":true,"model_id":"registry-alias","descriptor":{"model_id":"registry-alias","source_model_id":"qwen-1_8b","source_model_sha256":"${"a".repeat(64)}"}}""")
        }

        val prepared = runBlocking { client.getCurrentModel() }.getOrThrow()
        assertFalse(prepared.loaded)
        assertTrue(prepared.pipelinePrepared)
        assertEquals("registry-alias", prepared.modelId)
        assertEquals("qwen-1_8b", prepared.descriptor?.sourceModelId)
        assertEquals("registry-alias", prepared.descriptor?.modelId)

        route("/api/models/current") { _, reply ->
            reply(200, """{"loaded":true,"model_id":"qwen-1_8b"}""")
        }

        val full = runBlocking { client.getCurrentModel() }.getOrThrow()
        assertTrue(full.loaded)
        assertFalse(full.pipelinePrepared)
    }

    @Test
    fun `audit requests bounded server summaries and excludes content fields`() {
        route("/api/requests?limit=8") { req, reply ->
            assertEquals("GET", req.method)
            reply(
                200,
                """{"schema_version":2,"revision":1,"identity":{"boot_id":"test"},"observed_at":100,"validity":"valid","requests":[{"request_id":"req-1","deployment_id":"deployment","phase":"finished","execution_state":"stopped","terminal":"completed","outcome":"completed"}],"prompt":"must-not-be-present"}""",
            )
        }
        route("/api/cluster/review/tickets?limit=8&summary=1") { req, reply ->
            assertEquals("GET", req.method)
            reply(
                200,
                """{"count":1,"tickets":[{"ticket_id":"review-1","status":"pending","created_at":10,"target_node_id":"node-2","score":0,"vote_count":0,"transfer_reason":"must-not-be-present","votes":[{"comment":"secret"}]}]}""",
            )
        }

        val audit = runBlocking { client.getAuditData() }.getOrThrow()
        assertEquals("req-1", audit.requests.requests.single().requestId)
        assertEquals("stopped", audit.requests.requests.single().executionState)
        assertEquals("completed", audit.requests.requests.single().terminal)
        assertEquals("review-1", audit.reviewTickets.tickets.single().ticketId)
        assertEquals("node-2", audit.reviewTickets.tickets.single().targetNodeId)
    }

    @Test
    fun `connection health probe records bounded checks and skips absent auth`() {
        route("/api/cluster/status") { req, reply ->
            assertEquals("GET", req.method)
            reply(200, "{\"status\":\"ok\"}")
        }
        val report = runBlocking { client.probeConnectionHealth("wifi") }.getOrThrow()
        assertEquals("wifi", report.localNetworkType)
        assertEquals(ConnectionHealthState.PASS, report.checks.first { it.id == "cluster_status" }.state)
        assertEquals(ConnectionHealthState.SKIPPED, report.checks.first { it.id == "auth_session" }.state)
    }

    @Test
    fun `manual client diagnostics uses the log endpoint`() {
        route("/api/logs/client-error") { req, reply ->
            assertEquals("POST", req.method)
            assertTrue(req.body.contains("manual Android diagnostic upload"))
            assertTrue(req.body.contains("Bearer [REDACTED]"))
            reply(200, "{\"status\":\"ok\"}")
        }
        val result = runBlocking {
            client.reportClientError(
                ClientErrorReport(
                    message = "manual Android diagnostic upload",
                    source = "manual",
                    stack = "Authorization: Bearer [REDACTED]",
                )
            )
        }
        assertTrue("diagnostic upload failed: ${result.exceptionOrNull()}", result.isSuccess)
    }

    @Test
    fun `network unreachable returns failure`() {
        val unreachable = ApiClient(baseUrl = "http://127.0.0.1:1")
        val result = runBlocking { unreachable.chat(ChatRequest(message = "x")) }
        assertTrue(result.isFailure)
    }

    private fun intentOperationJson(status: String = "supplying", loaded: Boolean = false, id: String = "supply_abc"): String =
        """{"schema_version":2,"revision":7,"identity":{"boot_id":"test"},"observed_at":100,"validity":"valid","runnable":$loaded,"operation_id":"$id","intent":{"model_id":"qwen","engine":"auto","quant_type":"","strategy":"auto","timeout_seconds":3600},"status":"$status","reason_code":null,"reason":null,"config_id":"cfg-1","generation":7,"loaded_ready":$loaded,"nodes":{"phone":{"status":"verifying","bytes_received":3456789123,"total_bytes":4567891234,"reason_code":"SUPPLY_HASH_MISMATCH","reason":"checksum failed","layer_range":[4,8]}},"plan":{"node_count":1},"updated_at":123.5}"""

    @Test
    fun `intent catalog is path free and includes master binding`() = runBlocking {
        route("/api/models/intent/catalog") { request, reply ->
            assertEquals("GET", request.method)
            reply(200, """{"models":[{"model_id":"qwen","name":"Qwen","is_available":true,"available_formats":["gguf"],"path":"C:/private/model.gguf"}],"active_model_id":"qwen","binding":{"status":"connected","is_master":true,"master_host":"100.64.0.1","master_api_port":8000,"reason_code":null}}""")
        }
        val catalog = client.getModelIntentCatalog().getOrThrow()
        assertEquals("connected", catalog.binding.status)
        assertTrue(catalog.binding.isMaster)
        assertEquals("qwen", catalog.activeModelId)
        assertTrue(catalog.models.single().isAvailable)
        assertFalse(Gson().toJson(catalog).contains("private"))
    }

    @Test
    fun `intent current null succeeds but missing or invalid operation fails closed`() = runBlocking {
        route("/api/models/intent/current") { _, reply -> reply(200, "null") }
        val empty = client.getCurrentModelIntent()
        assertTrue(empty.isSuccess)
        assertEquals(null, empty.getOrThrow())
        for (body in listOf("", "{}", "null trailing", intentOperationJson("unknown"))) {
            route("/api/models/intent/current") { _, reply -> reply(200, body) }
            assertTrue("accepted $body", client.getCurrentModelIntent().isFailure)
        }
    }

    @Test
    fun `current failed local cut from an API client preserves its source choice and named failure`() = runBlocking {
        // Regression for the captured intent-before-selection.json from the physical device run.
        val id = "supply_4946a692602545db88e7994e91864adc"
        val captured = parseJson(intentOperationJson(status = "failed", id = id)).apply {
            getAsJsonObject("intent").apply {
                addProperty("model_id", "qwen3-5-9b")
                addProperty("engine", "llama_cpp")
                addProperty("strategy", "local_cut")
                addProperty("timeout_seconds", 900)
                add("node_ids", com.google.gson.JsonArray())
            }
            addProperty("reason_code", "SUPPLY_LOCAL_CUT_UNAVAILABLE")
            addProperty("reason", "Worker lacks the verified source or cutter")
        }
        route("/api/models/intent/current") { _, reply -> reply(200, captured.toString()) }
        val operation = client.getCurrentModelIntent().getOrThrow()!!
        assertEquals(id, operation.operationId)
        assertEquals("qwen3-5-9b", operation.intent.modelId)
        assertEquals("llama_cpp", operation.intent.engine)
        assertEquals("local_cut", operation.intent.strategy)
        assertEquals(900, operation.intent.timeoutSeconds)
        assertEquals("failed", operation.status)
        assertEquals("SUPPLY_LOCAL_CUT_UNAVAILABLE", operation.reasonCode)
        assertEquals("Worker lacks the verified source or cutter", operation.reason)
        assertTrue(operation.finished)
        assertFalse(operation.readyForChat)
    }

    @Test
    fun `current get and cancel accept every API strategy without rewriting the returned operation`() = runBlocking {
        for (strategy in listOf("auto", "local_cut", "distribute")) {
            val body = parseJson(intentOperationJson("failed")).apply {
                getAsJsonObject("intent").addProperty("strategy", strategy)
            }.toString()
            route("/api/models/intent/current") { _, reply -> reply(200, body) }
            route("/api/models/intent/supply_abc") { _, reply -> reply(200, body) }
            for (operation in listOf(client.getCurrentModelIntent().getOrThrow()!!,
                client.getModelIntent("supply_abc").getOrThrow(), client.cancelModelIntent("supply_abc").getOrThrow())) {
                assertEquals(strategy, operation.intent.strategy)
                assertEquals("failed", operation.status)
            }
        }
    }

    @Test
    fun `unknown returned strategies fail closed for current get and cancel`() = runBlocking {
        for (strategy in listOf("", "reuse", "LOCAL_CUT", "future_strategy")) {
            val body = parseJson(intentOperationJson()).apply {
                getAsJsonObject("intent").addProperty("strategy", strategy)
            }.toString()
            route("/api/models/intent/current") { _, reply -> reply(200, body) }
            route("/api/models/intent/supply_abc") { _, reply -> reply(200, body) }
            for (result in listOf(client.getCurrentModelIntent(), client.getModelIntent("supply_abc"),
                client.cancelModelIntent("supply_abc"))) {
                assertTrue("accepted strategy=$strategy", result.isFailure)
                assertEquals("intent_options_invalid", result.exceptionOrNull()!!.message)
            }
        }
    }

    @Test
    fun `UI submissions still reject explicit and unknown strategies before HTTP`() = runBlocking {
        var submissions = 0
        route("/api/models/intent") { request, reply ->
            submissions++
            assertEquals("auto", parseJson(request.body)["strategy"].asString)
            reply(202, intentOperationJson())
        }
        for (strategy in listOf("local_cut", "distribute", "", "reuse", "future_strategy")) {
            val result = client.submitModelIntent(ModelIntentRequest("qwen", strategy = strategy))
            assertTrue("submitted strategy=$strategy", result.isFailure)
            assertEquals("intent_options_invalid", result.exceptionOrNull()!!.message)
        }
        assertEquals(0, submissions)
        assertTrue(client.submitModelIntent(ModelIntentRequest("qwen")).isSuccess)
        assertEquals(1, submissions)
    }

    @Test
    fun `API strategies do not relax model engine or timeout validation`() = runBlocking {
        for (strategy in listOf("auto", "local_cut", "distribute")) {
            for ((field, value) in listOf("model_id" to com.google.gson.JsonPrimitive(""),
                "engine" to com.google.gson.JsonPrimitive("unknown"), "timeout_seconds" to com.google.gson.JsonPrimitive(0))) {
                val body = parseJson(intentOperationJson()).apply {
                    getAsJsonObject("intent").apply { addProperty("strategy", strategy); add(field, value) }
                }.toString()
                route("/api/models/intent/current") { _, reply -> reply(200, body) }
                assertTrue("relaxed $field for $strategy", client.getCurrentModelIntent().isFailure)
            }
        }
    }

    @Test
    fun `intent post uses exact master owned supply and prepare request`() = runBlocking {
        route("/api/models/intent") { request, reply ->
            assertEquals("POST", request.method)
            assertEquals(parseJson("""{"model_id":"qwen","engine":"auto","quant_type":"","strategy":"auto","timeout_seconds":3600}"""), parseJson(request.body))
            reply(202, intentOperationJson())
        }
        val operation = client.submitModelIntent(ModelIntentRequest("qwen")).getOrThrow()
        assertEquals("supply_abc", operation.operationId)
        assertEquals(7L, operation.generation)
        assertEquals("cfg-1", operation.configId)
        assertFalse(operation.readyForChat)
        assertEquals(3456789123L, operation.nodes.getValue("phone").bytesReceived)
        assertEquals("SUPPLY_HASH_MISMATCH", operation.nodes.getValue("phone").reasonCode)
        assertEquals(4, operation.nodes.getValue("phone").layerRange!!.asJsonArray[0].asInt)
        assertEquals(1, operation.plan!!.asJsonObject["node_count"].asInt)
    }

    @Test
    fun `intent get and delete address the same operation with proper methods`() = runBlocking {
        var requests = 0
        route("/api/models/intent/supply_abc") { request, reply ->
            requests++
            assertEquals(if (requests == 1) "GET" else "DELETE", request.method)
            assertEquals("", request.body)
            reply(200, intentOperationJson(if (requests == 1) "preparing" else "cancelled"))
        }
        assertEquals("preparing", client.getModelIntent("supply_abc").getOrThrow().status)
        assertEquals("cancelled", client.cancelModelIntent("supply_abc").getOrThrow().status)
        assertEquals(2, requests)
    }

    @Test
    fun `intent invalid ids options mismatched response and null post fail closed`() = runBlocking {
        for (id in listOf("", "../current", "supply_x/../current", "supply_x?secret=y", "other_x")) {
            assertTrue(client.getModelIntent(id).isFailure)
            assertTrue(client.cancelModelIntent(id).isFailure)
        }
        assertTrue(client.submitModelIntent(ModelIntentRequest("qwen", engine = "bad")).isFailure)
        route("/api/models/intent/supply_abc") { _, reply -> reply(200, intentOperationJson(id = "supply_other")) }
        assertTrue(client.getModelIntent("supply_abc").isFailure)
        route("/api/models/intent") { _, reply -> reply(200, "null") }
        assertTrue(client.submitModelIntent(ModelIntentRequest("qwen")).isFailure)
    }

    @Test
    fun `intent only loaded ready enables chat and all contract states parse`() = runBlocking {
        for (status in MODEL_INTENT_STATUSES) {
            route("/api/models/intent/current") { _, reply -> reply(200, intentOperationJson(status, loaded = true)) }
            val operation = client.getCurrentModelIntent().getOrThrow()!!
            assertEquals(status == "ready", operation.readyForChat)
        }
        route("/api/models/intent/current") { _, reply -> reply(200, intentOperationJson("ready", loaded = false)) }
        assertFalse(client.getCurrentModelIntent().getOrThrow()!!.readyForChat)
    }

    @Test
    fun `intent HTTP failures retain named reasons and authenticated requests`() = runBlocking {
        client = ApiClient("http://127.0.0.1:${server.port}", authStore = FakeAuthStore(storedSession()))
        route("/api/models/intent/catalog") { request, reply ->
            assertEquals("Bearer ${storedSession().accessToken}", request.headers["Authorization"])
            reply(409, """{"detail":{"reason_code":"INTENT_NODES_UNAVAILABLE","reason":"No admitted workers"}}""")
        }
        val error = client.getModelIntentCatalog().exceptionOrNull()!!
        assertTrue(error is ApiClientHttpException)
        assertEquals("INTENT_NODES_UNAVAILABLE", com.qlh.inference.modelIntentError(error).code)
        assertEquals("No admitted workers", com.qlh.inference.modelIntentError(error).reason)
    }

    @Test
    fun `intent network request is cancellable instead of returning a failure result`() = runBlocking {
        val entered = java.util.concurrent.CountDownLatch(1)
        val release = java.util.concurrent.CountDownLatch(1)
        route("/api/models/intent/current") { _, reply ->
            entered.countDown()
            release.await(5, java.util.concurrent.TimeUnit.SECONDS)
            reply(200, "null")
        }
        val request = async { client.getCurrentModelIntent() }
        try {
            assertTrue(kotlinx.coroutines.withContext(kotlinx.coroutines.Dispatchers.IO) {
                entered.await(5, java.util.concurrent.TimeUnit.SECONDS)
            })
            request.cancel()
            kotlinx.coroutines.withTimeout(1500) { request.join() }
            assertTrue(request.isCancelled)
        } finally { release.countDown() }
    }

    @Test
    fun `discovery candidate verifies HTTP master before encrypted first connect`() = runBlocking {
        val calls = mutableListOf<String>()
        route("/api/bootstrap/info") { request, reply ->
            calls += "info"
            assertEquals("GET", request.method)
            reply(200, """{"is_master":true}""")
        }
        route("/api/bootstrap/first-connect") { request, reply ->
            calls += "encrypted-bootstrap"
            assertEquals("POST", request.method)
            assertTrue(parseJson(request.body).has("credential_public_key"))
            reply(200, bootstrapResponseFor(request, "discovered-secret"))
        }
        val result = client.firstConnectVerifiedMaster(BootstrapRequest(nodeId = "android-discovery", hostname = "Phone"))
        assertEquals("discovered-secret", result.getOrThrow().clusterSecret)
        assertEquals(listOf("info", "encrypted-bootstrap"), calls)
    }

    @Test
    fun `unverified or retired discovery candidate never attempts bootstrap`() = runBlocking {
        var bootstrapCalls = 0
        route("/api/bootstrap/first-connect") { _, reply -> bootstrapCalls++; reply(500, "{}") }
        for (body in listOf("{}", """{"is_master":false}""")) {
            route("/api/bootstrap/info") { _, reply -> reply(200, body) }
            assertTrue(client.firstConnectVerifiedMaster(BootstrapRequest("android-discovery", hostname = "Phone")).isFailure)
        }
        route("/api/bootstrap/info") { _, reply -> reply(200, """{"is_master":true}""") }
        assertTrue(client.firstConnectVerifiedMaster(BootstrapRequest("android-discovery", hostname = "Phone")) { false }.isFailure)
        assertEquals(0, bootstrapCalls)
    }

    @Test
    fun `short no thinking chat serializes explicit false and token limit`() = runBlocking {
        route("/api/chat") { request, reply ->
            assertEquals("POST", request.method)
            val body = parseJson(request.body)
            assertEquals("1+1=? 只回答数字。", body["message"].asString)
            assertEquals(32, body["max_new_tokens"].asInt)
            assertTrue(body.has("show_thinking"))
            assertFalse(body["show_thinking"].asBoolean)
            assertTrue(body.has("enable_thinking"))
            assertFalse(body["enable_thinking"].asBoolean)
            reply(200, """{"content":"2","metrics":{"distributed_used":true,"model_operation_id":"supply_abc","model_config_id":"cfg-1","model_generation":7}}""")
        }
        val response = client.chat(ChatRequest("1+1=? 只回答数字。", maxNewTokens = 32, showThinking = false)).getOrThrow()
        assertEquals("2", response.content)
        assertEquals("supply_abc", response.metrics?.get("model_operation_id"))
    }


    @Test
    fun `thinking switch enables generation and display together`() = runBlocking {
        route("/api/chat") { request, reply ->
            val body = parseJson(request.body)
            assertTrue(body["show_thinking"].asBoolean)
            assertTrue(body["enable_thinking"].asBoolean)
            reply(200, """{"content":"answer"}""")
        }
        assertTrue(client.chat(ChatRequest("question", showThinking = true)).isSuccess)
    }

}
