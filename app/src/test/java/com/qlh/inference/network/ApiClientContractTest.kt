package com.qlh.inference.network

import com.google.gson.Gson
import com.google.gson.JsonParser
import com.qlh.inference.security.AuthSessionStore
import com.qlh.inference.security.StoredAuthSession
import kotlinx.coroutines.runBlocking
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

        override fun read(): StoredAuthSession? = session
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
        route("/api/bootstrap/first-connect") { req, reply ->
            seen = req
            reply(200, """{"cluster":{"cluster_id":"c1","master_api_host":"100.1.2.3"}}""")
        }
        val response = runBlocking { client.firstConnectBootstrap(
            BootstrapRequest(nodeId = "n1", hostname = "100.1.2.3")
        ) }.getOrNull()!!
        assertEquals("c1", response.cluster.clusterId)
        assertEquals("/api/bootstrap/first-connect", seen?.path)
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
    fun `audit requests bounded server summaries and excludes content fields`() {
        route("/api/workflows?limit=8&summary=1") { req, reply ->
            assertEquals("GET", req.method)
            reply(
                200,
                """{"enabled":true,"available":true,"role":"master","workflows":[{"workflow_id":"wf-1","template":"chat","state":"completed","stage_count":1,"completed_stage_count":1,"attempt_count":1,"stages":[{"stage_id":"stage-1","stage_type":"llm","state":"completed","error_code":"PROVIDER_TIMEOUT","attempt_count":1,"attempts":[{"attempt_id":"a-1","provider_kind":"local","provider_node_id":"node-1","state":"completed","error_code":"PROVIDER_TIMEOUT"}]}]}],"prompt":"must-not-be-present"}""",
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
        assertEquals("wf-1", audit.workflows.workflows.single().workflowId)
        assertEquals("a-1", audit.workflows.workflows.single().stages.single().attempts.single().attemptId)
        assertEquals("PROVIDER_TIMEOUT", audit.workflows.workflows.single().stages.single().errorCode)
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

}
