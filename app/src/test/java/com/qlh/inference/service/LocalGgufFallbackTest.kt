package com.qlh.inference.service

import com.google.gson.JsonArray
import com.google.gson.JsonObject
import com.qlh.inference.worker.ModelSupplyController
import kotlinx.coroutines.*
import org.junit.Assert.*
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import java.io.File
import java.net.ServerSocket
import java.util.concurrent.CopyOnWriteArrayList
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.atomic.AtomicLong
import java.util.concurrent.atomic.AtomicReference
import kotlin.concurrent.thread

/** Real HTTP, exact Python golden bytes and isolated storage; no device or shared registry. */
class LocalGgufFallbackTest {
    @get:Rule val temp = TemporaryFolder()
    private val token = "fallback-test-lease"
    private fun bytes(name: String) = requireNotNull(javaClass.classLoader!!.getResourceAsStream("local-cut/hybrid/$name"))
        .use { it.readBytes() }
    private fun manifest() = ModelSupplyManifest.parse(bytes("tail.supply.json").toString(Charsets.UTF_8))
    private fun source(): LocalGgufSource {
        val m = manifest(); val data = bytes("source.gguf.fixture")
        return LocalGgufSource(m.sourceModelId, m.sourceSha256, m.json["tokenizer_sha256"].asString,
            data.size.toLong()) { data.inputStream() }
    }
    private fun sources(resolve: suspend () -> LocalGgufSource) = object : LocalGgufSources {
        override suspend fun inventory(modelId: String, sourceSha256: String, current: suspend () -> Boolean) = listOf(source())
        override suspend fun resolve(manifest: ModelSupplyManifest, current: suspend () -> Boolean) = resolve()
    }
    private fun unavailable() = sources { throw ModelSupplyException("SUPPLY_LOCAL_SOURCE_UNAVAILABLE", "source revoked") }
    private fun sign(json: JsonObject): ModelSupplyManifest {
        json.remove("manifest_sha256")
        json.addProperty("manifest_sha256", supplySha256(canonicalSupplyJson(json).toByteArray(Charsets.UTF_8)))
        return ModelSupplyManifest.parse(json)
    }

    private data class Request(val line: String, val headers: Map<String, String>)
    private inner class Server(val m: ModelSupplyManifest, overrides: Map<String, ByteArray> = emptyMap()) : AutoCloseable {
        private val socket = ServerSocket(0)
        val port get() = socket.localPort
        val requests = CopyOnWriteArrayList<Request>()
        val sent = AtomicLong()
        private val failure = AtomicReference<Throwable?>()
        private val payloads = mapOf(m.weight.path to bytes("tail.gguf.fixture"), m.native.path to bytes("tail.native.json")) + overrides
        private val worker = thread(isDaemon = true, name = "fallback-loopback-http") {
            try {
                while (!socket.isClosed) socket.accept().use { client ->
                    client.soTimeout = 3000
                    val reader = client.getInputStream().bufferedReader(Charsets.ISO_8859_1)
                    val line = reader.readLine() ?: return@use
                    val headers = linkedMapOf<String, String>()
                    while (true) {
                        val header = reader.readLine() ?: break
                        if (header.isEmpty()) break
                        headers[header.substringBefore(':').lowercase()] = header.substringAfter(':').trim()
                    }
                    requests += Request(line, headers)
                    val name = line.substringBefore(" HTTP/").substringAfterLast('/')
                    val payload = payloads[name]
                    val body = payload ?: ByteArray(0)
                    client.getOutputStream().apply {
                        write(("HTTP/1.1 ${if (payload == null) "404 Missing" else "200 OK"}\r\n" +
                            "Content-Length: ${body.size}\r\nConnection: close\r\n\r\n").toByteArray(Charsets.US_ASCII))
                        write(body); flush(); sent.addAndGet(body.size.toLong())
                    }
                }
            } catch (error: Throwable) { if (!socket.isClosed) failure.set(error) }
        }
        fun assertRequests(vararg paths: String) {
            failure.get()?.let { throw AssertionError("HTTP server failed", it) }
            assertEquals(paths.map { "GET /api/models/supply/${m.operationId}/files/${m.nodeId}/$it HTTP/1.1" }, requests.map { it.line })
            requests.forEach {
                assertEquals(token, it.headers["x-qlh-supply-token"])
                assertEquals("identity", it.headers["accept-encoding"])
                assertNull("must not resume discarded local bytes", it.headers["range"])
            }
        }
        override fun close() {
            socket.close(); worker.join(4000)
            assertFalse("HTTP thread leaked", worker.isAlive)
            failure.get()?.let { throw AssertionError("HTTP server failed", it) }
        }
    }

    private inner class Operation(val m: ModelSupplyManifest, localSources: LocalGgufSources, val root: File = temp.newFolder()) {
        val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
        val bound = AtomicBoolean(true)
        val mode = AtomicBoolean(true)
        val authority = AtomicBoolean(true)
        val registrations = AtomicInteger()
        val hellos = AtomicInteger()
        val publications = AtomicInteger()
        val replies = CopyOnWriteArrayList<JsonObject>()
        val terminal = CompletableDeferred<JsonObject>()
        var beforePublish: () -> Unit = {}
        val controller = ModelSupplyController(m.nodeId, "127.0.0.1", root, { mode.get() }, { bound.get() }, { 123 }, { action ->
            beforePublish()
            checkSupply(bound.get() && mode.get(), "SUPPLY_MASTER_BINDING_CHANGED", "retired gate")
            action(); publications.incrementAndGet()
        }, { registrations.incrementAndGet() }, localSources = localSources)
        private val reply: suspend (JsonObject) -> Unit = {
            replies += it.deepCopy()
            if (it["action"]?.asString == "result") terminal.complete(it)
        }
        suspend fun inventory(): JsonObject {
            val done = CompletableDeferred<JsonObject>()
            controller.command(JsonObject().apply {
                addProperty("action", "inventory"); addProperty("operation_id", m.operationId)
                addProperty("model_id", m.sourceModelId); addProperty("source_model_sha256", m.sourceSha256)
            }, scope, { done.complete(it) }, {})
            return withTimeout(5000) { done.await() }
        }
        suspend fun start(port: Int, allow: Boolean? = true, timeout: Int = 300, edit: (JsonObject) -> Unit = {}) {
            val data = JsonObject().apply {
                addProperty("action", "ensure"); addProperty("operation_id", m.operationId); addProperty("node_id", m.nodeId)
                addProperty("strategy", "local_cut"); addProperty("api_port", port); addProperty("token", token)
                addProperty("local_cut_timeout_seconds", timeout)
                if (allow != null) addProperty("allow_fallback", allow)
                add("manifest", m.json.deepCopy())
            }
            edit(data)
            controller.command(data, scope, reply, { hellos.incrementAndGet() }, {
                if (!authority.get()) throw CancellationException("retired authority")
            })
        }
        suspend fun idle() = withTimeout(5000) { scope.coroutineContext[Job]!!.children.toList().forEach { it.join() } }
        suspend fun result(): JsonObject {
            val result = withTimeout(10_000) { terminal.await() }; idle(); return result
        }
        suspend fun cancel() = controller.command(JsonObject().apply {
            addProperty("action", "cancel"); addProperty("operation_id", m.operationId)
        }, scope, reply, {})
        fun assertNotPublished() {
            assertEquals(0, publications.get()); assertEquals(0, registrations.get()); assertEquals(0, hellos.get())
            assertFalse(root.resolve(ModelSupplyCurrent.POINTER).exists())
            assertFalse(root.listFiles().orEmpty().any { it.name.startsWith("segment-") })
        }
        suspend fun close() {
            controller.cancelAll(); scope.cancel(); withTimeout(5000) { scope.coroutineContext[Job]!!.join() }
        }
    }
    private fun assertAvailable(op: Operation, result: JsonObject, server: Server) {
        val total = op.m.files.sumOf { it.size }
        assertEquals(result.toString(), "available", result["status"].asString)
        assertEquals(total, result["downloaded_bytes"].asLong); assertEquals(total, result["actual_downloaded_bytes"].asLong)
        assertEquals(total, server.sent.get())
        assertEquals(0L, result["resumed_bytes"].asLong); assertEquals(0L, result["reused_bytes"].asLong)
        assertEquals(1, op.registrations.get()); assertEquals(1, op.hellos.get())
        val current = ModelSupplyCurrent(op.root).read(supplyBindingSha256("", "127.0.0.1", 0, 0))!!
        assertEquals(op.m.manifestSha256, current.manifestSha256)
        assertArrayEquals(bytes("tail.gguf.fixture"), ModelSupplyCurrent(op.root).directory(current).resolve(op.m.weight.path).readBytes())
        server.assertRequests(op.m.weight.path, op.m.native.path)
    }

    @Test fun `revoked registered source after inventory downloads same manifest with same lease`() = runBlocking {
        val m = manifest(); val full = temp.newFile().apply { writeBytes(bytes("source.gguf.fixture")) }
        val readable = AtomicBoolean(true)
        val registry = LocalGgufSourceRegistry(temp.newFolder()) {
            if (readable.get()) listOf(LocalGgufSourceRegistry.Document("content://test/full.gguf", full.length()) { full.inputStream() })
            else emptyList()
        }
        registry.register("content://test/full.gguf", m.sourceModelId)
        Server(m).use { server ->
            val op = Operation(m, registry)
            try {
                assertEquals(1, op.inventory().getAsJsonArray("sources").size()); readable.set(false)
                op.start(server.port); assertAvailable(op, op.result(), server)
            } finally { op.close() }
        }
    }

    @Test fun `unsupported layout falls back only with explicit boolean permission`() = runBlocking {
        for (allow in listOf(true, false, null)) {
            val m = manifest()
            Server(m).use { server ->
                val op = Operation(m, sources { throw ModelSupplyException("SUPPLY_LOCAL_CUT_UNSUPPORTED", "unsupported layout") })
                try {
                    op.start(server.port, allow); val result = op.result()
                    if (allow == true) assertAvailable(op, result, server) else {
                        assertEquals("failed", result["status"].asString)
                        assertEquals("SUPPLY_LOCAL_CUT_UNSUPPORTED", result["reason_code"].asString)
                        assertEquals(0L, result["actual_downloaded_bytes"].asLong)
                        server.assertRequests(); op.assertNotPublished()
                    }
                } finally { op.close() }
            }
        }
    }

    @Test fun `actual local budget expiry recovers without cancelling the operation`() = runBlocking {
        val m = manifest()
        Server(m).use { server ->
            val op = Operation(m, sources { awaitCancellation() })
            try { op.start(server.port, timeout = 1); assertAvailable(op, op.result(), server) }
            finally { op.close() }
        }
    }

    @Test fun `string fallback flag is rejected before HTTP`() = runBlocking {
        val m = manifest()
        Server(m).use { server ->
            val op = Operation(m, unavailable())
            try {
                op.start(server.port) { it.addProperty("allow_fallback", "true") }
                assertEquals("SUPPLY_INVALID_REQUEST", op.result()["reason_code"].asString)
                server.assertRequests(); op.assertNotPublished()
            } finally { op.close() }
        }
    }

    @Test fun `cancellation binding identity integrity and disk error codes never trigger fallback`() = runBlocking {
        for (code in listOf("SUPPLY_CANCELLED", "SUPPLY_MASTER_BINDING_CHANGED", "SUPPLY_IDENTITY_MISMATCH",
            "SUPPLY_SOURCE_DIGEST_MISMATCH", "SUPPLY_ARTIFACT_DIGEST_MISMATCH", "SUPPLY_TOKENIZER_DIGEST_MISMATCH",
            "SUPPLY_NATIVE_MANIFEST_MISMATCH", "SUPPLY_FULL_MODEL_FORBIDDEN", "SUPPLY_DISK_INSUFFICIENT")) {
            val m = manifest()
            Server(m).use { server ->
                val op = Operation(m, sources { throw ModelSupplyException(code, "must fail closed") })
                try {
                    op.start(server.port); val result = op.result()
                    assertEquals(code, result["reason_code"].asString); assertEquals("failed", result["status"].asString)
                    server.assertRequests(); op.assertNotPublished()
                } finally { op.close() }
            }
        }
    }

    @Test fun `resolved source SHA and tokenizer mismatch cannot become recoverable unavailability`() = runBlocking {
        for (tokenizer in listOf(false, true)) {
            val m = manifest()
            val bad = if (tokenizer) source().copy(tokenizerSha256 = "0".repeat(64)) else source().copy(sha256 = "0".repeat(64))
            Server(m).use { server ->
                val op = Operation(m, sources { bad })
                try {
                    op.start(server.port)
                    assertEquals(if (tokenizer) "SUPPLY_TOKENIZER_DIGEST_MISMATCH" else "SUPPLY_SOURCE_DIGEST_MISMATCH",
                        op.result()["reason_code"].asString)
                    server.assertRequests(); op.assertNotPublished()
                } finally { op.close() }
            }
        }
    }

    @Test fun `actual changed source bytes retain terminal digest failure instead of downloading`() = runBlocking {
        val m = manifest()
        val changed = bytes("source.gguf.fixture").apply { this[lastIndex] = (this[lastIndex].toInt() xor 1).toByte() }
        Server(m).use { server ->
            val op = Operation(m, sources { source().copy(open = { changed.inputStream() }) })
            try {
                op.start(server.port)
                assertEquals("SUPPLY_SOURCE_DIGEST_MISMATCH", op.result()["reason_code"].asString)
                server.assertRequests(); op.assertNotPublished()
            } finally { op.close() }
        }
    }

    @Test fun `real cancel connection retirement binding mode and authority changes prevent HTTP`() = runBlocking {
        for (retire in listOf("cancel", "connection", "binding", "mode", "authority")) {
            val m = manifest(); val entered = CompletableDeferred<Unit>(); val release = CompletableDeferred<Unit>()
            Server(m).use { server ->
                val op = Operation(m, sources {
                    entered.complete(Unit); release.await()
                    throw ModelSupplyException("SUPPLY_LOCAL_SOURCE_UNAVAILABLE", "revoked source")
                })
                try {
                    op.start(server.port); withTimeout(3000) { entered.await() }
                    when (retire) {
                        "cancel" -> op.cancel()
                        "connection" -> op.controller.cancelAll()
                        "binding" -> op.bound.set(false)
                        "mode" -> op.mode.set(false)
                        "authority" -> op.authority.set(false)
                    }
                    release.complete(Unit); op.idle()
                    server.assertRequests(); op.assertNotPublished()
                    assertFalse(op.replies.any { it["status"]?.asString == "available" })
                } finally { op.close() }
            }
        }
    }

    private fun seedPrefix(root: File, m: ModelSupplyManifest) {
        val prefix = bytes("tail.gguf.fixture").copyOf(128)
        val stage = root.resolve(".receive-${m.manifestSha256}").apply { mkdirs() }
        stage.resolve(m.weight.path + ".part").writeBytes(prefix)
        val journal = JsonObject().apply {
            addProperty("schema_version", 1); add("manifest", m.json.deepCopy())
            add("prefixes", JsonArray().apply { add(JsonObject().apply {
                addProperty("path", m.weight.path); addProperty("file_sha256", m.weight.sha256)
                addProperty("size_bytes", m.weight.size); addProperty("prefix_bytes", prefix.size)
                addProperty("prefix_sha256", supplySha256(prefix))
            }) })
            addProperty("journal_sha256", supplySha256(canonicalSupplyJson(this).toByteArray(Charsets.UTF_8)))
        }
        stage.resolve(ModelSupplyReceiver.RESUME_MARKER).writeText(canonicalSupplyJson(journal), Charsets.UTF_8)
    }

    @Test fun `discarded build and resume prefix count no network bytes while prior transfers survive`() = runBlocking {
        for (nativeFirst in listOf(false, true)) {
            val base = manifest()
            val m = if (!nativeFirst) base else sign(base.json.deepCopy().apply {
                val files = getAsJsonArray("files").map { it.deepCopy() }
                add("files", JsonArray().apply { add(files[1]); add(files[0]) })
            })
            val root = temp.newFolder(); seedPrefix(root, m)
            Server(m).use { server ->
                val progress = mutableListOf<ModelSupplyProgress>()
                val receiver = ModelSupplyReceiver(root, "127.0.0.1")
                val final = receiver.receive(m, server.port, token, { true }, { progress += it }, { it() },
                    allowLocalCutFallback = true, localWeight = { output, report ->
                        assertFalse(ModelSupplyReceiver.readResume(output.parentFile!!)!!.records.containsKey(m.weight.path))
                        output.writeBytes(ByteArray(77) { 42 }); report(77)
                        throw ModelSupplyException("SUPPLY_LOCAL_CUT_TIMEOUT", "partial build timed out")
                    })
                receiver.validateFiles(final, m)
                val fallback = progress.single { it.phase == "transfer_fallback_local_cut_unavailable" }
                val prior = if (nativeFirst) m.native.size else 0L
                assertEquals(prior, fallback.downloadedBytes); assertEquals(prior, fallback.actualDownloadedBytes)
                assertEquals(0L, fallback.resumedBytes); assertEquals(0L, fallback.reusedBytes)
                assertEquals(m.files.sumOf { it.size }, progress.last().actualDownloadedBytes)
                assertEquals(m.files.sumOf { it.size }, progress.last().downloadedBytes)
                assertEquals(0L, progress.last().resumedBytes); assertEquals(0L, progress.last().reusedBytes)
                assertArrayEquals(bytes("tail.gguf.fixture"), final.resolve(m.weight.path).readBytes())
                server.assertRequests(*m.files.map { it.path }.toTypedArray())
                assertEquals(m.files.sumOf { it.size }, server.sent.get())
            }
        }
    }

    @Test fun `binding is rechecked after fallback progress before starting HTTP`() = runBlocking {
        val m = manifest(); var current = true
        Server(m).use { server ->
            try {
                ModelSupplyReceiver(temp.newFolder(), "127.0.0.1").receive(m, server.port, token, { current }, {
                    if (it.phase == "transfer_fallback_local_cut_unavailable") current = false
                }, { fail("retired fallback published") }, allowLocalCutFallback = true, localWeight = { _, _ ->
                    throw ModelSupplyException("SUPPLY_LOCAL_SOURCE_UNAVAILABLE", "source revoked")
                })
                fail("retired fallback succeeded")
            } catch (error: ModelSupplyException) { assertEquals("SUPPLY_CANCELLED", error.reasonCode) }
            server.assertRequests()
        }
    }

    @Test fun `wrong local output SHA cannot trigger HTTP recovery`() = runBlocking {
        val m = manifest()
        Server(m).use { server ->
            try {
                ModelSupplyReceiver(temp.newFolder(), "127.0.0.1").receive(m, server.port, token, { true }, {},
                    { fail("bad local bytes published") }, allowLocalCutFallback = true,
                    localWeight = { output, _ -> output.writeBytes(ByteArray(m.weight.size.toInt())) })
                fail("wrong local SHA accepted")
            } catch (error: ModelSupplyException) { assertEquals("SUPPLY_ARTIFACT_DIGEST_MISMATCH", error.reasonCode) }
            server.assertRequests()
        }
    }

    @Test fun `corrupt downloaded fallback weight remains terminal without retry or publication`() = runBlocking {
        val m = manifest()
        val corrupt = bytes("tail.gguf.fixture").apply { this[lastIndex] = (this[lastIndex].toInt() xor 1).toByte() }
        Server(m, mapOf(m.weight.path to corrupt)).use { server ->
            val op = Operation(m, unavailable())
            try {
                op.start(server.port); val result = op.result()
                assertEquals("failed", result["status"].asString)
                assertEquals("SUPPLY_ARTIFACT_DIGEST_MISMATCH", result["reason_code"].asString)
                assertEquals(m.weight.size, result["actual_downloaded_bytes"].asLong)
                server.assertRequests(m.weight.path); op.assertNotPublished()
            } finally { op.close() }
        }
    }

    @Test fun `fallback native identity still rejects before publication even with valid file SHA`() = runBlocking {
        val base = manifest()
        val native = com.google.gson.JsonParser.parseString(bytes("tail.native.json").toString(Charsets.UTF_8)).asJsonObject
            .apply { addProperty("source_model_id", "another-source") }
        val data = canonicalSupplyJson(native).toByteArray(Charsets.UTF_8)
        val m = sign(base.json.deepCopy().apply {
            getAsJsonArray("files").map { it.asJsonObject }.single { it["kind"].asString == "manifest" }.apply {
                addProperty("size_bytes", data.size); addProperty("sha256", supplySha256(data))
            }
        })
        Server(m, mapOf(m.native.path to data)).use { server ->
            val op = Operation(m, unavailable())
            try {
                op.start(server.port); val result = op.result()
                assertEquals("SUPPLY_NATIVE_MANIFEST_MISMATCH", result["reason_code"].asString)
                assertEquals(m.files.sumOf { it.size }, result["actual_downloaded_bytes"].asLong)
                server.assertRequests(m.weight.path, m.native.path); op.assertNotPublished()
            } finally { op.close() }
        }
    }

    @Test fun `original publication gate can reject a complete fallback after binding changes`() = runBlocking {
        val m = manifest()
        Server(m).use { server ->
            val op = Operation(m, unavailable())
            try {
                op.beforePublish = { op.bound.set(false) }
                op.start(server.port); op.idle()
                server.assertRequests(m.weight.path, m.native.path); op.assertNotPublished()
                assertFalse(op.replies.any { it["status"]?.asString == "available" })
            } finally { op.close() }
        }
    }
}
