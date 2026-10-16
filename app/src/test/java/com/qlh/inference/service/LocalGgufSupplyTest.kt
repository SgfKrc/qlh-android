package com.qlh.inference.service

import com.google.gson.JsonObject
import com.qlh.inference.worker.ModelSupplyController
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withTimeout
import org.junit.Assert.*
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import java.io.BufferedReader
import java.io.InputStreamReader
import java.io.File
import java.net.ServerSocket
import java.util.concurrent.CopyOnWriteArrayList
import kotlin.concurrent.thread

class LocalGgufSupplyTest {
    @get:Rule val temp = TemporaryFolder()
    private fun registry(source: File, records: File = temp.newFolder(), available: () -> Boolean = { true }) =
        LocalGgufSourceRegistry(records) {
            if (!available()) emptyList() else listOf(LocalGgufSourceRegistry.Document("content://test/full.gguf", source.length()) { source.inputStream() })
        }
    private fun fullSource() = temp.newFile("full.gguf").apply { writeBytes(CutFixture.bytes("hybrid/source.gguf.fixture")) }
    private fun inventory(manifest: ModelSupplyManifest) = JsonObject().apply {
        addProperty("action", "inventory"); addProperty("operation_id", manifest.operationId)
        addProperty("model_id", manifest.sourceModelId); addProperty("source_model_sha256", "")
    }
    private fun ensure(manifest: ModelSupplyManifest, port: Int) = JsonObject().apply {
        addProperty("action", "ensure"); addProperty("operation_id", manifest.operationId); addProperty("node_id", manifest.nodeId)
        addProperty("strategy", "local_cut"); addProperty("api_port", port); addProperty("token", "test-lease")
        add("manifest", manifest.json.deepCopy())
    }
    private class NativeServer(private val native: ByteArray) : AutoCloseable {
        private val server = ServerSocket(0)
        val port get() = server.localPort
        val requests = CopyOnWriteArrayList<String>()
        private val worker = thread(isDaemon = true) {
            try {
                while (!server.isClosed) server.accept().use { socket ->
                    val reader = BufferedReader(InputStreamReader(socket.getInputStream()))
                    val request = reader.readLine() ?: return@use
                    requests += request
                    while (reader.readLine().orEmpty().isNotEmpty()) { }
                    val success = request.contains("segment.manifest.json")
                    val body = if (success) native else "weight download forbidden".toByteArray()
                    socket.getOutputStream().apply {
                        write("HTTP/1.1 ${if (success) "200 OK" else "500 forbidden"}\r\nContent-Length: ${body.size}\r\nConnection: close\r\n\r\n".toByteArray())
                        write(body); flush()
                    }
                }
            } catch (_: java.io.IOException) { }
        }
        override fun close() { server.close(); worker.join(3000) }
    }

    @Test fun `registration survives restart but stale or revoked source never advertises`() = runBlocking {
        val source = fullSource(); val records = temp.newFolder(); var readable = true
        val registry = registry(source, records) { readable }
        assertTrue(registry.inventory("golden", "", { true }).isEmpty())
        val registered = registry.register("content://test/full.gguf", "golden")
        assertEquals(CutFixture.manifest().sourceSha256, registered.sha256)
        assertTrue(registry.summary("content://test/full.gguf").startsWith("已登记 golden"))
        assertFalse(registry.summary("content://test/full.gguf").contains("???"))
        val restarted = registry(source, records) { readable }
        assertEquals(registered.sha256, restarted.inventory("golden", "", { true }).single().sha256)
        assertTrue(restarted.inventory("another", "", { true }).isEmpty())
        readable = false
        assertTrue(restarted.inventory("golden", "", { true }).isEmpty())
        readable = true
        val mutated = source.readBytes().apply { this[lastIndex] = (this[lastIndex].toInt() xor 1).toByte() }
        source.writeBytes(mutated)
        assertTrue(restarted.inventory("golden", "", { true }).isEmpty())
        expectCode("SUPPLY_LOCAL_SOURCE_UNAVAILABLE") { restarted.resolve(CutFixture.manifest(), { true }) }
    }

    @Test fun `source mutation cancellation and whole model assignment cannot publish`() = runBlocking {
        val manifest = CutFixture.manifest(); val source = CutFixture.source()
        expectCode("SUPPLY_CANCELLED") {
            GgufSegmentCutter.cut(source, manifest, temp.newFile(), { false }, {})
        }
        val changed = CutFixture.bytes("hybrid/source.gguf.fixture").apply { this[lastIndex] = (this[lastIndex].toInt() xor 1).toByte() }
        expectCode("SUPPLY_SOURCE_DIGEST_MISMATCH") {
            GgufSegmentCutter.cut(source.copy(open = { changed.inputStream() }), manifest, temp.newFile(), { true }, {})
        }
        val full = manifest.json.deepCopy().apply {
            getAsJsonArray("layer_range").set(0, com.google.gson.JsonPrimitive(0))
        }
        expectCode("SUPPLY_FULL_MODEL_FORBIDDEN") {
            GgufSegmentCutter.cut(source, ModelSupplyManifest(full, manifest.files), temp.newFile(), { true }, {})
        }
        var live = true
        expectCode("SUPPLY_CANCELLED") {
            GgufSegmentCutter.cut(source, manifest, temp.newFile(), { live }) { live = false }
        }
    }

    @Test fun `controller builds locally downloads only native manifest and selects through the existing owner`() = runBlocking {
        val manifest = CutFixture.manifest(); val root = temp.newFolder(); val registry = registry(fullSource())
        registry.register("content://test/full.gguf", "golden")
        val native = CutFixture.bytes("hybrid/tail.native.json")
        NativeServer(native).use { server ->
            val replies = CopyOnWriteArrayList<JsonObject>(); val done = CompletableDeferred<JsonObject>()
            var registrations = 0; var hellos = 0
            val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
            val controller = ModelSupplyController(manifest.nodeId, "127.0.0.1", root, { true }, { true }, { 123 }, { it() },
                { registrations++ }, localSources = registry)
            try {
                val inventoryDone = CompletableDeferred<JsonObject>()
                controller.command(inventory(manifest), scope, { replies += it; inventoryDone.complete(it) }, { hellos++ })
                val response = withTimeout(10_000) { inventoryDone.await() }
                assertTrue(response["local_cut"].asBoolean)
                assertEquals(GgufSegmentCutter.TOOL, response["local_cut_tool"].asString)
                assertEquals(manifest.sourceSha256, response.getAsJsonArray("sources").single().asJsonObject["source_model_sha256"].asString)
                controller.command(ensure(manifest, server.port), scope, {
                    replies += it
                    if (it["action"]?.asString == "result") done.complete(it)
                }, { hellos++ })
                val result = withTimeout(10_000) { done.await() }
                assertEquals(result.toString(), "available", result["status"].asString)
                assertEquals(native.size.toLong(), result["actual_downloaded_bytes"].asLong)
                assertEquals(1, registrations); assertEquals(1, hellos)
                assertEquals(1, server.requests.size)
                assertTrue(server.requests.single().contains("segment.manifest.json"))
                assertTrue(replies.any { it["phase"]?.asString == "building" })
                val current = ModelSupplyCurrent(root).read(supplyBindingSha256("", "127.0.0.1", 0, 0))!!
                assertEquals(manifest.manifestSha256, current.manifestSha256)
                assertArrayEquals(CutFixture.bytes("hybrid/tail.gguf.fixture"), ModelSupplyCurrent(root).directory(current).resolve(current.weight.path).readBytes())
            } finally { controller.cancelAll(); scope.cancel() }
        }
    }

    @Test fun `source mutation during building and wrong output SHA are rejected before publication`() = runBlocking {
        val manifest = CutFixture.manifest(); val native = CutFixture.bytes("hybrid/tail.native.json")
        NativeServer(native).use { server ->
            val root = temp.newFolder()
            val receiver = ModelSupplyReceiver(root, "127.0.0.1")
            var published = false
            expectCode("SUPPLY_ARTIFACT_DIGEST_MISMATCH") {
                receiver.receive(manifest, server.port, "lease", { true }, {}, { published = true; it() },
                    localWeight = { output, _ -> output.writeBytes(ByteArray(manifest.weight.size.toInt())) })
            }
            assertFalse(published)
            assertFalse(root.listFiles()!!.any { it.name.startsWith("segment-") })
            assertTrue(server.requests.isEmpty())
        }
    }

    @Test fun `connection retirement and cancel fence stop local cut before registration`() = runBlocking {
        val manifest = CutFixture.manifest(); val root = temp.newFolder(); val source = CutFixture.source()
        val entered = CompletableDeferred<Unit>(); val released = CompletableDeferred<Unit>()
        val sources = object : LocalGgufSources {
            override suspend fun inventory(modelId: String, sourceSha256: String, current: suspend () -> Boolean) = listOf(source)
            override suspend fun resolve(manifest: ModelSupplyManifest, current: suspend () -> Boolean): LocalGgufSource {
                entered.complete(Unit); released.await(); return source
            }
        }
        val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
        val replies = CopyOnWriteArrayList<JsonObject>()
        val controller = ModelSupplyController(manifest.nodeId, "127.0.0.1", root, { true }, { true }, { 123 }, { it() },
            { fail("retired local cut registered") }, localSources = sources)
        try {
            controller.command(ensure(manifest, 1), scope, { replies += it }, { fail("retired hello") })
            withTimeout(3000) { entered.await() }
            controller.command(JsonObject().apply {
                addProperty("action", "cancel"); addProperty("operation_id", manifest.operationId)
            }, scope, { replies += it }, {})
            released.complete(Unit)
            controller.cancelAll()
            assertFalse(replies.any { it["status"]?.asString == "available" })
            assertFalse(root.resolve(ModelSupplyCurrent.POINTER).exists())
            assertFalse(root.listFiles()!!.any { it.name.startsWith("segment-") })
        } finally { scope.cancel() }
    }

    @Test fun `source inventory never blocks TCP command processing and retires with connection`() = runBlocking {
        val manifest = CutFixture.manifest(); val root = temp.newFolder()
        val entered = CompletableDeferred<Unit>(); val release = CompletableDeferred<Unit>()
        val sources = object : LocalGgufSources {
            override suspend fun inventory(modelId: String, sourceSha256: String, current: suspend () -> Boolean): List<LocalGgufSource> {
                entered.complete(Unit); release.await()
                checkSupply(current(), "SUPPLY_CANCELLED", "retired inventory")
                return listOf(CutFixture.source())
            }
            override suspend fun resolve(manifest: ModelSupplyManifest, current: suspend () -> Boolean) = error("no ensure")
        }
        val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
        val replies = CopyOnWriteArrayList<JsonObject>()
        val controller = ModelSupplyController(manifest.nodeId, "127.0.0.1", root, { true }, { true }, { 123 }, { it() },
            { fail("inventory published") }, localSources = sources)
        try {
            withTimeout(3000) { controller.command(inventory(manifest), scope, { replies += it }, {}) }
            withTimeout(3000) { entered.await() }
            controller.cancelAll()
            release.complete(Unit)
            // Process a current command immediately while the retired inventory is being cancelled.
            controller.command(JsonObject().apply {
                addProperty("action", "cancel"); addProperty("operation_id", manifest.operationId)
            }, scope, { replies += it }, {})
            assertEquals(1, replies.size)
            assertEquals("SUPPLY_CANCELLED", replies.single()["reason_code"].asString)
        } finally { scope.cancel() }
    }

    private suspend fun expectCode(code: String, block: suspend () -> Unit) {
        try { block(); fail("expected $code") }
        catch (error: ModelSupplyException) { assertEquals(error.message, code, error.reasonCode) }
    }
}
