package com.qlh.inference.service

import com.google.gson.JsonObject
import com.google.gson.JsonParser
import com.qlh.inference.worker.GgufLayerWeightReader
import com.qlh.inference.worker.ModelSupplyController
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withTimeout
import okhttp3.OkHttpClient
import okhttp3.Protocol
import okhttp3.Response
import okhttp3.ResponseBody.Companion.toResponseBody
import org.junit.Assert.*
import org.junit.Assume.assumeNoException
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import java.io.ByteArrayOutputStream
import java.io.DataOutputStream
import java.io.File
import java.nio.file.Files

class ModelSupplyDedupGcTest {
    @get:Rule val temp = TemporaryFolder()
    private val node = "android_dedup"
    private val binding = supplyBindingSha256("cluster", "127.0.0.1", 1234, 1)
    private data class Fixture(val manifest: ModelSupplyManifest, val files: Map<String, ByteArray>)

    private fun sign(json: JsonObject): ModelSupplyManifest {
        json.remove("manifest_sha256")
        json.addProperty("manifest_sha256", supplySha256(canonicalSupplyJson(json).toByteArray()))
        return ModelSupplyManifest.parse(json)
    }
    private fun fixture(operation: Int = 1): Fixture {
        val buffer = ByteArrayOutputStream()
        DataOutputStream(buffer).use { out ->
            fun int(v: Int) = out.writeInt(Integer.reverseBytes(v))
            fun long(v: Long) = out.writeLong(java.lang.Long.reverseBytes(v))
            fun text(v: String) { val bytes = v.toByteArray(); long(bytes.size.toLong()); out.write(bytes) }
            out.write("GGUF".toByteArray()); int(3); long(2); long(3)
            text("tokenizer.ggml.model"); int(8); text("test")
            text("general.architecture"); int(8); text("test")
            text("test.block_count"); int(4); int(2)
            for (i in 0..1) { text("blk.$i.attn.weight"); int(1); long(1); int(0); long(i * 4L) }
        }
        val weight = buffer.toByteArray()
        val tokenizer = GgufLayerWeightReader.scan(weight.inputStream(), 2)!!.tokenizerSha256
        val native = JsonParser.parseString("""{
            "artifact":"segment.gguf","artifact_sha256":"${supplySha256(weight)}",
            "source_model_id":"source","source_model_sha256":"${"b".repeat(64)}",
            "tokenizer_sha256":"$tokenizer","hidden_size":1,"mode":"tail",
            "source_layer_range":[2,4],"n_layer":4,"kept_block_count":2,
            "gguf_metadata":{"architecture":"test","block_count":2,"head_count_kv":null,
                "key_length":null,"full_attention_interval":null},
            "conversion_proof":null,"layer_weight_bytes":[0,0,4,4]
        }""").asJsonObject
        val nativeBytes = canonicalSupplyJson(native).toByteArray()
        val json = JsonParser.parseString("""{
            "manifest_kind":"qlh.segment_supply.v1","operation_id":"supply_${operation.toString(16).padStart(32, '0')}",
            "node_id":"$node","source_model_id":"source","source_model_sha256":"${"b".repeat(64)}",
            "tokenizer_sha256":"$tokenizer","total_layers":4,"layer_range":[2,4],"segment_mode":"tail",
            "layer_weight_bytes":[0,0,4,4],"hidden_size":1,"artifact_sha256":"${supplySha256(weight)}",
            "format":"gguf","conversion_proof":null,
            "files":[{"path":"segment.gguf","kind":"weights","size_bytes":${weight.size},"sha256":"${supplySha256(weight)}"},
            {"path":"segment.manifest.json","kind":"manifest","size_bytes":${nativeBytes.size},"sha256":"${supplySha256(nativeBytes)}"}]
        }""").asJsonObject
        return Fixture(sign(json), mapOf("segment.gguf" to weight, "segment.manifest.json" to nativeBytes))
    }
    private fun root(): File = temp.newFolder().resolve("supply").apply { mkdir() }
    private fun publish(root: File, fixture: Fixture): File =
        root.resolve("segment-${fixture.manifest.manifestSha256}").apply {
            mkdir()
            fixture.files.forEach { (path, bytes) -> resolve(path).writeBytes(bytes) }
            resolve(ModelSupplyManifest.MARKER).writeText(canonicalSupplyJson(fixture.manifest.json))
        }
    private fun select(root: File, fixture: Fixture) {
        val current = ModelSupplyCurrent(root)
        current.complete(current.select(fixture.manifest, binding))
    }
    private fun client(fixture: Fixture, requests: MutableList<String>): OkHttpClient = OkHttpClient.Builder()
        .addInterceptor { chain ->
            val request = chain.request()
            val path = request.url.pathSegments.last()
            requests += path
            assertTrue(request.url.encodedPath.contains("/${fixture.manifest.operationId}/files/$node/"))
            Response.Builder().request(request).protocol(Protocol.HTTP_1_1).code(200).message("OK")
                .body(fixture.files.getValue(path).toResponseBody()).build()
        }.build()

    @Test fun `download verification survives rename and shares hashes with warm receive and registration`() = runBlocking {
        val root = root()
        val first = fixture()
        val files = PrivateDigestTestFiles(temp.newFolder())
        val requests = mutableListOf<String>()
        val initial = ModelSupplyReceiver(root, "127.0.0.1", client = client(first, requests),
            digests = files.digests, createLink = files::link).receive(
            first.manifest, 1, "secret", { true }, {}, { it() })
        assertEquals(first.files.size, files.reads.size)
        assertTrue(files.reads.all { it.name.endsWith(".part") })
        ModelSupplyReceiver(root, "unused", digests = files.digests).validateFiles(initial, first.manifest)
        assertEquals(first.manifest.weight.sha256, files.digests.verify(initial.resolve(first.manifest.weight.path)).digest)
        assertEquals(first.files.size, files.reads.size)
        requests.clear()
        val next = fixture(2)
        val progress = mutableListOf<ModelSupplyProgress>()
        val received = ModelSupplyReceiver(root, "127.0.0.1", client = client(next, requests),
            digests = files.digests, createLink = files::link).receive(
            next.manifest, 1, "secret", { true }, { progress += it }, { it() })
        ModelSupplyReceiver(root, "unused", digests = files.digests).validateFiles(received, next.manifest)
        assertEquals(next.manifest.weight.sha256, files.digests.verify(received.resolve(next.manifest.weight.path)).digest)
        assertTrue(requests.isEmpty())
        assertEquals(0L, progress.last().actualDownloadedBytes)
        assertEquals(next.manifest.files.sumOf { it.size }, progress.last().reusedBytes)
        assertTrue(Files.isSameFile(initial.resolve(first.manifest.weight.path).toPath(),
            received.resolve(next.manifest.weight.path).toPath()))
        assertEquals(first.files.size, files.reads.size)
    }

    @Test fun `cold donor is read once across hardlink target verification and final validation`() = runBlocking {
        val root = root()
        val first = fixture()
        publish(root, first)
        val next = fixture(2)
        val files = PrivateDigestTestFiles(temp.newFolder())
        val requests = mutableListOf<String>()
        val received = ModelSupplyReceiver(root, "127.0.0.1", client = client(next, requests),
            digests = files.digests, createLink = files::link).receive(
            next.manifest, 1, "secret", { true }, {}, { it() })
        ModelSupplyReceiver(root, "unused", digests = files.digests).validateFiles(received, next.manifest)
        assertTrue(requests.isEmpty())
        assertEquals(first.files.size, files.reads.size)
        assertEquals(1, files.reads.count { it.name == first.manifest.weight.path })
    }

    @Test fun `copy fallback always hashes new target bytes and later validation reuses that verification`() = runBlocking {
        val root = root()
        val first = fixture()
        publish(root, first)
        val next = fixture(2)
        val files = PrivateDigestTestFiles(temp.newFolder())
        val requests = mutableListOf<String>()
        val received = ModelSupplyReceiver(root, "127.0.0.1", client = client(next, requests),
            digests = files.digests, createLink = { _, _ -> throw UnsupportedOperationException("no links") }).receive(
            next.manifest, 1, "secret", { true }, {}, { it() })
        assertEquals(first.files.size * 2, files.reads.size)
        assertEquals(next.files.size, files.reads.count { it.parentFile?.name?.startsWith(".receive-") == true })
        ModelSupplyReceiver(root, "unused", digests = files.digests).validateFiles(received, next.manifest)
        assertEquals(first.files.size * 2, files.reads.size)
        assertTrue(requests.isEmpty())
    }

    @Test fun `copy fallback rejects corrupted copied bytes and downloads a verified replacement`() = runBlocking {
        val root = root()
        val first = fixture()
        val donor = publish(root, first)
        val files = PrivateDigestTestFiles(temp.newFolder())
        ModelSupplyReceiver(root, "unused", digests = files.digests).validateFiles(donor, first.manifest)
        val next = fixture(2)
        val requests = mutableListOf<String>()
        val received = ModelSupplyReceiver(root, "127.0.0.1", client = client(next, requests), digests = files.digests,
            createLink = { source, _ ->
                if (source.name == first.manifest.weight.path) {
                    val modified = Files.getLastModifiedTime(source.toPath())
                    source.writeBytes(source.readBytes().also { it[0] = (it[0].toInt() xor 1).toByte() })
                    Files.setLastModifiedTime(source.toPath(), modified)
                    files.changed(source)
                }
                throw UnsupportedOperationException("no links")
            }).receive(next.manifest, 1, "secret", { true }, {}, { it() })
        assertEquals(listOf(next.manifest.weight.path), requests)
        assertArrayEquals(next.files.getValue(next.manifest.weight.path), received.resolve(next.manifest.weight.path).readBytes())
        assertEquals(1, files.reads.count {
            it.name == next.manifest.weight.path && it.parentFile?.name?.startsWith(".receive-") == true
        })
        ModelSupplyReceiver(root, "unused", digests = files.digests).validateFiles(received, next.manifest)
    }

    @Test fun `GC ignores an unchanged hot digest and freshly rejects corrupted retirement bytes`() = runBlocking {
        val root = root()
        val first = fixture()
        val old = publish(root, first)
        val files = PrivateDigestTestFiles(temp.newFolder())
        ModelSupplyReceiver(root, "unused", digests = files.digests).validateFiles(old, first.manifest)
        val next = fixture(2)
        publish(root, next)
        select(root, next)
        val weight = old.resolve(first.manifest.weight.path)
        val modified = Files.getLastModifiedTime(weight.toPath())
        weight.writeBytes(weight.readBytes().also { it[0] = (it[0].toInt() xor 1).toByte() })
        Files.setLastModifiedTime(weight.toPath(), modified)
        assertEquals(first.manifest.weight.sha256, files.digests.verify(weight).digest)
        assertEquals(0, ModelSupplyGc(root).collect(node).deleted)
        assertTrue(old.exists())
    }

    @Test fun `restart and new operation reuse all bytes with a fresh bound manifest and zero HTTP`() = runBlocking {
        val root = root(); val first = fixture(); val requests = mutableListOf<String>()
        val old = ModelSupplyReceiver(root, "127.0.0.1", client = client(first, requests)).receive(
            first.manifest, 1, "secret", { true }, {}, { it() })
        assertEquals(first.files.keys, requests.toSet())
        select(root, first)
        requests.clear()
        val next = fixture(2); val progress = mutableListOf<ModelSupplyProgress>()
        val directory = ModelSupplyReceiver(root, "127.0.0.1", client = client(next, requests)).receive(
            next.manifest, 1, "secret", { true }, { progress += it }, { it() })
        assertTrue(requests.isEmpty())
        assertEquals(0L, progress.last().actualDownloadedBytes)
        assertEquals(next.manifest.files.sumOf { it.size }, progress.last().reusedBytes)
        assertEquals(progress.last().totalBytes, progress.last().downloadedBytes)
        assertEquals(next.manifest, ModelSupplyManifest.parse(directory.resolve(ModelSupplyManifest.MARKER).readText()))
        assertEquals(first.manifest, ModelSupplyManifest.parse(old.resolve(ModelSupplyManifest.MARKER).readText()))
        select(root, next)
        assertEquals(next.manifest, ModelSupplyCurrent(root).read(binding))
        assertEquals(1, ModelSupplyGc(root).collect(node).deleted)
        assertFalse(old.exists()); assertTrue(directory.exists())
    }

    @Test fun `same file SHA cannot donate across node source range role or geometry`() = runBlocking {
        for (field in listOf("node", "source-id", "source-sha", "range", "role", "geometry", "proof")) {
            val root = root(); val first = fixture()
            val foreign = first.manifest.json.deepCopy().apply {
                when (field) {
                    "node" -> addProperty("node_id", "other_node")
                    "source-id" -> addProperty("source_model_id", "other_source")
                    "source-sha" -> addProperty("source_model_sha256", "c".repeat(64))
                    "range" -> { add("layer_range", JsonParser.parseString("[1,3]")); addProperty("segment_mode", "middle")
                        add("layer_weight_bytes", JsonParser.parseString("[0,4,4,0]")) }
                    "role" -> addProperty("has_embedding", true)
                    "geometry" -> addProperty("hidden_size", 2)
                    "proof" -> add("conversion_proof", JsonObject().apply { addProperty("converter", "different") })
                }
            }
            publish(root, Fixture(sign(foreign), first.files))
            val next = fixture(2); val requests = mutableListOf<String>(); val progress = mutableListOf<ModelSupplyProgress>()
            ModelSupplyReceiver(root, "127.0.0.1", client = client(next, requests)).receive(
                next.manifest, 1, "secret", { true }, { progress += it }, { it() })
            assertEquals(field, next.files.keys, requests.toSet())
            assertEquals(field, 0L, progress.last().reusedBytes)
            assertEquals(field, progress.last().totalBytes, progress.last().actualDownloadedBytes)
            assertTrue(progress.any { it.phase == "transfer_fallback_local_cut_unavailable" })
        }
    }

    @Test fun `real SHA rejects same length and timestamp corruption and GC preserves the missing proof`() = runBlocking {
        val root = root(); val first = fixture(); val old = publish(root, first)
        val damaged = old.resolve(first.manifest.weight.path); val time = damaged.lastModified()
        damaged.writeBytes(damaged.readBytes().also { it[0] = (it[0].toInt() xor 1).toByte() })
        assertTrue(damaged.setLastModified(time))
        val next = fixture(2); val requests = mutableListOf<String>(); val progress = mutableListOf<ModelSupplyProgress>()
        ModelSupplyReceiver(root, "127.0.0.1", client = client(next, requests)).receive(
            next.manifest, 1, "secret", { true }, { progress += it }, { it() })
        assertEquals(listOf(next.manifest.weight.path), requests)
        assertEquals(next.manifest.weight.size, progress.last().actualDownloadedBytes)
        select(root, next)
        assertEquals(0, ModelSupplyGc(root).collect(node).deleted)
        assertTrue(old.exists())
    }

    @Test fun `current native and pending resolution are retained until their actual owners release`() = runBlocking {
        val root = root(); val first = fixture(); val old = publish(root, first)
        val next = fixture(2); val selected = publish(root, next); select(root, next)
        val native = ModelSupplyStorage.artifact(old.resolve(first.manifest.weight.path))!!
        assertEquals(0, ModelSupplyGc(root).collect(node).deleted)
        native.close(); native.close()
        val pending = ModelSupplyStorage.pending(root)
        assertEquals(0, ModelSupplyGc(root).collect(node).deleted)
        pending.close()
        assertEquals(1, ModelSupplyGc(root).collect(node).deleted)
        assertTrue(selected.exists()); assertFalse(old.exists())
    }

    @Test fun `registration rollback debt protects the previous current until terminal completion`() = runBlocking {
        val root = root(); val first = fixture(); val old = publish(root, first); select(root, first)
        val next = fixture(2); publish(root, next)
        val current = ModelSupplyCurrent(root); val registration = current.select(next.manifest, binding)
        assertEquals(0, ModelSupplyGc(root).collect(node).deleted)
        current.rollback(registration, binding)
        assertEquals(first.manifest, current.read(binding))
        assertEquals(1, ModelSupplyGc(root).collect(node).deleted)
        assertTrue(old.exists())
    }

    @Test fun `GC never touches another operation receive journal partial or complete staged targets`() = runBlocking {
        val root = root(); val first = fixture()
        val stage = root.resolve(".receive-${first.manifest.manifestSha256}").apply { mkdir() }
        val partial = stage.resolve(first.manifest.weight.path + ".part").apply { writeBytes(byteArrayOf(1, 2, 3)) }
        val journal = stage.resolve(ModelSupplyReceiver.RESUME_MARKER).apply { writeText("original journal owner") }
        stage.resolve(first.manifest.native.path).writeBytes(first.files.getValue(first.manifest.native.path))
        publish(root, fixture(2))
        assertEquals(1, ModelSupplyGc(root).collect(node).deleted)
        assertArrayEquals(byteArrayOf(1, 2, 3), partial.readBytes())
        assertEquals("original journal owner", journal.readText())
        assertTrue(stage.resolve(first.manifest.native.path).exists())
    }

    @Test fun `no valid marker or exact inventory means no deletion`() = runBlocking {
        for (damage in listOf("missing", "digest", "unknown-file", "nested-directory", "foreign-node")) {
            val root = root(); val first = fixture(); val directory = publish(root, first)
            when (damage) {
                "missing" -> Files.delete(directory.resolve(ModelSupplyManifest.MARKER).toPath())
                "digest" -> directory.resolve(ModelSupplyManifest.MARKER).writeText("{}")
                "unknown-file" -> directory.resolve("unknown.gguf").writeBytes(byteArrayOf(1))
                "nested-directory" -> directory.resolve("nested").mkdir()
                "foreign-node" -> {
                    Files.delete(directory.resolve(ModelSupplyManifest.MARKER).toPath())
                    val foreign = first.manifest.json.deepCopy().apply { addProperty("node_id", "other") }
                    val manifest = sign(foreign)
                    val foreignDirectory = root.resolve("segment-${manifest.manifestSha256}")
                    Files.move(directory.toPath(), foreignDirectory.toPath())
                    foreignDirectory.resolve(ModelSupplyManifest.MARKER).writeText(canonicalSupplyJson(manifest.json))
                }
            }
            assertEquals(damage, 0, ModelSupplyGc(root).collect(node).deleted)
            assertEquals(damage, 1, root.listFiles()!!.size)
        }
    }

    @Test fun `invalid or redirected current pointer prevents GC for every candidate`() = runBlocking {
        val root = root(); val first = fixture(); val directory = publish(root, first)
        root.resolve(ModelSupplyCurrent.POINTER).writeText("{\"binding_sha256\":\"bad\",\"manifest\":{}}")
        assertEquals(0, ModelSupplyGc(root).collect(node).deleted)
        assertTrue(directory.exists())
    }

    @Test fun `scan deletion byte and elapsed budgets bound each GC pass`() = runBlocking {
        val root = root()
        for (i in 1..4) publish(root, fixture(i))
        val total = fixture().manifest.files.sumOf { it.size }
        val noBytes = ModelSupplyGc(root, ModelSupplyGc.Limits(verifyBytes = total - 1)).collect(node)
        assertEquals(0, noBytes.deleted); assertEquals(0L, noBytes.verifiedBytes)
        val scan = ModelSupplyGc(root, ModelSupplyGc.Limits(scanEntries = 1)).collect(node)
        assertEquals(1, scan.scanned); assertEquals(1, scan.deleted)
        val deletion = ModelSupplyGc(root, ModelSupplyGc.Limits(deleteSegments = 1, verifyBytes = total)).collect(node)
        assertEquals(1, deletion.deleted); assertEquals(total, deletion.verifiedBytes)
        var nanos = 0L
        val time = ModelSupplyGc(root, ModelSupplyGc.Limits(elapsedNanos = 1), nanoTime = { nanos++ }).collect(node)
        assertEquals(0, time.deleted)
        assertEquals(2, root.listFiles()!!.size)
    }

    @Test fun `a new native reference or changed current during verification fences deletion`() = runBlocking {
        for (kind in listOf("native", "current", "retired")) {
            val root = root(); val first = fixture(); val directory = publish(root, first)
            var lease: ModelSupplyStorage.Lease? = null
            try {
                val result = ModelSupplyGc(root).collect(node, commit = { action ->
                    when (kind) {
                        "native" -> lease = ModelSupplyStorage.artifact(directory.resolve(first.manifest.weight.path))
                        "current" -> select(root, first)
                        "retired" -> throw ModelSupplyException("SUPPLY_CANCELLED", "authority retired")
                    }
                    action()
                })
                assertEquals(kind, 0, result.deleted)
                assertTrue(kind, directory.exists())
            } finally { lease?.close() }
        }
    }

    @Test fun `symlink artifacts and redirected roots are rejected without touching their targets`() = runBlocking {
        val root = root(); val first = fixture(); val directory = publish(root, first)
        val external = temp.newFile("external.gguf").apply { writeBytes(first.files.getValue(first.manifest.weight.path)) }
        val weight = directory.resolve(first.manifest.weight.path)
        Files.delete(weight.toPath())
        try { Files.createSymbolicLink(weight.toPath(), external.toPath()) }
        catch (error: Exception) { assumeNoException("Symbolic links unavailable on this filesystem", error) }
        assertEquals(0, ModelSupplyGc(root).collect(node).deleted)
        assertTrue(external.exists()); assertTrue(directory.exists())
        try {
            ModelSupplyPaths.child(directory, first.manifest.weight.path)
            fail("reparse path accepted")
        } catch (error: ModelSupplyException) { assertEquals("SUPPLY_UNSAFE_PATH", error.reasonCode) }
    }

    @Test fun `an in flight receiver owns donors until it finishes publishing`() = runBlocking {
        val root = root(); val first = fixture(); val old = publish(root, first)
        val next = fixture(2); val requests = mutableListOf<String>()
        val directory = ModelSupplyReceiver(root, "127.0.0.1", client = client(next, requests)).receive(
            next.manifest, 1, "secret", { true }, {}, { publish ->
                assertEquals(0, ModelSupplyGc(root).collect(node).deleted)
                assertTrue(old.isDirectory)
                publish()
                assertEquals(0, ModelSupplyGc(root).collect(node).deleted)
            })
        select(root, next)
        assertEquals(1, ModelSupplyGc(root).collect(node).deleted)
        assertTrue(directory.exists()); assertFalse(old.exists()); assertTrue(requests.isEmpty())
    }

    @Test fun `new unproved files added while hashing cannot be deleted through the old inventory`() = runBlocking {
        val root = root(); val first = fixture(); val directory = publish(root, first)
        val result = ModelSupplyGc(root).collect(node, commit = { action ->
            directory.resolve("new-owner-data").writeBytes(byteArrayOf(7))
            action()
        })
        assertEquals(0, result.deleted)
        assertTrue(directory.resolve(ModelSupplyManifest.MARKER).isFile)
        assertArrayEquals(first.files.getValue(first.manifest.weight.path), directory.resolve(first.manifest.weight.path).readBytes())
        assertArrayEquals(byteArrayOf(7), directory.resolve("new-owner-data").readBytes())
    }

    @Test fun `private child paths reject traversal before any file access`() {
        val root = root()
        for (path in listOf("../outside", "../supply/../outside", "nested/file")) {
            try { ModelSupplyPaths.child(root, path); fail("private path escaped: $path") }
            catch (error: ModelSupplyException) { assertEquals("SUPPLY_UNSAFE_PATH", error.reasonCode) }
        }
    }

    @Test fun `Windows directory junctions cannot become GC proof or a redirected receive root`() = runBlocking {
        if (!System.getProperty("os.name").orEmpty().startsWith("Windows")) return@runBlocking
        // The local exec runner creates this fixture: JVM child process creation is restricted on this host.
        val fixtureRoot = System.getenv("QLH_ANDROID_JUNCTION_TEST_ROOT")
        org.junit.Assume.assumeTrue("Run with a native Windows junction fixture", fixtureRoot != null)
        val root = File(fixtureRoot!!)
        val junction = root.resolve("segment-${"0".repeat(64)}")
        val external = junction.canonicalFile
        val first = fixture()
        first.files.forEach { (path, bytes) -> external.resolve(path).writeBytes(bytes) }
        external.resolve(ModelSupplyManifest.MARKER).writeText(canonicalSupplyJson(first.manifest.json))
        assertEquals(0, ModelSupplyGc(root).collect(node).deleted)
        assertTrue(external.resolve(first.manifest.weight.path).isFile)
        assertTrue(junction.exists())
        for (path in listOf(junction, junction.resolve(first.manifest.weight.path))) {
            try { ModelSupplyPaths.check(path); fail("directory junction accepted") }
            catch (error: ModelSupplyException) { assertEquals("SUPPLY_UNSAFE_PATH", error.reasonCode) }
        }
        assertEquals(0, ModelSupplyGc(junction).collect(node).deleted)
        assertTrue(external.resolve(ModelSupplyManifest.MARKER).isFile)
    }

    @Test fun `controller completes selection and bounded GC only after its terminal reply`() = runBlocking {
        val root = root(); val first = fixture(); val old = publish(root, first); select(root, first)
        val next = fixture(2); val reply = CompletableDeferred<JsonObject>()
        var oldExistedAtReply = false
        val controller = ModelSupplyController(node, "127.0.0.1", root, { true }, { true }, { 1L }, { it() }, {},
            bindingSha256 = binding)
        val command = JsonObject().apply {
            addProperty("action", "ensure"); addProperty("operation_id", next.manifest.operationId); addProperty("node_id", node)
            addProperty("strategy", "distribute"); addProperty("api_port", 1); addProperty("token", "secret")
            add("manifest", next.manifest.json)
        }
        controller.command(command, this, { event ->
            if (event["action"].asString == "result") {
                assertEquals(0, ModelSupplyGc(root).collect(node).deleted)
                oldExistedAtReply = old.exists(); reply.complete(event)
            }
        }, {})
        val result = withTimeout(3000) { reply.await() }
        withTimeout(3000) { coroutineContext[kotlinx.coroutines.Job]!!.children.toList().forEach { it.join() } }
        assertEquals("available", result["status"].asString)
        assertEquals(0L, result["actual_downloaded_bytes"].asLong)
        assertEquals(result["total_bytes"], result["reused_bytes"])
        assertTrue(oldExistedAtReply); assertFalse(old.exists())
        assertEquals(1, scanSuppliedSegmentRoots(root).size)
        assertEquals(next.manifest, ModelSupplyCurrent(root).read(binding))
    }
}
