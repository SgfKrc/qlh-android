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
import java.nio.file.attribute.FileTime
import com.google.gson.JsonArray
import kotlinx.coroutines.Job
import kotlinx.coroutines.launch

class ModelSupplyPartialGcTest {
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

    private fun stage(root: File, f: Fixture, length: Int = 11): File =
        root.resolve(".receive-${f.manifest.manifestSha256}").apply {
            mkdir()
            val prefix = f.files.getValue(f.manifest.weight.path).copyOf(length)
            resolve(f.manifest.weight.path + ".part").writeBytes(prefix)
            val journal = JsonObject().apply {
                addProperty("schema_version", 1); add("manifest", f.manifest.json.deepCopy())
                add("prefixes", JsonArray().apply { add(JsonObject().apply {
                    addProperty("path", f.manifest.weight.path); addProperty("file_sha256", f.manifest.weight.sha256)
                    addProperty("size_bytes", f.manifest.weight.size); addProperty("prefix_bytes", length)
                    addProperty("prefix_sha256", supplySha256(prefix))
                }) })
                addProperty("journal_sha256", supplySha256(canonicalSupplyJson(this).toByteArray()))
            }
            resolve(ModelSupplyReceiver.RESUME_MARKER).writeText(canonicalSupplyJson(journal))
        }

    private fun client(f: Fixture, ranges: MutableList<String> = mutableListOf()) = OkHttpClient.Builder()
        .addInterceptor { chain ->
            val request = chain.request(); val bytes = f.files.getValue(request.url.pathSegments.last())
            val range = request.header("Range"); if (range != null) ranges += range
            val offset = range?.removePrefix("bytes=")?.removeSuffix("-")?.toInt() ?: 0
            Response.Builder().request(request).protocol(Protocol.HTTP_1_1)
                .code(if (range == null) 200 else 206).message("OK")
                .apply { if (range != null) header("Content-Range", "bytes $offset-${bytes.size - 1}/${bytes.size}") }
                .body(bytes.copyOfRange(offset, bytes.size).toResponseBody()).build()
        }.build()

    @Test fun `cancel A resume B complete verifies a copied prefix and collects A`() = runBlocking {
        val root = root(); val a = fixture(); var current = true
        try {
            ModelSupplyReceiver(root, "127.0.0.1", checkpointIntervalBytes = 11, client = client(a)).receive(
                a.manifest, 1, "secret", { current }, { if (it.phase == "downloading") current = false }, { it() })
            fail("A should retire")
        } catch (error: ModelSupplyException) { assertEquals("SUPPLY_CANCELLED", error.reasonCode) }
        val old = root.resolve(".receive-${a.manifest.manifestSha256}")
        val donor = old.resolve(a.manifest.weight.path + ".part")
        assertEquals(11L, donor.length()); assertNotNull(ModelSupplyReceiver.readResume(old))
        val b = fixture(2); val ranges = mutableListOf<String>(); val progress = mutableListOf<ModelSupplyProgress>()
        val complete = ModelSupplyReceiver(root, "127.0.0.1", client = client(b, ranges)).receive(
            b.manifest, 1, "secret", { true }, { progress += it }, { action ->
                val target = root.resolve(".receive-${b.manifest.manifestSha256}").resolve(b.manifest.weight.path)
                assertFalse("prefix is copied, not linked", Files.isSameFile(donor.toPath(), target.toPath()))
                assertEquals(0, ModelSupplyGc(root).collect(node, completedManifest = b.manifest).deletedPartials)
                action()
            })
        assertEquals(listOf("bytes=11-"), ranges); assertEquals(11L, progress.last().resumedBytes); assertTrue(old.exists())
        select(root, b)
        val result = ModelSupplyGc(root).collect(node, completedManifest = b.manifest, stillCurrent = {
            assertFalse("hash and live callbacks stay outside the storage lock", Thread.holdsLock(ModelSupplyStorage.lock))
            true
        })
        assertEquals(1, result.deletedPartials); assertEquals(1, result.deleted)
        assertTrue(result.reclaimedPartialBytes > 11); assertFalse(old.exists()); assertTrue(complete.exists())
        assertArrayEquals(b.files.getValue(b.manifest.weight.path), complete.resolve(b.manifest.weight.path).readBytes())
    }

    @Test fun `unfinished unselected absent or damaged completion proof retains A`() = runBlocking {
        for (state in listOf("unfinished", "unselected", "no-proof", "same-operation", "corrupt-complete")) {
            val root = root(); val a = fixture(); val old = stage(root, a); val b = fixture(2)
            when (state) {
                "unfinished" -> stage(root, b, 17)
                "unselected" -> publish(root, b)
                "same-operation" -> { publish(root, a); select(root, a) }
                else -> { val complete = publish(root, b); select(root, b)
                    if (state == "corrupt-complete") complete.resolve(b.manifest.weight.path).writeBytes(ByteArray(b.manifest.weight.size.toInt())) }
            }
            val proof = when (state) { "no-proof" -> null; "same-operation" -> a.manifest; else -> b.manifest }
            assertEquals(state, 0, ModelSupplyGc(root).collect(node, completedManifest = proof).deletedPartials)
            assertTrue(state, old.resolve(ModelSupplyReceiver.RESUME_MARKER).exists())
        }
    }

    @Test fun `source node role range geometry conversion proof and file identity must match`() = runBlocking {
        for (field in listOf("node", "source-id", "source-sha", "range", "role", "geometry", "proof", "file-sha")) {
            val root = root(); val a = fixture()
            val foreign = sign(a.manifest.json.deepCopy().apply {
                when (field) {
                    "node" -> addProperty("node_id", "other_node")
                    "source-id" -> addProperty("source_model_id", "other_source")
                    "source-sha" -> addProperty("source_model_sha256", "c".repeat(64))
                    "range" -> { add("layer_range", JsonParser.parseString("[1,3]")); addProperty("segment_mode", "middle")
                        add("layer_weight_bytes", JsonParser.parseString("[0,4,4,0]")) }
                    "role" -> addProperty("has_embedding", true)
                    "geometry" -> addProperty("hidden_size", 2)
                    "proof" -> add("conversion_proof", JsonObject().apply { addProperty("converter", "different") })
                    "file-sha" -> getAsJsonArray("files")[1].asJsonObject.addProperty("sha256", "f".repeat(64))
                }
            })
            val old = stage(root, Fixture(foreign, a.files)); val b = fixture(2); publish(root, b); select(root, b)
            assertEquals(field, 0, ModelSupplyGc(root).collect(node, completedManifest = b.manifest).deletedPartials)
            assertTrue(field, old.exists())
        }
    }

    @Test fun `receiver and donor pins retain A including a pin acquired after hashing`() = runBlocking {
        for (phase in listOf("receiver", "donor", "late-donor")) {
            val root = root(); val a = fixture(); val old = stage(root, a); val b = fixture(2); publish(root, b); select(root, b)
            var lease: ModelSupplyStorage.Lease? = null
            try {
                if (phase == "receiver") lease = ModelSupplyStorage.pending(root)
                if (phase == "donor") lease = ModelSupplyStorage.transfer(root, old)
                val result = ModelSupplyGc(root).collect(node, completedManifest = b.manifest, commit = { action ->
                    if (phase == "late-donor") lease = ModelSupplyStorage.transfer(root, old)
                    action()
                })
                assertEquals(phase, 0, result.deletedPartials); assertTrue(old.exists())
            } finally { lease?.close() }
            assertEquals(1, ModelSupplyGc(root).collect(node, completedManifest = b.manifest).deletedPartials)
        }
    }

    @Test fun `unknown files malformed journal invalid checkpoint and unproved suffix are retained`() = runBlocking {
        for (damage in listOf("unknown", "nested", "bad-journal", "bad-prefix", "signed-wrong-prefix", "no-checkpoint", "suffix")) {
            val root = root(); val a = fixture(); val old = stage(root, a); val b = fixture(2); publish(root, b); select(root, b)
            when (damage) {
                "unknown" -> old.resolve("unknown-data").writeText("keep")
                "nested" -> old.resolve("nested").mkdir()
                "bad-journal" -> old.resolve(ModelSupplyReceiver.RESUME_MARKER).writeText("{}")
                "bad-prefix" -> old.resolve(a.manifest.weight.path + ".part").writeBytes(ByteArray(11))
                "signed-wrong-prefix" -> stage(root, Fixture(a.manifest,
                    a.files + (a.manifest.weight.path to ByteArray(a.manifest.weight.size.toInt()))))
                "no-checkpoint" -> old.resolve(a.manifest.native.path + ".part").writeBytes(byteArrayOf(7))
                "suffix" -> old.resolve(a.manifest.weight.path + ".part").appendBytes(byteArrayOf(7))
            }
            assertEquals(damage, 0, ModelSupplyGc(root).collect(node, completedManifest = b.manifest).deletedPartials)
            assertTrue(damage, old.resolve(ModelSupplyReceiver.RESUME_MARKER).exists())
        }
    }

    @Test fun `mutation of partial journal covering copy inventory directory or authority fences deletion`() = runBlocking {
        for (damage in listOf("partial", "journal", "cover", "cover-marker", "inventory", "directory", "selection", "authority")) {
            val root = root(); val a = fixture(); val old = stage(root, a); val b = fixture(2); val complete = publish(root, b); select(root, b)
            val result = ModelSupplyGc(root).collect(node, completedManifest = b.manifest, commit = { action ->
                when (damage) {
                    "partial" -> { val file = old.resolve(a.manifest.weight.path + ".part"); file.writeBytes(ByteArray(11))
                        Files.setLastModifiedTime(file.toPath(), FileTime.fromMillis(1000)) }
                    "journal" -> old.resolve(ModelSupplyReceiver.RESUME_MARKER).writeText("{}")
                    "cover" -> { val file = complete.resolve(b.manifest.weight.path); file.writeBytes(ByteArray(b.manifest.weight.size.toInt()))
                        Files.setLastModifiedTime(file.toPath(), FileTime.fromMillis(1000)) }
                    "cover-marker" -> complete.resolve(ModelSupplyManifest.MARKER).writeText("{}")
                    "inventory" -> old.resolve("arrived-after-hash").writeText("keep")
                    "directory" -> { Files.move(old.toPath(), root.resolve("retired-owner-backup").toPath()); old.mkdir() }
                    "selection" -> { val c = fixture(3); publish(root, c); select(root, c) }
                    "authority" -> throw ModelSupplyException("SUPPLY_CANCELLED", "retired")
                }
                action()
            })
            assertEquals(damage, 0, result.deletedPartials); assertTrue(damage, old.exists())
        }
    }

    @Test fun `job cancellation inside commit is checked again before deleting`() = runBlocking {
        val root = root(); val a = fixture(); val old = stage(root, a); val b = fixture(2); publish(root, b); select(root, b)
        val job = launch {
            ModelSupplyGc(root).collect(node, completedManifest = b.manifest, commit = { action ->
                coroutineContext[Job]!!.cancel(); action()
            })
        }
        job.join(); assertTrue(old.exists()); assertTrue(job.isCancelled)
    }

    @Test fun `partial symlink cannot authorize deletion or touch its external target`() = runBlocking {
        val root = root(); val a = fixture(); val old = stage(root, a); val b = fixture(2); publish(root, b); select(root, b)
        val part = old.resolve(a.manifest.weight.path + ".part")
        val external = temp.newFile().apply { writeBytes(part.readBytes()) }; Files.delete(part.toPath())
        try { Files.createSymbolicLink(part.toPath(), external.toPath()) }
        catch (error: Exception) { assumeNoException("Symbolic links unavailable", error) }
        assertEquals(0, ModelSupplyGc(root).collect(node, completedManifest = b.manifest).deletedPartials)
        assertTrue(old.exists()); assertEquals(11L, external.length())
    }

    @Test fun `Windows junction receive directory is rejected with valid external journal untouched`() = runBlocking {
        if (!System.getProperty("os.name").orEmpty().startsWith("Windows")) return@runBlocking
        val fixtureRoot = System.getenv("QLH_ANDROID_PARTIAL_JUNCTION_TEST_ROOT")
        org.junit.Assume.assumeTrue("Run with a native Windows partial junction fixture", fixtureRoot != null)
        val root = File(fixtureRoot!!); val a = fixture(); val b = fixture(2)
        val initial = root.resolve(".receive-${"0".repeat(64)}")
        val junction = root.resolve(".receive-${a.manifest.manifestSha256}")
        if (initial.exists()) Files.move(initial.toPath(), junction.toPath())
        val old = stage(root, a); publish(root, b); select(root, b)
        // canonicalFile does not resolve Windows junctions; the native fixture supplies a sibling target.
        val external = root.parentFile.resolve("partials-external")
        assertEquals(0, ModelSupplyGc(root).collect(node, completedManifest = b.manifest).deletedPartials)
        assertTrue(old.exists()); assertTrue(external.resolve(ModelSupplyReceiver.RESUME_MARKER).exists())
        assertEquals(11L, external.resolve(a.manifest.weight.path + ".part").length())
        try { ModelSupplyPaths.check(junction); fail("junction accepted") }
        catch (error: ModelSupplyException) { assertEquals("SUPPLY_UNSAFE_PATH", error.reasonCode) }
    }

    @Test fun `partial scan byte time and deletion work use shared finite budgets`() = runBlocking {
        val root = root(); val b = fixture(20); publish(root, b); select(root, b)
        for (i in 1..4) stage(root, fixture(i))
        val bytes = b.manifest.files.sumOf { it.size }
        assertEquals(0, ModelSupplyGc(root, ModelSupplyGc.Limits(verifyBytes = bytes - 1)).collect(node, completedManifest = b.manifest).deletedPartials)
        assertEquals(0, ModelSupplyGc(root, ModelSupplyGc.Limits(verifyBytes = bytes + 21)).collect(node, completedManifest = b.manifest).deletedPartials)
        assertEquals(1, ModelSupplyGc(root, ModelSupplyGc.Limits(deleteSegments = 1)).collect(node, completedManifest = b.manifest).deletedPartials)
        assertEquals(1, ModelSupplyGc(root, ModelSupplyGc.Limits(scanEntries = 1)).collect(node, completedManifest = b.manifest).scanned)
        var nanos = 0L
        assertEquals(0, ModelSupplyGc(root, ModelSupplyGc.Limits(elapsedNanos = 1), nanoTime = { nanos++ }).collect(node, completedManifest = b.manifest).deletedPartials)
    }

    @Test fun `controller supplies completion proof only after terminal reply`() = runBlocking {
        val root = root(); val a = fixture(); val old = stage(root, a)
        val donor = publish(root, fixture(3)); val b = fixture(2); val terminal = CompletableDeferred<Unit>()
        val controller = ModelSupplyController(node, "127.0.0.1", root, { true }, { true }, { 1L }, { it() }, {}, bindingSha256 = binding)
        controller.command(JsonObject().apply {
            addProperty("action", "ensure"); addProperty("operation_id", b.manifest.operationId); addProperty("node_id", node)
            addProperty("strategy", "distribute"); addProperty("api_port", 1); addProperty("token", "secret"); add("manifest", b.manifest.json)
        }, this, { event ->
            if (event["action"].asString == "result") {
                assertEquals("available", event["status"].asString); assertTrue(old.exists()); assertTrue(donor.exists()); terminal.complete(Unit)
            }
        }, {})
        withTimeout(3000) { terminal.await(); coroutineContext[Job]!!.children.toList().forEach { it.join() } }
        assertFalse(old.exists()); assertFalse(donor.exists()); assertEquals(b.manifest, ModelSupplyCurrent(root).read(binding))
    }
}
