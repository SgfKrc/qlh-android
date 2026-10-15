package com.qlh.inference.service

import com.google.gson.JsonObject
import com.google.gson.JsonParser
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.Job
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.async
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withTimeout
import kotlinx.coroutines.withContext
import okhttp3.MediaType
import okhttp3.OkHttpClient
import okhttp3.Protocol
import okhttp3.Response
import okhttp3.ResponseBody
import okio.Buffer
import okio.BufferedSource
import okio.Source
import okio.Timeout
import okio.buffer
import org.junit.Assert.*
import org.junit.Test
import org.junit.Rule
import org.junit.rules.TemporaryFolder
import java.io.ByteArrayOutputStream
import java.io.DataInputStream
import java.io.DataOutputStream
import java.io.File
import java.io.IOException
import java.net.SocketTimeoutException
import java.util.concurrent.atomic.AtomicReference
import java.net.ServerSocket
import java.security.MessageDigest
import kotlin.concurrent.thread

class ModelSupplyContractTest {
    @get:Rule val temp = TemporaryFolder()
    private val operation = "supply_" + "a".repeat(32)
    private val node = "android_12345678"

    private data class Fixture(val manifest: ModelSupplyManifest, val files: Map<String, ByteArray>)
    private fun fixture(
        extraWeightBytes: Int = 0,
        headerMetadata: JsonObject = JsonParser.parseString("""{
            "architecture":"test","block_count":2,"head_count_kv":null,
            "key_length":null,"full_attention_interval":null
        }""").asJsonObject,
    ): Fixture {
        // A header-only GGUF fixture: reader consumes no tensor payload.
        val buffer = ByteArrayOutputStream()
        DataOutputStream(buffer).use { out ->
            fun int(value: Int) = out.writeInt(Integer.reverseBytes(value))
            fun long(value: Long) = out.writeLong(java.lang.Long.reverseBytes(value))
            fun string(value: String) { val b=value.toByteArray(); long(b.size.toLong()); out.write(b) }
            val fields = headerMetadata.entrySet().filter { it.key != "architecture" && !it.value.isJsonNull }
            out.write("GGUF".toByteArray()); int(3); long(2); long(2L + fields.size)
            string("tokenizer.ggml.model"); int(8); string("test")
            string("general.architecture"); int(8); string(headerMetadata["architecture"].asString)
            val suffixes = mapOf("block_count" to "block_count", "head_count_kv" to "attention.head_count_kv",
                "key_length" to "attention.key_length", "full_attention_interval" to "full_attention_interval")
            for ((key, value) in fields) {
                string("${headerMetadata["architecture"].asString}.${suffixes.getValue(key)}")
                if (value.isJsonArray) {
                    int(9); int(4); long(value.asJsonArray.size().toLong())
                    value.asJsonArray.forEach { int(it.asInt) }
                } else { int(4); int(value.asInt) }
            }
            for (index in 0..1) { string("blk.$index.attn.weight"); int(1); long(1); int(0); long(index * 4L) }
        }
        val header = buffer.toByteArray()
        val weight = header.copyOf(header.size + extraWeightBytes)
        val scan = com.qlh.inference.worker.GgufLayerWeightReader.scan(weight.inputStream(), 2)!!
        val source = "b".repeat(64)
        val native = JsonParser.parseString("""{
            "artifact":"segment.gguf","artifact_sha256":"${supplySha256(weight)}",
            "source_model_id":"source","source_model_sha256":"$source",
            "tokenizer_sha256":"${scan.tokenizerSha256}","hidden_size":1,"mode":"tail",
            "source_layer_range":[2,4],"n_layer":4,"kept_block_count":2,
            "gguf_metadata":{"architecture":"test","block_count":2,"head_count_kv":null,
                "key_length":null,"full_attention_interval":null},
            "conversion_proof":null,"layer_weight_bytes":[0,0,4,4]
        }""").asJsonObject
        native.add("gguf_metadata", headerMetadata.deepCopy())
        val nativeBytes = canonicalSupplyJson(native).toByteArray()
        val manifest = JsonParser.parseString("""{
            "manifest_kind":"qlh.segment_supply.v1","operation_id":"$operation","node_id":"$node",
            "source_model_id":"source","source_model_sha256":"$source","tokenizer_sha256":"${scan.tokenizerSha256}",
            "total_layers":4,"layer_range":[2,4],"segment_mode":"tail","layer_weight_bytes":[0,0,4,4],
            "hidden_size":1,"artifact_sha256":"${supplySha256(weight)}","format":"gguf","conversion_proof":null,
            "files":[{"path":"segment.gguf","kind":"weights","size_bytes":${weight.size},"sha256":"${supplySha256(weight)}"},
                {"path":"segment.manifest.json","kind":"manifest","size_bytes":${nativeBytes.size},"sha256":"${supplySha256(nativeBytes)}"}]
        }""").asJsonObject
        resign(manifest)
        return Fixture(ModelSupplyManifest.parse(manifest), mapOf("segment.gguf" to weight, "segment.manifest.json" to nativeBytes))
    }
    private fun resign(j: JsonObject) {
        j.remove("manifest_sha256")
        j.addProperty("manifest_sha256", supplySha256(canonicalSupplyJson(j).toByteArray()))
    }
    private fun expect(code: String, body: () -> Unit) {
        try { body(); fail("expected $code") } catch (error: ModelSupplyException) { assertEquals(code, error.reasonCode) }
    }

    @Test fun `canonical bytes match ASCII sorted Python wire including proof Unicode`() {
        val json=JsonParser.parseString("""{"z":"中文😀\u007f","a":{"target":{"path":"../segment.gguf"}},"n":1.0}""")
        assertEquals("{\"a\":{\"target\":{\"path\":\"../segment.gguf\"}},\"n\":1.0,\"z\":\"\\u4e2d\\u6587\\ud83d\\ude00\\u007f\"}", canonicalSupplyJson(json))
        val manifest=fixture().manifest.json.deepCopy()
        manifest.add("conversion_proof", json)
        resign(manifest)
        assertEquals(json, ModelSupplyManifest.parse(manifest).json["conversion_proof"])
    }

    @Test fun `global bytes project to local span and extra file fields stay in digest`() {
        val j=fixture().manifest.json.deepCopy()
        j["files"].asJsonArray[0].asJsonObject.addProperty("transfer_hint", "opaque")
        resign(j)
        assertEquals(listOf(4L,4L), ModelSupplyManifest.parse(j).localLayerBytes)
        j.addProperty("schema_version", "invented")
        expect("SUPPLY_INVALID_MANIFEST") { ModelSupplyManifest.parse(j) }
    }

    @Test fun `path traversal full model and digest mismatch are refused`() {
        for (path in listOf("../file.gguf","/file.gguf","C:\\file.gguf","a/b.gguf",".file","file.part","supply.manifest.json","CON.gguf")) {
            expect("SUPPLY_UNSAFE_PATH") { ModelSupplyManifest.safeBasename(path) }
        }
        val j=fixture().manifest.json.deepCopy()
        j.add("layer_range", JsonParser.parseString("[0,4]"))
        expect("SUPPLY_FULL_MODEL_FORBIDDEN") { ModelSupplyManifest.parse(j) }
        val altered=fixture().manifest.json.deepCopy().apply { addProperty("hidden_size", 2) }
        expect("SUPPLY_MANIFEST_DIGEST_MISMATCH") { ModelSupplyManifest.parse(altered) }
    }

    @Test fun `resume is bound to master and atomically registers catalog without selecting or loading`() = runBlocking {
        val f=fixture(); val root=temp.newFolder("supply")
        val stage = interruptedDownload(root, f).stage
        val weight=f.files.getValue("segment.gguf")
        val requests=mutableListOf<String>()
        val server=ServerSocket(0)
        var error: Throwable?=null
        val worker=thread {
            try {
                repeat(2) { server.accept().use { socket ->
                    val input=socket.getInputStream().bufferedReader()
                    val line=input.readLine(); val headers=mutableMapOf<String,String>()
                    while(true) { val h=input.readLine(); if(h.isEmpty()) break; headers[h.substringBefore(':').lowercase()]=h.substringAfter(':').trim() }
                    requests += line
                    assertEquals("secret",headers["x-qlh-supply-token"])
                    val name=line.substringBefore(" HTTP").substringAfterLast('/')
                    val bytes=f.files.getValue(name)
                    val offset=if(name=="segment.gguf") 11 else 0
                    if(offset>0) assertEquals("bytes=11-",headers["range"])
                    val head=if(offset>0) "HTTP/1.1 206 Partial Content\r\nContent-Range: bytes $offset-${bytes.size-1}/${bytes.size}\r\n" else "HTTP/1.1 200 OK\r\n"
                    val out=socket.getOutputStream()
                    out.write((head+"Content-Length: ${bytes.size-offset}\r\nConnection: close\r\n\r\n").toByteArray())
                    out.write(bytes,offset,bytes.size-offset);out.flush()
                } }
            } catch(t:Throwable) { error=t }
        }
        try {
            val receiver=ModelSupplyReceiver(root,"127.0.0.1")
            val final=receiver.receive(f.manifest,server.localPort,"secret",{true},{_ ->},{ it() })
            assertEquals("segment-${f.manifest.manifestSha256}",final.name)
            assertTrue(final.resolve(ModelSupplyManifest.MARKER).isFile)
            assertEquals(1,scanSuppliedSegmentRoots(root).size)
            assertEquals(2,requests.size)
            assertTrue(requests.all { it.contains("/api/models/supply/$operation/files/$node/") })
            assertFalse(stage.exists())
            receiver.receive(f.manifest,server.localPort,"secret",{true},{_ ->},{ it() })
            assertEquals(2,requests.size)
        } finally { server.close();worker.join(3000) }
        assertNull(error)
    }

    @Test fun `redirect refusal digest failure and stale publish never advertise partial directories`() = runBlocking {
        val f=fixture()
        for (mode in listOf("redirect","range","digest","stale")) {
            val root=temp.newFolder(mode)
            val server=ServerSocket(0)
            val worker=thread {
                try { repeat(if(mode=="stale") 2 else 1) { server.accept().use { socket ->
                    val input=socket.getInputStream().bufferedReader();val request=input.readLine()
                    while(input.readLine().isNotEmpty()) Unit
                    val name=request.substringBefore(" HTTP").substringAfterLast('/')
                    val bytes=f.files.getValue(name).copyOf()
                    if(mode=="digest") bytes[0]=0
                    val head=when(mode) {
                        "redirect" -> "HTTP/1.1 302 Found\r\nLocation: http://other.invalid/artifact\r\nContent-Length: 0\r\n"
                        "range" -> "HTTP/1.1 206 Partial Content\r\nContent-Range: bytes 1-${bytes.size-1}/${bytes.size}\r\nContent-Length: ${bytes.size}\r\n"
                        else -> "HTTP/1.1 200 OK\r\nContent-Length: ${bytes.size}\r\n"
                    }
                    socket.getOutputStream().apply { write((head+"Connection: close\r\n\r\n").toByteArray()); if(mode!="redirect") write(bytes);flush() }
                } } } catch(_:Exception) { }
            }
            try {
                try {
                    ModelSupplyReceiver(root,"127.0.0.1").receive(f.manifest,server.localPort,"secret",{true},{_ ->},{ action ->
                        if(mode=="stale") throw ModelSupplyException("SUPPLY_CANCELLED","retired") else action()
                    })
                    fail("expected $mode failure")
                } catch(error:ModelSupplyException) {
                    assertEquals(mapOf("redirect" to "SUPPLY_REDIRECT_FORBIDDEN","range" to "SUPPLY_RANGE_INVALID", "digest" to "SUPPLY_ARTIFACT_DIGEST_MISMATCH", "stale" to "SUPPLY_CANCELLED")[mode],error.reasonCode)
                }
                assertTrue(scanSuppliedSegmentRoots(root).isEmpty())
            } finally { server.close();worker.join(3000) }
        }
    }

    @Test fun `cancel closes a blocked HTTP call and prevents publication`() = runBlocking {
        val f=fixture();val root=temp.newFolder("cancel");val server=ServerSocket(0)
        val accepted=CompletableDeferred<Unit>();val release=CompletableDeferred<Unit>()
        val serverJob=async(Dispatchers.IO) { server.accept().use { socket ->
            accepted.complete(Unit);release.await()
        } }
        val receiver=ModelSupplyReceiver(root,"127.0.0.1")
        val receiveJob=async(Dispatchers.IO) {
            runCatching { receiver.receive(f.manifest,server.localPort,"secret",{true},{_ ->},{ it() }) }
        }
        try {
            withTimeout(3000) { accepted.await() }
            receiver.cancel()
            val result=withTimeout(3000) { receiveJob.await() }
            assertTrue(result.isFailure)
            assertTrue(scanSuppliedSegmentRoots(root).isEmpty())
        } finally { release.complete(Unit);server.close();serverJob.await() }
    }
    /** Drive the real HTTP-body/write/journal path with deterministic 4 KiB reads. */
    private fun shortReadClient(
        f: Fixture,
        failAfter: Int? = null,
        afterRead: (String, Int) -> Unit = { _, _ -> },
    ): OkHttpClient = OkHttpClient.Builder().addInterceptor { chain ->
        val path = chain.request().url.pathSegments.last()
        val bytes = f.files.getValue(path)
        val source = object : Source {
            var position = 0
            override fun read(sink: Buffer, byteCount: Long): Long {
                val limit = if (path == f.manifest.weight.path) failAfter ?: bytes.size else bytes.size
                if (position == limit && limit < bytes.size) throw IOException("short-read disconnect")
                if (position == bytes.size) return -1
                val count = minOf(byteCount, 4096L, (limit - position).toLong()).toInt()
                sink.write(bytes, position, count)
                position += count
                afterRead(path, position)
                return count.toLong()
            }
            override fun timeout(): Timeout = Timeout.NONE
            override fun close() = Unit
        }.buffer()
        Response.Builder().request(chain.request()).protocol(Protocol.HTTP_1_1).code(200).message("OK")
            .body(object : ResponseBody() {
                override fun contentType(): MediaType? = null
                override fun contentLength(): Long = bytes.size.toLong()
                override fun source(): BufferedSource = source
            }).build()
    }.build()

    @Test fun `short reads aggregate live checks and progress by bounded written chunks`() = runBlocking {
        val chunk = 256 * 1024
        val f = fixture(extraWeightBytes = 512 * 1024)
        val root = temp.newFolder("short-reads")
        val checks = mutableListOf<Int>()
        val progress = mutableListOf<ModelSupplyProgress>()
        var weightRead = 0
        var reads = 0
        val client = shortReadClient(f, afterRead = { path, position ->
            reads++
            if (path == f.manifest.weight.path) weightRead = position
        })
        val final = ModelSupplyReceiver(root, "127.0.0.1", client = client).receive(
            f.manifest, 1, "secret", { checks += weightRead; true }, { progress += it }, { it() })
        val weightSize = f.files.getValue(f.manifest.weight.path).size
        assertTrue("must exercise many underlying short reads", reads > 128)
        assertTrue("live checks must scale with chunks", checks.size < 24)
        assertTrue("no participation read between short reads", checks.all { it % chunk == 0 || it == weightSize })
        val written = progress.filter { it.phase == "downloading" }
        val total = f.manifest.files.sumOf { it.size }
        val expectedChunks = f.manifest.files.sumOf { (it.size + chunk - 1) / chunk }.toInt()
        assertEquals(expectedChunks, written.size)
        var previous = 0L
        for (snapshot in written) {
            assertTrue(snapshot.downloadedBytes - previous in 1L..chunk.toLong())
            assertEquals(snapshot.downloadedBytes, snapshot.actualDownloadedBytes)
            previous = snapshot.downloadedBytes
        }
        assertEquals(total, progress.last().actualDownloadedBytes)
        f.files.forEach { (path, bytes) -> assertArrayEquals(bytes, final.resolve(path).readBytes()) }
    }

    @Test fun `binding change between short reads reports discarded network bytes and resumes only written bytes`() = runBlocking {
        val chunk = 256 * 1024
        for (committed in listOf(0, chunk)) {
            val f = fixture(extraWeightBytes = 512 * 1024)
            val root = temp.newFolder("short-read-binding-$committed")
            val progress = mutableListOf<ModelSupplyProgress>()
            val diagnostics = mutableListOf<ModelSupplyTransferDiagnostic>()
            var current = true
            var weightRead = 0
            val client = shortReadClient(f, afterRead = { path, position ->
                if (path == f.manifest.weight.path) {
                    weightRead = position
                    if (position == committed + 4096) current = false
                }
            })
            val failure = runCatching {
                ModelSupplyReceiver(root, "127.0.0.1", client = client).receive(f.manifest, 1, "secret",
                    { current }, { progress += it }, { fail("stale chunk published") }, diagnostics = { diagnostics += it })
            }.exceptionOrNull()
            assertTrue(failure is ModelSupplyException)
            assertEquals("SUPPLY_CANCELLED", (failure as ModelSupplyException).reasonCode)
            assertEquals("read aggregation must remain bounded", committed + chunk, weightRead)
            val stage = root.resolve(".receive-${f.manifest.manifestSha256}")
            assertEquals(committed.toLong(), stage.resolve(f.manifest.weight.path + ".part").length())
            assertEquals(committed.toLong(), progress.last().downloadedBytes)
            assertEquals(weightRead.toLong(), progress.last().actualDownloadedBytes)
            assertEquals(weightRead.toLong(), diagnostics.sumOf { it.actualNewBytes })
            assertTrue(diagnostics.last().terminal)
            if (committed > 0) checkpoint(root, f, committed.toLong())
            else assertFalse(stage.resolve(ModelSupplyReceiver.RESUME_MARKER).exists())
            assertTrue(scanSuppliedSegmentRoots(root).isEmpty())
            val next = Fixture(nextOperation(f), f.files)
            val (_, resumed) = downloaded(root, next, committed.toLong())
            assertEquals(committed.toLong(), resumed.resumedBytes)
            assertEquals(next.manifest.files.sumOf { it.size } - committed, resumed.actualDownloadedBytes)
        }
    }

    @Test fun `IO failure within an aggregated chunk preserves and resumes its safely written short tail`() = runBlocking {
        val prefixBytes = 256 * 1024 + 11
        val f = fixture(extraWeightBytes = 512 * 1024)
        val root = temp.newFolder("short-read-disconnect")
        val progress = mutableListOf<ModelSupplyProgress>()
        val failure = runCatching {
            ModelSupplyReceiver(root, "127.0.0.1", client = shortReadClient(f, failAfter = prefixBytes))
                .receive(f.manifest, 1, "secret", { true }, { progress += it }, { fail("disconnect published") })
        }.exceptionOrNull()
        assertTrue(failure is IOException)
        assertEquals("short-read disconnect", failure!!.message)
        checkpoint(root, f, prefixBytes.toLong())
        assertEquals(prefixBytes.toLong(), progress.last().actualDownloadedBytes)
        assertEquals(prefixBytes.toLong(), progress.last().downloadedBytes)
        val next = Fixture(nextOperation(f), f.files)
        val (_, resumed) = downloaded(root, next, prefixBytes.toLong())
        assertEquals(prefixBytes.toLong(), resumed.resumedBytes)
        assertEquals(next.manifest.files.sumOf { it.size } - prefixBytes, resumed.actualDownloadedBytes)
    }

    @Test fun `coroutine cancellation between short reads stops before writing and reports all consumed network bytes`() = runBlocking {
        val f = fixture(extraWeightBytes = 512 * 1024)
        val root = temp.newFolder("short-read-cancellation")
        val transferJob = Job()
        val progress = mutableListOf<ModelSupplyProgress>()
        var reads = 0
        val client = shortReadClient(f, afterRead = { _, _ ->
            if (++reads == 2) transferJob.cancel()
        })
        try {
            val failure = runCatching {
                withContext(transferJob) {
                    ModelSupplyReceiver(root, "127.0.0.1", client = client).receive(f.manifest, 1, "secret",
                        { true }, { progress += it }, { fail("cancelled chunk published") })
                }
            }.exceptionOrNull()
            assertTrue(failure is CancellationException)
            assertEquals("ensureActive must run between underlying reads", 2, reads)
            val stage = root.resolve(".receive-${f.manifest.manifestSha256}")
            assertEquals(0L, stage.resolve(f.manifest.weight.path + ".part").length())
            assertEquals(0L, progress.last().downloadedBytes)
            assertEquals(8192L, progress.last().actualDownloadedBytes)
            assertFalse(stage.resolve(ModelSupplyReceiver.RESUME_MARKER).exists())
            assertTrue(scanSuppliedSegmentRoots(root).isEmpty())
        } finally { transferJob.cancel() }
    }

    private fun publishFixture(root: java.io.File, fixture: Fixture): java.io.File {
        val directory = root.resolve("segment-${fixture.manifest.manifestSha256}").apply { mkdir() }
        fixture.files.forEach { (path, bytes) -> directory.resolve(path).writeBytes(bytes) }
        directory.resolve(ModelSupplyManifest.MARKER).writeText(canonicalSupplyJson(fixture.manifest.json))
        return directory
    }
    private fun nextOperation(fixture: Fixture): ModelSupplyManifest {
        val next = fixture.manifest.json.deepCopy().apply { addProperty("operation_id", "supply_" + "e".repeat(32)) }
        resign(next)
        return ModelSupplyManifest.parse(next)
    }

    @Test fun `new operation strongly reuses private bytes and publishes its own marker without HTTP`() = runBlocking {
        val f = fixture(); val root = temp.newFolder("reuse")
        val old = publishFixture(root, f); val next = nextOperation(f)
        val progress = mutableListOf<ModelSupplyProgress>()
        val final = ModelSupplyReceiver(root, "127.0.0.1").receive(next, 1, "secret", { true }, { progress += it }, { it() })
        assertTrue(old.isDirectory)
        assertNotEquals(old, final)
        assertEquals(next.operationId, ModelSupplyManifest.parse(final.resolve(ModelSupplyManifest.MARKER).readText()).operationId)
        assertEquals(2, scanSuppliedSegmentRoots(root).size)
        val last = progress.last()
        assertEquals(last.totalBytes, last.downloadedBytes)
        assertEquals(last.totalBytes, last.reusedBytes)
        assertEquals(0L, last.actualDownloadedBytes)
        assertEquals(0L, last.resumedBytes)
    }

    @Test fun `reuse checks live SHA even when size and mtime are unchanged and redownloads only damaged bytes`() = runBlocking {
        val f = fixture(); val root = temp.newFolder("corrupt-reuse")
        val old = publishFixture(root, f); val next = nextOperation(f)
        val damaged = old.resolve("segment.gguf"); val time = damaged.lastModified()
        val corruption = damaged.readBytes().also { it[0] = 0 }
        damaged.writeBytes(corruption); damaged.setLastModified(time)
        val server = ServerSocket(0)
        val requests = mutableListOf<String>(); var error: Throwable? = null
        val peer = thread {
            try { server.accept().use { socket ->
                val input = socket.getInputStream().bufferedReader()
                requests += input.readLine()
                while (input.readLine().isNotEmpty()) Unit
                val bytes = f.files.getValue("segment.gguf")
                socket.getOutputStream().apply {
                    write(("HTTP/1.1 200 OK\r\nContent-Length: ${bytes.size}\r\nConnection: close\r\n\r\n").toByteArray())
                    write(bytes); flush()
                }
            } } catch (t: Throwable) { error = t }
        }
        val progress = mutableListOf<ModelSupplyProgress>()
        try {
            val final = ModelSupplyReceiver(root, "127.0.0.1").receive(next, server.localPort, "secret", { true }, { progress += it }, { it() })
            assertArrayEquals(f.files.getValue("segment.gguf"), final.resolve("segment.gguf").readBytes())
            assertArrayEquals(corruption, damaged.readBytes())
            assertEquals(1, requests.size)
            assertTrue(requests.single().contains("/${next.operationId}/files/$node/segment.gguf"))
            assertEquals(f.files.getValue("segment.gguf").size.toLong(), progress.last().actualDownloadedBytes)
            assertEquals(f.files.getValue("segment.manifest.json").size.toLong(), progress.last().reusedBytes)
        } finally { server.close(); peer.join(3000) }
        assertNull(error)
    }

    private fun revisedFixture(f: Fixture, update: (JsonObject, JsonObject) -> Unit): Fixture {
        val manifest = f.manifest.json.deepCopy()
        val native = JsonParser.parseString(f.files.getValue(f.manifest.native.path).toString(Charsets.UTF_8)).asJsonObject
        update(manifest, native)
        val nativeBytes = canonicalSupplyJson(native).toByteArray(Charsets.UTF_8)
        manifest["files"].asJsonArray.single { it.asJsonObject["path"].asString == f.manifest.native.path }.asJsonObject.apply {
            addProperty("size_bytes", nativeBytes.size); addProperty("sha256", supplySha256(nativeBytes))
        }
        resign(manifest)
        return Fixture(ModelSupplyManifest.parse(manifest), f.files + (f.manifest.native.path to nativeBytes))
    }

    @Test fun `receipt checks each actual header value for matching scalars and per layer arrays`() = runBlocking {
        for (array in listOf(false, true)) {
            val values = JsonParser.parseString(if (array) """{
                "architecture":"test","block_count":2,"head_count_kv":[2,4],
                "key_length":[128,256],"full_attention_interval":[4,8]
            }""" else """{
                "architecture":"test","block_count":2,"head_count_kv":2,
                "key_length":128,"full_attention_interval":4
            }""").asJsonObject
            val good = revisedFixture(fixture(headerMetadata = values)) { _, native ->
                native.addProperty("block_count", 4)
                native.addProperty("nextn_predict_layers", 1)
                native["gguf_metadata"].asJsonObject.addProperty("nextn_predict_layers", 0)
            }
            val root = temp.newFolder("metadata-matching-$array")
            val final = ModelSupplyReceiver(root, "127.0.0.1", client = shortReadClient(good)).receive(
                good.manifest, 1, "secret", { true }, {}, { it() })
            ModelSupplyReceiver(root, "unused").validateFiles(final, good.manifest)
            for (key in listOf("block_count", "head_count_kv", "key_length", "full_attention_interval")) {
                val bad = revisedFixture(good) { _, native -> native["gguf_metadata"].asJsonObject.addProperty(key, 99) }
                val badRoot = temp.newFolder("metadata-mismatch-$array-$key")
                val failure = runCatching {
                    ModelSupplyReceiver(badRoot, "127.0.0.1", client = shortReadClient(bad)).receive(
                        bad.manifest, 1, "secret", { true }, {}, { fail("header mismatch published: $key") })
                }.exceptionOrNull()
                assertTrue(key, failure is ModelSupplyException)
                assertEquals(key, "SUPPLY_GGUF_METADATA_MISMATCH", (failure as ModelSupplyException).reasonCode)
                assertTrue("missing named mismatch for $key: ${failure.message}",
                    failure.message.orEmpty().contains("$key: mismatch"))
                assertTrue(scanSuppliedSegmentRoots(badRoot).isEmpty())
            }
            val absentDeclaration = revisedFixture(good) { _, native ->
                native["gguf_metadata"].asJsonObject.add("head_count_kv", JsonParser.parseString("null"))
            }
            val failed = runCatching {
                val absentRoot = temp.newFolder("metadata-false-absence-$array")
                ModelSupplyReceiver(absentRoot, "127.0.0.1", client = shortReadClient(absentDeclaration)).receive(
                    absentDeclaration.manifest, 1, "secret", { true }, {}, { fail("actual value treated as absent") })
            }.exceptionOrNull()
            assertTrue(failed is ModelSupplyException)
            assertEquals("SUPPLY_GGUF_METADATA_MISMATCH", (failed as ModelSupplyException).reasonCode)
        }
    }

    @Test fun `downloaded receipt rejects missing mismatched null and typed header declarations before publication`() = runBlocking {
        val changes = linkedMapOf<String, (JsonObject) -> Unit>(
            "legacy" to { it.remove("gguf_metadata") },
            "null-contract" to { it.add("gguf_metadata", JsonParser.parseString("null")) },
            "array-contract" to { it.add("gguf_metadata", JsonParser.parseString("[]")) },
            "source-block-count" to { it["gguf_metadata"].asJsonObject.addProperty("block_count", 4) },
            "null-block-count" to { it["gguf_metadata"].asJsonObject.add("block_count", JsonParser.parseString("null")) },
            "float-block-count" to { it["gguf_metadata"].asJsonObject.addProperty("block_count", 2.0) },
            "bool-block-count" to { it["gguf_metadata"].asJsonObject.addProperty("block_count", true) },
            "architecture" to { it["gguf_metadata"].asJsonObject.addProperty("architecture", "other") },
        )
        for (key in listOf("block_count", "head_count_kv", "key_length", "full_attention_interval")) {
            changes["missing-$key"] = { it["gguf_metadata"].asJsonObject.remove(key) }
            if (key != "block_count") changes["actual-absent-$key"] = { it["gguf_metadata"].asJsonObject.addProperty(key, 1) }
        }
        for ((name, change) in changes) {
            val f = revisedFixture(fixture()) { _, native -> change(native) }
            val root = temp.newFolder("metadata-$name")
            val failure = runCatching {
                ModelSupplyReceiver(root, "127.0.0.1", client = shortReadClient(f)).receive(
                    f.manifest, 1, "secret", { true }, {}, { fail("invalid metadata published: $name") })
            }.exceptionOrNull()
            assertTrue(name, failure is ModelSupplyException)
            assertEquals(name, "SUPPLY_GGUF_METADATA_MISMATCH", (failure as ModelSupplyException).reasonCode)
            val stage = root.resolve(".receive-${f.manifest.manifestSha256}")
            assertTrue(stage.resolve(f.manifest.weight.path).isFile)
            assertFalse(stage.resolve(ModelSupplyManifest.MARKER).exists())
            assertFalse(root.resolve("segment-${f.manifest.manifestSha256}").exists())
            assertFalse(root.resolve(ModelSupplyCurrent.POINTER).exists())
        }
    }

    @Test fun `receiver error joins every named missing typedvalue and mismatch detail`() = runBlocking {
        val f = revisedFixture(fixture()) { _, native ->
            native["gguf_metadata"].asJsonObject.apply {
                remove("head_count_kv")
                addProperty("key_length", true)
                addProperty("full_attention_interval", 4)
            }
        }
        val root = temp.newFolder("metadata-named-differences")
        val failure = runCatching {
            ModelSupplyReceiver(root, "127.0.0.1", client = shortReadClient(f)).receive(
                f.manifest, 1, "secret", { true }, {}, { fail("metadata differences published") })
        }.exceptionOrNull()
        assertTrue(failure is ModelSupplyException)
        val error = failure as ModelSupplyException
        assertEquals("SUPPLY_GGUF_METADATA_MISMATCH", error.reasonCode)
        assertEquals(listOf(
            "head_count_kv: missing from declared metadata",
            "key_length: typedvalue in declared metadata",
            "full_attention_interval: mismatch between actual and declared values",
        ), error.message.orEmpty().split("; "))
        assertFalse(root.resolve("segment-${f.manifest.manifestSha256}").exists())
        assertFalse(root.resolve(ModelSupplyCurrent.POINTER).exists())
    }

    @Test fun `published legacy receipt is revalidated and remains unchanged on refusal`() = runBlocking {
        val legacy = revisedFixture(fixture()) { _, native -> native.remove("gguf_metadata") }
        val root = temp.newFolder("metadata-published-legacy")
        val old = publishFixture(root, legacy)
        val marker = old.resolve(ModelSupplyManifest.MARKER).readBytes()
        val failure = runCatching {
            ModelSupplyReceiver(root, "127.0.0.1").receive(legacy.manifest, 1, "secret", { true }, {},
                { fail("legacy receipt republished") })
        }.exceptionOrNull()
        assertTrue(failure is ModelSupplyException)
        assertEquals("SUPPLY_GGUF_METADATA_MISMATCH", (failure as ModelSupplyException).reasonCode)
        assertArrayEquals(marker, old.resolve(ModelSupplyManifest.MARKER).readBytes())
        legacy.files.forEach { (path, bytes) -> assertArrayEquals(bytes, old.resolve(path).readBytes()) }
        assertFalse(root.resolve(ModelSupplyCurrent.POINTER).exists())
    }

    @Test fun `failed legacy stage cannot bypass metadata validation by reusing every file`() = runBlocking {
        val legacy = revisedFixture(fixture()) { _, native -> native.remove("gguf_metadata") }
        val root = temp.newFolder("metadata-stage-legacy")
        val first = runCatching {
            ModelSupplyReceiver(root, "127.0.0.1", client = shortReadClient(legacy)).receive(
                legacy.manifest, 1, "secret", { true }, {}, { fail("legacy download published") })
        }.exceptionOrNull()
        assertTrue(first is ModelSupplyException)
        assertEquals("SUPPLY_GGUF_METADATA_MISMATCH", (first as ModelSupplyException).reasonCode)
        val next = nextOperation(legacy)
        val progress = mutableListOf<ModelSupplyProgress>()
        val second = runCatching {
            ModelSupplyReceiver(root, "127.0.0.1").receive(next, 1, "secret", { true }, { progress += it },
                { fail("legacy reused stage published") })
        }.exceptionOrNull()
        assertTrue(second is ModelSupplyException)
        assertEquals("SUPPLY_GGUF_METADATA_MISMATCH", (second as ModelSupplyException).reasonCode)
        assertEquals(next.files.sumOf { it.size }, progress.last().reusedBytes)
        assertEquals(0L, progress.last().actualDownloadedBytes)
        assertTrue(scanSuppliedSegmentRoots(root).isEmpty())
        assertFalse(root.resolve(ModelSupplyCurrent.POINTER).exists())
    }

    @Test fun `new authoritative receipt can reuse legacy weights while old ready cache stays unchanged`() = runBlocking {
        val good = fixture()
        val legacy = revisedFixture(good) { _, native -> native.remove("gguf_metadata") }
        val root = temp.newFolder("metadata-upgrade-receipt")
        val old = publishFixture(root, legacy)
        val marker = old.resolve(ModelSupplyManifest.MARKER).readBytes()
        val next = Fixture(nextOperation(good), good.files)
        val (final, progress) = downloaded(root, next, expectedFiles = setOf(next.manifest.native.path))
        assertEquals(next.manifest.weight.size, progress.reusedBytes)
        assertEquals(next.manifest.native.size, progress.actualDownloadedBytes)
        assertArrayEquals(marker, old.resolve(ModelSupplyManifest.MARKER).readBytes())
        legacy.files.forEach { (path, bytes) -> assertArrayEquals(bytes, old.resolve(path).readBytes()) }
        assertArrayEquals(good.files.getValue(good.manifest.native.path), final.resolve(next.manifest.native.path).readBytes())
        ModelSupplyReceiver(root, "unused").validateFiles(final, next.manifest)
    }

    /** A complete failed stage must come from the real receiver, never a fabricated positive journal. */
    private suspend fun failedNative(root: File, f: Fixture, checkpointBytes: Long = 64L * 1024 * 1024): Pair<Fixture, File> {
        val bad = revisedFixture(f) { _, native -> native.addProperty("mode", "middle") }
        try { downloaded(root, bad, checkpointBytes = checkpointBytes); fail("middle native advertised as tail") }
        catch (failure: ModelSupplyException) { assertEquals("SUPPLY_NATIVE_MANIFEST_MISMATCH", failure.reasonCode) }
        val stage = root.resolve(".receive-${bad.manifest.manifestSha256}")
        bad.files.forEach { (path, bytes) -> assertArrayEquals(bytes, stage.resolve(path).readBytes()) }
        assertFalse(stage.resolve(bad.manifest.weight.path + ".part").exists())
        val journal = JsonParser.parseString(stage.resolve(ModelSupplyReceiver.RESUME_MARKER).readText()).asJsonObject
        assertEquals(bad.manifest.json, journal["manifest"])
        assertEquals(supplySha256(canonicalSupplyJson(journal.deepCopy().apply { remove("journal_sha256") }).toByteArray()),
            journal["journal_sha256"].asString)
        assertTrue(scanSuppliedSegmentRoots(root).isEmpty())
        assertFalse(stage.resolve(ModelSupplyManifest.MARKER).exists())
        assertFalse(root.resolve(ModelSupplyCurrent.POINTER).exists())
        return bad to stage
    }

    @Test fun `native mode failure preserves complete weights for a new operation that downloads only corrected native`() = runBlocking {
        // Cover both a production-style periodic journal and a file smaller than the checkpoint interval.
        for (checkpointBytes in listOf(64L * 1024, 64L * 1024 * 1024)) {
            val f = fixture(extraWeightBytes = 512 * 1024)
            val root = temp.newFolder("complete-stage-$checkpointBytes")
            val (bad, stage) = failedNative(root, f, checkpointBytes)
            val oldWeight = stage.resolve(bad.manifest.weight.path).readBytes()
            val next = Fixture(nextOperation(f), f.files)
            val (final, progress) = downloaded(root, next, expectedFiles = setOf(next.manifest.native.path))
            assertEquals(next.manifest.weight.size, progress.reusedBytes)
            assertEquals(next.manifest.native.size, progress.actualDownloadedBytes)
            assertEquals(0L, progress.resumedBytes)
            assertEquals(next.manifest.files.sumOf { it.size }, progress.downloadedBytes)
            assertArrayEquals(oldWeight, stage.resolve(bad.manifest.weight.path).readBytes())
            assertArrayEquals(f.files.getValue(f.manifest.native.path), final.resolve(next.manifest.native.path).readBytes())
            assertEquals(1, scanSuppliedSegmentRoots(root).size)
            assertFalse(stage.resolve(ModelSupplyManifest.MARKER).exists())
            assertFalse(root.resolve(ModelSupplyCurrent.POINTER).exists())
        }
    }

    @Test fun `complete failed stage cannot donate weights through a missing or invalid journal`() = runBlocking {
        for (damage in listOf("missing", "unsigned", "manifest-digest", "prefix-binding", "format", "directory")) {
            val f = fixture(extraWeightBytes = 128 * 1024)
            val root = temp.newFolder("complete-journal-$damage")
            val (_, stage) = failedNative(root, f, 64L * 1024)
            val marker = stage.resolve(ModelSupplyReceiver.RESUME_MARKER)
            when (damage) {
                "missing" -> assertTrue(marker.delete())
                "directory" -> java.nio.file.Files.move(stage.toPath(), root.resolve(".receive-${"0".repeat(64)}").toPath())
                else -> {
                    val json = JsonParser.parseString(marker.readText()).asJsonObject
                    when (damage) {
                        "unsigned" -> json["prefixes"].asJsonArray[0].asJsonObject.addProperty("prefix_sha256", "0".repeat(64))
                        "manifest-digest" -> json["manifest"].asJsonObject.addProperty("hidden_size", 2)
                        "prefix-binding" -> json["prefixes"].asJsonArray[0].asJsonObject.addProperty("file_sha256", "0".repeat(64))
                        "format" -> {
                            json["manifest"].asJsonObject.addProperty("format", "ggml")
                            resign(json["manifest"].asJsonObject)
                        }
                    }
                    if (damage != "unsigned") {
                        json.remove("journal_sha256")
                        json.addProperty("journal_sha256", supplySha256(canonicalSupplyJson(json).toByteArray()))
                    }
                    marker.writeText(canonicalSupplyJson(json))
                }
            }
            val next = Fixture(nextOperation(f), f.files)
            val (_, progress) = downloaded(root, next)
            assertEquals(damage, 0L, progress.reusedBytes)
            assertEquals(damage, 0L, progress.resumedBytes)
            assertEquals(damage, next.manifest.files.sumOf { it.size }, progress.actualDownloadedBytes)
        }
    }

    @Test fun `complete failed stage reuse requires every validated source identity component`() = runBlocking {
        for (field in listOf("source_model_id", "source_model_sha256", "tokenizer_sha256", "conversion_proof")) {
            val f = fixture()
            val root = temp.newFolder("complete-source-$field")
            val foreign = revisedFixture(f) { manifest, native ->
                for (json in listOf(manifest, native)) when (field) {
                    "source_model_id" -> json.addProperty(field, "other-source")
                    "conversion_proof" -> json.add(field, JsonObject().apply { addProperty("producer", "other-proof") })
                    else -> json.addProperty(field, "0".repeat(64))
                }
            }
            failedNative(root, foreign)
            val next = Fixture(nextOperation(f), f.files)
            val (_, progress) = downloaded(root, next)
            assertEquals(field, 0L, progress.reusedBytes)
            assertEquals(field, 0L, progress.resumedBytes)
            assertEquals(field, next.manifest.files.sumOf { it.size }, progress.actualDownloadedBytes)
            assertEquals(1, scanSuppliedSegmentRoots(root).size)
        }
    }

    @Test fun `complete failed stage verifies live SHA and size even when timestamp is unchanged`() = runBlocking {
        for (damage in listOf("bytes", "size")) {
            val f = fixture()
            val root = temp.newFolder("complete-weight-$damage")
            val (_, stage) = failedNative(root, f)
            val weight = stage.resolve(f.manifest.weight.path)
            val time = weight.lastModified()
            weight.writeBytes(if (damage == "size") weight.readBytes().dropLast(1).toByteArray()
                else weight.readBytes().also { it[0] = (it[0].toInt() xor 1).toByte() })
            assertTrue(weight.setLastModified(time))
            val next = Fixture(nextOperation(f), f.files)
            val (_, progress) = downloaded(root, next)
            assertEquals(damage, 0L, progress.reusedBytes)
            assertEquals(damage, next.manifest.files.sumOf { it.size }, progress.actualDownloadedBytes)
            assertFalse(stage.resolve(ModelSupplyManifest.MARKER).exists())
        }
    }

    @Test fun `complete failed stage remains a byte donor and cannot bypass native validation`() = runBlocking {
        val f = fixture()
        val root = temp.newFolder("complete-native-still-required")
        val (bad, stage) = failedNative(root, f)
        val next = Fixture(nextOperation(bad), bad.files)
        val progress = mutableListOf<ModelSupplyProgress>()
        val failure = runCatching {
            ModelSupplyReceiver(root, "127.0.0.1").receive(next.manifest, 1, "secret", { true }, { progress += it },
                { fail("invalid native published through reuse") })
        }.exceptionOrNull()
        assertTrue(failure is ModelSupplyException)
        assertEquals("SUPPLY_NATIVE_MANIFEST_MISMATCH", (failure as ModelSupplyException).reasonCode)
        assertEquals(next.manifest.files.sumOf { it.size }, progress.last().reusedBytes)
        assertEquals(0L, progress.last().actualDownloadedBytes)
        assertTrue(scanSuppliedSegmentRoots(root).isEmpty())
        assertFalse(stage.resolve(ModelSupplyManifest.MARKER).exists())
        assertFalse(root.resolve(ModelSupplyCurrent.POINTER).exists())
    }

    @Test fun `complete staged weight reuse rechecks the current fence after hashing before creating a link`() = runBlocking {
        val f = fixture()
        val root = temp.newFolder("complete-reuse-fence")
        val (_, stage) = failedNative(root, f)
        val next = Fixture(nextOperation(f), f.files)
        var current = true
        val failure = runCatching {
            ModelSupplyReceiver(root, "127.0.0.1").receive(next.manifest, 1, "secret", { current }, {
                if (it.phase == "verifying_reuse") current = false
            }, { fail("stale reuse published") })
        }.exceptionOrNull()
        assertTrue(failure is ModelSupplyException)
        assertEquals("SUPPLY_CANCELLED", (failure as ModelSupplyException).reasonCode)
        assertFalse(root.resolve(".receive-${next.manifest.manifestSha256}").resolve(next.manifest.weight.path).exists())
        assertArrayEquals(f.files.getValue(f.manifest.weight.path), stage.resolve(f.manifest.weight.path).readBytes())
        assertTrue(scanSuppliedSegmentRoots(root).isEmpty())
    }

    @Test fun `successful controller only registers refreshes HELLO and reports matching identities`() = runBlocking {
        val f = fixture(); val root = temp.newFolder("controller-success")
        val stage = root.resolve(".receive-${f.manifest.manifestSha256}").apply { mkdir() }
        f.files.forEach { (path, bytes) -> stage.resolve(path).writeBytes(bytes) }
        val result = CompletableDeferred<JsonObject>(); val order = mutableListOf<String>()
        val controller = com.qlh.inference.worker.ModelSupplyController(node, "127.0.0.1", root,
            { true }, { true }, { 1L }, { it() }, { assertEquals(f.manifest, it); order += "register" })
        val command = JsonParser.parseString("""{"action":"ensure","operation_id":"$operation","node_id":"$node","strategy":"distribute","api_port":1,"token":"secret"}""").asJsonObject
        command.add("manifest", f.manifest.json)
        controller.command(command, this, { if (it["action"].asString == "result") { order += "result"; result.complete(it) } }, { order += "hello" })
        val reply = withTimeout(3000) { result.await() }
        assertEquals(listOf("register", "hello", "result"), order)
        assertEquals("available", reply["status"].asString)
        for (key in listOf("manifest_sha256", "artifact_sha256", "layer_range", "source_model_sha256")) assertEquals(f.manifest.json[key], reply[key])
        assertEquals(1, scanSuppliedSegmentRoots(root).size)
        val binding = supplyBindingSha256("", "127.0.0.1", 0, 0)
        assertEquals(f.manifest, ModelSupplyCurrent(root).read(binding))
        assertEquals(reply["total_bytes"].asLong, reply["downloaded_bytes"].asLong)
        assertEquals(reply["total_bytes"].asLong, reply["reused_bytes"].asLong)
        assertEquals(0L, reply["actual_downloaded_bytes"].asLong)
        assertEquals(0L, reply["resumed_bytes"].asLong)
    }

    @Test fun `controller retirement cancels before publication or registration`() = runBlocking {
        val f = fixture(); val root = temp.newFolder("controller-cancel")
        val stage = root.resolve(".receive-${f.manifest.manifestSha256}").apply { mkdir() }
        f.files.forEach { (path, bytes) -> stage.resolve(path).writeBytes(bytes) }
        val entered = CompletableDeferred<Unit>(); val release = CompletableDeferred<Unit>()
        val controller = com.qlh.inference.worker.ModelSupplyController(node, "127.0.0.1", root,
            { true }, { true }, { 1L }, { entered.complete(Unit); release.await(); it() }, { fail("retired registration") })
        val command = JsonParser.parseString("""{"action":"ensure","operation_id":"$operation","node_id":"$node","strategy":"distribute","api_port":1,"token":"secret"}""").asJsonObject
        command.add("manifest", f.manifest.json)
        controller.command(command, this, { assertNotEquals("result", it["action"].asString) }, { fail("retired hello") })
        withTimeout(3000) { entered.await() }
        controller.cancelAll(); release.complete(Unit)
        withTimeout(3000) { coroutineContext[kotlinx.coroutines.Job]!!.children.toList().forEach { it.join() } }
        assertTrue(scanSuppliedSegmentRoots(root).isEmpty())
        assertFalse(stage.resolve(ModelSupplyManifest.MARKER).exists())
    }

    @Test fun `publication rechecks certificate after waiting for the mode gate`() = runBlocking {
        val f = fixture(); val root = temp.newFolder("controller-fence")
        val stage = root.resolve(".receive-${f.manifest.manifestSha256}").apply { mkdir() }
        f.files.forEach { (path, bytes) -> stage.resolve(path).writeBytes(bytes) }
        var authorized = true
        val controller = com.qlh.inference.worker.ModelSupplyController(node, "127.0.0.1", root,
            { true }, { true }, { 1L }, { authorized = false; it() }, { fail("expired registration") })
        val command = JsonParser.parseString("""{"action":"ensure","operation_id":"$operation","node_id":"$node","strategy":"distribute","api_port":1,"token":"secret"}""").asJsonObject
        command.add("manifest", f.manifest.json)
        controller.command(command, this, {}, { fail("expired hello") }, {
            if (!authorized) throw kotlinx.coroutines.CancellationException("lease expired")
        })
        withTimeout(3000) { coroutineContext[kotlinx.coroutines.Job]!!.children.toList().forEach { it.join() } }
        assertTrue(scanSuppliedSegmentRoots(root).isEmpty())
        assertFalse(stage.resolve(ModelSupplyManifest.MARKER).exists())
    }

    private val controllerBinding get() = supplyBindingSha256("", "127.0.0.1", 0, 0)
    private fun ensure(manifest: ModelSupplyManifest) = JsonObject().apply {
        addProperty("action", "ensure"); addProperty("operation_id", manifest.operationId)
        addProperty("node_id", node); addProperty("strategy", "distribute")
        addProperty("api_port", 1); addProperty("token", "secret"); add("manifest", manifest.json)
    }
    private suspend fun joinChildren() {
        val parent = kotlinx.coroutines.currentCoroutineContext()[kotlinx.coroutines.Job]!!
        withTimeout(3000) { parent.children.toList().forEach { if (it != kotlinx.coroutines.currentCoroutineContext()[kotlinx.coroutines.Job]) it.join() } }
    }

    @Test fun `current survives recreation and remains bound without guessing among operation directories`() {
        val f = fixture(); val root = temp.newFolder("current-restart")
        publishFixture(root, f)
        val next = nextOperation(f)
        publishFixture(root, Fixture(next, f.files))
        val store = ModelSupplyCurrent(root)
        assertNull(store.read(controllerBinding))
        store.select(next, controllerBinding)
        assertEquals(next, ModelSupplyCurrent(root).read(controllerBinding))
        assertEquals(2, scanSuppliedSegmentRoots(root).size)
        assertNull(ModelSupplyCurrent(root).read(supplyBindingSha256("other", "127.0.0.1", 0, 0)))
        assertEquals(controllerBinding, supplyBindingSha256("", " 127.0.0.1 ", 0, 0))
    }

    @Test fun `selected supply wins SAF digest conflicts and identical private copies with stable artifact id`() {
        val f = fixture(); val root = temp.newFolder("candidate-selection")
        val old = publishFixture(root, f); val next = nextOperation(f)
        val directory = publishFixture(root, Fixture(next, f.files))
        ModelSupplyCurrent(root).select(next, controllerBinding)
        val recovered = ModelSupplyCurrent(root).read(controllerBinding)!!
        val descriptor = LayerArtifactManifestParser.parse(f.files.getValue("segment.manifest.json").toString(Charsets.UTF_8), f.manifest.native.path).getOrThrow()
        val selected = LayerArtifactCandidate(descriptor, directory.resolve(next.weight.path).path, next.native.path, LayerArtifactCandidate.Root.INTERNAL)
        val oldCopy = selected.copy(reference = old.resolve(next.weight.path).path)
        val saf = selected.copy(root = LayerArtifactCandidate.Root.SAF, reference = "content://saf/old-tail.gguf",
            descriptor = descriptor.copy(artifactName = "old-tail.gguf", artifactSha256 = "f".repeat(64)))
        val sameBytesDifferentName = saf.copy(descriptor = descriptor.copy(artifactName = "renamed.gguf"))
        val independent = saf.copy(descriptor = saf.descriptor.copy(startLayer = 0, endLayerExclusive = 2, mode = "head"), reference = "content://saf/head.gguf")
        val candidates = preferCurrentSupplyCandidates(listOf(saf, oldCopy, sameBytesDifferentName, selected, independent), recovered, selected.reference)
        assertEquals(listOf(selected, independent), candidates)
        assertEquals(candidates, unambiguousLayerArtifactCandidates(candidates))
        val modelId = com.qlh.inference.worker.AndroidWorkerCapabilities.artifactModelId(selected.descriptor.artifactName, selected.descriptor.artifactSha256)
        assertEquals("segment.gguf", modelId)
        val chosen = chooseLayerArtifact(candidates, next.sourceModelId, next.sourceSha256, next.start, next.end,
            next.artifactSha256, offeredArtifactModelId = modelId) as LayerArtifactChoice.Selected
        assertEquals(selected, chosen.artifact)
        for (sha in listOf(saf.descriptor.artifactSha256, "a".repeat(64))) {
            assertTrue(chooseLayerArtifact(candidates, next.sourceModelId, next.sourceSha256, next.start, next.end,
                sha, offeredArtifactModelId = modelId) is LayerArtifactChoice.ManualRequired)
        }
        assertTrue(chooseLayerArtifact(candidates, next.sourceModelId, next.sourceSha256, next.start, next.end,
            next.artifactSha256, offeredArtifactModelId = "renamed.gguf") is LayerArtifactChoice.ManualRequired)
        assertEquals(listOf(saf, independent), preferCurrentSupplyCandidates(listOf(saf, independent), null, ""))
        // Missing/failed current verification reserves its range rather than selecting old SAF bytes.
        assertEquals(listOf(independent), preferCurrentSupplyCandidates(listOf(saf, oldCopy, independent), recovered, selected.reference))
        assertTrue(preferCurrentSupplyCandidates(listOf(selected.copy(descriptor = descriptor.copy(tokenizerSha256 = "a".repeat(64)))), recovered, selected.reference).isEmpty())
    }

    @Test fun `selected recovery still rejects modified bytes and broken current metadata`() = runBlocking {
        val f = fixture(); val root = temp.newFolder("current-corrupt")
        val directory = publishFixture(root, f)
        ModelSupplyCurrent(root).select(f.manifest, controllerBinding)
        val weight = directory.resolve(f.manifest.weight.path)
        val mtime = weight.lastModified()
        weight.writeBytes(weight.readBytes().also { it[0] = 0 }); weight.setLastModified(mtime)
        try {
            ModelSupplyReceiver(root, "unused").validateFiles(directory, ModelSupplyCurrent(root).read(controllerBinding)!!)
            fail("damaged current accepted")
        } catch (error: ModelSupplyException) { assertEquals("SUPPLY_ARTIFACT_DIGEST_MISMATCH", error.reasonCode) }
        root.resolve(ModelSupplyCurrent.POINTER).writeText("{}")
        expect("SUPPLY_CURRENT_INVALID") { ModelSupplyCurrent(root).read(controllerBinding) }
    }

    @Test fun `failure after selection restores previous current and includes final reuse snapshot`() = runBlocking {
        val f = fixture(); val root = temp.newFolder("current-failed")
        publishFixture(root, f); ModelSupplyCurrent(root).select(f.manifest, controllerBinding)
        val next = nextOperation(f); val result = CompletableDeferred<JsonObject>()
        val controller = com.qlh.inference.worker.ModelSupplyController(node, "127.0.0.1", root,
            { true }, { true }, { 1L }, { it() }, {})
        controller.command(ensure(next), this, { if (it["action"].asString == "result") result.complete(it) }, {
            assertEquals(next, ModelSupplyCurrent(root).read(controllerBinding))
            throw java.io.IOException("HELLO failed")
        })
        val reply = withTimeout(3000) { result.await() }
        joinChildren()
        assertEquals("failed", reply["status"].asString)
        assertEquals(reply["total_bytes"].asLong, reply["reused_bytes"].asLong)
        assertEquals(0L, reply["actual_downloaded_bytes"].asLong)
        assertEquals(f.manifest, ModelSupplyCurrent(root).read(controllerBinding))
        assertEquals(2, scanSuppliedSegmentRoots(root).size)
    }

    @Test fun `cancel during HELLO restores previous current before returning cancellation snapshot`() = runBlocking {
        val f = fixture(); val root = temp.newFolder("current-cancel")
        publishFixture(root, f); ModelSupplyCurrent(root).select(f.manifest, controllerBinding)
        val next = nextOperation(f); val entered = CompletableDeferred<Unit>(); val release = CompletableDeferred<Unit>()
        val replies = mutableListOf<JsonObject>()
        val controller = com.qlh.inference.worker.ModelSupplyController(node, "127.0.0.1", root,
            { true }, { true }, { 1L }, { it() }, {})
        controller.command(ensure(next), this, { replies += it }, {
            assertEquals(next, ModelSupplyCurrent(root).read(controllerBinding)); entered.complete(Unit); release.await()
        })
        withTimeout(3000) { entered.await() }
        val cancel = JsonObject().apply { addProperty("action", "cancel"); addProperty("operation_id", next.operationId) }
        controller.command(cancel, this, { replies += it }, {})
        assertEquals(f.manifest, ModelSupplyCurrent(root).read(controllerBinding))
        release.complete(Unit); joinChildren()
        val result = replies.single { it["action"].asString == "result" }
        assertEquals("SUPPLY_CANCELLED", result["reason_code"].asString)
        assertEquals(result["total_bytes"].asLong, result["reused_bytes"].asLong)
        assertEquals(0L, result["actual_downloaded_bytes"].asLong)
        assertEquals(0L, result["resumed_bytes"].asLong)
        assertEquals(f.manifest, ModelSupplyCurrent(root).read(controllerBinding))
    }

    @Test fun `retirement after fresh selection removes current so restart cannot restore cancelled operation`() = runBlocking {
        val f = fixture(); val root = temp.newFolder("current-retired")
        publishFixture(root, f)
        val entered = CompletableDeferred<Unit>(); val release = CompletableDeferred<Unit>()
        val controller = com.qlh.inference.worker.ModelSupplyController(node, "127.0.0.1", root,
            { true }, { true }, { 1L }, { it() }, {})
        controller.command(ensure(f.manifest), this, { assertNotEquals("result", it["action"].asString) }, {
            entered.complete(Unit); release.await()
        })
        withTimeout(3000) { entered.await() }
        assertEquals(f.manifest, ModelSupplyCurrent(root).read(controllerBinding))
        controller.cancelAll()
        assertNull(ModelSupplyCurrent(root).read(controllerBinding))
        release.complete(Unit); joinChildren()
        assertNull(ModelSupplyCurrent(root).read(controllerBinding))
        assertEquals(1, scanSuppliedSegmentRoots(root).size)
    }

    @Test fun `controller resume final snapshot counts actual and resumed bytes without requiring last progress message`() = runBlocking {
        val f = fixture(); val root = temp.newFolder("controller-resume")
        interruptedDownload(root, f)
        val weight = f.files.getValue(f.manifest.weight.path)
        val server = ServerSocket(0); var error: Throwable? = null
        val peer = thread {
            try {
                repeat(2) { server.accept().use { socket ->
                    val input = socket.getInputStream().bufferedReader(); val request = input.readLine()
                    val headers = mutableMapOf<String, String>()
                    while (true) { val line = input.readLine(); if (line.isEmpty()) break; headers[line.substringBefore(':').lowercase()] = line.substringAfter(':').trim() }
                    assertEquals("secret", headers["x-qlh-supply-token"])
                    val name = request.substringBefore(" HTTP").substringAfterLast('/')
                    val bytes = f.files.getValue(name); val offset = if (name == f.manifest.weight.path) 11 else 0
                    if (offset > 0) assertEquals("bytes=11-", headers["range"])
                    val head = if (offset > 0) "HTTP/1.1 206 Partial Content\r\nContent-Range: bytes $offset-${bytes.size - 1}/${bytes.size}\r\n" else "HTTP/1.1 200 OK\r\n"
                    socket.getOutputStream().apply {
                        write((head + "Content-Length: ${bytes.size - offset}\r\nConnection: close\r\n\r\n").toByteArray())
                        write(bytes, offset, bytes.size - offset); flush()
                    }
                } }
            } catch (t: Throwable) { error = t }
        }
        val result = CompletableDeferred<JsonObject>()
        val controller = com.qlh.inference.worker.ModelSupplyController(node, "127.0.0.1", root,
            { true }, { true }, { 1L }, { it() }, {})
        try {
            controller.command(ensure(f.manifest).apply { addProperty("api_port", server.localPort) }, this,
                { if (it["action"].asString == "result") result.complete(it) }, {})
            val reply = withTimeout(3000) { result.await() }; joinChildren()
            val total = f.manifest.files.sumOf { it.size }
            assertEquals("available", reply["status"].asString)
            assertEquals(total, reply["total_bytes"].asLong)
            assertEquals(total, reply["downloaded_bytes"].asLong)
            assertEquals(total - 11, reply["actual_downloaded_bytes"].asLong)
            assertEquals(11L, reply["resumed_bytes"].asLong)
            assertEquals(0L, reply["reused_bytes"].asLong)
        } finally { controller.cancelAll(); server.close(); peer.join(3000) }
        assertNull(error)
    }

    @Test fun `transfer digest failure final snapshot retains actual bytes and never selects current`() = runBlocking {
        val f = fixture(); val root = temp.newFolder("controller-digest-failure")
        val server = ServerSocket(0); var error: Throwable? = null
        val damaged = f.files.getValue(f.manifest.weight.path).copyOf().also { it[0] = 0 }
        val peer = thread {
            try { server.accept().use { socket ->
                val input = socket.getInputStream().bufferedReader()
                val request = input.readLine()
                assertTrue(request.contains("/${f.manifest.operationId}/files/$node/${f.manifest.weight.path}"))
                while (input.readLine().isNotEmpty()) Unit
                socket.getOutputStream().apply {
                    write(("HTTP/1.1 200 OK\r\nContent-Length: ${damaged.size}\r\nConnection: close\r\n\r\n").toByteArray())
                    write(damaged); flush()
                }
            } } catch (t: Throwable) { error = t }
        }
        val result = CompletableDeferred<JsonObject>()
        val controller = com.qlh.inference.worker.ModelSupplyController(node, "127.0.0.1", root,
            { true }, { true }, { 1L }, { it() }, { fail("damaged registration") })
        try {
            controller.command(ensure(f.manifest).apply { addProperty("api_port", server.localPort) }, this,
                { if (it["action"].asString == "result") result.complete(it) }, { fail("damaged HELLO") })
            val reply = withTimeout(3000) { result.await() }; joinChildren()
            assertEquals("failed", reply["status"].asString)
            assertEquals("SUPPLY_ARTIFACT_DIGEST_MISMATCH", reply["reason_code"].asString)
            assertEquals(damaged.size.toLong(), reply["actual_downloaded_bytes"].asLong)
            assertEquals(damaged.size.toLong(), reply["downloaded_bytes"].asLong)
            assertEquals(f.manifest.files.sumOf { it.size }, reply["total_bytes"].asLong)
            assertEquals(0L, reply["reused_bytes"].asLong)
            assertEquals(0L, reply["resumed_bytes"].asLong)
            assertNull(ModelSupplyCurrent(root).read(controllerBinding))
            assertTrue(scanSuppliedSegmentRoots(root).isEmpty())
        } finally { controller.cancelAll(); server.close(); peer.join(3000) }
        assertNull(error)
    }

    @Test fun `mode or binding rejection at registration leaves published bytes unselected across restart`() = runBlocking {
        val f = fixture()
        for (reason in listOf("SUPPLY_MODE_DISABLED", "SUPPLY_MASTER_BINDING_CHANGED")) {
            val root = temp.newFolder(reason)
            val stage = root.resolve(".receive-${f.manifest.manifestSha256}").apply { mkdir() }
            f.files.forEach { (path, bytes) -> stage.resolve(path).writeBytes(bytes) }
            var gates = 0; val result = CompletableDeferred<JsonObject>()
            val controller = com.qlh.inference.worker.ModelSupplyController(node, "127.0.0.1", root,
                { true }, { true }, { 1L }, {
                    gates++
                    if (gates == 2) throw ModelSupplyException(reason, "registration gate changed")
                    it()
                }, { fail("rejected registration") })
            controller.command(ensure(f.manifest), this, { if (it["action"].asString == "result") result.complete(it) }, { fail("rejected HELLO") })
            val reply = withTimeout(3000) { result.await() }; joinChildren()
            assertEquals(reason, reply["reason_code"].asString)
            assertEquals(reply["total_bytes"].asLong, reply["reused_bytes"].asLong)
            assertEquals(1, scanSuppliedSegmentRoots(root).size)
            assertNull(ModelSupplyCurrent(root).read(controllerBinding))
        }
    }


    private data class Interrupted(val stage: File, val partial: File, val length: Long, val journal: JsonObject)

    /** Acceptance always starts with bytes and checkpoints produced by the receiver. */
    private suspend fun interruptedDownload(root: File, f: Fixture, prefixBytes: Int = 11,
                                           mode: String = "disconnect", checkpointBytes: Long = 64L * 1024 * 1024,
                                           checkpointSeen: (Long) -> Unit = {}): Interrupted {
        val server = ServerSocket(0)
        val release = CompletableDeferred<Unit>()
        val error = AtomicReference<Throwable?>()
        val peer = thread {
            try { server.accept().use { socket ->
                val input = socket.getInputStream().bufferedReader()
                val request = input.readLine()
                assertTrue(request.contains("/${f.manifest.operationId}/files/$node/${f.manifest.weight.path}"))
                val headers = mutableMapOf<String, String>()
                while (true) { val line = input.readLine(); if (line.isEmpty()) break; headers[line.substringBefore(':').lowercase()] = line.substringAfter(':').trim() }
                assertNull(headers["range"])
                val bytes = f.files.getValue(f.manifest.weight.path)
                socket.getOutputStream().apply {
                    write(("HTTP/1.1 200 OK\r\nContent-Length: ${bytes.size}\r\nConnection: close\r\n\r\n").toByteArray())
                    write(bytes, 0, prefixBytes); flush()
                }
                if (mode == "timeout") runBlocking { release.await() }
            } } catch (t: Throwable) { error.set(t) }
        }
        val receiver = ModelSupplyReceiver(root, "127.0.0.1", checkpointBytes, if (mode == "timeout") 200 else 60_000)
        var current = true
        try {
            try {
                receiver.receive(f.manifest, server.localPort, "secret", { current }, { progress ->
                    val checkpoint = root.resolve(".receive-${f.manifest.manifestSha256}").resolve(ModelSupplyReceiver.RESUME_MARKER)
                    if (checkpoint.isFile) {
                        val record = JsonParser.parseString(checkpoint.readText()).asJsonObject["prefixes"].asJsonArray.first().asJsonObject
                        checkpointSeen(record["prefix_bytes"].asLong)
                    }
                    if (mode == "cancel" && progress.downloadedBytes >= prefixBytes && progress.phase == "downloading") current = false
                }, { fail("interrupted transfer published") })
                fail("expected interrupted transfer")
            } catch (failure: Exception) {
                when (mode) {
                    "cancel" -> assertEquals("SUPPLY_CANCELLED", (failure as ModelSupplyException).reasonCode)
                    "timeout" -> assertTrue(failure is SocketTimeoutException)
                    else -> assertTrue(failure is IOException)
                }
            }
        } finally { release.complete(Unit); server.close(); peer.join(3000) }
        assertNull(error.get())
        assertFalse(peer.isAlive)
        return checkpoint(root, f, prefixBytes.toLong())
    }

    private fun checkpoint(root: File, f: Fixture, expectedBytes: Long): Interrupted {
        val stage = root.resolve(".receive-${f.manifest.manifestSha256}")
        val journal = JsonParser.parseString(stage.resolve(ModelSupplyReceiver.RESUME_MARKER).readText()).asJsonObject
        val record = journal["prefixes"].asJsonArray.single().asJsonObject
        val partial = stage.resolve(record["path"].asString + ".part")
        val length = record["prefix_bytes"].asLong
        assertEquals(expectedBytes, length)
        assertEquals(f.manifest.json, journal["manifest"])
        assertEquals(f.manifest.weight.sha256, record["file_sha256"].asString)
        assertEquals(f.manifest.weight.size, record["size_bytes"].asLong)
        assertEquals(length, partial.length())
        assertEquals(supplySha256(partial.readBytes()), record["prefix_sha256"].asString)
        assertEquals(supplySha256(canonicalSupplyJson(journal.deepCopy().apply { remove("journal_sha256") }).toByteArray()), journal["journal_sha256"].asString)
        assertTrue(scanSuppliedSegmentRoots(root).isEmpty())
        assertFalse(stage.resolve(ModelSupplyManifest.MARKER).exists())
        return Interrupted(stage, partial, length, journal)
    }

    private suspend fun downloaded(root: File, f: Fixture, expectedPrefix: Long = 0, ignoreRange: Boolean = false,
                                   corruptSuffix: Boolean = false, expectedFiles: Set<String> = f.files.keys,
                                   checkpointBytes: Long = 64L * 1024 * 1024): Pair<File, ModelSupplyProgress> {
        val server = ServerSocket(0)
        val error = AtomicReference<Throwable?>()
        val requests = mutableListOf<String>()
        val peer = thread {
            try { repeat(expectedFiles.size) { server.accept().use { socket ->
                val input = socket.getInputStream().bufferedReader()
                val request = input.readLine(); requests += request
                val headers = mutableMapOf<String, String>()
                while (true) { val line = input.readLine(); if (line.isEmpty()) break; headers[line.substringBefore(':').lowercase()] = line.substringAfter(':').trim() }
                assertTrue(request.contains("/${f.manifest.operationId}/files/$node/"))
                assertEquals("secret", headers["x-qlh-supply-token"])
                val name = request.substringBefore(" HTTP").substringAfterLast('/')
                assertTrue("unexpected download: $name", name in expectedFiles)
                val bytes = f.files.getValue(name).copyOf()
                val requested = if (name == f.manifest.weight.path) expectedPrefix else 0
                assertEquals(if (requested > 0) "bytes=$requested-" else null, headers["range"])
                val offset = if (ignoreRange) 0 else requested.toInt()
                if (corruptSuffix && name == f.manifest.weight.path) bytes[maxOf(offset, bytes.lastIndex)] = (bytes.last().toInt() xor 1).toByte()
                val head = if (offset > 0) "HTTP/1.1 206 Partial Content\r\nContent-Range: bytes $offset-${bytes.size - 1}/${bytes.size}\r\n" else "HTTP/1.1 200 OK\r\n"
                socket.getOutputStream().apply {
                    write((head + "Content-Length: ${bytes.size - offset}\r\nConnection: close\r\n\r\n").toByteArray())
                    write(bytes, offset, bytes.size - offset); flush()
                }
            } } } catch (t: Throwable) { if (!server.isClosed) error.set(t) }
        }
        val progress = mutableListOf<ModelSupplyProgress>()
        try {
            val result = ModelSupplyReceiver(root, "127.0.0.1", checkpointBytes).receive(f.manifest, server.localPort, "secret", { true }, { progress += it }, { it() })
            assertEquals(expectedFiles.size, requests.size)
            assertFalse(result.resolve(ModelSupplyReceiver.RESUME_MARKER).exists())
            assertEquals(f.manifest.json, ModelSupplyManifest.parse(result.resolve(ModelSupplyManifest.MARKER).readText()).json)
            assertArrayEquals(f.files.getValue(f.manifest.weight.path), result.resolve(f.manifest.weight.path).readBytes())
            return result to progress.last()
        } finally { server.close(); peer.join(3000); assertFalse(peer.isAlive); assertNull(error.get()) }
    }

    @Test fun `disconnect timeout and cancellation preserve production checkpoint for a new operation`() = runBlocking {
        for (mode in listOf("disconnect", "timeout", "cancel")) {
            val f = fixture(); val root = temp.newFolder(mode)
            val prefix = interruptedDownload(root, f, mode = mode)
            val oldBytes = prefix.partial.readBytes()
            val next = Fixture(nextOperation(f), f.files)
            val (directory, progress) = downloaded(root, next, prefix.length)
            assertEquals(prefix.length, progress.resumedBytes)
            assertEquals(next.manifest.files.sumOf { it.size } - prefix.length, progress.actualDownloadedBytes)
            assertEquals(0L, progress.reusedBytes)
            assertArrayEquals(oldBytes, prefix.partial.readBytes())
            assertTrue(prefix.stage.resolve(ModelSupplyReceiver.RESUME_MARKER).isFile)
            assertEquals(1, scanSuppliedSegmentRoots(root).size)
            assertEquals(next.manifest.manifestSha256, ModelSupplyManifest.parse(directory.resolve(ModelSupplyManifest.MARKER).readText()).manifestSha256)
            assertFalse(root.resolve(ModelSupplyCurrent.POINTER).exists())
        }
    }

    @Test fun `unrecorded legacy partial never sends Range within its operation or a new operation`() = runBlocking {
        for (sameOperation in listOf(true, false)) for (complete in listOf(true, false)) {
            val f = fixture(); val root = temp.newFolder()
            val stage = root.resolve(".receive-${f.manifest.manifestSha256}").apply { mkdir() }
            val bytes = f.files.getValue(f.manifest.weight.path)
            stage.resolve(f.manifest.weight.path + ".part").writeBytes(if (complete) bytes else bytes.copyOfRange(0, 11))
            val next = if (sameOperation) f else Fixture(nextOperation(f), f.files)
            val (_, progress) = downloaded(root, next)
            assertEquals(0L, progress.resumedBytes)
            assertEquals(next.manifest.files.sumOf { it.size }, progress.actualDownloadedBytes)
        }
    }

    @Test fun `bad live prefix missing checkpoint and invalid source marker force download from zero`() = runBlocking {
        for (damage in listOf("bytes", "no-journal", "no-marker", "no-checkpoint", "unsigned", "source", "file-sha", "file-size", "directory")) {
            val f = fixture(); val root = temp.newFolder(damage)
            val prefix = interruptedDownload(root, f)
            val marker = prefix.stage.resolve(ModelSupplyReceiver.RESUME_MARKER)
            when (damage) {
                "bytes" -> {
                    val time = prefix.partial.lastModified()
                    prefix.partial.writeBytes(prefix.partial.readBytes().also { it[0] = (it[0].toInt() xor 1).toByte() })
                    assertTrue(prefix.partial.setLastModified(time))
                }
                "no-journal" -> marker.delete()
                "directory" -> java.nio.file.Files.move(prefix.stage.toPath(), root.resolve(".receive-${"0".repeat(64)}").toPath())
                else -> {
                    val json = prefix.journal.deepCopy()
                    when (damage) {
                        "no-marker" -> json.remove("manifest")
                        "no-checkpoint" -> json.add("prefixes", com.google.gson.JsonArray())
                        "unsigned" -> json["prefixes"].asJsonArray[0].asJsonObject.addProperty("prefix_sha256", "0".repeat(64))
                        "source" -> json["manifest"].asJsonObject.addProperty("source_model_sha256", "0".repeat(64))
                        "file-sha" -> json["prefixes"].asJsonArray[0].asJsonObject.addProperty("file_sha256", "0".repeat(64))
                        "file-size" -> json["prefixes"].asJsonArray[0].asJsonObject.addProperty("size_bytes", f.manifest.weight.size + 1)
                    }
                    // Negative tests may corrupt a production journal; no positive test fabricates one.
                    if (damage != "unsigned") {
                        json.remove("journal_sha256")
                        json.addProperty("journal_sha256", supplySha256(canonicalSupplyJson(json).toByteArray()))
                    }
                    marker.writeText(canonicalSupplyJson(json))
                }
            }
            val (_, progress) = downloaded(root, Fixture(nextOperation(f), f.files))
            assertEquals(damage, 0L, progress.resumedBytes)
            assertEquals(damage, f.manifest.files.sumOf { it.size }, progress.actualDownloadedBytes)
        }
    }

    @Test fun `a valid different source marker cannot donate identical bytes to another source`() = runBlocking {
        val f = fixture(); val root = temp.newFolder("foreign-source")
        interruptedDownload(root, f)
        val json = nextOperation(f).json.deepCopy().apply { addProperty("source_model_sha256", "0".repeat(64)) }
        resign(json)
        val next = Fixture(ModelSupplyManifest.parse(json), f.files)
        // Stop after the authoritative full download: the intentionally mismatching native source must still fail.
        try { downloaded(root, next); fail("foreign native source accepted") }
        catch (failure: ModelSupplyException) { assertEquals("SUPPLY_NATIVE_MANIFEST_MISMATCH", failure.reasonCode) }
        assertTrue(scanSuppliedSegmentRoots(root).isEmpty())
    }

    @Test fun `same operation discards crash tail after last checkpoint and revalidates live prefix`() = runBlocking {
        val f = fixture(); val root = temp.newFolder("crash-tail")
        val prefix = interruptedDownload(root, f)
        prefix.partial.appendBytes("uncheckpointed tail".toByteArray())
        val (_, progress) = downloaded(root, f, prefix.length)
        assertEquals(prefix.length, progress.resumedBytes)
    }

    @Test fun `server ignoring Range restarts hashing and counts the full fresh response`() = runBlocking {
        val f = fixture(); val root = temp.newFolder("ignored-range")
        val prefix = interruptedDownload(root, f)
        val (_, progress) = downloaded(root, Fixture(nextOperation(f), f.files), prefix.length, ignoreRange = true)
        assertEquals(0L, progress.resumedBytes)
        assertEquals(f.manifest.files.sumOf { it.size }, progress.actualDownloadedBytes)
    }

    @Test fun `final strong SHA rejects a corrupt suffix after a trusted cross operation prefix`() = runBlocking {
        val f = fixture(); val root = temp.newFolder("bad-suffix")
        val prefix = interruptedDownload(root, f)
        try { downloaded(root, Fixture(nextOperation(f), f.files), prefix.length, corruptSuffix = true); fail("bad final digest accepted") }
        catch (failure: ModelSupplyException) { assertEquals("SUPPLY_ARTIFACT_DIGEST_MISMATCH", failure.reasonCode) }
        assertTrue(scanSuppliedSegmentRoots(root).isEmpty())
        assertFalse(root.resolve(ModelSupplyCurrent.POINTER).exists())
    }


    @Test fun `streaming checkpoints advance by interval and the newest production prefix survives interruption`() = runBlocking {
        val f = fixture(extraWeightBytes = 512 * 1024)
        val root = temp.newFolder("periodic-prefix")
        val checkpoints = mutableListOf<Long>()
        val interval = 64L * 1024
        val prefix = interruptedDownload(root, f, prefixBytes = 384 * 1024, checkpointBytes = interval,
            checkpointSeen = { length -> if (checkpoints.lastOrNull() != length) checkpoints += length })
        assertTrue(checkpoints.size >= 4)
        assertTrue(checkpoints.zipWithNext().all { (before, after) -> after - before >= interval })
        assertTrue(checkpoints.size <= prefix.length / interval)
        val (_, progress) = downloaded(root, Fixture(nextOperation(f), f.files), prefix.length)
        assertEquals(prefix.length, progress.resumedBytes)
        assertEquals(f.manifest.files.sumOf { it.size } - prefix.length, progress.actualDownloadedBytes)
    }

    @Test fun `cross operation prefix follows actual file SHA and size when the basename changes`() = runBlocking {
        val f = fixture(); val root = temp.newFolder("renamed-prefix")
        val prefix = interruptedDownload(root, f)
        val weightName = "renamed.gguf"
        val nativeName = "renamed.manifest.json"
        val native = JsonParser.parseString(f.files.getValue(f.manifest.native.path).toString(Charsets.UTF_8)).asJsonObject
            .apply { addProperty("artifact", weightName) }
        val nativeBytes = canonicalSupplyJson(native).toByteArray()
        val json = nextOperation(f).json.deepCopy()
        json["files"].asJsonArray[0].asJsonObject.addProperty("path", weightName)
        json["files"].asJsonArray[1].asJsonObject.apply {
            addProperty("path", nativeName); addProperty("size_bytes", nativeBytes.size); addProperty("sha256", supplySha256(nativeBytes))
        }
        resign(json)
        val next = Fixture(ModelSupplyManifest.parse(json), mapOf(weightName to f.files.getValue(f.manifest.weight.path), nativeName to nativeBytes))
        val (_, progress) = downloaded(root, next, prefix.length)
        assertEquals(prefix.length, progress.resumedBytes)
        assertEquals(next.manifest.files.sumOf { it.size } - prefix.length, progress.actualDownloadedBytes)
    }

    @Test fun `a self consistent corrupted prefix checkpoint still cannot override the full manifest SHA`() = runBlocking {
        val f = fixture(); val root = temp.newFolder("forged-prefix")
        val prefix = interruptedDownload(root, f)
        prefix.partial.writeBytes(prefix.partial.readBytes().also { it[0] = (it[0].toInt() xor 1).toByte() })
        val journal = prefix.journal.deepCopy()
        journal["prefixes"].asJsonArray[0].asJsonObject.addProperty("prefix_sha256", supplySha256(prefix.partial.readBytes()))
        journal.remove("journal_sha256")
        journal.addProperty("journal_sha256", supplySha256(canonicalSupplyJson(journal).toByteArray()))
        prefix.stage.resolve(ModelSupplyReceiver.RESUME_MARKER).writeText(canonicalSupplyJson(journal))
        try { downloaded(root, Fixture(nextOperation(f), f.files), prefix.length); fail("corrupt full artifact accepted") }
        catch (failure: ModelSupplyException) { assertEquals("SUPPLY_ARTIFACT_DIGEST_MISMATCH", failure.reasonCode) }
        assertTrue(scanSuppliedSegmentRoots(root).isEmpty())
        assertFalse(root.resolve(ModelSupplyCurrent.POINTER).exists())
    }

    @Test fun `transfer diagnostics aggregate five second windows and only fresh bytes without identifiers`() = runBlocking {
        val f = fixture(extraWeightBytes = 512 * 1024)
        val root = temp.newFolder("transfer-timings")
        val snapshots = mutableListOf<ModelSupplyTransferDiagnostic>()
        var nanos = 0L
        val client = shortReadClient(f, afterRead = { _, _ -> nanos += 100_000_000L })
        val final = ModelSupplyReceiver(root, "127.0.0.1", client = client, nanoTime = { nanos }).receive(
            f.manifest, 1, "sensitive-token-never-log", { nanos += 1_000_000L; true },
            { nanos += 2_000_000L }, { it() }, diagnostics = { snapshots += it },
        )
        assertTrue("must aggregate across multiple windows", snapshots.size >= 3)
        assertEquals(1, snapshots.count { it.terminal })
        assertTrue(snapshots.last().terminal)
        assertTrue(snapshots.filterNot { it.terminal }.all { it.elapsedNanos >= 5_000_000_000L })
        val actual = f.manifest.files.sumOf { it.size }
        assertEquals(actual, snapshots.sumOf { it.actualNewBytes })
        assertEquals(actual, snapshots.last().actualTotalBytes)
        assertTrue(snapshots.sumOf { it.readNanos } > 0)
        assertTrue(snapshots.sumOf { it.liveCheckNanos } > 0)
        assertTrue(snapshots.sumOf { it.progressNanos } > 0)
        for (snapshot in snapshots) {
            assertTrue(snapshot.readNanos + snapshot.liveCheckNanos + snapshot.writeHashNanos +
                snapshot.checkpointNanos + snapshot.progressNanos <= snapshot.elapsedNanos)
            assertTrue("log fields must be numeric only", Regex("transfer_perf(?: [a-z_]+=[0-9]+)+").matches(snapshot.logMessage()))
            assertFalse(snapshot.logMessage().contains("sensitive-token"))
            assertFalse(snapshot.logMessage().contains(f.manifest.operationId))
            assertFalse(snapshot.logMessage().contains(f.manifest.weight.path))
        }
        f.files.forEach { (path, bytes) -> assertArrayEquals(bytes, final.resolve(path).readBytes()) }
    }

    @Test fun `diagnostic failure cannot replace IO failure or lose a durable short tail`() = runBlocking {
        val prefixBytes = 256 * 1024 + 11
        val f = fixture(extraWeightBytes = 512 * 1024)
        val root = temp.newFolder("diagnostic-failure")
        val snapshots = mutableListOf<ModelSupplyTransferDiagnostic>()
        val progress = mutableListOf<ModelSupplyProgress>()
        val failure = runCatching {
            ModelSupplyReceiver(root, "127.0.0.1", client = shortReadClient(f, failAfter = prefixBytes))
                .receive(f.manifest, 1, "secret", { true }, { progress += it }, { fail("failed transfer published") },
                    diagnostics = { snapshots += it; throw IOException("logger unavailable") })
        }.exceptionOrNull()
        assertTrue(failure is IOException)
        assertEquals("short-read disconnect", failure!!.message)
        checkpoint(root, f, prefixBytes.toLong())
        assertEquals(prefixBytes.toLong(), progress.last().actualDownloadedBytes)
        assertEquals(prefixBytes.toLong(), snapshots.sumOf { it.actualNewBytes })
        assertTrue(snapshots.last().terminal)
        assertTrue(scanSuppliedSegmentRoots(root).isEmpty())
    }

}
