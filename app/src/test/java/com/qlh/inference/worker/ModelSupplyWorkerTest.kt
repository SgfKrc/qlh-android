package com.qlh.inference.worker

import com.google.gson.JsonObject
import com.google.gson.JsonParser
import androidx.datastore.core.DataStore
import androidx.datastore.preferences.core.Preferences
import androidx.datastore.preferences.core.mutablePreferencesOf
import com.qlh.inference.data.SettingsDataStore
import com.qlh.inference.security.ClusterCredentialStorage
import com.qlh.inference.service.ModelSupplyException
import com.qlh.inference.service.LocalGgufSource
import com.qlh.inference.service.LocalGgufSources
import com.qlh.inference.service.supplyBindingSha256
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withTimeout
import kotlinx.coroutines.yield
import org.junit.Assert.*
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import com.qlh.inference.service.ModelSupplyManifest
import com.qlh.inference.service.canonicalSupplyJson
import com.qlh.inference.service.supplySha256
import com.qlh.inference.service.ModelSupplyCurrent
import java.io.ByteArrayOutputStream
import java.io.File
import java.net.ServerSocket
import java.io.DataInputStream
import java.io.DataOutputStream
import kotlin.concurrent.thread

class ModelSupplyWorkerTest {
    @get:Rule val temp=TemporaryFolder()
    private val operation="supply_"+"1".repeat(32)
    private val node="android_test"
    private fun manifest(): JsonObject {
        val json=JsonParser.parseString("""{
            "manifest_kind":"qlh.segment_supply.v1","operation_id":"$operation","node_id":"$node",
            "source_model_id":"source","source_model_sha256":"${"a".repeat(64)}","tokenizer_sha256":"${"b".repeat(64)}",
            "total_layers":4,"layer_range":[2,4],"segment_mode":"tail","layer_weight_bytes":[0,0,4,4],
            "hidden_size":1,"artifact_sha256":"${"c".repeat(64)}","format":"gguf","conversion_proof":null,
            "files":[{"path":"segment.gguf","size_bytes":10,"sha256":"${"c".repeat(64)}","kind":"weights"},
            {"path":"segment.manifest.json","size_bytes":10,"sha256":"${"d".repeat(64)}","kind":"manifest"}]
        }""").asJsonObject
        json.addProperty("manifest_sha256",supplySha256(canonicalSupplyJson(json).toByteArray()))
        return json
    }
    @Test fun `inventory and local cut failure are named without SAF or native load`()=runBlocking {
        val replies=mutableListOf<JsonObject>()
        var refreshed=false
        val controller=ModelSupplyController(node,"bound-master",temp.newFolder(),{true},{true},{123L},{it()},{fail("must not register local cut")})
        val inventory=JsonParser.parseString("""{"action":"inventory","operation_id":"$operation","model_id":"source","source_model_sha256":"${"a".repeat(64)}","control_certificate":{}}""").asJsonObject
        controller.command(inventory,this,{replies+=it},{refreshed=true})
        assertEquals("inventory_result",replies.last()["action"].asString)
        assertFalse(replies.last()["local_cut"].asBoolean)
        assertTrue(replies.last()["sources"].asJsonArray.isEmpty)
        assertEquals(123L,replies.last()["memory_available_bytes"].asLong)
        val ensure=JsonParser.parseString("""{"action":"ensure","operation_id":"$operation","node_id":"$node","strategy":"local_cut","api_port":8000,"token":"secret"}""").asJsonObject
        ensure.add("manifest",manifest())
        controller.command(ensure,this,{replies+=it},{refreshed=true})
        assertEquals("SUPPLY_LOCAL_CUT_UNAVAILABLE",replies.last()["reason_code"].asString)
        assertEquals(ensure["manifest"].asJsonObject["manifest_sha256"],replies.last()["manifest_sha256"])
        assertFalse(refreshed)
    }
    @Test fun `inventory reports registered artifact only while its files exist`()=runBlocking {
        // #135：master 曾只依据 hello 上报的 capabilities 快照判断「段可复用」。运维清掉
        // 段缓存后快照仍持旧清单 ⇒ 误判可复用 ⇒ 改判 distribute ⇒ 接收端找不到文件 ⇒
        // 退化成真下载（比本地裁层更慢）。清点必须每次重读磁盘，文件没了就报空。
        val replies=mutableListOf<JsonObject>()
        val root=temp.newFolder()
        val binding=supplyBindingSha256("","bound-master",0,0)
        val controller=ModelSupplyController(node,"bound-master",root,{true},{true},{123L},{it()},{})
        val manifest=ModelSupplyManifest.parse(manifest().toString())
        val directory=File(root,"segment-${manifest.manifestSha256}").apply{mkdirs()}
        File(directory,"supply.manifest.json").writeText(canonicalSupplyJson(manifest.json),Charsets.UTF_8)
        File(directory,"segment.gguf").writeBytes(ByteArray(10))
        File(directory,"segment.manifest.json").writeBytes(ByteArray(10))
        val registration=ModelSupplyCurrent(root).select(manifest,binding)
        try {
            val inventory=JsonParser.parseString("""{"action":"inventory","operation_id":"$operation","model_id":"source","source_model_sha256":"${"a".repeat(64)}"}""").asJsonObject

            controller.command(inventory,this,{replies+=it},{})
            val reported=replies.last()["artifacts"].asJsonArray
            assertEquals(1,reported.size())
            val item=reported[0].asJsonObject
            // 逐字段等于 supply manifest —— 正是 master 侧复用判据的输入。
            assertEquals(listOf(2,4),item["layer_range"].asJsonArray.map{it.asInt})
            assertEquals("c".repeat(64),item["artifact_sha256"].asString)
            assertEquals("source",item["source_model_id"].asString)
            assertEquals("a".repeat(64),item["source_model_sha256"].asString)
            assertEquals("b".repeat(64),item["tokenizer_sha256"].asString)

            // 模拟运维清理：删掉段权重，指针与 manifest 原样留着。
            assertTrue(File(directory,"segment.gguf").delete())
            replies.clear()
            controller.command(inventory,this,{replies+=it},{})
            assertTrue("段文件已消失时必须报空，否则 master 会误判可复用",
                replies.last()["artifacts"].asJsonArray.isEmpty)
        } finally { ModelSupplyCurrent(root).rollback(registration,binding) }
    }
    @Test fun `inventory reports no artifact before any registration`()=runBlocking {
        val replies=mutableListOf<JsonObject>()
        val controller=ModelSupplyController(node,"bound-master",temp.newFolder(),{true},{true},{123L},{it()},{})
        controller.command(JsonParser.parseString("""{"action":"inventory","operation_id":"$operation","model_id":"","source_model_sha256":""}""").asJsonObject,
            this,{replies+=it},{})
        assertEquals("inventory_result",replies.last()["action"].asString)
        assertTrue(replies.last()["artifacts"].asJsonArray.isEmpty)
    }
    @Test fun `stale binding refuses commands before network or registration`()=runBlocking {
        val replies=mutableListOf<JsonObject>()
        val controller=ModelSupplyController(node,"bound-master",temp.newFolder(),{true},{false},{0L},{it()},{fail("stale registration")})
        controller.command(JsonParser.parseString("""{"action":"inventory","operation_id":"$operation"}""").asJsonObject,this,{replies+=it},{fail("stale hello")})
        assertEquals("SUPPLY_MASTER_BINDING_CHANGED",replies.single()["reason_code"].asString)
    }
    @Test fun `supply shares outer control fence and outbound decoration`()=runBlocking {
        val cert=JsonParser.parseString("""{"term":2,"expires_at_ms":123456789}""").asJsonObject
        val fence=TcpControlFence(required=true,validate={assertEquals(cert,it)})
        val server=ServerSocket(0)
        var error:Throwable?=null
        val peer=thread {
            try { server.accept().use { socket ->
                val output=DataOutputStream(socket.getOutputStream())
                val bytes="""{"type":"model_supply","format":"json","data":{"action":"inventory","operation_id":"$operation","control_certificate":$cert}}""".toByteArray()
                output.writeInt(bytes.size);output.write(bytes);output.flush()
                val input=DataInputStream(socket.getInputStream());val reply=ByteArray(input.readInt());input.readFully(reply)
                val outer=JsonParser.parseString(reply.toString(Charsets.UTF_8)).asJsonObject
                assertEquals("model_supply",outer["type"].asString)
                assertEquals(cert,outer["data"].asJsonObject["control_certificate"])
            } } catch(t:Throwable){error=t}
        }
        try {
            val socket=java.net.Socket("127.0.0.1",server.localPort)
            val transport=SocketTaskWorkerTransport(socket,controlFence=fence)
            val event=transport.receiveEvent() as TaskWorkerInboundEvent.ModelSupply
            assertEquals(operation,event.data["operation_id"].asString)
            transport.sendModelSupply(JsonObject().apply{addProperty("action","inventory_result")})
            transport.close()
        } finally { server.close();peer.join(3000) }
        assertNull(error)
    }
    @Test fun `unconfigured certificate authority never accepts a claimed HA lease`() {
        val fence=TcpControlFence()
        try { fence.admit(JsonObject().apply{add("control_certificate",JsonObject())});fail("claimed authority") }
        catch(error:TaskWorkerProtocolException){assertEquals("control_fence_unavailable",error.code)}
        val required=TcpControlFence(required=true)
        try { required.admit(JsonObject());fail("missing certificate") }
        catch(error:TaskWorkerProtocolException){assertEquals("control_certificate_missing",error.code)}
    }
    @Test fun `inventory permits unresolved source hash but ensure manifest remains strict`() = runBlocking {
        val replies = mutableListOf<JsonObject>()
        val controller = ModelSupplyController(node, "bound-master", temp.newFolder(), { true }, { true }, { 123L }, { it() }, { fail("no assets to register") })
        val inventory = JsonParser.parseString("""{"action":"inventory","operation_id":"$operation","model_id":"source","source_model_sha256":""}""").asJsonObject
        controller.command(inventory, this, { replies += it }, { fail("inventory never refreshes hello") })
        assertEquals("inventory_result", replies.last()["action"].asString)
        assertEquals("llama_cpp", replies.last()["engines"].asJsonArray.single().asString)
        assertFalse(replies.last()["local_cut"].asBoolean)
        assertTrue(replies.last()["sources"].asJsonArray.isEmpty)
        inventory.addProperty("source_model_sha256", "bad-hash")
        controller.command(inventory, this, { replies += it }, { fail("invalid inventory refresh") })
        assertEquals("SUPPLY_INVALID_REQUEST", replies.last()["reason_code"].asString)
        val invalidManifest = manifest().apply {
            addProperty("source_model_sha256", "")
            remove("manifest_sha256")
            addProperty("manifest_sha256", supplySha256(canonicalSupplyJson(this).toByteArray()))
        }
        val ensure = JsonParser.parseString("""{"action":"ensure","operation_id":"$operation","node_id":"$node","strategy":"distribute","api_port":8000,"token":"secret"}""").asJsonObject
        ensure.add("manifest", invalidManifest)
        controller.command(ensure, this, { replies += it }, { fail("invalid manifest refresh") })
        assertEquals("SUPPLY_INVALID_MANIFEST", replies.last()["reason_code"].asString)
    }

    private fun inventory(modelId: String = "", sourceSha: String = "") = JsonObject().apply {
        addProperty("action", "inventory"); addProperty("operation_id", operation)
        addProperty("model_id", modelId); addProperty("source_model_sha256", sourceSha)
    }

    private fun inventorySources(
        read: suspend (String, String, suspend () -> Boolean) -> List<LocalGgufSource> = { _, _, _ ->
            error("budget inventory must not inspect local sources")
        },
    ) = object : LocalGgufSources {
        override suspend fun inventory(modelId: String, sourceSha256: String, current: suspend () -> Boolean) =
            read(modelId, sourceSha256, current)
        override suspend fun resolve(manifest: ModelSupplyManifest, current: suspend () -> Boolean): LocalGgufSource =
            error("inventory must not resolve a source")
    }

    @Test fun `empty inventory samples fresh budgets without reading sources or changing registered segments`() = runBlocking {
        var disk = 1024L
        var diskReads = 0
        val storage = object : File(temp.newFolder().absolutePath) {
            override fun getUsableSpace(): Long { diskReads++; return disk }
        }
        val root = object : File(storage, "supply") {
            override fun getParentFile(): File = storage
        }.apply { mkdir() }
        val parsed = ModelSupplyManifest.parse(published(root, operation))
        val binding = supplyBindingSha256("", "bound-master", 0, 0)
        val selected = ModelSupplyCurrent(root)
        selected.complete(selected.select(parsed, binding))
        fun snapshot() = root.walkTopDown().filter { it.isFile }.associate {
            it.relativeTo(root).path to (it.lastModified() to it.readBytes().toList())
        }
        val previous = snapshot()
        var enabled = true
        var memory = 0L
        var modeReads = 0
        var memoryReads = 0
        var authorizations = 0
        val replies = mutableListOf<JsonObject>()
        val controller = ModelSupplyController(node, "bound-master", root,
            distributed = { modeReads++; enabled }, bindingCurrent = { true },
            memoryAvailable = { memoryReads++; memory }, publicationGate = { fail("inventory publication") },
            available = { fail("inventory registration") }, localSources = inventorySources())
        for ((index, budget) in listOf(4096L, -1L, 8192L).withIndex()) {
            memory = budget
            disk = 1024L * (index + 1)
            enabled = index != 1
            val done = CompletableDeferred<JsonObject>()
            controller.command(inventory(), this, { replies += it; done.complete(it) }, { fail("inventory hello") },
                authorize = { authorizations++ })
            val response = withTimeout(3000) { done.await() }
            assertEquals(response.toString(), "inventory_result", response["action"].asString)
            assertEquals(enabled, response["distributed"].asBoolean)
            assertEquals(budget.coerceAtLeast(0), response["memory_available_bytes"].asLong)
            assertEquals(disk, response["disk_free_bytes"].asLong)
            assertTrue(response["sources"].asJsonArray.isEmpty)
            assertTrue(response["local_cut"].asBoolean)
            assertEquals("llama_cpp", response["engines"].asJsonArray.single().asString)
            assertFalse(response.has("status"))
            assertEquals(index + 1, modeReads)
            assertEquals(index + 1, memoryReads)
            assertEquals(index + 1, diskReads)
            assertTrue(authorizations >= (index + 1) * 2)
            assertEquals(previous, snapshot())
            assertEquals(parsed.manifestSha256, selected.read(binding)!!.manifestSha256)
        }
        assertEquals(3, replies.size)
        controller.cancelAll()
        assertEquals(previous, snapshot())
    }

    @Test fun `model or source hash selection still inventories matching sources`() = runBlocking {
        val source = LocalGgufSource("source", "a".repeat(64), "b".repeat(64), 123L) {
            error("controller must not open source contents")
        }
        val reads = mutableListOf<Pair<String, String>>()
        val sources = inventorySources { modelId, sourceSha, current ->
            assertTrue(current())
            reads += modelId to sourceSha
            listOf(source)
        }
        val controller = ModelSupplyController(node, "bound-master", temp.newFolder(), { true }, { true }, { 123L },
            { fail("inventory publication") }, { fail("inventory registration") }, localSources = sources)
        val selections = listOf("source" to "", "" to source.sha256, "source" to source.sha256)
        for ((index, selection) in selections.withIndex()) {
            val done = CompletableDeferred<JsonObject>()
            controller.command(inventory(selection.first, selection.second), this, { done.complete(it) }, { fail("inventory hello") })
            val response = withTimeout(3000) { done.await() }
            assertEquals("inventory_result", response["action"].asString)
            assertEquals(selections.take(index + 1), reads)
            val advertised = response["sources"].asJsonArray.single().asJsonObject
            assertEquals(source.modelId, advertised["source_model_id"].asString)
            assertEquals(source.sha256, advertised["source_model_sha256"].asString)
            assertEquals(source.tokenizerSha256, advertised["tokenizer_sha256"].asString)
            assertEquals(source.size, advertised["size_bytes"].asLong)
        }
        controller.cancelAll()
    }

    @Test fun `empty inventory rejects unauthorized requests before sampling budgets`() = runBlocking {
        val replies = mutableListOf<JsonObject>()
        val controller = ModelSupplyController(node, "bound-master", temp.newFolder(), { error("unauthorized mode read") },
            { true }, { error("unauthorized memory read") }, { fail("inventory publication") },
            { fail("inventory registration") }, localSources = inventorySources())
        controller.command(inventory(), this, { replies += it }, { fail("inventory hello") },
            authorize = { throw ModelSupplyException("SUPPLY_TEST_UNAUTHORIZED", "denied") })
        assertEquals("SUPPLY_TEST_UNAUTHORIZED", replies.single()["reason_code"].asString)
        controller.cancelAll()
    }

    @Test fun `empty inventory cannot publish budgets after binding retirement generation change or cancellation`() = runBlocking {
        for (change in listOf("binding", "generation", "cancel")) {
            val entered = CompletableDeferred<Unit>()
            val release = CompletableDeferred<Unit>()
            val replies = mutableListOf<JsonObject>()
            var bound = true
            val controller = ModelSupplyController(node, "bound-master", temp.newFolder(),
                { entered.complete(Unit); release.await(); true }, { bound }, { 123L },
                { fail("inventory publication") }, { fail("inventory registration") }, localSources = inventorySources())
            withTimeout(3000) {
                coroutineScope {
                    controller.command(inventory(), this, { replies += it }, { fail("inventory hello") })
                    try {
                        entered.await()
                        when (change) {
                            "binding" -> bound = false
                            "generation" -> controller.cancelAll()
                            "cancel" -> controller.command(cancel(), this, { replies += it }, { fail("cancel hello") })
                        }
                    } finally { release.complete(Unit) }
                }
            }
            assertFalse(change, replies.any { it["action"].asString == "inventory_result" })
            controller.cancelAll()
        }
    }


    private fun ensure(operationId: String = operation, token: String = "secret", supplied: JsonObject? = null): JsonObject {
        val parsed = supplied ?: manifest().apply {
            addProperty("operation_id", operationId)
            remove("manifest_sha256")
            addProperty("manifest_sha256", supplySha256(canonicalSupplyJson(this).toByteArray()))
        }
        return JsonObject().apply {
            addProperty("action", "ensure"); addProperty("operation_id", operationId); addProperty("node_id", node)
            addProperty("strategy", "distribute"); addProperty("api_port", 1); addProperty("token", token)
            add("manifest", parsed)
        }
    }

    private fun cancel(operationId: String = operation) = JsonObject().apply {
        addProperty("action", "cancel"); addProperty("operation_id", operationId)
    }

    private fun published(root: File, operationId: String): JsonObject {
        val buffer = ByteArrayOutputStream()
        DataOutputStream(buffer).use { out ->
            fun int(value: Int) = out.writeInt(Integer.reverseBytes(value))
            fun long(value: Long) = out.writeLong(java.lang.Long.reverseBytes(value))
            fun string(value: String) { val bytes = value.toByteArray(); long(bytes.size.toLong()); out.write(bytes) }
            out.write("GGUF".toByteArray()); int(3); long(2); long(3)
            string("tokenizer.ggml.model"); int(8); string("test")
            string("general.architecture"); int(8); string("test")
            string("test.block_count"); int(4); int(2)
            for (index in 0..1) { string("blk.$index.attn.weight"); int(1); long(1); int(0); long(index * 4L) }
        }
        val weight = buffer.toByteArray()
        val scan = GgufLayerWeightReader.scan(weight.inputStream(), 2)!!
        val native = JsonParser.parseString("""{
            "artifact":"segment.gguf","artifact_sha256":"${supplySha256(weight)}",
            "source_model_id":"source","source_model_sha256":"${"a".repeat(64)}",
            "tokenizer_sha256":"${scan.tokenizerSha256}","hidden_size":1,"mode":"tail",
            "source_layer_range":[2,4],"n_layer":4,"kept_block_count":2,
            "gguf_metadata":{"architecture":"test","block_count":2,"head_count_kv":null,
                "key_length":null,"full_attention_interval":null},
            "conversion_proof":null,"layer_weight_bytes":[0,0,4,4]
        }""").asJsonObject
        val nativeBytes = canonicalSupplyJson(native).toByteArray()
        val parsed = manifest().apply {
            addProperty("operation_id", operationId)
            addProperty("tokenizer_sha256", scan.tokenizerSha256)
            addProperty("artifact_sha256", supplySha256(weight))
            val files = getAsJsonArray("files")
            files[0].asJsonObject.apply { addProperty("size_bytes", weight.size); addProperty("sha256", supplySha256(weight)) }
            files[1].asJsonObject.apply { addProperty("size_bytes", nativeBytes.size); addProperty("sha256", supplySha256(nativeBytes)) }
            remove("manifest_sha256")
            addProperty("manifest_sha256", supplySha256(canonicalSupplyJson(this).toByteArray()))
        }
        val directory = root.resolve("segment-${parsed["manifest_sha256"].asString}").apply { mkdir() }
        directory.resolve("segment.gguf").writeBytes(weight)
        directory.resolve("segment.manifest.json").writeBytes(nativeBytes)
        directory.resolve(ModelSupplyManifest.MARKER).writeText(canonicalSupplyJson(parsed))
        return parsed
    }

    @Test fun `cancel before ensure refuses all work while inventory and another operation still register`() = runBlocking {
        val root = temp.newFolder()
        val replies = mutableListOf<JsonObject>()
        val registered = mutableListOf<String>()
        var refreshed = 0
        val controller = ModelSupplyController(node, "127.0.0.1", root, { true }, { true }, { 123L }, { it() }, { registered += it.operationId })
        controller.command(cancel(), this, { replies += it }, { fail("cancel refresh") })
        controller.command(ensure(), this, { replies += it }, { fail("late ensure refresh") })
        assertEquals("SUPPLY_CANCELLED", replies.last()["reason_code"].asString)
        assertEquals(2, replies.size)
        assertTrue(root.listFiles()!!.isEmpty())
        controller.command(JsonObject().apply {
            addProperty("action", "inventory"); addProperty("operation_id", operation)
            addProperty("model_id", "source"); addProperty("source_model_sha256", "")
        },
            this, { replies += it }, { fail("inventory refresh") })
        assertEquals("inventory_result", replies.last()["action"].asString)
        val other = "supply_" + "2".repeat(32)
        val parsed = published(root, other)
        val completed = CompletableDeferred<Unit>()
        controller.command(ensure(other, supplied = parsed), this, {
            replies += it
            if (it["status"]?.asString == "available") completed.complete(Unit)
        }, { refreshed++ })
        withTimeout(3000) { completed.await() }
        assertEquals(listOf(other), registered)
        assertEquals(1, refreshed)
        assertEquals("available", replies.last()["status"].asString)
        val previous = root.resolve(ModelSupplyCurrent.POINTER).readBytes()
        controller.command(ensure(), this, { replies += it }, { fail("late retry refresh") })
        assertEquals("SUPPLY_CANCELLED", replies.last()["reason_code"].asString)
        assertArrayEquals(previous, root.resolve(ModelSupplyCurrent.POINTER).readBytes())
    }

    @Test fun `cancel during ensure admission deterministically fences the later continuation`() = runBlocking {
        val root = temp.newFolder()
        val entered = CompletableDeferred<Unit>()
        val release = CompletableDeferred<Unit>()
        val replies = mutableListOf<JsonObject>()
        var pause = true
        val controller = ModelSupplyController(node, "127.0.0.1", root, {
            if (pause) { pause = false; entered.complete(Unit); release.await() }
            true
        }, { true }, { 0L }, { fail("cancelled publication") }, { fail("cancelled registration") })
        val delayed = launch { controller.command(ensure(), this, { replies += it }, { fail("cancelled hello") }) }
        try {
            withTimeout(3000) { entered.await() }
            controller.command(cancel(), this, { replies += it }, { fail("cancel refresh") })
        } finally { release.complete(Unit) }
        withTimeout(3000) { delayed.join() }
        assertEquals(2, replies.size)
        assertTrue(replies.all { it["reason_code"].asString == "SUPPLY_CANCELLED" })
        assertTrue(root.listFiles()!!.isEmpty())
    }

    @Test fun `cancelled late ensure never retires an unrelated active operation`() = runBlocking {
        val root = temp.newFolder()
        val other = "supply_" + "3".repeat(32)
        val parsed = published(root, other)
        val entered = CompletableDeferred<Unit>()
        val release = CompletableDeferred<Unit>()
        val completed = CompletableDeferred<Unit>()
        val replies = mutableListOf<JsonObject>()
        var gateCalls = 0
        var refreshed = 0
        val controller = ModelSupplyController(node, "127.0.0.1", root, { true }, { true }, { 0L }, {
            gateCalls++
            entered.complete(Unit)
            release.await()
            it()
        }, { assertEquals(other, it.operationId) })
        controller.command(ensure(other, supplied = parsed), this, {
            replies += it
            if (it["status"]?.asString == "available") completed.complete(Unit)
        }, { refreshed++ })
        try {
            withTimeout(3000) { entered.await() }
            controller.command(cancel(), this, { replies += it }, { fail("wrong cancel hello") })
            controller.command(ensure(), this, { replies += it }, { fail("late ensure hello") })
            assertEquals("SUPPLY_CANCELLED", replies.last()["reason_code"].asString)
        } finally { release.complete(Unit) }
        withTimeout(3000) { completed.await() }
        assertEquals(1, gateCalls)
        assertEquals(1, refreshed)
        assertEquals(other, replies.last()["operation_id"].asString)
        assertTrue(root.resolve(ModelSupplyCurrent.POINTER).isFile)
    }

    @Test fun `explicit cancel fences a running operation and its repeated ensure at publication`() = runBlocking {
        val root = temp.newFolder()
        val parsed = published(root, operation)
        val entered = CompletableDeferred<Unit>()
        val release = CompletableDeferred<Unit>()
        val replies = mutableListOf<JsonObject>()
        val controller = ModelSupplyController(node, "127.0.0.1", root, { true }, { true }, { 0L }, {
            entered.complete(Unit); release.await(); it()
        }, { fail("cancelled registration") })
        controller.command(ensure(supplied = parsed), this, { replies += it }, { fail("cancelled hello") })
        try {
            withTimeout(3000) { entered.await() }
            controller.command(cancel(), this, { replies += it }, { fail("cancel hello") })
            controller.command(ensure(supplied = parsed), this, { replies += it }, { fail("late ensure hello") })
        } finally { release.complete(Unit) }
        yield()
        assertFalse(replies.any { it["status"]?.asString == "available" })
        assertEquals("SUPPLY_CANCELLED", replies.last()["reason_code"].asString)
        assertFalse(root.resolve(ModelSupplyCurrent.POINTER).exists())
        assertTrue(root.resolve("segment-${parsed["manifest_sha256"].asString}").isDirectory)
    }

    @Test fun `cancelAll fences a pending admission and permits fresh commands in the controller`() = runBlocking {
        val root = temp.newFolder()
        val entered = CompletableDeferred<Unit>()
        val release = CompletableDeferred<Unit>()
        val replies = mutableListOf<JsonObject>()
        var pause = true
        val controller = ModelSupplyController(node, "127.0.0.1", root, { true }, {
            if (pause) { pause = false; entered.complete(Unit); release.await() }
            true
        }, { 0L }, { it() }, { fail("no registration") })
        val delayed = launch { controller.command(ensure(), this, { replies += it }, { fail("retired hello") }) }
        try {
            withTimeout(3000) { entered.await() }
            controller.cancelAll()
        } finally { release.complete(Unit) }
        withTimeout(3000) { delayed.join() }
        assertEquals("SUPPLY_CANCELLED", replies.single()["reason_code"].asString)
        val completed = CompletableDeferred<Unit>()
        controller.command(ensure(token = ""), this, { replies += it; completed.complete(Unit) }, { fail("invalid token hello") })
        withTimeout(3000) { completed.await() }
        assertEquals("SUPPLY_INVALID_REQUEST", replies.last()["reason_code"].asString)
        assertTrue(root.listFiles()!!.isEmpty())
    }

    @Test fun `explicit tombstones are bounded to this controller and do not replay after replacement`() = runBlocking {
        val root = temp.newFolder()
        val replies = mutableListOf<JsonObject>()
        fun controller() = ModelSupplyController(node, "127.0.0.1", root, { true }, { true }, { 0L }, { it() }, { fail("no registration") })
        val first = controller()
        val operations = (0..256).map { "supply_" + it.toString(16).padStart(32, '0') }
        for (operationId in operations) first.command(cancel(operationId), this, { replies += it }, { fail("cancel hello") })
        first.command(ensure(operations.last()), this, { replies += it }, { fail("tombstone hello") })
        assertEquals("SUPPLY_CANCELLED", replies.last()["reason_code"].asString)
        val evicted = CompletableDeferred<Unit>()
        first.command(ensure(operations.first(), token = ""), this, { replies += it; evicted.complete(Unit) }, { fail("invalid token hello") })
        withTimeout(3000) { evicted.await() }
        assertEquals("SUPPLY_INVALID_REQUEST", replies.last()["reason_code"].asString)
        val replaced = CompletableDeferred<Unit>()
        controller().command(ensure(operations.last(), token = ""), this, { replies += it; replaced.complete(Unit) }, { fail("invalid token hello") })
        withTimeout(3000) { replaced.await() }
        assertEquals("SUPPLY_INVALID_REQUEST", replies.last()["reason_code"].asString)
        assertTrue(root.listFiles()!!.isEmpty())
    }

    @Test fun `unauthorized and stale binding cancels cannot poison future operation admission`() = runBlocking {
        val root = temp.newFolder()
        val replies = mutableListOf<JsonObject>()
        var bound = false
        val controller = ModelSupplyController(node, "127.0.0.1", root, { true }, { bound }, { 0L }, { it() }, { fail("no registration") })
        controller.command(cancel(), this, { replies += it }, { fail("stale hello") })
        assertEquals("SUPPLY_MASTER_BINDING_CHANGED", replies.last()["reason_code"].asString)
        bound = true
        controller.command(cancel(), this, { replies += it }, { fail("unauthorized hello") },
            authorize = { throw com.qlh.inference.service.ModelSupplyException("SUPPLY_TEST_UNAUTHORIZED", "denied") })
        assertEquals("SUPPLY_TEST_UNAUTHORIZED", replies.last()["reason_code"].asString)
        val completed = CompletableDeferred<Unit>()
        controller.command(ensure(token = ""), this, { replies += it; completed.complete(Unit) }, { fail("invalid token hello") })
        withTimeout(3000) { completed.await() }
        assertEquals("SUPPLY_INVALID_REQUEST", replies.last()["reason_code"].asString)
    }

    private class ParticipationStore(var prefs: Preferences) : DataStore<Preferences> {
        var reads = 0
        var afterRead: (() -> Unit)? = null
        override val data: Flow<Preferences> = flow {
            val snapshot = prefs
            reads++
            afterRead?.invoke()
            emit(snapshot)
        }
        override suspend fun updateData(transform: suspend (Preferences) -> Preferences): Preferences =
            transform(prefs).also { prefs = it }
    }

    private fun participationPrefs(
        mode: String = SettingsDataStore.MODE_DISTRIBUTED,
        bootstrapped: Boolean = true,
        cluster: String = "cluster-a",
        host: String = "127.0.0.1",
        port: Int = 8888,
        epoch: Int = 7,
    ): Preferences = mutablePreferencesOf().apply {
        this[SettingsDataStore.KEY_INFERENCE_MODE] = mode
        this[SettingsDataStore.KEY_BOOTSTRAPPED] = bootstrapped
        this[SettingsDataStore.KEY_CLUSTER_ID] = cluster
        this[SettingsDataStore.KEY_MASTER_TCP_HOST] = host
        this[SettingsDataStore.KEY_MASTER_TCP_PORT] = port
        this[SettingsDataStore.KEY_CLUSTER_SECRET_EPOCH] = epoch
        this[SettingsDataStore.KEY_CLUSTER_SECRET] = "legacy-must-not-be-read"
    }.toPreferences()

    private fun metadataSettings(store: ParticipationStore) = SettingsDataStore(store, object : ClusterCredentialStorage {
        override fun read(): String? = throw AssertionError("metadata must not decrypt credentials")
        override fun save(clusterSecret: String) = throw AssertionError("metadata must not migrate credentials")
        override fun clear() = throw AssertionError("metadata must not clear credentials")
    })

    @Test fun `participation reads one coherent fresh preferences snapshot without credentials`() = runBlocking {
        val store = ParticipationStore(participationPrefs())
        val settings = metadataSettings(store)
        store.afterRead = {
            store.prefs = participationPrefs(SettingsDataStore.MODE_LOCAL, false, "cluster-b", "192.0.2.2", 9999, 8)
        }
        val first = settings.getWorkerParticipationSnapshot()
        assertEquals(1, store.reads)
        assertTrue(first.participationCurrent("cluster-a", "127.0.0.1", 8888, 7))
        assertEquals(SettingsDataStore.MODE_DISTRIBUTED, first.inferenceMode)
        assertEquals("cluster-a", first.clusterId)
        assertEquals(7, first.clusterSecretEpoch)
        val second = settings.getWorkerParticipationSnapshot()
        assertEquals(2, store.reads)
        assertEquals(SettingsDataStore.MODE_LOCAL, second.inferenceMode)
        assertFalse(second.bootstrapped)
        assertEquals("cluster-b", second.clusterId)
        assertEquals("192.0.2.2", second.masterTcpHost)
        assertEquals(9999, second.masterTcpPort)
        assertEquals(8, second.clusterSecretEpoch)
        assertFalse(second.participationCurrent("cluster-a", "127.0.0.1", 8888, 7))
    }

    @Test fun `participation fences every mode and binding component from the same snapshot`() = runBlocking {
        val store = ParticipationStore(participationPrefs())
        val settings = metadataSettings(store)
        val rejected = listOf(
            participationPrefs(mode = SettingsDataStore.MODE_LOCAL),
            participationPrefs(mode = SettingsDataStore.MODE_FALLBACK),
            participationPrefs(bootstrapped = false),
            participationPrefs(cluster = "cluster-b"),
            participationPrefs(host = "192.0.2.2"),
            participationPrefs(port = 9999),
            participationPrefs(epoch = 8),
        )
        for (prefs in rejected) {
            store.prefs = prefs
            assertFalse(settings.getWorkerParticipationSnapshot().participationCurrent("cluster-a", "127.0.0.1", 8888, 7))
        }
        assertEquals(rejected.size, store.reads)
        store.prefs = participationPrefs(mode = "thin", host = "[::1]", epoch = 0)
        assertTrue(settings.getWorkerParticipationSnapshot().participationCurrent("cluster-a", "::1", 8888, 1))
    }

    @Test fun `ensure refuses mixed admission generations using the combined participation check`() = runBlocking {
        val root = temp.newFolder()
        val store = ParticipationStore(participationPrefs(mode = SettingsDataStore.MODE_LOCAL))
        val settings = metadataSettings(store)
        val replies = mutableListOf<JsonObject>()
        val controller = ModelSupplyController(node, "127.0.0.1", root,
            distributed = {
                // Separate reads could accept old binding together with this new mode.
                store.prefs = participationPrefs(cluster = "cluster-b")
                true
            },
            bindingCurrent = { settings.getWorkerParticipationSnapshot().bindingCurrent("cluster-a", "127.0.0.1", 8888, 7) },
            memoryAvailable = { 0L }, publicationGate = { fail("mixed generation publication") },
            available = { fail("mixed generation registration") },
            participationCurrent = { settings.getWorkerParticipationSnapshot().participationCurrent("cluster-a", "127.0.0.1", 8888, 7) })
        controller.command(ensure(), this, { replies += it }, { fail("mixed generation hello") })
        assertEquals("SUPPLY_CANCELLED", replies.single()["reason_code"].asString)
        assertEquals(2, store.reads)
        assertTrue(root.listFiles()!!.isEmpty())
    }

    @Test fun `current uses one fresh combined read and fences mode or old binding before publication`() = runBlocking {
        for (changed in listOf(participationPrefs(mode = SettingsDataStore.MODE_LOCAL), participationPrefs(epoch = 8))) {
            val root = temp.newFolder()
            val parsed = published(root, operation)
            val store = ParticipationStore(participationPrefs())
            val settings = metadataSettings(store)
            val replies = mutableListOf<JsonObject>()
            val stopped = CompletableDeferred<Unit>()
            var currentCalls = 0
            var bindingCalls = 0
            var modeCalls = 0
            val controller = ModelSupplyController(node, "127.0.0.1", root,
                distributed = { modeCalls++; true },
                bindingCurrent = { bindingCalls++; true },
                memoryAvailable = { 0L }, publicationGate = { fail("stale participation publication") },
                available = { fail("stale participation registration") },
                participationCurrent = {
                    currentCalls++
                    val current = settings.getWorkerParticipationSnapshot().participationCurrent("cluster-a", "127.0.0.1", 8888, 7)
                    // Admission is valid; the next current() must see the changed metadata.
                    if (currentCalls == 1) store.prefs = changed
                    if (!current) stopped.complete(Unit)
                    current
                })
            controller.command(ensure(supplied = parsed), this, { replies += it }, { fail("stale participation hello") })
            withTimeout(3000) { stopped.await() }
            yield()
            assertTrue(currentCalls >= 2)
            assertEquals(currentCalls, store.reads)
            assertEquals(1, bindingCalls)
            assertEquals(1, modeCalls)
            assertTrue(replies.isEmpty())
            assertFalse(root.resolve(ModelSupplyCurrent.POINTER).exists())
            controller.cancelAll()
        }
    }

    @Test fun `publication gate still rejects mode and binding changes after the last current check`() = runBlocking {
        for (changed in listOf(participationPrefs(mode = SettingsDataStore.MODE_LOCAL), participationPrefs(cluster = "cluster-b"))) {
            val root = temp.newFolder()
            val parsed = published(root, operation)
            val store = ParticipationStore(participationPrefs())
            val settings = metadataSettings(store)
            val entered = CompletableDeferred<Unit>()
            val release = CompletableDeferred<Unit>()
            val rejected = CompletableDeferred<Unit>()
            val replies = mutableListOf<JsonObject>()
            val controller = ModelSupplyController(node, "127.0.0.1", root,
                distributed = { true }, bindingCurrent = { true }, memoryAvailable = { 0L },
                publicationGate = { action ->
                    entered.complete(Unit)
                    release.await()
                    val snapshot = settings.getWorkerParticipationSnapshot()
                    if (!snapshot.participationCurrent("cluster-a", "127.0.0.1", 8888, 7)) {
                        rejected.complete(Unit)
                        throw ModelSupplyException("SUPPLY_MASTER_BINDING_CHANGED", "participation changed")
                    }
                    action()
                }, available = { fail("stale gate registration") },
                participationCurrent = { settings.getWorkerParticipationSnapshot().participationCurrent("cluster-a", "127.0.0.1", 8888, 7) })
            controller.command(ensure(supplied = parsed), this, { replies += it }, { fail("stale gate hello") })
            try {
                withTimeout(3000) { entered.await() }
                store.prefs = changed
            } finally { release.complete(Unit) }
            withTimeout(3000) { rejected.await() }
            yield()
            assertFalse(replies.any { it["status"]?.asString == "available" })
            assertFalse(root.resolve(ModelSupplyCurrent.POINTER).exists())
            controller.cancelAll()
        }
    }

}
