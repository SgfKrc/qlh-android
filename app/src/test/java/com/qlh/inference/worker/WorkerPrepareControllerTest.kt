package com.qlh.inference.worker

import com.google.gson.JsonObject
import com.google.gson.JsonParser
import com.qlh.inference.status.AndroidRuntimeStatus
import com.qlh.inference.status.AndroidRuntimeStatusInvalidation
import com.qlh.inference.status.ModelRuntimeStatus
import com.qlh.inference.ui.modelRuntimeStatusLabel
import kotlinx.coroutines.CoroutineStart
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.NonCancellable
import kotlinx.coroutines.async
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withContext
import kotlinx.coroutines.withTimeout
import kotlinx.coroutines.yield
import org.junit.Assert.*
import org.junit.Test

class WorkerPrepareControllerTest {
    private val node = "android_test"
    private val largeGeneration = 1_791_800_000_000_000_013L
    private fun identity(generation: Long = largeGeneration, operation: String = "supply_one", artifact: String = "c".repeat(64)): JsonObject =
        JsonParser.parseString("""{
            "operation_id":"$operation","config_id":"config:$generation","generation":$generation,
            "assignment_id":"assignment:$generation","node_id":"$node","model_id":"master-hf",
            "model_sha256":"${"a".repeat(64)}","stage_model_identity":{"model_id":"tail.gguf","engine":"llama_cpp","format":"gguf","revision":"local","sha256":"$artifact"},
            "artifact_sha256":"$artifact","manifest_sha256":"","source_model_id":"gguf-source",
            "source_model_sha256":"${"b".repeat(64)}","tokenizer_sha256":"${"d".repeat(64)}",
            "layer_range":[2,4],"total_layers":4,"hidden_size":1,"segment_mode":"tail"
        }""").asJsonObject
    private fun request(action: String, identity: JsonObject = identity()): JsonObject = JsonObject().apply {
        addProperty("action", action); add("identity", identity)
        if (action == "prepare") { addProperty("context_size", 4096); addProperty("middle_channel", "keep_head_layer_out") }
        if (action in setOf("abort", "finalize", "release")) addProperty("cleanup_id", "cleanup:$action")
    }
    private fun offer(identity: JsonObject = identity(), tuple: Boolean = true): TaskWorkerEnvelope {
        val i = WorkerPrepareIdentity.parse(identity)
        val root = mapOf("context_size" to 4096, "want_hidden" to false)
        val payload = mutableMapOf<String, Any?>("stage_type" to "layer_forward", "model_identity" to i.stageIdentity,
            "layer_range" to listOf(2L, 4L), "hidden_spec" to mapOf("n_embd" to 1L),
            "middle_channel" to "keep_head_layer_out", "root_input" to root)
        if (tuple) payload["prepare_identity"] = mapOf("operation_id" to i.text("operation_id"), "config_id" to i.text("config_id"),
            "generation" to i.generation, "assignment_id" to i.text("assignment_id"), "artifact_sha256" to i.text("artifact_sha256"))
        return TaskWorkerEnvelope("qlh.task_worker", 3, "stage_offer", "msg_offer_test", 10L, payload)
    }
    private open class Runtime : WorkerPrepareRuntime {
        var current: WorkerPrepareIdentity? = null
        var loads = 0
        var verifications = 0
        var unloads = 0
        var failLoad = false
        var failUnload = false
        var unownedSlot = false
        override fun currentPreparedIdentity(): WorkerPrepareIdentity? = current
        override suspend fun load(identity: WorkerPrepareIdentity) {
            verifications++
            if (failLoad) throw WorkerPrepareException("prepare_artifact_identity_mismatch", "verify/load failed")
            if (current?.text("artifact_sha256") != identity.text("artifact_sha256")) loads++
            current = identity
        }
        override suspend fun unload() { unloads++; if (failUnload) throw IllegalStateException("native free failed"); current = null }
        override fun loaded(identity: WorkerPrepareIdentity): Boolean = current == identity
        override fun hasRuntime(): Boolean = current != null || unownedSlot
    }
    private suspend fun transact(controller: WorkerPrepareController, request: JsonObject, scope: kotlinx.coroutines.CoroutineScope): JsonObject {
        val replies = mutableListOf<JsonObject>()
        controller.command(request, scope, { replies += it }, {})
        controller.awaitIdle()
        return replies.single()
    }

    @Test fun `real native ready must precede prepared and commit precedes stage`() = runBlocking {
        val started = CompletableDeferred<Unit>(); val finish = CompletableDeferred<Unit>()
        val runtime = object : Runtime() {
            override suspend fun load(identity: WorkerPrepareIdentity) { started.complete(Unit); finish.await(); super.load(identity) }
        }
        val c = WorkerPrepareController(node, runtime); val replies = mutableListOf<JsonObject>()
        assertEquals("prepare_not_committed", c.validateStage(offer(tuple = false)))
        c.command(request("prepare"), this, { replies += it }, {})
        withTimeout(5000) { started.await() }
        assertTrue(replies.isEmpty()); assertEquals("prepare_not_committed", c.validateStage(offer()))
        finish.complete(Unit); c.awaitIdle()
        assertEquals("prepared", replies.single()["status"].asString); assertTrue(replies.single()["loaded"].asBoolean)
        assertEquals(identity(), replies.single()["identity"])
        assertEquals("prepare_not_committed", c.validateStage(offer()))
        val commit = transact(c, request("commit"), this)
        assertEquals("ready", commit["status"].asString); assertNull(c.validateStage(offer()))
        assertEquals("prepare_identity_mismatch", c.validateStage(offer(tuple = false)))
    }

    @Test fun `sha reuse verifies new operation and still requires a new committed generation`() = runBlocking {
        val runtime = Runtime(); val c = WorkerPrepareController(node, runtime)
        transact(c, request("prepare"), this); transact(c, request("commit"), this); transact(c, request("finalize"), this)
        val next = identity(largeGeneration + 1, "supply_two")
        assertEquals("prepared", transact(c, request("prepare", next), this)["status"].asString)
        assertEquals(1, runtime.loads); assertEquals(2, runtime.verifications)
        assertEquals("prepare_not_committed", c.validateStage(offer(next)))
        assertEquals("prepare_not_committed", c.validateStage(offer()))
        transact(c, request("commit", next), this)
        assertNull(c.validateStage(offer(next))); assertEquals("prepare_identity_mismatch", c.validateStage(offer()))
        val stale = transact(c, request("release"), this)
        assertEquals("released", stale["status"].asString); assertFalse(stale["loaded"].asBoolean)
        assertTrue(runtime.loaded(WorkerPrepareIdentity.parse(next)))
        assertNull(c.validateStage(offer(next)))
    }

    @Test fun `same loaded artifact contract permits a new generation but fences changed content or load spec`() {
        val loaded = WorkerPrepareIdentity.parse(identity())
        val nextWire = identity(largeGeneration + 1, "supply_two").apply {
            addProperty("manifest_sha256", "e".repeat(64))
        }
        val next = WorkerPrepareIdentity.parse(nextWire)
        assertTrue(loaded.sameLoadedArtifact(next))

        val changedArtifact = nextWire.deepCopy().apply {
            addProperty("artifact_sha256", "f".repeat(64))
            getAsJsonObject("stage_model_identity").addProperty("sha256", "f".repeat(64))
        }
        assertFalse(loaded.sameLoadedArtifact(WorkerPrepareIdentity.parse(changedArtifact)))

        val changedSpec = identity(largeGeneration + 1, "supply_two").apply {
            addProperty("load_spec_digest", "f".repeat(64))
        }
        assertFalse(loaded.sameLoadedArtifact(WorkerPrepareIdentity.parse(changedSpec)))
        val loadedWithSpec = WorkerPrepareIdentity.parse(identity().apply {
            addProperty("load_spec_digest", "f".repeat(64))
        })
        assertTrue(loadedWithSpec.sameLoadedArtifact(WorkerPrepareIdentity.parse(changedSpec)))
        assertFalse(loadedWithSpec.sameLoadedArtifact(WorkerPrepareIdentity.parse(changedSpec.deepCopy().apply {
            addProperty("load_spec_digest", "e".repeat(64))
        })))
        for ((key, changed) in mapOf("source_model_sha256" to "e".repeat(64),
            "tokenizer_sha256" to "e".repeat(64), "node_id" to "other_node")) {
            assertFalse(key, loaded.sameLoadedArtifact(WorkerPrepareIdentity.parse(nextWire.deepCopy().apply {
                addProperty(key, changed)
            })))
        }
    }

    @Test fun `native failure emits error no prepared and fences prior runtime`() = runBlocking {
        val runtime = Runtime(); val c = WorkerPrepareController(node, runtime)
        transact(c, request("prepare"), this); transact(c, request("commit"), this); transact(c, request("finalize"), this)
        runtime.failLoad = true
        val next = identity(largeGeneration + 1, "supply_two", "e".repeat(64))
        val ack = transact(c, request("prepare", next), this)
        assertEquals("error", ack["status"].asString); assertFalse(ack["loaded"].asBoolean)
        assertNull(runtime.current); assertEquals("prepare_not_committed", c.validateStage(offer()))
        assertEquals("released", transact(c, request("abort", next), this)["status"].asString)
    }

    @Test fun `abort before commit really unloads and stale commit cannot resurrect candidate`() = runBlocking {
        val runtime = Runtime(); val c = WorkerPrepareController(node, runtime)
        transact(c, request("prepare"), this)
        val ack = transact(c, request("abort"), this)
        assertEquals("released", ack["status"].asString); assertFalse(ack["loaded"].asBoolean)
        assertEquals("cleanup:abort", ack["cleanup_id"].asString); assertNull(runtime.current)
        assertEquals("error", transact(c, request("commit"), this)["status"].asString)
        assertEquals("error", transact(c, request("prepare"), this)["status"].asString)
    }

    @Test fun `abort after commit safely fences previous one-slot runtime`() = runBlocking {
        val runtime = Runtime(); val c = WorkerPrepareController(node, runtime)
        transact(c, request("prepare"), this); transact(c, request("commit"), this); transact(c, request("finalize"), this)
        val next = identity(largeGeneration + 1, "supply_two", "e".repeat(64))
        transact(c, request("prepare", next), this); transact(c, request("commit", next), this)
        val ack = transact(c, request("abort", next), this)
        assertEquals("released", ack["status"].asString); assertFalse(ack["loaded"].asBoolean)
        assertNull(runtime.current); assertEquals("prepare_not_committed", c.validateStage(offer()))
    }

    @Test fun `finalize preserves active runtime and refuses later abort`() = runBlocking {
        val runtime = Runtime(); val c = WorkerPrepareController(node, runtime)
        transact(c, request("prepare"), this); transact(c, request("commit"), this)
        val ack = transact(c, request("finalize"), this)
        assertEquals("finalized", ack["status"].asString); assertTrue(ack["loaded"].asBoolean)
        assertEquals("finalized", transact(c, request("finalize"), this)["status"].asString)
        assertEquals("error", transact(c, request("abort"), this)["status"].asString)
        assertEquals(0, runtime.unloads); assertNull(c.validateStage(offer()))
    }

    @Test fun `release waits actual native unload and duplicates do not free a new context`() = runBlocking {
        val entered = CompletableDeferred<Unit>(); val finish = CompletableDeferred<Unit>()
        val runtime = object : Runtime() { override suspend fun unload() { entered.complete(Unit); finish.await(); super.unload() } }
        val c = WorkerPrepareController(node, runtime)
        transact(c, request("prepare"), this); transact(c, request("commit"), this)
        val replies = mutableListOf<JsonObject>()
        c.command(request("release"), this, { replies += it }, {})
        entered.await(); assertTrue(replies.isEmpty()); assertEquals("prepare_not_committed", c.validateStage(offer()))
        finish.complete(Unit); c.awaitIdle()
        assertEquals("released", replies.single()["status"].asString); assertFalse(replies.single()["loaded"].asBoolean)
        assertEquals("cleanup:release", replies.single()["cleanup_id"].asString)
        val next = identity(largeGeneration + 1, "supply_two")
        transact(c, request("prepare", next), this); transact(c, request("commit", next), this)
        assertEquals("released", transact(c, request("release"), this)["status"].asString)
        assertNull(c.validateStage(offer(next))); assertEquals(1, runtime.unloads)
    }

    @Test fun `release for prior process identity settles without unloading newer runtime`() = runBlocking {
        val runtime = Runtime()
        val newer = identity(largeGeneration + 1, "new_process_assignment")
        runtime.current = WorkerPrepareIdentity.parse(newer)
        val restarted = WorkerPrepareController(node, runtime)

        val ack = transact(restarted, request("release", identity()), this)

        assertEquals("released", ack["status"].asString)
        assertFalse(ack["loaded"].asBoolean)
        assertEquals("cleanup:release", ack["cleanup_id"].asString)
        assertTrue(runtime.loaded(WorkerPrepareIdentity.parse(newer)))
        assertEquals(0, runtime.unloads)
    }

    @Test fun `release from a future generation cannot clear an older native slot`() = runBlocking {
        val runtime = Runtime().apply { current = WorkerPrepareIdentity.parse(identity()) }
        val restarted = WorkerPrepareController(node, runtime)

        val ack = transact(restarted, request("release", identity(largeGeneration + 1, "future_assignment")), this)

        assertEquals("error", ack["status"].asString)
        assertFalse(ack["loaded"].asBoolean)
        assertTrue(runtime.loaded(WorkerPrepareIdentity.parse(identity())))
        assertEquals(0, runtime.unloads)
    }

    @Test fun `remote release refreshes the native status card only after actual unload without a mode switch`() = runBlocking {
        val changes = AndroidRuntimeStatusInvalidation()
        val entered = CompletableDeferred<Unit>()
        val finish = CompletableDeferred<Unit>()
        val runtime = object : Runtime() {
            override suspend fun unload() = changes.afterModelChange {
                entered.complete(Unit)
                finish.await()
                super.unload()
            }
        }
        val c = WorkerPrepareController(node, runtime)
        transact(c, request("prepare"), this); transact(c, request("commit"), this)
        val ui = MutableStateFlow(AndroidRuntimeStatus())
        val observer = launch(start = CoroutineStart.UNDISPATCHED) {
            changes.revisions.collect {
                changes.readSnapshot {
                    AndroidRuntimeStatus(inferenceMode = "distributed", model = ModelRuntimeStatus(
                        loaded = runtime.hasRuntime(), name = if (runtime.hasRuntime()) "native crop" else "",
                        loadedLayerRange = runtime.current?.range.orEmpty(),
                    ))
                }?.let { ui.value = it }
            }
        }
        try {
            assertEquals("已加载层段 [2,4)：native crop", modelRuntimeStatusLabel(ui.value.model))
            val replies = mutableListOf<JsonObject>()
            c.command(request("release"), this, { replies += it }, {})
            withTimeout(5000) { entered.await() }
            assertTrue(ui.value.model.loaded)
            assertEquals(0L, changes.revisions.value)
            assertTrue(replies.isEmpty())
            finish.complete(Unit)
            withTimeout(5000) {
                c.awaitIdle()
                ui.first { !it.model.loaded }
            }
            assertEquals("尚未加载模型", modelRuntimeStatusLabel(ui.value.model))
            assertEquals("distributed", ui.value.inferenceMode)
            assertEquals("released", replies.single()["status"].asString)
            // An old release after re-prepare must not notify an unload of the new owner.
            val next = identity(largeGeneration + 1, "supply_two")
            transact(c, request("prepare", next), this); transact(c, request("commit", next), this)
            val revision = changes.revisions.value
            transact(c, request("release"), this)
            assertEquals(revision, changes.revisions.value)
            assertTrue(runtime.loaded(WorkerPrepareIdentity.parse(next)))
        } finally {
            finish.complete(Unit)
            observer.cancel()
        }
    }

    @Test fun `failed remote free refreshes actual loaded facts instead of pretending the slot is empty`() = runBlocking {
        val changes = AndroidRuntimeStatusInvalidation()
        val runtime = object : Runtime() {
            override suspend fun unload() = changes.afterModelChange { super.unload() }
        }
        val c = WorkerPrepareController(node, runtime)
        transact(c, request("prepare"), this); transact(c, request("commit"), this)
        val refreshed = CompletableDeferred<AndroidRuntimeStatus>()
        val observer = launch(start = CoroutineStart.UNDISPATCHED) {
            changes.revisions.collect { revision ->
                val snapshot = changes.readSnapshot {
                    AndroidRuntimeStatus(model = ModelRuntimeStatus(loaded = runtime.hasRuntime()))
                }
                if (revision > 0 && snapshot != null) refreshed.complete(snapshot)
            }
        }
        try {
            runtime.failUnload = true
            val failed = transact(c, request("release"), this)
            assertEquals("prepare_unload_failed", failed["error"].asString)
            assertTrue(withTimeout(5000) { refreshed.await() }.model.loaded)
            assertTrue(runtime.hasRuntime())
            runtime.failUnload = false
            assertEquals("released", transact(c, request("release"), this)["status"].asString)
            assertFalse(changes.readSnapshot {
                AndroidRuntimeStatus(model = ModelRuntimeStatus(loaded = runtime.hasRuntime()))
            }!!.model.loaded)
        } finally {
            observer.cancel()
        }
    }

    @Test fun `failed unload never sends released and retry requires successful free`() = runBlocking {
        val runtime = Runtime(); val c = WorkerPrepareController(node, runtime)
        transact(c, request("prepare"), this); transact(c, request("commit"), this)
        runtime.failUnload = true
        val failed = transact(c, request("release"), this)
        assertEquals("error", failed["status"].asString); assertEquals("prepare_unload_failed", failed["error"].asString)
        assertTrue(failed["loaded"].asBoolean); assertEquals("prepare_not_committed", c.validateStage(offer()))
        runtime.failUnload = false
        val retry = transact(c, request("release"), this)
        assertEquals("released", retry["status"].asString); assertFalse(retry["loaded"].asBoolean); assertNull(runtime.current)
    }

    @Test fun `disconnect joins noncancellable native load then unloads without late ACK`() = runBlocking {
        val entered = CompletableDeferred<Unit>(); val finish = CompletableDeferred<Unit>()
        val runtime = object : Runtime() {
            override suspend fun load(identity: WorkerPrepareIdentity) = withContext(NonCancellable) {
                entered.complete(Unit); finish.await(); super.load(identity)
            }
        }
        val c = WorkerPrepareController(node, runtime); val replies = mutableListOf<JsonObject>()
        c.command(request("prepare"), this, { replies += it }, {})
        entered.await(); c.cancelAll()
        val retirement = async { c.retireAndJoin() }; yield()
        assertFalse(retirement.isCompleted); assertEquals(0, runtime.unloads)
        finish.complete(Unit); withTimeout(5000) { retirement.await() }
        assertTrue(replies.isEmpty()); assertNull(runtime.current)
    }

    @Test fun `abort cancels inflight JNI joins it and sends only real cleanup ACK`() = runBlocking {
        val entered = CompletableDeferred<Unit>(); val finish = CompletableDeferred<Unit>()
        val runtime = object : Runtime() {
            override suspend fun load(identity: WorkerPrepareIdentity) = withContext(NonCancellable) {
                entered.complete(Unit); finish.await(); super.load(identity)
            }
        }
        val c = WorkerPrepareController(node, runtime); val replies = mutableListOf<JsonObject>()
        c.command(request("prepare"), this, { replies += it }, {})
        entered.await(); c.command(request("abort"), this, { replies += it }, {})
        yield(); assertTrue(replies.isEmpty())
        finish.complete(Unit); withTimeout(5000) { c.awaitIdle() }
        assertEquals(1, replies.size); assertEquals("abort", replies.single()["phase"].asString)
        assertEquals("released", replies.single()["status"].asString); assertNull(runtime.current)
    }

    @Test fun `expired authority after JNI suppresses all ACK and reclaims candidate`() = runBlocking {
        val runtime = Runtime(); var authority = true
        val c = WorkerPrepareController(node, object : WorkerPrepareRuntime by runtime {
            override suspend fun load(identity: WorkerPrepareIdentity) { runtime.load(identity); authority = false }
        })
        val replies = mutableListOf<JsonObject>()
        c.command(request("prepare"), this, { replies += it }, { if (!authority) throw TaskWorkerProtocolException("expired", "control_certificate_expired") })
        c.awaitIdle(); assertTrue(replies.isEmpty()); assertNull(runtime.current)
    }

    @Test fun `participation loss during native load fences runtime and suppresses ACK`() = runBlocking {
        val runtime = Runtime(); var participating = true
        val c = WorkerPrepareController(node, object : WorkerPrepareRuntime by runtime {
            override suspend fun load(identity: WorkerPrepareIdentity) { runtime.load(identity); participating = false }
        }, { participating })
        val replies = mutableListOf<JsonObject>()
        c.command(request("prepare"), this, { replies += it }, {})
        c.awaitIdle(); assertTrue(replies.isEmpty()); assertNull(runtime.current)
    }

    @Test fun `cleanup waits stage ownership and stage cannot run after retirement`() = runBlocking {
        val runtime = Runtime(); val c = WorkerPrepareController(node, runtime)
        transact(c, request("prepare"), this); transact(c, request("commit"), this)
        val entered = CompletableDeferred<Unit>(); val finish = CompletableDeferred<Unit>()
        val stage = launch { c.executeStage(offer()) { entered.complete(Unit); finish.await(); TaskWorkerStageExecution(emptyMap(), emptyMap()) } }
        entered.await()
        val replies = mutableListOf<JsonObject>()
        c.command(request("release"), this, { replies += it }, {}); yield()
        assertEquals(0, runtime.unloads); assertTrue(replies.isEmpty())
        finish.complete(Unit); stage.join(); c.awaitIdle()
        assertNull(runtime.current); assertEquals("released", replies.single()["status"].asString)
        try { c.executeStage(offer()) { fail("no native stage"); error("unreachable") }; fail("retired stage") }
        catch (e: AndroidFullWorkerStageException) { assertEquals("prepare_not_committed", e.code) }
    }

    @Test fun `queued prepare cancelled behind a stage proves no candidate and preserves current runtime`() = runBlocking {
        val runtime = Runtime(); val c = WorkerPrepareController(node, runtime)
        transact(c, request("prepare"), this); transact(c, request("commit"), this); transact(c, request("finalize"), this)
        val entered = CompletableDeferred<Unit>(); val finish = CompletableDeferred<Unit>()
        val stage = launch { c.executeStage(offer()) { entered.complete(Unit); finish.await(); TaskWorkerStageExecution(emptyMap(), emptyMap()) } }
        entered.await()
        val next = identity(largeGeneration + 1, "queued_supply")
        val replies = mutableListOf<JsonObject>()
        c.command(request("prepare", next), this, { replies += it }, {}); yield()
        c.command(request("abort", next), this, { replies += it }, {}); yield()
        assertEquals(1, runtime.loads); assertTrue(replies.isEmpty())
        finish.complete(Unit); stage.join(); c.awaitIdle()
        assertEquals("released", replies.single()["status"].asString); assertFalse(replies.single()["loaded"].asBoolean)
        assertEquals(0, runtime.unloads); assertNull(c.validateStage(offer()))
        for (action in listOf("release", "finalize")) assertFalse(transact(c, request(action, next), this)["loaded"].asBoolean)
        assertEquals("error", transact(c, request("prepare", next), this)["status"].asString)
    }

    @Test fun `late abort does not cancel duplicate prepare of finalized active generation`() = runBlocking {
        val runtime = Runtime(); var block = false
        val entered = CompletableDeferred<Unit>(); val finish = CompletableDeferred<Unit>()
        val c = WorkerPrepareController(node, runtime, participationCurrent = {
            if (block) { entered.complete(Unit); finish.await() }
            true
        })
        transact(c, request("prepare"), this); transact(c, request("commit"), this); transact(c, request("finalize"), this)
        block = true
        val replies = mutableListOf<JsonObject>()
        c.command(request("prepare"), this, { replies += it }, {}); entered.await()
        c.command(request("abort"), this, { replies += it }, {}); yield()
        finish.complete(Unit); c.awaitIdle()
        assertEquals(listOf("prepared", "error"), replies.map { it["status"].asString })
        assertEquals(0, runtime.unloads); assertNull(c.validateStage(offer()))
    }

    @Test fun `nanosecond generation above 2 to 53 roundtrips exactly through JSON and stage decoder`() = runBlocking {
        val wire = request("prepare")
        val decodedRequest = WorkerPrepareRequest.parse(JsonParser.parseString(wire.toString()).asJsonObject)
        assertEquals(largeGeneration, decodedRequest.identity.generation)
        val ack = decodedRequest.ack("prepared", true)
        assertEquals(largeGeneration, JsonParser.parseString(ack.toString()).asJsonObject["identity"].asJsonObject["generation"].asLong)
        val i = decodedRequest.identity
        val stage = TaskWorkerProtocol.buildStageOffer(
            identity = TaskWorkerAttemptIdentity("wf_prepare_01", "stage_01", "att_prepare_01", "lease_prepare_01", 1),
            requestId = "request_01", stageType = "layer_forward", providerId = "remote_android_test", leaseExpiresAtMs = 200L,
            rootInput = mapOf("context_size" to 4096, "want_hidden" to false), dependencies = emptyMap(), modelIdentity = i.stageIdentity,
            messageId = "msg_prepare_01", sentAtMs = 100L,
            stageFields = mapOf("layer_range" to listOf(2, 4), "handoff_at" to 2, "hidden_sha256" to "a".repeat(64),
                "hidden_spec" to mapOf("n_tokens" to 1, "n_embd" to 1, "dtype" to "float32"), "middle_channel" to "keep_head_layer_out",
                "prepare_identity" to (offer().payload["prepare_identity"] as Map<*, *>)),
        )
        val bytes = TaskWorkerProtocol.encode(stage)
        val decoded = TaskWorkerProtocol.decode(bytes)
        val generation = (decoded.payload["prepare_identity"] as Map<*, *>)["generation"]
        assertTrue(generation is Long); assertEquals(largeGeneration, generation)
        assertArrayEquals(bytes, TaskWorkerProtocol.encode(decoded))
        val c = WorkerPrepareController(node, Runtime())
        transact(c, request("prepare"), this); transact(c, request("commit"), this)
        assertNull(c.validateStage(decoded))
        // Adjacent nanoseconds round to the same Double. Wire decoding and gating must
        // reject them even when every other identity key is unchanged.
        val adjacentTuple = (decoded.payload["prepare_identity"] as Map<*, *>).toMutableMap()
        adjacentTuple["generation"] = largeGeneration + 1
        val adjacentOffer = TaskWorkerProtocol.decode(TaskWorkerProtocol.encode(decoded.copy(
            payload = decoded.payload + ("prepare_identity" to adjacentTuple))))
        assertEquals("prepare_identity_mismatch", c.validateStage(adjacentOffer))
        val adjacentIdentity = identity().apply { addProperty("generation", largeGeneration + 1) }
        val adjacentWire = JsonParser.parseString(adjacentIdentity.toString()).asJsonObject
        assertNotEquals(i, WorkerPrepareIdentity.parse(adjacentWire))
        assertEquals(2, setOf(i, WorkerPrepareIdentity.parse(adjacentWire)).size)
        assertEquals("error", transact(c, request("commit", adjacentWire), this)["status"].asString)
        assertEquals("error", transact(c, request("release", adjacentWire), this)["status"].asString)
        assertNull(c.validateStage(decoded))
        val rounded = (decoded.payload["prepare_identity"] as Map<*, *>).toMutableMap()
        rounded["generation"] = largeGeneration.toDouble()
        assertEquals("prepare_identity_mismatch", c.validateStage(decoded.copy(payload = decoded.payload + ("prepare_identity" to rounded))))
    }

    @Test fun `disconnect real retirement then reconnect durable cleanup replay proves unloaded`() = runBlocking {
        val runtime = Runtime(); val c = WorkerPrepareController(node, runtime)
        transact(c, request("prepare"), this); transact(c, request("commit"), this)
        c.cancelAll(); c.retireAndJoin()
        assertNull(runtime.current); assertEquals(1, runtime.unloads)
        for (action in listOf("abort", "release", "finalize")) {
            val ack = transact(c, request(action), this)
            assertEquals(if (action == "finalize") "finalized" else "released", ack["status"].asString)
            assertFalse(ack["loaded"].asBoolean); assertEquals("cleanup:$action", ack["cleanup_id"].asString)
        }
        assertEquals(1, runtime.unloads)
        assertEquals("prepare_not_committed", c.validateStage(offer()))
        val next = identity(largeGeneration + 1, "supply_two")
        transact(c, request("prepare", next), this); transact(c, request("commit", next), this)
        for (action in listOf("abort", "release", "finalize")) {
            val ack = transact(c, request(action), this)
            assertFalse(ack["loaded"].asBoolean); assertNull(c.validateStage(offer(next)))
        }
        assertEquals(1, runtime.unloads)
    }

    @Test fun `fresh process with no native runtime can clear unknown old debt then never unload new owner`() = runBlocking {
        val runtime = Runtime(); val c = WorkerPrepareController(node, runtime)
        for (action in listOf("release", "abort", "finalize")) {
            val ack = transact(c, request(action), this)
            assertEquals(if (action == "finalize") "finalized" else "released", ack["status"].asString)
            assertFalse(ack["loaded"].asBoolean)
        }
        assertEquals(0, runtime.unloads)
        val next = identity(largeGeneration + 1, "supply_two")
        transact(c, request("prepare", next), this); transact(c, request("commit", next), this)
        for (action in listOf("release", "abort", "finalize")) {
            assertFalse(transact(c, request(action), this)["loaded"].asBoolean)
            assertNull(c.validateStage(offer(next)))
        }
        val unknown = identity(largeGeneration - 1, "never_known")
        for (action in listOf("release", "abort", "finalize")) {
            val ack = transact(c, request(action, unknown), this)
            assertEquals(if (action == "finalize") "finalized" else "released", ack["status"].asString)
            assertFalse(ack["loaded"].asBoolean)
            assertNull(c.validateStage(offer(next)))
        }
        assertEquals(0, runtime.unloads); assertTrue(runtime.loaded(WorkerPrepareIdentity.parse(next)))
    }

    @Test fun `old finalize while new candidate is loaded cannot finalize or unload the new candidate`() = runBlocking {
        val runtime = Runtime(); val c = WorkerPrepareController(node, runtime)
        transact(c, request("prepare"), this); transact(c, request("commit"), this); transact(c, request("finalize"), this)
        val next = identity(largeGeneration + 1, "supply_two")
        transact(c, request("prepare", next), this)
        val ack = transact(c, request("finalize"), this)
        assertEquals("finalized", ack["status"].asString); assertFalse(ack["loaded"].asBoolean)
        assertTrue(runtime.loaded(WorkerPrepareIdentity.parse(next))); assertEquals(0, runtime.unloads)
        assertEquals("prepare_not_committed", c.validateStage(offer(next)))
        transact(c, request("commit", next), this); assertNull(c.validateStage(offer(next)))
    }

    @Test fun `cleanup cannot claim no runtime when an unowned native slot still exists`() = runBlocking {
        val runtime = Runtime().apply { unownedSlot = true }
        val c = WorkerPrepareController(node, runtime)
        for (action in listOf("abort", "release", "finalize")) {
            val ack = transact(c, request(action), this)
            assertEquals("error", ack["status"].asString); assertFalse(ack["loaded"].asBoolean)
        }
        assertEquals(0, runtime.unloads); assertNull(runtime.current); assertTrue(runtime.hasRuntime())
    }

    @Test fun `master source identity differs without cross-format conflation`() {
        val parsed = WorkerPrepareIdentity.parse(identity())
        assertNotEquals(parsed.text("model_sha256"), parsed.text("source_model_sha256"))
        assertNotEquals(parsed.text("model_id"), parsed.text("source_model_id"))
        assertEquals(largeGeneration, parsed.generation)
        assertEquals(identity(), parsed.wire())
    }

    @Test fun `exact schema rejects full model cleanup smuggling and imprecise integers`() {
        val bad = listOf(
            request("prepare").apply { addProperty("cleanup_id", "unexpected") },
            request("commit").apply { addProperty("context_size", 4096) },
            request("abort").apply { remove("cleanup_id") },
            request("release").apply { addProperty("abort", true) },
            request("prepare").apply { getAsJsonObject("identity").add("layer_range", JsonParser.parseString("[0,4]")); getAsJsonObject("identity").addProperty("segment_mode", "head") },
            request("prepare").apply { getAsJsonObject("identity").addProperty("generation", 1.0) },
            request("prepare").apply { getAsJsonObject("identity").getAsJsonObject("stage_model_identity").addProperty("extra", true) },
            request("prepare").apply { getAsJsonObject("identity").addProperty("unknown", true) },
            request("prepare").apply { addProperty("context_size", 2048) },
        )
        bad.forEach { data -> try { WorkerPrepareRequest.parse(data); fail("accepted $data") } catch (_: WorkerPrepareException) {} }
    }
    private fun policy(): JsonObject = JsonParser.parseString("""{
        "schema_version":1,"device":"cpu","dtype":"artifact","quant_type":"artifact","context_size":4096,
        "kv_device":"cpu","kv_dtype":"float16","kv_policy":"preallocated","components":["final_norm","layers","lm_head"],
        "materialization":"gguf_segment","kv_cache_bytes":0,"batch_size":4096,"sequence_count":1
    }""").asJsonObject

    private fun boundIdentity(spec: JsonObject = policy()): JsonObject = identity().apply {
        val bytes = com.qlh.inference.service.canonicalSupplyJson(spec).toByteArray(Charsets.UTF_8)
        val sha = java.security.MessageDigest.getInstance("SHA-256").digest(bytes)
            .joinToString("") { "%02x".format(it.toInt() and 255) }
        addProperty("load_spec_digest", sha)
    }

    @Test fun `frozen policy is loaded once and actual receipt survives commit`() = runBlocking {
        val spec = policy(); val id = boundIdentity(spec)
        var policyLoads = 0
        val runtime = object : Runtime() {
            override suspend fun loadWithSpec(identity: WorkerPrepareIdentity, spec: JsonObject) {
                assertEquals(policy(), spec); policyLoads++; super.load(identity)
            }
            override suspend fun receipt(identity: WorkerPrepareIdentity, spec: JsonObject): JsonObject = JsonObject().apply {
                add("load_spec", spec.deepCopy()); addProperty("materialized_bytes", 17)
                addProperty("memory_scope", "weight_tensors"); add("layer_range", identity.wire()["layer_range"])
                addProperty("artifact_sha256", identity.text("artifact_sha256"))
            }
        }
        val c = WorkerPrepareController(node, runtime)
        val prepare = request("prepare", id).apply { add("load_spec", spec) }
        val first = transact(c, prepare, this)
        assertEquals("prepared", first["status"].asString)
        assertEquals(17, first.getAsJsonObject("runtime_receipt")["materialized_bytes"].asInt)
        assertEquals(first, transact(c, prepare, this)); assertEquals(1, policyLoads)
        val committed = transact(c, request("commit", id), this)
        assertEquals("ready", committed["status"].asString)
        assertEquals(first["runtime_receipt"], committed["runtime_receipt"])
        // A stage tuple with the old fields omits the frozen policy digest.
        assertEquals("prepare_identity_mismatch", c.validateStage(offer(id)))
    }

    @Test fun `missing actual receipt aborts the newly materialized native slot`() = runBlocking {
        val id = boundIdentity()
        val runtime = object : Runtime() {
            override suspend fun loadWithSpec(identity: WorkerPrepareIdentity, spec: JsonObject) { super.load(identity) }
        }
        val c = WorkerPrepareController(node, runtime)
        val response = transact(c, request("prepare", id).apply { add("load_spec", policy()) }, this)
        assertEquals("error", response["status"].asString)
        assertEquals("prepare_runtime_receipt_missing", response["error"].asString)
        assertFalse(runtime.hasRuntime()); assertNull(c.committedIdentity())
    }

    @Test fun `policy digest context device and duplicate components are exact`() {
        val valid = request("prepare", boundIdentity()).apply { add("load_spec", policy()) }
        assertNotNull(WorkerPrepareRequest.parse(valid).loadSpec)
        val bad = listOf(
            valid.deepCopy().apply { remove("load_spec") },
            valid.deepCopy().apply { getAsJsonObject("load_spec").addProperty("kv_cache_bytes", 1) },
            valid.deepCopy().apply { getAsJsonObject("load_spec").addProperty("context_size", 2048) },
            valid.deepCopy().apply { getAsJsonObject("load_spec").addProperty("device", "cuda:0") },
            valid.deepCopy().apply { getAsJsonObject("load_spec").add("components", JsonParser.parseString("[\"final_norm\",\"layers\",\"layers\",\"lm_head\"]")) },
        )
        bad.forEach { data -> try { WorkerPrepareRequest.parse(data); fail("accepted $data") } catch (_: WorkerPrepareException) {} }
    }

}
