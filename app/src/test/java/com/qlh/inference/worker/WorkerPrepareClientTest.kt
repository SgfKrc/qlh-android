package com.qlh.inference.worker

import com.google.gson.JsonObject
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.NonCancellable
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withContext
import kotlinx.coroutines.withTimeout
import kotlinx.coroutines.yield
import org.junit.Assert.*
import org.junit.Test
import java.util.concurrent.atomic.AtomicInteger

class WorkerPrepareClientTest {
    private class Transport : TaskWorkerTransport {
        val inbound = Channel<TaskWorkerInboundEvent>(Channel.UNLIMITED)
        val outbound = Channel<TaskWorkerEnvelope>(Channel.UNLIMITED)
        val acks = Channel<JsonObject>(Channel.UNLIMITED)
        override suspend fun send(envelope: TaskWorkerEnvelope) { outbound.send(envelope) }
        override suspend fun sendModelPrepare(data: JsonObject) { acks.send(data) }
        override suspend fun receiveEvent(): TaskWorkerInboundEvent = inbound.receiveCatching().getOrNull() ?: TaskWorkerInboundEvent.Closed
        override suspend fun receive(): TaskWorkerEnvelope? = (receiveEvent() as? TaskWorkerInboundEvent.Envelope)?.value
        override suspend fun close() { inbound.close() }
        suspend fun push(envelope: TaskWorkerEnvelope) { inbound.send(TaskWorkerInboundEvent.Envelope(envelope)) }
        suspend fun prepare(action: String) {
            val wire = WorkerPrepareTransportTest.wire().apply {
                addProperty("action", action)
                if (action != "prepare") { remove("context_size"); remove("middle_channel") }
                if (action in setOf("abort", "release", "finalize")) addProperty("cleanup_id", "cleanup_replay")
            }
            inbound.send(TaskWorkerInboundEvent.ModelPrepare(wire))
        }
        suspend fun next(type: String): TaskWorkerEnvelope = withTimeout(5000) {
            var message = outbound.receive()
            while (message.messageType != type) message = outbound.receive()
            message
        }
        suspend fun ack(): JsonObject = withTimeout(5000) { acks.receive() }
        suspend fun hello() {
            next(TaskWorkerProtocol.HELLO)
            push(TaskWorkerProtocol.buildHelloAck("master_test", true, 3, "", "msg_ack_test", System.currentTimeMillis()))
        }
    }
    private open class Runtime : WorkerPrepareRuntime {
        @Volatile var identity: WorkerPrepareIdentity? = null
        val loads = AtomicInteger(0); val frees = AtomicInteger(0)
        val retired = CompletableDeferred<Unit>()
        override suspend fun load(identity: WorkerPrepareIdentity) { loads.incrementAndGet(); this.identity = identity }
        override suspend fun unload() { frees.incrementAndGet(); identity = null; retired.complete(Unit) }
        override fun loaded(identity: WorkerPrepareIdentity): Boolean = this.identity == identity
        override fun hasRuntime(): Boolean = identity != null
    }
    private fun offer(tuple: Boolean): TaskWorkerEnvelope {
        val i = WorkerPrepareIdentity.parse(WorkerPrepareTransportTest.wire()["identity"].asJsonObject)
        val fields = mutableMapOf<String, Any?>("layer_range" to i.range, "handoff_at" to 2,
            "hidden_sha256" to "e".repeat(64), "hidden_spec" to mapOf("n_tokens" to 1, "n_embd" to 1, "dtype" to "float32"),
            "middle_channel" to "keep_head_layer_out")
        if (tuple) fields["prepare_identity"] = mapOf("operation_id" to i.text("operation_id"), "config_id" to i.text("config_id"),
            "assignment_id" to i.text("assignment_id"), "artifact_sha256" to i.text("artifact_sha256"), "generation" to i.generation)
        val now = System.currentTimeMillis()
        return TaskWorkerProtocol.buildStageOffer(TaskWorkerAttemptIdentity("wf_prepare_client", "stage_test", "att_prepare_client", "lease_prepare_client", 1),
            "request_test", "layer_forward", "remote_android_test", now + 30000,
            // Legacy context_size is advisory on the stage path. It cannot reload the prepared 4096 context.
            mapOf("want_hidden" to false, "context_size" to 2048), emptyMap(), i.stageIdentity, "msg_prepare_client", now, fields)
    }
    private fun client(transport: TaskWorkerTransportFactory, controller: WorkerPrepareController, calls: AtomicInteger = AtomicInteger()): TaskWorkerClient =
        TaskWorkerClient("127.0.0.1", 8888, "android_test", capabilities = {
            mapOf("stage_types" to listOf("layer_forward"), "engines" to listOf("llama_cpp"), "models" to emptyList<Map<String, Any?>>(), "max_concurrency" to 1)
        }, transportFactory = transport, modelPrepareHandler = controller, stageHandler = TaskWorkerStageHandler {
            calls.incrementAndGet(); TaskWorkerStageExecution(mapOf("token_argmax" to 2), emptyMap())
        })

    @Test fun `client rejects unprepared layer offers before accept and never lazy loads stages`() = runBlocking {
        val transport = Transport(); val runtime = Runtime(); val c = WorkerPrepareController("android_test", runtime); val calls = AtomicInteger()
        val client = client({ _, _ -> transport }, c, calls)
        try {
            client.start(); transport.hello()
            transport.push(offer(false))
            assertFalse(transport.next(TaskWorkerProtocol.STAGE_ACCEPT).payload["accepted"] as Boolean)
            assertEquals(0, calls.get()); assertEquals(0, runtime.loads.get())
            transport.prepare("prepare"); assertEquals("prepared", transport.ack()["status"].asString)
            transport.push(offer(true)); assertFalse(transport.next(TaskWorkerProtocol.STAGE_ACCEPT).payload["accepted"] as Boolean)
            transport.prepare("commit"); assertEquals("ready", transport.ack()["status"].asString)
            transport.push(offer(true)); assertTrue(transport.next(TaskWorkerProtocol.STAGE_ACCEPT).payload["accepted"] as Boolean)
            transport.next(TaskWorkerProtocol.STAGE_RESULT)
            assertEquals(1, calls.get()); assertEquals(1, runtime.loads.get())
        } finally { client.stopAndJoin() }
        assertFalse(runtime.hasRuntime())
    }

    @Test fun `client reconnect only follows real retirement and can replay old release debt`() = runBlocking {
        val first = Transport(); val second = Transport(); val connections = AtomicInteger()
        val runtime = Runtime(); val c = WorkerPrepareController("android_test", runtime)
        val client = client({ _, _ -> if (connections.getAndIncrement() == 0) first else second }, c)
        try {
            client.start(); first.hello(); first.prepare("prepare"); first.ack(); first.prepare("commit"); first.ack()
            first.close(); withTimeout(5000) { runtime.retired.await() }
            assertFalse(runtime.hasRuntime())
            second.hello(); second.prepare("release")
            val ack = second.ack()
            assertEquals("released", ack["status"].asString); assertFalse(ack["loaded"].asBoolean)
            assertEquals("cleanup_replay", ack["cleanup_id"].asString); assertEquals(1, runtime.loads.get())
        } finally { client.stopAndJoin() }
    }

    @Test fun `UI cancellation during native prepare closes connection joins JNI and emits no load ACK`() = runBlocking {
        val entered = CompletableDeferred<Unit>(); val finish = CompletableDeferred<Unit>()
        val runtime = object : Runtime() {
            override suspend fun load(identity: WorkerPrepareIdentity) = withContext(NonCancellable) {
                entered.complete(Unit); finish.await(); super.load(identity)
            }
        }
        val transport = Transport(); val c = WorkerPrepareController("android_test", runtime)
        val client = client({ _, _ -> transport }, c)
        try {
            client.start(); transport.hello(); transport.prepare("prepare")
            withTimeout(5000) { entered.await() }
            assertTrue(client.cancelActive()); yield(); assertEquals(0, runtime.frees.get())
            finish.complete(Unit); withTimeout(5000) { runtime.retired.await() }
            assertFalse(runtime.hasRuntime()); assertNull(transport.acks.tryReceive().getOrNull())
        } finally { finish.complete(Unit); client.stopAndJoin() }
    }
}
