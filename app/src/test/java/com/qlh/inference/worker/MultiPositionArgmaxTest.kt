package com.qlh.inference.worker

import com.qlh.inference.service.probeMultiPositionArgmax
import kotlinx.coroutines.runBlocking
import org.junit.Assert.*
import org.junit.Test
import org.junit.runner.RunWith
import org.junit.runners.Parameterized
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.Base64

internal object MultiPositionFixtures {
    val identity = TaskWorkerAttemptIdentity("wf_multipos_001", "tail", "att_multipos_001", "lease_multipos_001", 1)
    val model = mapOf("model_id" to "tail", "engine" to "llama_cpp", "format" to "gguf",
        "revision" to "local", "sha256" to "a".repeat(64))
    val positions = listOf(41, 42, 43)
    val tokens = listOf(2, 1, 4)
    val output: Map<String, Any?> = mapOf("token_argmax" to 4,
        "token_argmax_by_position" to tokens, "positions" to positions, "n_vocab" to 5, "n_tokens" to 3)
    fun offer(): TaskWorkerEnvelope {
        val bytes = ByteBuffer.allocate(3 * 4 * 4).order(ByteOrder.LITTLE_ENDIAN)
        repeat(12) { bytes.putFloat(it.toFloat()) }
        return TaskWorkerProtocol.buildStageOffer(identity, "request_multipos", "layer_forward", "android_worker",
            2000, mapOf("hidden_f32" to Base64.getEncoder().encodeToString(bytes.array()), "want_hidden" to false, "pos_base" to 0),
            emptyMap(), model, "msg_multipos_offer", 1000, mapOf(
                "layer_range" to listOf(2, 4), "handoff_at" to 2, "hidden_sha256" to "b".repeat(64),
                "hidden_spec" to mapOf("n_tokens" to 3, "n_embd" to 4, "dtype" to "float32"),
                "token_output" to "all_positions", "positions" to positions))
    }
    fun withPayload(payload: Map<String, Any?>): TaskWorkerEnvelope {
        @Suppress("UNCHECKED_CAST")
        val root = payload["root_input"] as Map<String, Any?>
        return offer().copy(payload = payload + ("input_sha256" to TaskWorkerProtocol.stageInputSha256(root, emptyMap())))
    }
    fun result(output: Map<String, Any?>) = TaskWorkerProtocol.buildStageResult(identity, "android_worker",
        output, emptyMap(), "msg_multipos_result", 1001)
    fun executor(onLoad: () -> Unit = {}, onLayer: (LayerForwardRequest) -> Result<LayerForwardResult> = {
        Result.success(LayerForwardResult(4, null, tokens, positions, 5))
    }) = AndroidFullWorkerStageExecutor(expectedModelIdentity = { model },
        ensureModelLoaded = { onLoad(); Result.success(Unit) },
        generate = { _, _, _, _ -> error("layer offer must not generate text") }, layerForward = onLayer)
    fun protocolError(action: () -> Unit): TaskWorkerProtocolException = try {
        action(); throw AssertionError("expected protocol rejection")
    } catch (error: TaskWorkerProtocolException) { error }
}

@RunWith(Parameterized::class)
class MultiPositionArgmaxInvalidRequestTest(private val name: String, private val payload: Map<String, Any?>) {
    companion object {
        @JvmStatic @Parameterized.Parameters(name = "{0}")
        fun cases(): Collection<Array<Any>> {
            val valid = MultiPositionFixtures.offer().payload
            val cases = mutableListOf<Array<Any>>()
            fun add(name: String, field: String, value: Any?) { cases.add(arrayOf(name, valid + (field to value))) }
            for (value in listOf(null, true, false, 1, listOf("all_positions"), "last", "", "ALL_POSITIONS")) {
                add("token_output=$value", "token_output", value)
            }
            cases.add(arrayOf("implicit positions", valid - "positions"))
            for (value in listOf(null, "7,8,9", emptyList<Int>(), listOf(7, 8), listOf(7, 8, 9, 10),
                listOf(7, 9, 10), listOf(7, 7, 8), listOf(9, 8, 7), listOf(-1, 0, 1),
                listOf(7.0, 8, 9), listOf(7.5, 8, 9), listOf("7", 8, 9), listOf(true, 8, 9),
                listOf(null, 8, 9), listOf(Int.MAX_VALUE, 2147483648L, 2147483649L))) {
                add("positions=$value", "positions", value)
            }
            for (value in listOf(null, emptyList<Int>(), listOf(1, 1), listOf(1, 1, 1, 1), listOf(1, 2, 1),
                listOf(-1, -1, -1), listOf(1.0, 1, 1), listOf(1.5, 1, 1), listOf(true, true, true),
                listOf("1", "1", "1"), listOf(2147483648L, 2147483648L, 2147483648L))) {
                add("seq_ids=$value", "seq_ids", value)
            }
            @Suppress("UNCHECKED_CAST")
            val root = valid["root_input"] as Map<String, Any?>
            cases.add(arrayOf("implicit want_hidden", valid + ("root_input" to (root - "want_hidden"))))
            for (value in listOf(null, true, "false", 0)) {
                add("want_hidden=$value", "root_input", root + ("want_hidden" to value))
            }
            for (value in listOf(null, 0, -1, 3.0, 3.5, true, "3", 2147483648L)) {
                add("n_tokens=$value", "hidden_spec", mapOf("n_tokens" to value, "n_embd" to 4, "dtype" to "float32"))
            }
            add("full inference cannot request tokens", "stage_type", "full_inference")
            return cases
        }
    }
    @Test fun `codec rejects malformed token request on validation and decode`() {
        val envelope = MultiPositionFixtures.withPayload(payload)
        MultiPositionFixtures.protocolError { TaskWorkerProtocol.validate(envelope) }
        MultiPositionFixtures.protocolError { TaskWorkerProtocol.decode(TaskWorkerProtocol.canonicalBytes(envelope)) }
    }
    @Test fun `direct executor rejects before loading or native execution`() = runBlocking {
        var loads = 0
        var calls = 0
        val executor = MultiPositionFixtures.executor(onLoad = { loads++ }, onLayer = { calls++; error("must not run") })
        try {
            executor.execute(MultiPositionFixtures.withPayload(payload))
            fail("accepted $name")
        } catch (error: AndroidFullWorkerStageException) {
            assertEquals("invalid_token_output", error.code)
        }
        assertEquals(0, loads)
        assertEquals(0, calls)
    }
}

@RunWith(Parameterized::class)
class MultiPositionArgmaxInvalidResultTest(private val name: String, private val output: Map<String, Any?>) {
    companion object {
        @JvmStatic @Parameterized.Parameters(name = "{0}")
        fun cases(): Collection<Array<Any>> {
            val valid = MultiPositionFixtures.output
            val cases = mutableListOf<Array<Any>>()
            fun add(name: String, field: String, value: Any?) { cases.add(arrayOf(name, valid + (field to value))) }
            for (field in valid.keys) cases.add(arrayOf("missing $field", valid - field))
            // Any single new field starts the group; none may bypass validation.
            for (field in listOf("positions", "n_vocab", "token_argmax_by_position")) {
                cases.add(arrayOf("only $field", mapOf(field to valid[field])))
            }
            for (field in listOf("n_vocab", "n_tokens")) {
                for (value in listOf(null, 0, -1, 5.0, 5.5, true, "5", 2147483648L)) add("$field=$value", field, value)
            }
            for (value in listOf(null, emptyList<Int>(), listOf(2, 4), listOf(2, 1, 4, 4), listOf(2, 1, 5),
                listOf(-1, 1, 4), listOf(2.0, 1, 4), listOf(2.5, 1, 4), listOf("2", 1, 4),
                listOf(true, 1, 4), listOf(null, 1, 4), listOf(2147483648L, 1, 4))) {
                add("tokens=$value", "token_argmax_by_position", value)
            }
            for (value in listOf(null, emptyList<Int>(), listOf(7, 8), listOf(7, 8, 9, 10), listOf(7, 9, 10),
                listOf(7, 7, 8), listOf(9, 8, 7), listOf(-1, 0, 1), listOf(7.0, 8, 9),
                listOf("7", 8, 9), listOf(true, 8, 9), listOf(2147483648L, 2147483649L, 2147483650L))) {
                add("positions=$value", "positions", value)
            }
            for (value in listOf(null, -1, 5, 3, 4.0, 4.5, true, "4", 2147483648L)) {
                add("scalar=$value", "token_argmax", value)
            }
            return cases
        }
    }
    @Test fun `result builder checks complete and bounded position group`() {
        assertEquals("invalid_token_output", MultiPositionFixtures.protocolError {
            MultiPositionFixtures.result(output)
        }.code)
    }
    @Test fun `wire decoder rejects even with a correct digest`() {
        val valid = MultiPositionFixtures.result(MultiPositionFixtures.output)
        val malformed = valid.copy(payload = valid.payload + mapOf("output" to output,
            "output_sha256" to TaskWorkerProtocol.stageOutputSha256(output)))
        assertEquals("invalid_token_output", MultiPositionFixtures.protocolError {
            TaskWorkerProtocol.decode(TaskWorkerProtocol.canonicalBytes(malformed))
        }.code)
    }
}

class MultiPositionArgmaxTest {
    @Test fun `requested positions flow through executor into existing result output`() = runBlocking {
        var seen: LayerForwardRequest? = null
        val executor = MultiPositionFixtures.executor(onLayer = {
            seen = it
            Result.success(LayerForwardResult(4, null, listOf(2, 1, 4), listOf(41, 42, 43), 5))
        })
        val offer = MultiPositionFixtures.offer()
        val admitted = TaskWorkerProtocol.decode(TaskWorkerProtocol.encode(offer))
        val result = executor.execute(admitted)
        assertEquals("all_positions", seen?.tokenOutput)
        assertFalse(requireNotNull(seen).wantHidden)
        assertArrayEquals(intArrayOf(41, 42, 43), seen?.positions)
        assertNull(seen?.seqIds)
        assertEquals(0, seen?.posBase) // Scheduler default must not override explicit positions 41..43.
        assertEquals(MultiPositionFixtures.output + ("layer_range" to listOf(2, 4)), result.output)
        val wire = MultiPositionFixtures.result(result.output)
        val decoded = TaskWorkerProtocol.decode(TaskWorkerProtocol.encode(wire))
        assertEquals(TaskWorkerProtocol.canonicalJson(result.output),
            TaskWorkerProtocol.canonicalJson(decoded.payload["output"]))
        assertEquals(TaskWorkerProtocol.stageOutputSha256(result.output), decoded.payload["output_sha256"])
    }

    @Test fun `explicit one sequence is preserved`() = runBlocking {
        val offer = MultiPositionFixtures.offer()
        val executor = MultiPositionFixtures.executor(onLayer = {
            assertArrayEquals(intArrayOf(11, 11, 11), it.seqIds)
            Result.success(LayerForwardResult(4, null, listOf(2, 1, 4), listOf(41, 42, 43), 5))
        })
        executor.execute(offer.copy(payload = offer.payload + ("seq_ids" to listOf(11, 11L, 11))))
        Unit
    }

    @Test fun `one position and maximum int32 boundary are valid`() {
        val offer = MultiPositionFixtures.offer()
        val payload = offer.payload + mapOf("hidden_spec" to mapOf("n_tokens" to 1, "n_embd" to 4, "dtype" to "float32"),
            "positions" to listOf(Int.MAX_VALUE), "seq_ids" to listOf(Int.MAX_VALUE))
        TaskWorkerProtocol.validate(MultiPositionFixtures.withPayload(payload))
        val result = MultiPositionFixtures.result(mapOf("n_tokens" to 1, "n_vocab" to Int.MAX_VALUE,
            "positions" to listOf(Int.MAX_VALUE), "token_argmax" to Int.MAX_VALUE - 1,
            "token_argmax_by_position" to listOf(Int.MAX_VALUE - 1)))
        assertNotNull(TaskWorkerProtocol.decode(TaskWorkerProtocol.encode(result)))
    }

    @Test fun `unrequested forward retains scalar result and old request shape`() = runBlocking {
        val offer = MultiPositionFixtures.offer()
        val executor = MultiPositionFixtures.executor(onLayer = {
            assertNull(it.tokenOutput)
            assertNull(it.positions)
            Result.success(LayerForwardResult(4, null))
        })
        val legacy = offer.copy(payload = offer.payload - "token_output" - "positions")
        TaskWorkerProtocol.validate(legacy)
        val result = executor.execute(legacy)
        assertEquals(mapOf("token_argmax" to 4, "n_tokens" to 3, "layer_range" to listOf(2, 4)), result.output)
        assertNotNull(TaskWorkerProtocol.decode(TaskWorkerProtocol.encode(MultiPositionFixtures.result(result.output))))
    }

    @Test fun `v2 cannot carry all positions`() {
        MultiPositionFixtures.protocolError { TaskWorkerProtocol.validate(MultiPositionFixtures.offer().copy(version = 2)) }
    }

    @Test fun `executor rejects incomplete wrong or unsolicited native position groups`() = runBlocking {
        val invalid = listOf(
            LayerForwardResult(4, null),
            LayerForwardResult(4, null, listOf(2, 4), listOf(41, 42, 43), 5),
            LayerForwardResult(4, null, listOf(2, 1, 4), listOf(8, 9, 10), 5),
            LayerForwardResult(4, null, listOf(2, 1, 5), listOf(41, 42, 43), 5),
            LayerForwardResult(3, null, listOf(2, 1, 4), listOf(41, 42, 43), 5),
            LayerForwardResult(4, null, listOf(2, 1, 4), listOf(41, 42, 43), 0),
            LayerForwardResult(4, FloatArray(12), listOf(2, 1, 4), listOf(41, 42, 43), 5),
            LayerForwardResult(4, null, listOf(2, 1, 4), listOf(41, 42, 43), null),
        )
        for (out in invalid) {
            try {
                MultiPositionFixtures.executor(onLayer = { Result.success(out) }).execute(MultiPositionFixtures.offer())
                fail("accepted invalid native result $out")
            } catch (error: AndroidFullWorkerStageException) { assertEquals("invalid_token_output", error.code) }
        }
        val offer = MultiPositionFixtures.offer()
        try {
            MultiPositionFixtures.executor().execute(offer.copy(payload = offer.payload - "token_output"))
            fail("accepted unsolicited multi-position result")
        } catch (error: AndroidFullWorkerStageException) { assertEquals("invalid_token_output", error.code) }
    }

    @Test fun `multi-position result rejects inline hidden output`() {
        assertMixedHiddenRejected("hidden_out_f32", "AAAAAA==")
    }

    @Test fun `multi-position result rejects hidden output reference`() {
        assertMixedHiddenRejected("hidden_out_ref", mapOf("hidden_id" to "hidden_tail", "total_bytes" to 48))
    }

    private fun assertMixedHiddenRejected(field: String, hidden: Any) {
        // Presence is forbidden even for JSON null, matching the shared PC contract.
        for (value in listOf(hidden, null)) {
            val output = MultiPositionFixtures.output + (field to value)
            assertEquals("invalid_token_output", MultiPositionFixtures.protocolError {
                MultiPositionFixtures.result(output)
            }.code)
            val valid = MultiPositionFixtures.result(MultiPositionFixtures.output)
            val mixed = valid.copy(payload = valid.payload + mapOf("output" to output,
                "output_sha256" to TaskWorkerProtocol.stageOutputSha256(output)))
            assertEquals("invalid_token_output", MultiPositionFixtures.protocolError {
                TaskWorkerProtocol.decode(TaskWorkerProtocol.canonicalBytes(mixed))
            }.code)
        }
    }

    @Test fun `runner failure produces no result and clears execution state`() = runBlocking {
        val executor = MultiPositionFixtures.executor(onLayer = { Result.failure(IllegalStateException("bad logit row")) })
        try { executor.execute(MultiPositionFixtures.offer()); fail("expected runner failure") }
        catch (error: AndroidFullWorkerStageException) { assertEquals("worker_execution_failed", error.code) }
        assertFalse(executor.isExecutionInFlight(MultiPositionFixtures.identity))
    }

    @Test fun `capability is optional and requires a positive native probe and layer stage`() {
        val legacy = AndroidWorkerCapabilities.build(layerRanges = listOf(listOf(2, 4)))
        assertFalse(legacy.containsKey("multi_position_argmax"))
        assertFalse(AndroidWorkerCapabilities.build(multiPositionArgmax = true).containsKey("multi_position_argmax"))
        val supported = AndroidWorkerCapabilities.build(layerRanges = listOf(listOf(2, 4)), multiPositionArgmax = true)
        assertEquals(true, supported["multi_position_argmax"])
        for (caps in listOf(legacy, supported, legacy + ("multi_position_argmax" to false))) {
            val hello = TaskWorkerProtocol.buildHello("android_worker", caps, "msg_multipos_hello", 1000)
            assertNotNull(TaskWorkerProtocol.decode(TaskWorkerProtocol.encode(hello)))
        }
        for (bad in listOf(null, "true", 1, listOf(true))) {
            MultiPositionFixtures.protocolError {
                TaskWorkerProtocol.buildHello("android_worker", legacy + ("multi_position_argmax" to bad),
                    "msg_multipos_hello", 1000)
            }
        }
    }

    @Test fun `old missing or failed native query never advertises support`() {
        assertTrue(probeMultiPositionArgmax { true })
        assertFalse(probeMultiPositionArgmax { false })
        assertFalse(probeMultiPositionArgmax { throw UnsatisfiedLinkError("old so") })
        assertFalse(probeMultiPositionArgmax { throw NoClassDefFoundError("unavailable") })
        assertFalse(probeMultiPositionArgmax { throw IllegalStateException("probe failed") })
    }
}
