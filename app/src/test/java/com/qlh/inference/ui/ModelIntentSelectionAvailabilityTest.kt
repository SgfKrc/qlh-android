package com.qlh.inference.ui

import com.google.gson.Gson
import com.qlh.inference.network.ModelIntentCatalog
import com.qlh.inference.network.PipelineEngineAvailability
import com.qlh.inference.network.ServerModelSummary
import org.junit.Assert.*
import org.junit.Test

class ModelIntentSelectionAvailabilityTest {
    private val unsupported = "SUPPLY_MODEL_RUNTIME_UNSUPPORTED"
    private val fullSource = "SUPPLY_FULL_SOURCE_REQUIRED"

    private fun model(
        pytorch: PipelineEngineAvailability? = null,
        llama: PipelineEngineAvailability? = null,
        auto: String? = "pytorch",
        diskAvailable: Boolean = false,
    ) = ServerModelSummary(
        modelId = "fixture", name = "Fixture", isAvailable = diskAvailable,
        unavailableReason = "$unsupported: global summary must not decide selection",
        preferredEngine = "llama_cpp", pipelineAutoEngine = auto,
        pipelineAutoUnavailableReason = "$unsupported: aggregate text must not decide selection",
        pipelineEngineAvailability = mapOf("pytorch" to pytorch, "llama_cpp" to llama),
    )

    private fun blocked(code: String, reason: String) = PipelineEngineAvailability(
        status = "unavailable", reasonCode = code, reason = reason,
    )

    @Test fun `catalog JSON carries per engine errors and owner selected automatic engine`() {
        val catalog = Gson().fromJson("""{
            "models": [{"model_id":"qwen3-0.6b","name":"Qwen3","is_available":true,
                "supported_engines":[],"pipeline_auto_engine":"pytorch",
                "pipeline_auto_unavailable_reason":"opaque aggregate",
                "pipeline_engine_availability":{
                    "pytorch":{"status":"unavailable","runtime_model_type":"qwen3",
                        "reason_code":"SUPPLY_MODEL_RUNTIME_UNSUPPORTED","reason":"HF qwen3 unsupported"},
                    "llama_cpp":{"status":"unavailable","runtime_model_type":"qwen3",
                        "reason_code":"SUPPLY_MODEL_RUNTIME_UNSUPPORTED","reason":"GGUF qwen3 unsupported"}
                }}]
        }""", ModelIntentCatalog::class.java)
        val source = catalog.models.single()
        assertEquals("pytorch", source.pipelineAutoEngine)
        assertEquals("opaque aggregate", source.pipelineAutoUnavailableReason)
        assertEquals("qwen3", source.pipelineEngineAvailability?.get("pytorch")?.runtimeModelType)
        assertTrue(source.supportedEngines.orEmpty().isEmpty())
        for (engine in listOf("auto", "pytorch", "llama_cpp")) {
            val decision = modelIntentSelectionAvailability(source, engine)
            assertFalse(engine, decision.enabled)
            assertEquals(unsupported, decision.reasonCode)
            assertTrue(decision.message.contains(if (engine == "llama_cpp") "GGUF qwen3" else "HF qwen3"))
            assertFalse(decision.message.contains("获取"))
        }
    }

    @Test fun `cropped GGUF blocks automatic and LL but absent HF remains acquirable`() {
        val source = model(
            pytorch = PipelineEngineAvailability(status = "missing"),
            llama = blocked(fullSource, "GGUF source is pre-cut"), auto = "llama_cpp",
        )
        for (engine in listOf("auto", "llama_cpp")) {
            val decision = modelIntentSelectionAvailability(source, engine)
            assertFalse(decision.enabled)
            assertEquals(fullSource, decision.reasonCode)
            assertTrue(decision.message.contains("GGUF source is pre-cut"))
        }
        val acquireHf = modelIntentSelectionAvailability(source, "pytorch")
        assertTrue(acquireHf.enabled)
        assertEquals("", acquireHf.reasonCode)
        assertTrue(acquireHf.message.contains("来源尚未就绪"))
    }

    @Test fun `blocked automatic PyTorch does not seal explicit proven LL source`() {
        val source = model(
            pytorch = blocked(unsupported, "HF unsupported"),
            llama = PipelineEngineAvailability(status = "supported", runtimeModelType = "qwen3_5",
                source = "provenance", contentVerification = "supply_owner_required"),
        )
        assertFalse(modelIntentSelectionAvailability(source, "auto").enabled)
        assertFalse(modelIntentSelectionAvailability(source, "pytorch").enabled)
        val ll = modelIntentSelectionAvailability(source, "llama_cpp")
        assertTrue(ll.enabled)
        assertEquals("", ll.reasonCode)
        assertTrue(ll.message.contains("主节点在供给时验证"))
        assertFalse(ll.message.contains("HF unsupported"))
        assertFalse(ll.message.contains("global summary"))
    }

    @Test fun `supported automatic PyTorch is selectable when LL is unsupported`() {
        val source = model(
            pytorch = PipelineEngineAvailability(status = "supported"),
            llama = blocked(unsupported, "LL unsupported"),
        )
        assertTrue(modelIntentSelectionAvailability(source, "auto").enabled)
        assertTrue(modelIntentSelectionAvailability(source, "pytorch").enabled)
        assertFalse(modelIntentSelectionAvailability(source, "llama_cpp").enabled)
    }

    @Test fun `automatic uses pipeline auto engine rather than preferred engine`() {
        val source = model(
            pytorch = blocked(unsupported, "HF unsupported"),
            llama = PipelineEngineAvailability(status = "supported"), auto = "pytorch",
        )
        assertEquals("llama_cpp", source.preferredEngine)
        val automatic = modelIntentSelectionAvailability(source, "auto")
        assertFalse(automatic.enabled)
        assertTrue(automatic.message.startsWith("自动（PyTorch）"))
        assertTrue(modelIntentSelectionAvailability(source, "llama_cpp").enabled)
    }

    @Test fun `missing unknown and uncoded metadata remain with acquisition owner`() {
        for (status in listOf(null, "missing", "unknown", "unavailable", "future_status")) {
            val source = model(pytorch = PipelineEngineAvailability(status = status,
                reasonCode = "", reason = "$unsupported in prose only"))
            for (engine in listOf("auto", "pytorch")) {
                val decision = modelIntentSelectionAvailability(source, engine)
                assertTrue("$status / $engine", decision.enabled)
                assertEquals("", decision.reasonCode)
                assertTrue(decision.message.contains("主节点"))
                assertFalse(decision.message.contains(unsupported))
            }
        }
    }

    @Test fun `legacy global unavailable reason does not become an engine fact`() {
        val source = ServerModelSummary(modelId = "old-server", isAvailable = false,
            unavailableReason = "$unsupported: qwen3", preferredEngine = "pytorch")
        for (engine in listOf("auto", "pytorch", "llama_cpp")) {
            val decision = modelIntentSelectionAvailability(source, engine)
            assertTrue(decision.enabled)
            assertTrue(decision.message.contains("获取并验证"))
            assertFalse(decision.message.contains(unsupported))
        }
    }

    @Test fun `absent automatic mapping does not infer a refusal from other engine records`() {
        val source = model(pytorch = blocked(unsupported, "HF unsupported"),
            llama = blocked(fullSource, "LL cropped"), auto = null)
        assertTrue(modelIntentSelectionAvailability(source, "auto").enabled)
        assertFalse(modelIntentSelectionAvailability(source, "pytorch").enabled)
        assertFalse(modelIntentSelectionAvailability(source, "llama_cpp").enabled)
        assertTrue(modelIntentSelectionAvailability(source.copy(pipelineAutoEngine = "  "), "auto").enabled)
    }

    @Test fun `null JSON maps entries and reason fields keep unknown availability selectable`() {
        val bodies = listOf(
            """{"model_id":"fixture","pipeline_engine_availability":null,"pipeline_auto_engine":null}""",
            """{"model_id":"fixture","pipeline_auto_engine":"pytorch",
                "pipeline_engine_availability":{"pytorch":null}}""",
            """{"model_id":"fixture","pipeline_auto_engine":"pytorch",
                "pipeline_engine_availability":{"pytorch":{"status":null,"reason_code":null,"reason":null}}}""",
        )
        for (body in bodies) {
            val source = Gson().fromJson(body, ServerModelSummary::class.java)
            assertTrue(modelIntentSelectionAvailability(source, "auto").enabled)
            assertTrue(modelIntentSelectionAvailability(source, "pytorch").enabled)
        }
    }

    @Test fun `typed reason code blocks even when disk and status suggest availability`() {
        val source = model(pytorch = PipelineEngineAvailability(status = "supported",
            reasonCode = "SUPPLY_FUTURE_INCOMPATIBILITY", reason = "Selected source is incompatible"), diskAvailable = true)
        val decision = modelIntentSelectionAvailability(source, "auto")
        assertFalse(decision.enabled)
        assertEquals("SUPPLY_FUTURE_INCOMPATIBILITY", decision.reasonCode)
        assertTrue(decision.message.contains("Selected source is incompatible"))
    }

    @Test fun `blank coded reason uses neutral refusal without claiming a download can fix it`() {
        val source = model(llama = PipelineEngineAvailability(status = "unavailable",
            reasonCode = "  $fullSource  ", reason = "  "))
        val decision = modelIntentSelectionAvailability(source, "llama_cpp")
        assertFalse(decision.enabled)
        assertEquals(fullSource, decision.reasonCode)
        assertTrue(decision.message.contains("llama.cpp"))
        assertTrue(decision.message.contains(fullSource))
        assertFalse(decision.message.contains("获取"))
    }

    @Test fun `engine changes recompute enabled and text from the chosen format only`() {
        val source = model(pytorch = blocked(unsupported, "HF unsupported"),
            llama = PipelineEngineAvailability(status = "unknown"))
        val automatic = modelIntentSelectionAvailability(source, "auto")
        val ll = modelIntentSelectionAvailability(source, "llama_cpp")
        val hf = modelIntentSelectionAvailability(source, "pytorch")
        assertFalse(automatic.enabled)
        assertTrue(ll.enabled)
        assertTrue(ll.message.startsWith("llama.cpp"))
        assertFalse(ll.message.contains("HF unsupported"))
        assertFalse(hf.enabled)
        assertTrue(hf.message.startsWith("PyTorch"))
    }

    @Test fun `provenance metadata does not bypass a runtime refusal or assert loaded content`() {
        val proof = PipelineEngineAvailability(status = "supported", source = "provenance",
            contentVerification = "supply_owner_required")
        val source = model(llama = proof, auto = "llama_cpp")
        val selectable = modelIntentSelectionAvailability(source, "auto")
        assertTrue(selectable.enabled)
        assertTrue(selectable.message.contains("完整模型内容仍由主节点"))
        val invalid = modelIntentSelectionAvailability(source.copy(
            pipelineEngineAvailability = mapOf("llama_cpp" to proof.copy(reasonCode = fullSource, reason = "Pre-cut"))), "auto")
        assertFalse(invalid.enabled)
        assertEquals(fullSource, invalid.reasonCode)
    }

    @Test fun `selection preserves the operation gate and rejects an empty registry ID`() {
        val source = model(pytorch = PipelineEngineAvailability(status = "supported"))
        assertTrue(modelIntentSelectionAvailability(source, "pytorch", canChooseModel = true).enabled)
        assertFalse(modelIntentSelectionAvailability(source, "pytorch", canChooseModel = false).enabled)
        val invalid = modelIntentSelectionAvailability(source.copy(modelId = " "), "pytorch")
        assertFalse(invalid.enabled)
        assertTrue(invalid.message.contains("模型标识缺失"))
    }

    @Test fun `proof metadata deserializes without converting it into a full content receipt`() {
        val source = Gson().fromJson("""{"model_id":"qwen3-5-9b","is_available":true,
            "preferred_engine":"pytorch","has_gguf":true,"supported_engines":["pytorch","llama_cpp"],
            "pipeline_auto_engine":"pytorch","pipeline_engine_availability":{
                "llama_cpp":{"status":"supported","runtime_model_type":"qwen3_5",
                    "reason_code":"","reason":"","source":"provenance",
                    "content_verification":"supply_owner_required"}}}""", ServerModelSummary::class.java)
        val evidence = source.pipelineEngineAvailability?.get("llama_cpp")
        assertEquals("provenance", evidence?.source)
        assertEquals("supply_owner_required", evidence?.contentVerification)
        assertEquals(listOf("pytorch", "llama_cpp"), source.supportedEngines)
        assertTrue(source.hasGguf)
        assertTrue(modelIntentSelectionAvailability(source, "llama_cpp").enabled)
        assertEquals("pytorch", source.preferredEngine)
        assertEquals("pytorch", source.pipelineAutoEngine)
    }
}
