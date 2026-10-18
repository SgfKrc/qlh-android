package com.qlh.inference.worker

import com.google.gson.JsonElement
import com.google.gson.JsonParser
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Assert.assertThrows
import org.junit.Test

class EngineRegistryTest {
    private fun jsonValue(value: JsonElement): Any? = when {
        value.isJsonNull -> null
        value.isJsonObject -> value.asJsonObject.entrySet().associate { it.key to jsonValue(it.value) }
        value.isJsonArray -> value.asJsonArray.map(::jsonValue)
        value.asJsonPrimitive.isBoolean -> value.asBoolean
        value.asJsonPrimitive.isString -> value.asString
        else -> value.asString.toLongOrNull() ?: value.asDouble
    }

    @Suppress("UNCHECKED_CAST")
    private fun fixture(): Map<String, Any?> {
        val stream = requireNotNull(javaClass.classLoader!!.getResourceAsStream("engine-capabilities-v1.json"))
        return stream.bufferedReader().use { jsonValue(JsonParser.parseReader(it)) as Map<String, Any?> }
    }

    @Test
    fun `entire Kotlin registry equals the versioned Python fixture`() {
        val actual = JsonParser.parseString(TaskWorkerProtocol.canonicalJson(EngineRegistry.snapshot()))
        val expected = JsonParser.parseString(TaskWorkerProtocol.canonicalJson(fixture()["registry"] as Map<String, Any?>))
        assertEquals(expected, actual)
    }

    @Test
    @Suppress("UNCHECKED_CAST")
    fun `real Android builder and protocol match shared runtime cases`() {
        val cases = fixture()["builder_cases"] as List<Map<String, Any?>>
        cases.filter { it["family"] == "llama_cpp" }.forEach { case ->
            val kind = case["runtime_kind"] as String
            val probe = (case["probed_operations"] as? List<String>)?.toSet()
            val full = case["full_inference_available"] as Boolean
            val ranges = (case["layer_ranges"] as List<List<Number>>).map { range -> range.map { it.toInt() } }
            val caps = AndroidWorkerCapabilities.build(
                modelId = "fixture_model", modelSha256 = "a".repeat(64), resourceAdmitted = true,
                fullInferenceAvailable = full, layerRanges = ranges,
                runtimeKind = kind, probedOperations = probe,
            )
            assertEquals(case["name"].toString(),
                JsonParser.parseString(TaskWorkerProtocol.canonicalJson(case["engine_capabilities"])),
                JsonParser.parseString(TaskWorkerProtocol.canonicalJson(caps["engine_capabilities"])))
            assertEquals(case["android_stage_types"], caps["stage_types"])
            if (!full) {
                assertFalse((caps["stage_types"] as List<*>).contains("full_inference"))
                assertTrue(EngineRegistry.validateAdvertisement(caps["engine_capabilities"]).all {
                    EngineRegistry.FULL_REQUEST !in it.operations
                })
            }
            val hello = TaskWorkerProtocol.buildHello(
                nodeId = "android_registry_fixture", capabilities = caps,
                messageId = "msg_registry_fixture", sentAtMs = 1L,
            )
            TaskWorkerProtocol.validate(hello)
            val decoded = TaskWorkerProtocol.decode(TaskWorkerProtocol.encode(hello))
            assertEquals(TaskWorkerProtocol.canonicalJson(caps),
                TaskWorkerProtocol.canonicalJson(decoded.payload["capabilities"]))
        }
    }

    @Test
    @Suppress("UNCHECKED_CAST")
    fun `real Android protocol rejects every shared invalid capability advertisement`() {
        val cases = fixture()["invalid_cases"] as List<Map<String, Any?>>
        cases.forEach { case ->
            val caps = case["capabilities"] as Map<String, Any?>
            val error = assertThrows(case["name"].toString(), TaskWorkerProtocolException::class.java) {
                TaskWorkerProtocol.validate(TaskWorkerProtocol.buildHello(
                    nodeId = "android_registry_fixture", capabilities = caps,
                    messageId = "msg_registry_invalid", sentAtMs = 1L,
                ))
            }
            assertEquals(case["name"].toString(), "invalid_capabilities", error.code)
        }
    }

    @Test
    fun `server cannot build a layer inventory and binding probe only narrows`() {
        assertThrows(IllegalArgumentException::class.java) {
            AndroidWorkerCapabilities.build(runtimeKind = "llama_server", layerRanges = listOf(listOf(2, 4)))
        }
        val schema = EngineRegistry.advertisement("llama_cpp", "llama_server",
            setOf(EngineRegistry.FULL_REQUEST, EngineRegistry.LAYER_RANGE, EngineRegistry.FORWARD_LAYERS))
        assertEquals(setOf(EngineRegistry.FULL_REQUEST), EngineRegistry.validateAdvertisement(schema).single().operations)
    }
}
