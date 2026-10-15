package com.qlh.inference.service

import com.google.gson.JsonElement
import com.google.gson.JsonObject
import com.google.gson.JsonParser
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class GgufMetadataContractTest {
    private fun metadata(): JsonObject = JsonParser.parseString("""{
        "architecture":"qwen35","block_count":2,"head_count_kv":[2,4],
        "key_length":128,"full_attention_interval":null
    }""").asJsonObject
    private fun actual(json: JsonObject = metadata()): Map<String, JsonElement> = json.entrySet().associate { it.key to it.value }
    private val keys = listOf("block_count", "head_count_kv", "key_length", "full_attention_interval")

    @Test fun `matching scalars arrays and explicit absence pass without source block or nextn values`() {
        assertTrue(GgufMetadataContract.matches(actual(), metadata(), 2))
        val expected = metadata().apply { remove("architecture"); addProperty("n_embd", 2048); addProperty("nextn_predict_layers", 0) }
        assertTrue(GgufMetadataContract.matches(actual(), expected, 2))
        assertFalse(GgufMetadataContract.matches(actual(), metadata(), 4))
    }

    @Test fun `every critical key must be explicitly present on both sides`() {
        for (key in keys) {
            assertFalse(key, GgufMetadataContract.matches(actual(), metadata().apply { remove(key) }, 2))
            assertFalse(key, GgufMetadataContract.matches(actual(metadata().apply { remove(key) }), metadata(), 2))
        }
        for (invalid in listOf<JsonElement?>(null, JsonParser.parseString("null"), JsonParser.parseString("[]"), JsonParser.parseString("1"))) {
            assertFalse(GgufMetadataContract.matches(actual(), invalid, 2))
        }
        assertFalse(GgufMetadataContract.matches(null, metadata(), 2))
    }

    @Test fun `null denotes real optional absence and never matches a value or missing key`() {
        for (key in keys.drop(1)) {
            val absent = metadata().apply { add(key, JsonParser.parseString("null")) }
            val present = metadata().apply { addProperty(key, 4) }
            assertTrue(GgufMetadataContract.matches(actual(absent), absent, 2))
            assertFalse(GgufMetadataContract.matches(actual(absent), present, 2))
            assertFalse(GgufMetadataContract.matches(actual(present), absent, 2))
            assertFalse(GgufMetadataContract.matches(actual(absent), absent.deepCopy().apply { remove(key) }, 2))
        }
        val noBlocks = metadata().apply { add("block_count", JsonParser.parseString("null")) }
        assertFalse(GgufMetadataContract.matches(actual(noBlocks), noBlocks, 2))
    }

    @Test fun `each value mismatch and array shape or order mismatch fails`() {
        for (key in keys) {
            assertFalse(key, GgufMetadataContract.matches(actual(), metadata().apply { addProperty(key, 99) }, 2))
        }
        for (value in listOf("[4,2]", "[2]", "[2,4,6]", "2", "[[2,4]]")) {
            assertFalse(value, GgufMetadataContract.matches(actual(), metadata().apply { add("head_count_kv", JsonParser.parseString(value)) }, 2))
        }
    }

    @Test fun `integer lookalikes are invalid even when Gson numeric equality would match`() {
        for (value in listOf("true", "false", "2.0", "2e0", "\"2\"", "[2.0,4]", "[2,true]", "[2,null]", "[\"2\",4]")) {
            val typed = metadata().apply { add("head_count_kv", JsonParser.parseString(value)) }
            assertFalse(value, GgufMetadataContract.matches(actual(), typed, 2))
            assertFalse(value, GgufMetadataContract.matches(actual(typed), typed, 2))
        }
        for (value in listOf("2.0", "2e0", "true", "[2]")) {
            val typed = metadata().apply { add("block_count", JsonParser.parseString(value)) }
            assertFalse(value, GgufMetadataContract.matches(actual(typed), typed, 2))
        }
    }

    @Test fun `large signed and unsigned integers compare without double or long truncation`() {
        for (value in listOf("9007199254740993", "18446744073709551615", "-9223372036854775808")) {
            val exact = metadata().apply { add("key_length", JsonParser.parseString(value)) }
            assertTrue(GgufMetadataContract.matches(actual(exact), exact, 2))
            val different = exact.deepCopy().apply { add("key_length", JsonParser.parseString((value.toBigInteger() - java.math.BigInteger.ONE).toString())) }
            assertFalse(GgufMetadataContract.matches(actual(exact), different, 2))
        }
    }

    @Test fun `architecture must resolve actual header fields and match when declared`() {
        for (value in listOf("\"llama\"", "null", "1", "true", "[]")) {
            assertFalse(GgufMetadataContract.matches(actual(), metadata().apply { add("architecture", JsonParser.parseString(value)) }, 2))
        }
        assertFalse(GgufMetadataContract.matches(actual(metadata().apply { remove("architecture") }), metadata(), 2))
    }

    @Test fun `matching metadata has no differences and matches delegates to the same verdict`() {
        assertTrue(GgufMetadataContract.differences(actual(), metadata(), 2).isEmpty())
        val variants = listOf(metadata(), metadata().apply { remove("head_count_kv") },
            metadata().apply { addProperty("key_length", true) }, metadata().apply { addProperty("key_length", 256) })
        for (declared in variants) {
            assertEquals(GgufMetadataContract.differences(actual(), declared, 2).isEmpty(),
                GgufMetadataContract.matches(actual(), declared, 2))
        }
    }

    @Test fun `missing diagnostics name each field and the side with no declaration`() {
        for (key in keys) {
            assertEquals(listOf("$key: missing from declared metadata"),
                GgufMetadataContract.differences(actual(), metadata().apply { remove(key) }, 2))
            assertEquals(listOf("$key: missing from actual metadata"),
                GgufMetadataContract.differences(actual(metadata().apply { remove(key) }), metadata(), 2))
        }
        assertEquals(listOf("architecture: missing from actual metadata"),
            GgufMetadataContract.differences(actual(metadata().apply { remove("architecture") }), metadata(), 2))
    }

    @Test fun `typedvalue diagnostics name scalar and array fields on either side`() {
        for (key in listOf("head_count_kv", "key_length")) {
            for (value in listOf("true", "128.0", "\"128\"", "[2.0,4]", "[2,true]", "[2,null]", "[[2]]")) {
                val typed = metadata().apply { add(key, JsonParser.parseString(value)) }
                assertEquals("$key=$value", listOf("$key: typedvalue in declared metadata"),
                    GgufMetadataContract.differences(actual(), typed, 2))
                assertEquals("$key=$value", listOf("$key: typedvalue in actual metadata"),
                    GgufMetadataContract.differences(actual(typed), metadata(), 2))
            }
        }
        for (value in listOf("null", "2.0", "true", "[2]")) {
            assertEquals(listOf("block_count: typedvalue in declared metadata"),
                GgufMetadataContract.differences(actual(), metadata().apply { add("block_count", JsonParser.parseString(value)) }, 2))
        }
        assertEquals(listOf("architecture: typedvalue in declared metadata"),
            GgufMetadataContract.differences(actual(), metadata().apply { addProperty("architecture", true) }, 2))
    }

    @Test fun `mismatch diagnostics identify array order scalar and explicit null differences`() {
        for ((key, value) in listOf("head_count_kv" to "[4,2]", "head_count_kv" to "2", "key_length" to "256",
            "key_length" to "null", "full_attention_interval" to "4", "architecture" to "\"llama\"")) {
            assertEquals(listOf("$key: mismatch between actual and declared values"),
                GgufMetadataContract.differences(actual(), metadata().apply { add(key, JsonParser.parseString(value)) }, 2))
        }
        assertEquals(listOf("block_count: mismatch with segment layer count (4)"),
            GgufMetadataContract.differences(actual(), metadata(), 4))
    }

    @Test fun `all named differences are collected in stable field order`() {
        val declared = metadata().apply {
            remove("head_count_kv")
            addProperty("key_length", true)
            addProperty("full_attention_interval", 4)
            addProperty("architecture", "llama")
        }
        assertEquals(listOf(
            "head_count_kv: missing from declared metadata",
            "key_length: typedvalue in declared metadata",
            "full_attention_interval: mismatch between actual and declared values",
            "architecture: mismatch between actual and declared values",
        ), GgufMetadataContract.differences(actual(), declared, 2))
    }

    @Test fun `missing or malformed metadata container still names the missing critical fields`() {
        val missing = GgufMetadataContract.differences(actual(), null, 2)
        assertTrue(missing.contains("gguf_metadata: missing declared metadata"))
        for (key in keys) assertTrue(missing.contains("$key: missing from declared metadata"))
        for (value in listOf("null", "[]", "true")) {
            val malformed = GgufMetadataContract.differences(actual(), JsonParser.parseString(value), 2)
            assertTrue(malformed.contains("gguf_metadata: typedvalue in declared metadata (expected object)"))
            for (key in keys) assertTrue(malformed.contains("$key: missing from declared metadata"))
        }
    }

}
