package com.qlh.inference.service

import com.google.gson.JsonElement

/** Receipt contract for header KVs; source-model top-level fields are not cut metadata. */
internal object GgufMetadataContract {
    private val requiredKeys = setOf("block_count", "head_count_kv", "key_length", "full_attention_interval")
    private val integerPattern = Regex("-?(0|[1-9][0-9]*)")

    fun matches(actual: Map<String, JsonElement>?, declared: JsonElement?, layerCount: Int): Boolean =
        differences(actual, declared, layerCount).isEmpty()

    /** Stable field names and categories; collect every difference without retaining raw values. */
    fun differences(actual: Map<String, JsonElement>?, declared: JsonElement?, layerCount: Int): List<String> {
        val problems = mutableListOf<String>()
        val expected = declared?.takeIf { it.isJsonObject }?.asJsonObject
        if (actual == null) problems += "gguf_metadata: missing actual metadata"
        if (declared == null) {
            problems += "gguf_metadata: missing declared metadata"
        } else if (expected == null) {
            problems += "gguf_metadata: typedvalue in declared metadata (expected object)"
        }
        for (key in requiredKeys) {
            val left = actual?.get(key)
            val right = expected?.get(key)
            if (left == null) problems += "$key: missing from actual metadata"
            if (right == null) problems += "$key: missing from declared metadata"
            val leftValid = left != null && isValue(key, left)
            val rightValid = right != null && isValue(key, right)
            if (left != null && !leftValid) problems += "$key: typedvalue in actual metadata"
            if (right != null && !rightValid) problems += "$key: typedvalue in declared metadata"
            if (leftValid && rightValid && !sameValue(left!!, right!!)) {
                problems += "$key: mismatch between actual and declared values"
            }
            // The header describes this segment, even when the source manifest describes more blocks.
            if (key == "block_count" && leftValid && (layerCount <= 0 || left!!.asBigInteger != layerCount.toBigInteger())) {
                problems += "$key: mismatch with segment layer count ($layerCount)"
            }
        }
        val architecture = actual?.get("architecture")
        if (architecture == null) {
            problems += "architecture: missing from actual metadata"
        } else if (!isArchitecture(architecture)) {
            problems += "architecture: typedvalue in actual metadata"
        }
        if (expected?.has("architecture") == true) {
            val value = expected["architecture"]
            if (!isArchitecture(value)) {
                problems += "architecture: typedvalue in declared metadata"
            } else if (architecture != null && isArchitecture(architecture) && value.asString != architecture.asString) {
                problems += "architecture: mismatch between actual and declared values"
            }
        }
        return problems
    }

    private fun isArchitecture(value: JsonElement): Boolean = value.isJsonPrimitive &&
        value.asJsonPrimitive.isString && value.asString.isNotEmpty()

    private fun isValue(key: String, value: JsonElement): Boolean = if (key == "block_count") {
        isInteger(value)
    } else {
        value.isJsonNull || isInteger(value) || (value.isJsonArray && value.asJsonArray.all(::isInteger))
    }

    private fun isInteger(value: JsonElement): Boolean = value.isJsonPrimitive &&
        value.asJsonPrimitive.isNumber && integerPattern.matches(value.toString())

    private fun sameValue(actual: JsonElement, expected: JsonElement): Boolean = when {
        actual.isJsonNull || expected.isJsonNull -> actual.isJsonNull && expected.isJsonNull
        actual.isJsonArray && expected.isJsonArray -> {
            val left = actual.asJsonArray
            val right = expected.asJsonArray
            left.size() == right.size() && left.zip(right).all { (a, b) ->
                isInteger(a) && isInteger(b) && a.asBigInteger == b.asBigInteger
            }
        }
        else -> isInteger(actual) && isInteger(expected) && actual.asBigInteger == expected.asBigInteger
    }
}
