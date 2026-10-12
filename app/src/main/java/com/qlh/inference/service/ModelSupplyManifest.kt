package com.qlh.inference.service

import com.google.gson.JsonElement
import com.google.gson.JsonObject
import com.google.gson.JsonParser
import java.security.MessageDigest

internal class ModelSupplyException(val reasonCode: String, message: String) : Exception(message)
internal fun checkSupply(ok: Boolean, code: String, detail: String) {
    if (!ok) throw ModelSupplyException(code, detail)
}
internal fun supplySha256(bytes: ByteArray): String = MessageDigest.getInstance("SHA-256")
    .digest(bytes).joinToString("") { "%02x".format(it.toInt() and 255) }

/** Python ASCII, sorted-key, compact JSON. Numeric tokens arrive in Python canonical form. */
internal fun canonicalSupplyJson(value: JsonElement): String = when {
    value.isJsonNull -> "null"
    value.isJsonObject -> value.asJsonObject.entrySet().sortedWith { a, b -> compareUnicodeKeys(a.key, b.key) }
        .joinToString(",", "{", "}") { (key, child) -> asciiJsonString(key) + ":" + canonicalSupplyJson(child) }
    value.isJsonArray -> value.asJsonArray.joinToString(",", "[", "]") { canonicalSupplyJson(it) }
    value.asJsonPrimitive.isString -> asciiJsonString(value.asString)
    value.asJsonPrimitive.isBoolean -> value.asBoolean.toString()
    else -> value.asString.also {
        checkSupply(Regex("-?(0|[1-9][0-9]*)(\\.[0-9]+)?([eE][+-]?[0-9]+)?").matches(it),
            "SUPPLY_INVALID_MANIFEST", "invalid JSON number")
    }
}
private fun compareUnicodeKeys(a: String, b: String): Int {
    val aa = a.codePoints().toArray(); val bb = b.codePoints().toArray()
    for (i in 0 until minOf(aa.size, bb.size)) if (aa[i] != bb[i]) return aa[i].compareTo(bb[i])
    return aa.size.compareTo(bb.size)
}
private fun asciiJsonString(text: String): String = buildString {
    append('"')
    for (char in text) when (char) {
        '"' -> append("\\\""); '\\' -> append("\\\\"); '\b' -> append("\\b")
        '\u000c' -> append("\\f"); '\n' -> append("\\n"); '\r' -> append("\\r"); '\t' -> append("\\t")
        else -> if (char.code < 32 || char.code >= 127) append("\\u%04x".format(char.code)) else append(char)
    }
    append('"')
}

/** Proof paths are opaque metadata. Only validated basenames reach the filesystem. */
internal data class ModelSupplyManifest(val json: JsonObject, val files: List<SupplyFile>) {
    data class SupplyFile(val path: String, val size: Long, val sha256: String, val kind: String)
    val operationId get() = json["operation_id"].asString
    val nodeId get() = json["node_id"].asString
    val sourceModelId get() = json["source_model_id"].asString
    val sourceSha256 get() = json["source_model_sha256"].asString
    val manifestSha256 get() = json["manifest_sha256"].asString
    val artifactSha256 get() = json["artifact_sha256"].asString
    val start get() = json["layer_range"].asJsonArray[0].asInt
    val end get() = json["layer_range"].asJsonArray[1].asInt
    val weight get() = files.single { it.kind == "weights" }
    val native get() = files.single { it.kind == "manifest" && it.path in nativeNames(weight.path) }
    val localLayerBytes get() = json["layer_weight_bytes"].asJsonArray.map { it.asLong }.subList(start, end)
    fun result(status: String, reason: String = "") = JsonObject().apply {
        addProperty("action", "result"); addProperty("operation_id", operationId); addProperty("node_id", nodeId)
        addProperty("status", status); addProperty("reason_code", reason)
        for (key in listOf("manifest_sha256", "artifact_sha256", "layer_range", "source_model_sha256")) add(key, json[key].deepCopy())
    }
    companion object {
        const val MARKER = "supply.manifest.json"
        val OPERATION = Regex("supply_[0-9a-f]{32}")
        private val SHA = Regex("[0-9a-f]{64}")
        private val ID = Regex("[A-Za-z0-9_.:-]{1,128}")
        private val REQUIRED = setOf("manifest_kind", "operation_id", "node_id", "source_model_id", "source_model_sha256",
            "tokenizer_sha256", "total_layers", "layer_range", "segment_mode", "layer_weight_bytes", "hidden_size",
            "artifact_sha256", "conversion_proof", "files", "manifest_sha256", "format")
        private val OPTIONAL = setOf("model_type", "has_embedding", "has_lm_head", "tie_word_embeddings")
        fun nativeNames(path: String) = setOf(path.removeSuffix(".gguf") + ".manifest.json", path + ".manifest.json")
        fun text(json: JsonObject, key: String): String {
            val v = json[key]
            checkSupply(v != null && v.isJsonPrimitive && v.asJsonPrimitive.isString, "SUPPLY_INVALID_MANIFEST", key)
            return v.asString
        }
        fun integer(v: JsonElement?, min: Long = 0, max: Long = Long.MAX_VALUE): Long {
            checkSupply(v != null && v.isJsonPrimitive && v.asJsonPrimitive.isNumber &&
                Regex("0|[1-9][0-9]*").matches(v.asString), "SUPPLY_INVALID_MANIFEST", "exact integer")
            val n = v!!.asString.toLongOrNull()
            checkSupply(n != null && n in min..max, "SUPPLY_INVALID_MANIFEST", "integer bounds")
            return n!!
        }
        fun safeBasename(path: String) {
            val stem = path.substringBefore('.').uppercase()
            checkSupply(path.length in 1..255 && Regex("[A-Za-z0-9_.-]+").matches(path) && !path.startsWith('.') &&
                !path.endsWith('.') && !path.endsWith(".part", true) && path.lowercase() != MARKER &&
                stem !in setOf("CON", "PRN", "AUX", "NUL") && !Regex("(COM|LPT)[1-9]").matches(stem),
                "SUPPLY_UNSAFE_PATH", "unsafe basename")
        }
        fun parse(raw: String): ModelSupplyManifest = parse(JsonParser.parseString(raw).asJsonObject)
        fun parse(input: JsonObject): ModelSupplyManifest {
            val j = input.deepCopy()
            checkSupply(j.keySet().containsAll(REQUIRED) && j.keySet().all { it in REQUIRED + OPTIONAL }, "SUPPLY_INVALID_MANIFEST", "manifest fields")
            checkSupply(text(j, "manifest_kind") == "qlh.segment_supply.v1", "SUPPLY_INVALID_MANIFEST", "manifest kind")
            checkSupply(OPERATION.matches(text(j, "operation_id")), "SUPPLY_INVALID_OPERATION", "operation id")
            for (key in listOf("node_id", "source_model_id")) checkSupply(ID.matches(text(j, key)) && text(j, key) !in setOf(".", ".."), "SUPPLY_INVALID_MANIFEST", key)
            for (key in listOf("source_model_sha256", "tokenizer_sha256", "artifact_sha256", "manifest_sha256")) checkSupply(SHA.matches(text(j, key)), "SUPPLY_INVALID_MANIFEST", key)
            val total = integer(j["total_layers"], 1, 100_000).toInt()
            integer(j["hidden_size"], 1, Int.MAX_VALUE.toLong())
            val range = j["layer_range"]
            checkSupply(range.isJsonArray && range.asJsonArray.size() == 2, "SUPPLY_INVALID_MANIFEST", "range")
            val start = integer(range.asJsonArray[0], 0, total.toLong()).toInt()
            val end = integer(range.asJsonArray[1], 0, total.toLong()).toInt()
            checkSupply(start < end && !(start == 0 && end == total), "SUPPLY_FULL_MODEL_FORBIDDEN", "proper segment required")
            checkSupply(text(j, "segment_mode") == (if (start == 0) "head" else if (end == total) "tail" else "middle"), "SUPPLY_INVALID_MANIFEST", "mode")
            val bytes = j["layer_weight_bytes"]
            checkSupply(bytes.isJsonArray && bytes.asJsonArray.size() == total, "SUPPLY_INVALID_MANIFEST", "global layer bytes")
            bytes.asJsonArray.forEachIndexed { index, v ->
                val n = integer(v)
                checkSupply(if (index in start until end) n > 0 else n == 0L, "SUPPLY_INVALID_MANIFEST", "layer bytes")
            }
            checkSupply(j["conversion_proof"].isJsonNull || j["conversion_proof"].isJsonObject, "SUPPLY_INVALID_MANIFEST", "proof")
            for (key in OPTIONAL - "model_type") if (j.has(key)) checkSupply(j[key].isJsonPrimitive && j[key].asJsonPrimitive.isBoolean, "SUPPLY_INVALID_MANIFEST", key)
            checkPublic(j)
            val inventory = j["files"]
            checkSupply(inventory.isJsonArray && inventory.asJsonArray.size() in 1..1024, "SUPPLY_INVALID_MANIFEST", "files")
            val files = inventory.asJsonArray.map { v ->
                checkSupply(v.isJsonObject, "SUPPLY_INVALID_MANIFEST", "file object")
                val f = v.asJsonObject
                checkSupply(f.keySet().containsAll(setOf("path", "size_bytes", "sha256", "kind")), "SUPPLY_INVALID_MANIFEST", "file fields")
                val path = text(f, "path"); safeBasename(path)
                val sha = text(f, "sha256"); checkSupply(SHA.matches(sha), "SUPPLY_INVALID_MANIFEST", "file sha")
                val kind = text(f, "kind"); checkSupply(kind in setOf("weights", "manifest", "support"), "SUPPLY_INVALID_MANIFEST", "file kind")
                SupplyFile(path, integer(f["size_bytes"]), sha, kind)
            }
            checkSupply(files.map { it.path.lowercase() }.distinct().size == files.size, "SUPPLY_UNSAFE_PATH", "duplicate basename")
            checkSupply(text(j, "format") == "gguf", "SUPPLY_FORMAT_UNSUPPORTED", "Android supports GGUF")
            val weights = files.filter { it.kind == "weights" }
            checkSupply(weights.size == 1 && weights[0].path.endsWith(".gguf") && weights[0].size > 0, "SUPPLY_INVALID_MANIFEST", "weight inventory")
            val weight = weights.single()
            checkSupply(files.count { it.kind == "manifest" && it.path in nativeNames(weight.path) } == 1, "SUPPLY_INVALID_MANIFEST", "native manifest")
            checkSupply(weight.sha256 == text(j, "artifact_sha256"), "SUPPLY_ARTIFACT_DIGEST_MISMATCH", "weight digest")
            val unsigned = j.deepCopy().apply { remove("manifest_sha256") }
            checkSupply(supplySha256(canonicalSupplyJson(unsigned).toByteArray(Charsets.UTF_8)) == text(j, "manifest_sha256"), "SUPPLY_MANIFEST_DIGEST_MISMATCH", "manifest digest")
            return ModelSupplyManifest(j, files)
        }
        private fun checkPublic(v: JsonElement) {
            when {
                v.isJsonObject -> v.asJsonObject.entrySet().forEach { (key, child) -> checkPublic(com.google.gson.JsonPrimitive(key)); checkPublic(child) }
                v.isJsonArray -> v.asJsonArray.forEach(::checkPublic)
                v.isJsonPrimitive && v.asJsonPrimitive.isString -> {
                    val s = v.asString
                    checkSupply(!s.startsWith('/') && !s.startsWith('\\') && !s.startsWith("~/") && !Regex("[A-Za-z]:[\\\\/]").containsMatchIn(s), "SUPPLY_UNSAFE_MANIFEST", "absolute metadata path")
                }
            }
        }
    }
}
