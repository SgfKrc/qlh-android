package com.qlh.inference.worker

import com.google.gson.Gson
import com.google.gson.JsonElement
import com.google.gson.JsonObject
import com.qlh.inference.service.ModelSupplyManifest
import com.qlh.inference.service.canonicalSupplyJson

internal class WorkerPrepareException(val code: String, detail: String) : IllegalStateException(detail)
internal fun checkPrepare(ok: Boolean, code: String = "prepare_invalid_request", detail: String) {
    if (!ok) throw WorkerPrepareException(code, detail)
}

/** Immutable wire identity. ACKs echo the original JSON, including master and artifact identities. */
internal class WorkerPrepareIdentity private constructor(private val value: JsonObject) {
    // Gson JsonPrimitive equality can compare LazilyParsedNumber via Double. A JSON-parsed
    // generation and its adjacent nanosecond must remain different in every owner/cleanup check.
    private val exactKey = canonicalSupplyJson(value)
    fun wire(): JsonObject = value.deepCopy()
    fun text(key: String): String = value[key].asString
    val generation: Long get() = value["generation"].asLong
    val range: List<Int> get() = value["layer_range"].asJsonArray.map { it.asInt }
    val totalLayers: Int get() = value["total_layers"].asInt
    val hiddenSize: Int get() = value["hidden_size"].asInt
    val mode: String get() = text("segment_mode")
    val stageIdentity: Map<String, Any?> get() = value["stage_model_identity"].asJsonObject.entrySet().associate { it.key to it.value.asString }
    fun stageTuple(): JsonObject = JsonObject().apply {
        for (key in STAGE_KEYS + OPTIONAL_KEYS) if (value.has(key)) add(key, value[key].deepCopy())
    }
    override fun equals(other: Any?): Boolean = other is WorkerPrepareIdentity && exactKey == other.exactKey
    override fun hashCode(): Int = exactKey.hashCode()

    fun verifyManifest(manifest: ModelSupplyManifest) {
        val m = manifest.json
        for (key in listOf("operation_id", "node_id", "artifact_sha256", "manifest_sha256", "source_model_id",
            "source_model_sha256", "tokenizer_sha256", "layer_range", "total_layers", "hidden_size", "segment_mode")) {
            checkPrepare(value[key] == m[key], "prepare_manifest_mismatch", "manifest differs at $key")
        }
        checkPrepare(stageIdentity["model_id"] == AndroidWorkerCapabilities.artifactModelId(manifest.weight.path, manifest.artifactSha256),
            "prepare_artifact_identity_mismatch", "artifact model id differs")
    }

    companion object {
        val OPTIONAL_KEYS = setOf("load_spec_digest")
        private val ID = Regex("[A-Za-z0-9_.:/-]{1,256}")
        val STAGE_KEYS = setOf("operation_id", "config_id", "generation", "assignment_id", "artifact_sha256")
        private val KEYS = STAGE_KEYS + setOf("node_id", "model_id", "model_sha256", "stage_model_identity", "manifest_sha256",
            "source_model_id", "source_model_sha256", "tokenizer_sha256", "layer_range", "total_layers", "hidden_size", "segment_mode")
        private val MODEL_KEYS = setOf("model_id", "engine", "format", "revision", "sha256")
        private val SHA = Regex("[0-9a-f]{64}")
        fun text(j: JsonObject, key: String, empty: Boolean = false): String {
            val v = j[key]
            checkPrepare(v != null && v.isJsonPrimitive && v.asJsonPrimitive.isString, detail = "$key must be a string")
            return v.asString.also { checkPrepare((empty || it.isNotBlank()) && ((empty && it.isEmpty()) || ID.matches(it)), detail = "invalid $key") }
        }
        fun integer(v: JsonElement?, min: Long = 0L, max: Long = Long.MAX_VALUE): Long {
            checkPrepare(v != null && v.isJsonPrimitive && v.asJsonPrimitive.isNumber && Regex("0|[1-9][0-9]*").matches(v.asString), detail = "exact integer required")
            val n = v!!.asString.toLongOrNull()
            checkPrepare(n != null && n in min..max, detail = "integer bounds")
            return n!!
        }
        fun parse(j: JsonObject): WorkerPrepareIdentity {
            checkPrepare(j.keySet() - OPTIONAL_KEYS == KEYS, detail = "identity fields")
            if (j.has("load_spec_digest")) checkPrepare(SHA.matches(j["load_spec_digest"].asString), detail = "load spec digest")
            for (key in STAGE_KEYS - "generation" + setOf("node_id", "model_id", "source_model_id", "model_sha256", "manifest_sha256", "source_model_sha256", "tokenizer_sha256", "segment_mode")) {
                text(j, key, key == "manifest_sha256")
            }
            for (key in listOf("artifact_sha256", "model_sha256", "source_model_sha256", "tokenizer_sha256", "manifest_sha256")) {
                checkPrepare((key == "manifest_sha256" && j[key].asString.isEmpty()) || SHA.matches(j[key].asString), detail = "invalid $key")
            }
            integer(j["generation"], 1)
            val total = integer(j["total_layers"], 1, Int.MAX_VALUE.toLong()).toInt()
            integer(j["hidden_size"], 1, Int.MAX_VALUE.toLong())
            val range = j["layer_range"]
            checkPrepare(range.isJsonArray && range.asJsonArray.size() == 2, detail = "layer range")
            val start = integer(range.asJsonArray[0], 0, total.toLong()).toInt()
            val end = integer(range.asJsonArray[1], 0, total.toLong()).toInt()
            checkPrepare(start < end && !(start == 0 && end == total), "prepare_full_model_forbidden", "proper segment required")
            checkPrepare(j["segment_mode"].asString == if (start == 0) "head" else if (end == total) "tail" else "middle", detail = "segment mode")
            val stage = j["stage_model_identity"]
            checkPrepare(stage.isJsonObject && stage.asJsonObject.keySet() == MODEL_KEYS, detail = "stage identity fields")
            MODEL_KEYS.forEach { text(stage.asJsonObject, it) }
            checkPrepare(stage.asJsonObject["engine"].asString == "llama_cpp" && stage.asJsonObject["format"].asString == "gguf",
                "prepare_engine_unsupported", "Android requires llama_cpp GGUF")
            checkPrepare(stage.asJsonObject["sha256"] == j["artifact_sha256"], "prepare_artifact_identity_mismatch", "stage SHA differs")
            return WorkerPrepareIdentity(j.deepCopy())
        }
        fun validateStageTuple(value: Map<*, *>) {
            checkPrepare(value.keys - OPTIONAL_KEYS == STAGE_KEYS, detail = "prepare_identity fields")
            val j = Gson().toJsonTree(value).asJsonObject
            (STAGE_KEYS - "generation").forEach { text(j, it) }
            integer(j["generation"], 1)
            checkPrepare(SHA.matches(j["artifact_sha256"].asString), detail = "prepare_identity artifact SHA")
        }
        fun tupleMatches(identity: WorkerPrepareIdentity, value: Any?): Boolean {
            val map = value as? Map<*, *> ?: return false
            return try {
                validateStageTuple(map)
                val parsed = Gson().toJsonTree(map).asJsonObject
                integer(parsed["generation"], 1) == identity.generation &&
                    (STAGE_KEYS - "generation").all { parsed[it].asString == identity.text(it) } &&
                    parsed.has("load_spec_digest") == identity.value.has("load_spec_digest") &&
                    (!parsed.has("load_spec_digest") || parsed["load_spec_digest"] == identity.value["load_spec_digest"])
            } catch (_: WorkerPrepareException) { false }
        }
    }
}

internal data class WorkerPrepareRequest(val action: String, val identity: WorkerPrepareIdentity, val cleanupId: String?, val loadSpec: JsonObject? = null) {
    fun ack(status: String, loaded: Boolean, error: String? = null): JsonObject = JsonObject().apply {
        addProperty("action", "ack"); addProperty("phase", action); add("identity", identity.wire())
        addProperty("status", status); addProperty("loaded", loaded)
        error?.let { addProperty("error", it) }; cleanupId?.let { addProperty("cleanup_id", it) }
    }
    companion object {
        fun parse(data: JsonObject): WorkerPrepareRequest {
            val action = WorkerPrepareIdentity.text(data, "action")
            checkPrepare(action in setOf("prepare", "commit", "abort", "finalize", "release"), detail = "action")
            val fields = (if (data.has("load_spec")) setOf("load_spec") else emptySet()) + setOf("action", "identity") + if (action == "prepare") setOf("context_size", "middle_channel")
                else if (action in setOf("abort", "finalize")) setOf("cleanup_id")
                else if (action == "release" && data.has("cleanup_id")) setOf("cleanup_id") else emptySet()
            checkPrepare(data.keySet() == fields, detail = "request fields")
            if (action == "prepare") {
                checkPrepare(WorkerPrepareIdentity.integer(data["context_size"]) == 4096L, detail = "context_size must be 4096")
                checkPrepare(WorkerPrepareIdentity.text(data, "middle_channel") == "keep_head_layer_out", detail = "middle_channel")
            }
            checkPrepare(data["identity"].isJsonObject, detail = "identity object")
            val identity = WorkerPrepareIdentity.parse(data["identity"].asJsonObject)
            val spec = if (data.has("load_spec")) data["load_spec"].asJsonObject.deepCopy() else null
            if (action == "prepare" && identity.wire().has("load_spec_digest")) {
                checkPrepare(spec != null, "prepare_load_spec_missing", "load spec required")
                WorkerPrepareLoadSpec.validate(requireNotNull(spec), identity, data["context_size"].asInt)
            } else checkPrepare(spec == null, detail = "unexpected load spec")
            return WorkerPrepareRequest(action, identity,
                if (data.has("cleanup_id")) WorkerPrepareIdentity.text(data, "cleanup_id") else null, spec)
        }
    }
}

/** Android's resolved CPU GGUF policy; unknown materialization never falls back. */
internal object WorkerPrepareLoadSpec {
    private val keys = setOf("device", "dtype", "quant_type", "context_size", "kv_device", "kv_dtype", "kv_policy",
        "components", "materialization", "kv_cache_bytes", "batch_size", "sequence_count", "schema_version")
    fun validate(spec: JsonObject, identity: WorkerPrepareIdentity, context: Int) {
        checkPrepare(spec.keySet() == keys, detail = "load spec fields")
        val expected = mutableSetOf("layers")
        if (identity.mode == "head") expected.add("embedding")
        if (identity.mode == "tail") expected.addAll(setOf("final_norm", "lm_head"))
        val components = spec["components"].asJsonArray.map { it.asString }
        checkPrepare(components == components.sorted() && components.size == expected.size && components.toSet() == expected, detail = "load spec components")
        for ((key, value) in mapOf("device" to "cpu", "dtype" to "artifact", "quant_type" to "artifact",
            "kv_device" to "cpu", "kv_dtype" to "float16", "kv_policy" to "preallocated", "materialization" to "gguf_segment")) {
            checkPrepare(spec[key].asString == value, "prepare_load_spec_unsupported", "unsupported $key")
        }
        checkPrepare(WorkerPrepareIdentity.integer(spec["schema_version"]) == 1L &&
            WorkerPrepareIdentity.integer(spec["context_size"]) == context.toLong() && context == 4096 &&
            WorkerPrepareIdentity.integer(spec["batch_size"]) == context.toLong() &&
            WorkerPrepareIdentity.integer(spec["sequence_count"]) == 1L, detail = "load spec dimensions")
        WorkerPrepareIdentity.integer(spec["kv_cache_bytes"])
        val sha = java.security.MessageDigest.getInstance("SHA-256")
            .digest(canonicalSupplyJson(spec).toByteArray(Charsets.UTF_8)).joinToString("") { "%02x".format(it.toInt() and 255) }
        checkPrepare(sha == identity.text("load_spec_digest"), "prepare_load_spec_digest_mismatch", "load spec digest differs")
    }
}
