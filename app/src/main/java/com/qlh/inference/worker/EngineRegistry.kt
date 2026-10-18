package com.qlh.inference.worker

/** Kotlin projection of qlh_core.contracts.engines, checked against the shared v1 fixture. */
object EngineRegistry {
    const val SCHEMA_VERSION = 1
    const val FULL_REQUEST = "full_request"
    const val LAYER_RANGE = "layer_range"
    const val FORWARD_LAYERS = "forward_layers"
    const val ANDROID_RUNTIME = "llama_cpp_android"

    data class RuntimeSpec(
        val family: String,
        val runtimeKind: String,
        val artifactKind: String?,
        val layerRangeMode: String,
        val operations: Set<String>,
        val capabilities: Set<String>,
        val artifactFormats: List<String>,
        val architectures: Map<String, List<String>>,
        val precisions: Map<String, List<String>>,
        val inputKinds: List<String>,
        val outputKinds: List<String>,
        val nativeAbi: String,
        val optimizationConstraints: List<String>,
    ) {
        fun toMap(): Map<String, Any?> = linkedMapOf(
            "family" to family, "runtime_kind" to runtimeKind,
            "operations" to operations.sorted(), "native_abi" to nativeAbi,
        )
    }

    val entries: List<RuntimeSpec> = listOf(
        RuntimeSpec(
            "llama_cpp",
            "llama_cpp_binding",
            "gguf",
            "precut_artifact",
            setOf("forward_layers", "full_request", "layer_range"),
            setOf("chat", "chat_image", "chat_stream", "forward_layers", "load_model", "unload_model"),
            listOf("gguf"),
            mapOf("forward_layers" to listOf("qwen", "qwen2", "qwen3_5", "qwen3_5_text"), "full_request" to listOf("qwen", "qwen2", "qwen3_5", "qwen3_5_text", "llama", "mistral", "gemma", "gemma2", "phi3"), "layer_range" to listOf("qwen", "qwen2", "qwen3_5", "qwen3_5_text")),
            mapOf("forward_layers" to listOf("gguf"), "full_request" to listOf("gguf"), "layer_range" to listOf("gguf")),
            listOf("messages", "token_ids", "hidden_f32"),
            listOf("text", "hidden_f32", "logits_f32"),
            "llama_cpp_python.v1",
            listOf("distributed_no_full_model", "verified_precut_manifest")
        ),
        RuntimeSpec(
            "llama_cpp",
            "llama_cpp_keep_head",
            "gguf",
            "precut_artifact",
            setOf("forward_layers", "layer_range"),
            setOf("forward_layers"),
            listOf("gguf"),
            mapOf("forward_layers" to listOf("qwen", "qwen2", "qwen3_5", "qwen3_5_text"), "layer_range" to listOf("qwen", "qwen2", "qwen3_5", "qwen3_5_text")),
            mapOf("forward_layers" to listOf("gguf"), "layer_range" to listOf("gguf")),
            listOf("token_ids", "hidden_f32"),
            listOf("hidden_f32", "logits_f32"),
            "qlh_keep_head.v1",
            listOf("distributed_no_full_model", "verified_precut_manifest", "keep_head_hidden_extraction")
        ),
        RuntimeSpec(
            "llama_cpp",
            "llama_server",
            "gguf",
            "none",
            setOf("full_request"),
            setOf("chat", "chat_stream", "load_model", "unload_model"),
            listOf("gguf"),
            mapOf("full_request" to listOf("qwen", "qwen2", "qwen3_5", "qwen3_5_text", "llama", "mistral", "gemma", "gemma2", "phi3")),
            mapOf("full_request" to listOf("gguf")),
            listOf("messages"),
            listOf("text"),
            "llama_server_http.v1",
            listOf("distributed_no_full_model", "full_request_only")
        ),
        RuntimeSpec(
            "llama_cpp",
            "llama_cpp_android",
            "gguf",
            "precut_artifact",
            setOf("forward_layers", "full_request", "layer_range"),
            setOf("chat", "chat_stream", "forward_layers", "load_model", "unload_model"),
            listOf("gguf"),
            mapOf("forward_layers" to listOf("qwen", "qwen2", "qwen3_5", "qwen3_5_text"), "full_request" to listOf("qwen", "qwen2", "qwen3_5", "qwen3_5_text", "llama", "mistral", "gemma", "gemma2", "phi3"), "layer_range" to listOf("qwen", "qwen2", "qwen3_5", "qwen3_5_text")),
            mapOf("forward_layers" to listOf("gguf"), "full_request" to listOf("gguf"), "layer_range" to listOf("gguf")),
            listOf("messages", "token_ids", "hidden_f32"),
            listOf("text", "hidden_f32", "logits_f32"),
            "qlh_llama_jni.v1",
            listOf("distributed_no_full_model", "verified_precut_manifest")
        ),
        RuntimeSpec(
            "pytorch",
            "pytorch_cpu",
            "safetensors",
            "dynamic",
            setOf("forward_layers", "full_request", "layer_range"),
            setOf("chat", "chat_image", "chat_stream", "ensure_full_model", "forward_layers", "load_layer_range", "load_model", "unload_model"),
            listOf("safetensors"),
            mapOf("forward_layers" to listOf("qwen", "qwen2", "qwen3_5", "qwen3_5_text"), "full_request" to listOf("qwen", "qwen2", "qwen3_5", "qwen3_5_text", "llama", "mistral", "gemma", "gemma2", "phi3"), "layer_range" to listOf("qwen", "qwen2", "qwen3_5", "qwen3_5_text")),
            mapOf("forward_layers" to listOf("fp32"), "full_request" to listOf("fp32", "fp16", "int8", "int4"), "layer_range" to listOf("fp32")),
            listOf("messages", "token_ids", "hidden_f32"),
            listOf("text", "hidden_f32", "logits_f32"),
            "torch_module.v1",
            listOf("distributed_no_full_model", "cpu_fp32_materialization")
        ),
        RuntimeSpec(
            "pytorch",
            "pytorch_cuda",
            "safetensors",
            "dynamic",
            setOf("forward_layers", "full_request", "layer_range"),
            setOf("chat", "chat_image", "chat_stream", "ensure_full_model", "forward_layers", "load_layer_range", "load_model", "unload_model"),
            listOf("safetensors"),
            mapOf("forward_layers" to listOf("qwen", "qwen2", "qwen3_5", "qwen3_5_text"), "full_request" to listOf("qwen", "qwen2", "qwen3_5", "qwen3_5_text", "llama", "mistral", "gemma", "gemma2", "phi3"), "layer_range" to listOf("qwen", "qwen2", "qwen3_5", "qwen3_5_text")),
            mapOf("forward_layers" to listOf("fp32", "fp16"), "full_request" to listOf("fp32", "fp16", "int8", "int4"), "layer_range" to listOf("fp32", "fp16")),
            listOf("messages", "token_ids", "hidden_f32"),
            listOf("text", "hidden_f32", "logits_f32"),
            "torch_module.v1",
            listOf("distributed_no_full_model", "quantized_layers_materialize_fp16")
        ),
        RuntimeSpec(
            "island",
            "island_remote",
            null,
            "none",
            setOf("full_request"),
            setOf("chat", "chat_stream", "load_model", "unload_model"),
            listOf(),
            mapOf("full_request" to listOf("qwen", "qwen2", "qwen3_5", "qwen3_5_text", "llama", "mistral", "gemma", "gemma2", "phi3")),
            mapOf("full_request" to listOf("remote")),
            listOf("messages"),
            listOf("text"),
            "openai_chat.v1",
            listOf("distributed_no_full_model")
        ),
        RuntimeSpec(
            "external_api",
            "external_api_remote",
            null,
            "none",
            setOf("full_request"),
            setOf("chat", "chat_stream", "load_model", "unload_model"),
            listOf(),
            mapOf("full_request" to listOf("qwen", "qwen2", "qwen3_5", "qwen3_5_text", "llama", "mistral", "gemma", "gemma2", "phi3")),
            mapOf("full_request" to listOf("remote")),
            listOf("messages"),
            listOf("text"),
            "openai_chat.v1",
            listOf("distributed_no_full_model")
        ),
    )

    fun requireRuntime(family: String, kind: String): RuntimeSpec =
        requireNotNull(entries.find { it.family == family && it.runtimeKind == kind }) {
            "unknown engine runtime: $family/$kind"
        }

    fun snapshot(): Map<String, Any?> = mapOf(
        "schema_version" to SCHEMA_VERSION,
        "runtimes" to entries.map { spec -> spec.toMap() + mapOf(
            "artifact_kind" to spec.artifactKind, "layer_range_mode" to spec.layerRangeMode,
            "capabilities" to spec.capabilities.sorted(),
            "artifact_formats" to spec.artifactFormats, "architectures" to spec.architectures,
            "precisions" to spec.precisions, "input_kinds" to spec.inputKinds,
            "output_kinds" to spec.outputKinds, "optimization_constraints" to spec.optimizationConstraints,
        ) },
    )

    fun advertisement(
        family: String, kind: String, probedOperations: Set<String>? = null,
    ): Map<String, Any?> {
        val spec = requireRuntime(family, kind)
        val effective = spec.operations.intersect(probedOperations ?: spec.operations)
        return mapOf("schema_version" to SCHEMA_VERSION,
            "runtimes" to listOf(spec.copy(operations = effective).toMap()))
    }

    fun validateAdvertisement(value: Any?): List<RuntimeSpec> {
        val schema = value as? Map<*, *> ?: error("engine_capabilities must be an object")
        require(schema.keys == setOf("schema_version", "runtimes")) { "invalid engine schema fields" }
        val version = schema["schema_version"]
        require((version is Int || version is Long) && (version as Number).toLong() == 1L) {
            "unsupported engine capability schema version"
        }
        val runtimes = schema["runtimes"] as? List<*> ?: error("engine runtimes must be a list")
        require(runtimes.size <= entries.size) { "too many engine runtimes" }
        val seen = mutableSetOf<Pair<String, String>>()
        return runtimes.map { raw ->
            val item = raw as? Map<*, *> ?: error("engine runtime must be an object")
            require(item.keys == setOf("family", "runtime_kind", "operations", "native_abi")) { "invalid runtime fields" }
            val family = item["family"] as? String ?: error("invalid engine family")
            val kind = item["runtime_kind"] as? String ?: error("invalid runtime kind")
            val spec = requireRuntime(family, kind)
            require(item["native_abi"] == spec.nativeAbi) { "unknown or mismatched engine native ABI" }
            val ops = item["operations"] as? List<*> ?: error("invalid runtime operations")
            require(ops.all { it is String && it in spec.operations } && ops.distinct().size == ops.size) {
                "engine runtime operations exceed the static contract"
            }
            require(seen.add(family to kind)) { "duplicate engine runtime" }
            spec.copy(operations = ops.filterIsInstance<String>().toSet())
        }
    }

    fun validateWorkerCapabilities(capabilities: Map<String, Any?>) {
        if (!capabilities.containsKey("engine_capabilities")) return
        val runtimes = validateAdvertisement(capabilities["engine_capabilities"])
        val engines = capabilities["engines"] as? List<*> ?: emptyList<Any>()
        require(runtimes.all { it.family in engines }) { "undeclared engine capability family" }
        val stages = capabilities["stage_types"] as? List<*> ?: emptyList<Any>()
        require("full_inference" !in stages || runtimes.any { FULL_REQUEST in it.operations }) {
            "full_inference requires a full_request runtime"
        }
        require("layer_forward" !in stages || runtimes.any { FORWARD_LAYERS in it.operations }) {
            "layer_forward requires a forward_layers runtime"
        }
        val hasLayers = listOf("layer_ranges", "layer_artifacts").any {
            (capabilities[it] as? List<*>)?.isNotEmpty() == true
        } || capabilities["layer_budget"] != null
        require(!hasLayers || runtimes.any { LAYER_RANGE in it.operations && FORWARD_LAYERS in it.operations }) {
            "layer ranges/artifacts/budget require a layer runtime"
        }
        val models = (capabilities["models"] as? List<*>)?.filterIsInstance<Map<*, *>>().orEmpty()
        val artifacts = (capabilities["layer_artifacts"] as? List<*>)?.filterIsInstance<Map<*, *>>().orEmpty()
        artifacts.forEach { artifact ->
            val family = models.find { it["model_id"] == artifact["model_id"] }?.get("engine")
            require(runtimes.any { it.family == family && LAYER_RANGE in it.operations && FORWARD_LAYERS in it.operations }) {
                "layer artifact engine has no layer runtime"
            }
        }
    }
}
