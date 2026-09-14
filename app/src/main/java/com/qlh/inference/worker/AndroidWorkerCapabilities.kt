package com.qlh.inference.worker

/** Builds the conservative capability snapshot sent by an Android Full Worker. */
object AndroidWorkerCapabilities {
    fun modelIdentity(
        modelId: String,
        modelFormat: String,
        modelRevision: String,
        modelSha256: String,
        resourceAdmitted: Boolean,
    ): Map<String, Any?>? = if (
        resourceAdmitted && modelId.isNotBlank() && modelSha256.matches(Regex("[a-fA-F0-9]{64}"))
    ) {
        mapOf(
            "model_id" to modelId,
            "engine" to "llama_cpp",
            "format" to modelFormat,
            "revision" to modelRevision,
            "sha256" to modelSha256.lowercase(),
        )
    } else {
        null
    }

    fun build(
        modelId: String = "",
        modelFormat: String = "gguf",
        modelRevision: String = "local",
        modelSha256: String = "",
        resourceAdmitted: Boolean = false,
        resourceReason: String = "resource_gate_not_confirmed",
    ): Map<String, Any?> {
        val normalizedReason = if (resourceAdmitted) "" else resourceReason.ifBlank {
            "resource_gate_not_confirmed"
        }
        val model = modelIdentity(
            modelId, modelFormat, modelRevision, modelSha256, resourceAdmitted,
        )
        return mapOf(
            "stage_types" to listOf("full_inference"),
            "engines" to listOf("llama_cpp"),
            "models" to (model?.let { listOf(it) } ?: emptyList<Map<String, Any?>>()),
            "max_concurrency" to 1,
            "resource_gate" to mapOf(
                "admitted" to resourceAdmitted,
                "reason_code" to normalizedReason,
            ),
        )
    }
}
