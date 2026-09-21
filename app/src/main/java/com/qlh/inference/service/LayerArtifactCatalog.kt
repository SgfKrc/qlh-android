package com.qlh.inference.service

import com.google.gson.JsonArray
import com.google.gson.JsonObject
import com.google.gson.JsonParser
import java.io.File

/** Metadata required to bind one GGUF artifact to a source-model layer range. */
data class LayerArtifactDescriptor(
    val artifactName: String,
    val startLayer: Int,
    val endLayerExclusive: Int,
    val artifactSha256: String,
    val sourceModelSha256: String,
    val architecture: String,
)

/** Parser for the main repository's layer artifact manifests. */
object LayerArtifactManifestParser {
    private val sha256Pattern = Regex("[0-9a-fA-F]{64}")

    fun parse(raw: String, manifestName: String): Result<LayerArtifactDescriptor> = runCatching {
        val root = JsonParser.parseString(raw).asJsonObject
        val artifact = string(root, "artifact")
            .ifBlank { File(manifestName).nameWithoutExtension + ".gguf" }
        val range = explicitRange(root) ?: derivedTailRange(root)
            ?: error("manifest has no layer range")
        val artifactSha = string(root, "artifact_sha256").lowercase()
        require(sha256Pattern.matches(artifactSha)) {
            "manifest artifact_sha256 must be a 64-char hex digest"
        }
        val sourceSha = listOf(
            string(root, "source_model_sha256"),
            string(root, "model_sha256"),
            string(root, "source_sha256"),
        ).firstOrNull { it.isNotBlank() }?.lowercase().orEmpty()
        if (sourceSha.isNotBlank()) {
            require(sha256Pattern.matches(sourceSha)) {
                "manifest source model digest is invalid"
            }
        }
        require(range.first >= 0 && range.second > range.first) {
            "manifest layer range must be non-empty"
        }
        LayerArtifactDescriptor(
            artifactName = File(artifact).name,
            startLayer = range.first,
            endLayerExclusive = range.second,
            artifactSha256 = artifactSha,
            sourceModelSha256 = sourceSha,
            architecture = string(root, "architecture"),
        )
    }

    private fun explicitRange(root: JsonObject): Pair<Int, Int>? {
        val value = root.get("layer_range") ?: root.get("source_layer_range") ?: return null
        if (value !is JsonArray || value.size() != 2) return null
        val start = value[0].asInt
        val end = value[1].asInt
        return start to end
    }

    private fun derivedTailRange(root: JsonObject): Pair<Int, Int>? {
        val start = root.get("first_local_layer_maps_to")?.asInt ?: return null
        val keptBlocks = root.get("kept_block_count")?.asInt ?: return null
        val nextn = root.get("nextn_predict_layers")?.asInt ?: 0
        return start to (start + keptBlocks - nextn)
    }

    private fun string(root: JsonObject, name: String): String =
        root.get(name)?.takeUnless { it.isJsonNull }?.asString?.trim().orEmpty()
}
