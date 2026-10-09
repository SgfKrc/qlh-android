package com.qlh.inference.service

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class LayerArtifactCatalogTest {
    @Test
    fun `windows artifact path is normalized for android storage lookup`() {
        val result = LayerArtifactManifestParser.parse(
            raw = """
                {
                  "artifact": "build\\cross-framework-layer-poc\\out\\qwen35-2b-q4km-mid4-16.gguf",
                  "source_model_sha256": "${"a".repeat(64)}",
                  "source_model_id": "qwen3-5-2b",
                  "hidden_size": 2048,
                  "tokenizer_sha256": "${"c".repeat(64)}",
                  "artifact_sha256": "${"b".repeat(64)}",
                  "source_layer_range": [4, 16],
                  "mode": "middle",
                  "architecture": "qwen35"
                }
            """.trimIndent(),
            manifestName = "qwen35-2b-q4km-mid4-16.manifest.json",
        ).getOrThrow()

        assertEquals("qwen35-2b-q4km-mid4-16.gguf", result.artifactName)
        assertEquals(4, result.startLayer)
        assertEquals(16, result.endLayerExclusive)
        assertEquals("middle", result.mode)
        assertEquals("qwen3-5-2b", result.sourceModelId)
        assertEquals(2048, result.hiddenSize)
        assertEquals("c".repeat(64), result.tokenizerSha256)
    }

    @Test
    fun `legacy manifest remains usable without model preflight contract`() {
        val legacy = LayerArtifactManifestParser.parse(
            raw = """
                {
                  "artifact": "legacy.gguf",
                  "source_model_sha256": "${"a".repeat(64)}",
                  "artifact_sha256": "${"b".repeat(64)}",
                  "source_layer_range": [4, 16],
                  "mode": "middle",
                  "architecture": "qwen35"
                }
            """.trimIndent(),
            manifestName = "legacy.manifest.json",
        ).getOrThrow()
        val hiddenOnly = LayerArtifactManifestParser.parse(
            raw = """
                {
                  "artifact": "new-legacy.gguf",
                  "source_model_sha256": "${"a".repeat(64)}",
                  "artifact_sha256": "${"b".repeat(64)}",
                  "source_layer_range": [4, 16],
                  "hidden_size": 2048,
                  "mode": "middle",
                  "architecture": "qwen35"
                }
            """.trimIndent(),
            manifestName = "new-legacy.manifest.json",
        ).getOrThrow()

        assertEquals(null, legacy.sourceModelId)
        assertEquals(null, legacy.hiddenSize)
        assertEquals(null, legacy.tokenizerSha256)
        assertEquals(null, hiddenOnly.sourceModelId)
        assertEquals(2048, hiddenOnly.hiddenSize)
        assertEquals(null, hiddenOnly.tokenizerSha256)
    }

    @Test
    fun `incomplete model preflight manifest contract is rejected`() {
        val result = LayerArtifactManifestParser.parse(
            raw = """
                {
                  "artifact": "segment.gguf",
                  "source_model_sha256": "${"a".repeat(64)}",
                  "source_model_id": "qwen3-5-2b",
                  "hidden_size": 2048,
                  "artifact_sha256": "${"b".repeat(64)}",
                  "source_layer_range": [4, 16],
                  "mode": "middle",
                  "architecture": "qwen35"
                }
            """.trimIndent(),
            manifestName = "segment.manifest.json",
        )

        assertTrue(result.isFailure)
    }

    @Test
    fun `invalid or missing manifest mode is rejected fail closed`() {
        fun manifest(modeLine: String): String = """
            {
              "artifact": "segment.gguf",
              "source_model_sha256": "${"a".repeat(64)}",
              "artifact_sha256": "${"b".repeat(64)}",
              "source_layer_range": [4, 16],
              $modeLine
              "architecture": "qwen35"
            }
        """.trimIndent()

        assertTrue(
            LayerArtifactManifestParser.parse(
                manifest("\"mode\": \"whole\","),
                "segment.manifest.json",
            ).isFailure,
        )
        assertTrue(
            LayerArtifactManifestParser.parse(
                manifest(""),
                "segment.manifest.json",
            ).isFailure,
        )
    }
}
