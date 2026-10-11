package com.qlh.inference.service

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class LayerArtifactCatalogTest {
    private val sourceDigest = "a".repeat(64)
    private val artifactDigest = "b".repeat(64)

    private fun candidate(
        root: LayerArtifactCandidate.Root,
        source: String = sourceDigest,
        digest: String = artifactDigest,
        sourceId: String? = "qwen3-5-2b",
        start: Int = 4,
        end: Int = 16,
        mode: String = "middle",
    ): LayerArtifactCandidate = LayerArtifactCandidate(
        descriptor = LayerArtifactDescriptor(
            artifactName = "segment.gguf",
            startLayer = start,
            endLayerExclusive = end,
            artifactSha256 = digest,
            sourceModelSha256 = source,
            sourceModelId = sourceId,
            hiddenSize = if (sourceId == null) null else 2048,
            tokenizerSha256 = if (sourceId == null) null else "e".repeat(64),
            architecture = "qwen35",
            mode = mode,
        ),
        reference = if (root == LayerArtifactCandidate.Root.SAF) {
            "content://models/segment.gguf"
        } else {
            "$FALLBACK_LAYER_ARTIFACT_DIRECTORY/segment.gguf"
        },
        manifestName = "segment.manifest.json",
        root = root,
    )

    @Test
    fun `merging deduplicates identical artifacts with SAF priority but retains alternatives`() {
        val saf = candidate(LayerArtifactCandidate.Root.SAF)
        val fallback = candidate(LayerArtifactCandidate.Root.FALLBACK)
        val otherDigest = fallback.copy(descriptor = fallback.descriptor.copy(
            artifactSha256 = "c".repeat(64),
        ))
        val otherSource = fallback.copy(descriptor = fallback.descriptor.copy(
            sourceModelSha256 = "d".repeat(64),
        ))

        val merged = mergeLayerArtifactCandidates(
            listOf(saf, saf), listOf(fallback, otherDigest, otherSource),
        )

        assertEquals(3, merged.size)
        assertEquals(saf, merged.first())
        assertEquals(
            setOf(sourceDigest, "d".repeat(64)),
            merged.map { it.descriptor.sourceModelSha256 }.toSet(),
        )
    }

    @Test
    fun `selection uses exact master source digest id and half open range then prefers SAF`() {
        val fallback = candidate(LayerArtifactCandidate.Root.FALLBACK)
        val saf = candidate(LayerArtifactCandidate.Root.SAF)
        val chosen = chooseLayerArtifact(
            listOf(fallback, saf), "qwen3-5-2b", sourceDigest, 4, 16,
            offeredArtifactSha256 = artifactDigest.uppercase(),
        ) as LayerArtifactChoice.Selected

        assertEquals(saf, chosen.artifact)
        assertEquals(LayerArtifactCandidate.Root.SAF, chosen.artifact.root)
    }

    @Test
    fun `legacy artifact without source model id matches only by exact source digest`() {
        val legacy = candidate(LayerArtifactCandidate.Root.FALLBACK, sourceId = null)
        val chosen = chooseLayerArtifact(listOf(legacy), "master-alias", sourceDigest, 4, 16)

        assertEquals(legacy, (chosen as LayerArtifactChoice.Selected).artifact)
        assertTrue(chooseLayerArtifact(
            listOf(legacy), "master-alias", "d".repeat(64), 4, 16,
        ) is LayerArtifactChoice.ManualRequired)
    }

    @Test
    fun `incorrect model digest range or offered artifact digest never selects a candidate`() {
        val artifact = candidate(LayerArtifactCandidate.Root.SAF)
        val requests = listOf(
            chooseLayerArtifact(listOf(artifact), "other-model", sourceDigest, 4, 16),
            chooseLayerArtifact(listOf(artifact), "qwen3-5-2b", "c".repeat(64), 4, 16),
            chooseLayerArtifact(listOf(artifact), "qwen3-5-2b", sourceDigest, 5, 16),
            chooseLayerArtifact(listOf(artifact), "qwen3-5-2b", sourceDigest, 4, 15),
            chooseLayerArtifact(
                listOf(artifact), "qwen3-5-2b", sourceDigest, 4, 16,
                offeredArtifactSha256 = "d".repeat(64),
            ),
        )

        assertTrue(requests.all {
            it is LayerArtifactChoice.ManualRequired && it.reason == "layer_artifact_not_found"
        })
    }

    @Test
    fun `different artifact digests for one range require the master artifact digest`() {
        val saf = candidate(LayerArtifactCandidate.Root.SAF)
        val fallback = candidate(
            LayerArtifactCandidate.Root.FALLBACK, digest = "c".repeat(64),
        )
        val merged = mergeLayerArtifactCandidates(listOf(saf), listOf(fallback))
        val ambiguous = chooseLayerArtifact(merged, "qwen3-5-2b", sourceDigest, 4, 16)
        val exact = chooseLayerArtifact(
            merged, "qwen3-5-2b", sourceDigest, 4, 16,
            offeredArtifactSha256 = "c".repeat(64),
        )

        assertEquals(
            "layer_artifact_ambiguous",
            (ambiguous as LayerArtifactChoice.ManualRequired).reason,
        )
        assertEquals(fallback, (exact as LayerArtifactChoice.Selected).artifact)
    }

    @Test
    fun `only uniquely determined ranges can be advertised before a master offer`() {
        val first = candidate(LayerArtifactCandidate.Root.SAF)
        val conflict = candidate(LayerArtifactCandidate.Root.FALLBACK, digest = "c".repeat(64))
        val independent = candidate(LayerArtifactCandidate.Root.FALLBACK, start = 16, end = 24)
        assertEquals(
            listOf(independent),
            unambiguousLayerArtifactCandidates(listOf(first, conflict, independent)),
        )
    }

    @Test
    fun `different segment modes are not silently deduplicated even with equal digest`() {
        val saf = candidate(LayerArtifactCandidate.Root.SAF)
        val tail = candidate(LayerArtifactCandidate.Root.FALLBACK, mode = "tail")
        val choice = chooseLayerArtifact(
            mergeLayerArtifactCandidates(listOf(saf), listOf(tail)),
            "qwen3-5-2b", sourceDigest, 4, 16,
            offeredArtifactSha256 = artifactDigest,
        )

        assertEquals(
            "layer_artifact_ambiguous",
            (choice as LayerArtifactChoice.ManualRequired).reason,
        )
    }

    @Test
    fun `different source ids remain selectable after deduplication`() {
        val saf = candidate(LayerArtifactCandidate.Root.SAF, sourceId = "other-model")
        val fallback = candidate(LayerArtifactCandidate.Root.FALLBACK)
        val merged = mergeLayerArtifactCandidates(listOf(saf), listOf(fallback))
        val choice = chooseLayerArtifact(merged, "qwen3-5-2b", sourceDigest, 4, 16)

        assertEquals(2, merged.size)
        assertEquals(fallback, (choice as LayerArtifactChoice.Selected).artifact)
    }

    @Test
    fun `offer model id resolves identical bytes under different artifact names`() {
        val saf = candidate(LayerArtifactCandidate.Root.SAF)
        val fallback = candidate(LayerArtifactCandidate.Root.FALLBACK).copy(
            descriptor = saf.descriptor.copy(artifactName = "other model.gguf"),
        )
        val merged = mergeLayerArtifactCandidates(listOf(saf), listOf(fallback))
        val choice = chooseLayerArtifact(
            merged, "qwen3-5-2b", sourceDigest, 4, 16,
            offeredArtifactSha256 = artifactDigest,
            offeredArtifactModelId = "other_model.gguf",
        )

        assertEquals(2, merged.size)
        assertEquals(fallback, (choice as LayerArtifactChoice.Selected).artifact)
        val missing = chooseLayerArtifact(
            merged, "qwen3-5-2b", sourceDigest, 4, 16,
            offeredArtifactSha256 = artifactDigest,
            offeredArtifactModelId = "unknown.gguf",
        ) as LayerArtifactChoice.ManualRequired
        assertEquals("layer_artifact_not_found", missing.reason)
    }

    @Test
    fun `conflicting preflight metadata is not erased by deduplication`() {
        val saf = candidate(LayerArtifactCandidate.Root.SAF)
        val fallback = candidate(LayerArtifactCandidate.Root.FALLBACK).copy(
            descriptor = saf.descriptor.copy(tokenizerSha256 = "d".repeat(64)),
        )

        assertEquals(2, mergeLayerArtifactCandidates(listOf(saf), listOf(fallback)).size)
    }

    @Test
    fun `unreadable storage needs named diagnosis and SAF subdirectory guidance`() {
        val choice = chooseLayerArtifact(
            emptyList(), "qwen3-5-2b", sourceDigest, 4, 16,
            storageUnavailable = true,
        ) as LayerArtifactChoice.ManualRequired

        assertEquals("artifact_storage_unavailable", choice.reason)
        assertTrue(choice.prompt.contains("SAF"))
        assertTrue(choice.prompt.contains("Download/QLH/models"))
        assertFalse(choice.prompt.contains("MANAGE_EXTERNAL_STORAGE"))
    }

    @Test
    fun `fallback unavailable diagnostic projection omits local paths`() {
        val diagnostic = LayerArtifactDiagnostic(
            manifestName = "default-storage",
            errorCode = LayerArtifactDiagnostic.FALLBACK_UNAVAILABLE,
            detail = "请在设置中授权 Download/QLH/models 子目录",
        )

        assertFalse(diagnostic.usable)
        val projected = diagnostic.toAdvertisement()
        assertEquals("fallback_storage_unavailable", projected["error_code"])
        assertEquals("default-storage", projected["manifest"])
        assertEquals(null, projected["layer_range"])
        assertFalse(projected.values.any { it.toString().contains("/sdcard/") })
        assertFalse(projected.containsKey("detail"))
    }

    @Test
    fun `invalid master digest cannot use an artifact digest as a substitute`() {
        val choice = chooseLayerArtifact(
            listOf(candidate(LayerArtifactCandidate.Root.SAF)),
            "qwen3-5-2b", "", 4, 16,
            offeredArtifactSha256 = artifactDigest,
        )

        assertEquals("model_identity_missing", (choice as LayerArtifactChoice.ManualRequired).reason)
    }

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
        val inferred = LayerArtifactManifestParser.parse(
            raw = """{"artifact_sha256":"${"b".repeat(64)}","source_model_sha256":"${"a".repeat(64)}","source_layer_range":[4,16],"mode":"middle"}""",
            manifestName = "segment.manifest.json",
        ).getOrThrow()
        assertEquals("segment.gguf", inferred.artifactName)
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
