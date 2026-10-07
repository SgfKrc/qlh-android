package com.qlh.inference.service

import com.qlh.inference.service.ModelManager.LayerArtifact
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * ★ 2026-10-07（DIST-NEXT-6）：层段工件 inventory 的结构化诊断。
 *
 * fail-closed 语义不变（不可用工件绝不返回），但每条 manifest 必须留下**可区分的**
 * 原因：「读不到」「manifest 非法」「工件不存在」「摘要不符」「不属于本次源模型」
 * 此前都被压成同一个「没有该区间」。广告面只带 error code 与身份/区间，不含本地路径。
 */
class LayerArtifactDiagnosticTest {
    private val sourceSha = "a".repeat(64)
    private val artifactSha = "b".repeat(64)

    private fun manifest(
        artifact: String = "qwen35-2b-mid4-16.gguf",
        mode: String = "middle",
        range: String = "[4, 16]",
        source: String = sourceSha,
    ): String = """
        {
          "artifact": "$artifact",
          "source_model_sha256": "$source",
          "artifact_sha256": "$artifactSha",
          "source_layer_range": $range,
          "mode": "$mode",
          "architecture": "qwen35"
        }
    """.trimIndent()

    @Test
    fun `usable manifest reports ok and keeps its identity`() {
        val classification = classifyLayerArtifact(
            manifestName = "mid.manifest.json",
            raw = manifest(),
            probeArtifact = {
                LayerArtifactProbe(sizeBytes = 1_234L, digestMatches = true)
            },
        )

        assertTrue(classification.usable)
        assertEquals(LayerArtifactDiagnostic.ERROR_OK, classification.diagnostic.errorCode)
        assertEquals(4, classification.diagnostic.startLayer)
        assertEquals(16, classification.diagnostic.endLayerExclusive)
        assertEquals("middle", classification.diagnostic.mode)
        assertEquals("qwen35", classification.diagnostic.architecture)
        assertTrue(classification.diagnostic.artifactPresent)
        assertEquals(1_234L, classification.diagnostic.artifactSizeBytes)
    }

    @Test
    fun `every failure mode keeps a distinct stable error code`() {
        // ① 读不到 manifest
        val unreadable = classifyLayerArtifact(
            manifestName = "gone.manifest.json",
            raw = null,
            probeArtifact = { throw AssertionError("unreadable must not probe") },
        )
        assertEquals(
            LayerArtifactDiagnostic.MANIFEST_UNREADABLE,
            unreadable.diagnostic.errorCode,
        )
        assertFalse(unreadable.usable)

        // ② manifest 非法（缺 mode）
        val invalid = classifyLayerArtifact(
            manifestName = "bad.manifest.json",
            raw = """{"artifact":"x.gguf","artifact_sha256":"$artifactSha",
                |"source_layer_range":[4,16]}""".trimMargin(),
            probeArtifact = { throw AssertionError("invalid must not probe") },
        )
        assertEquals(
            LayerArtifactDiagnostic.MANIFEST_INVALID,
            invalid.diagnostic.errorCode,
        )
        assertTrue(invalid.diagnostic.detail.isNotBlank())

        // ③ 不属于本次请求的源模型
        val foreign = classifyLayerArtifact(
            manifestName = "other.manifest.json",
            raw = manifest(),
            expectedModelSha256 = "c".repeat(64),
            probeArtifact = { throw AssertionError("foreign must not probe") },
        )
        assertEquals(
            LayerArtifactDiagnostic.SOURCE_DIGEST_MISMATCH,
            foreign.diagnostic.errorCode,
        )
        // 身份仍带出来：主节点据此知道「哪个区间因源模型不符被剔除」
        assertEquals(4, foreign.diagnostic.startLayer)

        // ④ 工件文件不存在
        val missing = classifyLayerArtifact(
            manifestName = "mid.manifest.json",
            raw = manifest(),
            probeArtifact = { null },
        )
        assertEquals(
            LayerArtifactDiagnostic.ARTIFACT_MISSING,
            missing.diagnostic.errorCode,
        )
        assertFalse(missing.diagnostic.artifactPresent)

        // ⑤ 摘要不符（存在但内容不是 manifest 声明的那份）
        val mismatch = classifyLayerArtifact(
            manifestName = "mid.manifest.json",
            raw = manifest(),
            probeArtifact = { LayerArtifactProbe(sizeBytes = 9L, digestMatches = false) },
        )
        assertEquals(
            LayerArtifactDiagnostic.ARTIFACT_DIGEST_MISMATCH,
            mismatch.diagnostic.errorCode,
        )
        assertTrue(mismatch.diagnostic.artifactPresent)

        // 六种结果互不相同
        assertEquals(
            5,
            setOf(
                unreadable.diagnostic.errorCode,
                invalid.diagnostic.errorCode,
                foreign.diagnostic.errorCode,
                missing.diagnostic.errorCode,
                mismatch.diagnostic.errorCode,
            ).size,
        )
    }

    @Test
    fun `advertisement carries the reason without any local path`() {
        val classification = classifyLayerArtifact(
            manifestName = "mid.manifest.json",
            raw = manifest(),
            probeArtifact = { null },
        )

        val advertised = classification.diagnostic.toAdvertisement()
        assertEquals(
            setOf(
                "error_code", "manifest", "architecture", "mode",
                "layer_range", "artifact_present",
            ),
            advertised.keys,
        )
        assertEquals("artifact_missing", advertised["error_code"])
        assertEquals(listOf(4, 16), advertised["layer_range"])
        assertEquals("middle", advertised["mode"])
        // 本地路径 / URI / 摘要绝不能出现在广告里
        assertTrue(
            advertised.values.filterIsInstance<String>()
                .none { it.contains("/") || it.contains("\\") || it.length == 64 },
        )
    }

    @Test
    fun `inventory separates usable artifacts from failure counts`() {
        val inventory = LayerArtifactInventory(
            artifacts = emptyList<LayerArtifact>(),
            diagnostics = listOf(
                LayerArtifactDiagnostic(
                    "a.manifest.json", LayerArtifactDiagnostic.ARTIFACT_MISSING,
                ),
                LayerArtifactDiagnostic(
                    "b.manifest.json", LayerArtifactDiagnostic.ARTIFACT_MISSING,
                ),
                LayerArtifactDiagnostic(
                    "c.manifest.json", LayerArtifactDiagnostic.MANIFEST_INVALID,
                ),
            ),
            root = "internal",
            manifestCount = 3,
            artifactFileCount = 0,
        )

        assertEquals(3, inventory.failures.size)
        assertEquals(2, inventory.failureCounts()["artifact_missing"])
        assertEquals(1, inventory.failureCounts()["manifest_invalid"])
    }
}
