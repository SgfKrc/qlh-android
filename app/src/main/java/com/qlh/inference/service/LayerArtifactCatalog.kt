package com.qlh.inference.service

import com.google.gson.JsonArray
import com.google.gson.JsonObject
import com.google.gson.JsonParser
import com.qlh.inference.service.ModelManager.LayerArtifact
import java.io.File

/** Metadata required to bind one GGUF artifact to a source-model layer range. */
data class LayerArtifactDescriptor(
    val artifactName: String,
    val startLayer: Int,
    val endLayerExclusive: Int,
    val artifactSha256: String,
    val sourceModelSha256: String,
    val architecture: String,
    /**
     * ★ 2026-10-05（DIST-3 实测缺口）：工件在源模型里的**段类型**，取值同主仓
     * `scripts/cut_layers.py` 的 `mode`：`"head"` / `"middle"` / `"tail"`（另有
     * 整模不算段）。
     *
     * 为什么必须广告出去：`layer_ranges` 只说「本节点覆盖哪些层」，**不区分它是
     * 首段 / 中间段 / 末段** ⇒ master 只比对区间覆盖就会放行**段类型不匹配**的分配。
     * 2026-10-05 三机实测正是如此：Y700 加载 `mid8-24`
     * （`mode=middle`，`[8,24)`）却按区间被分到末段位次 `[20,24)`
     * ⇒ `remote worker reported a Stage error`。
     *
     * ⚠️ **2026-10-09 更正**：本注释原写「中间段工件没有 `lm_head`/`final_norm`」——
     * **与实际产物矛盾**。实测 `build/keephead/q35-2b-cut-16-20.manifest.json`
     * （`mode=middle`）的 `tensors_kept=55`，正是 3 个 linear 层×14 + 1 个 full 层×11
     * + **2 个非 blk 张量**（`token_embd` 与 `output_norm`；该模型 tie embeddings，
     * 故无独立 `output.weight`）⇒ **中间段工件是带 `output_norm` 的**。
     * 真正导致失败的机制是**段角色与位次不符**（middle 工件的张量集合按"中间位次"切出，
     * 被派到末段位次后职责对不上），**不是"缺少某些张量"**。
     * 三机实测结论不变，只是归因写错了。
     *
     * Missing or unknown modes are rejected so an ambiguous artifact is never advertised.
     */
    val mode: String,
)

/** Parser for the main repository's layer artifact manifests. */
object LayerArtifactManifestParser {    private val sha256Pattern = Regex("[0-9a-fA-F]{64}")
    private val segmentModes = setOf("head", "middle", "tail")

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
        val mode = string(root, "mode").lowercase()
        require(mode in segmentModes) {
            "manifest mode must be one of ${segmentModes.sorted().joinToString(", ")}"
        }
        LayerArtifactDescriptor(
            // Manifests are often generated on Windows and contain backslash
            // paths. Android's File.name does not treat '\\' as a separator.
            artifactName = artifact.substringAfterLast('/').substringAfterLast('\\'),
            startLayer = range.first,
            endLayerExclusive = range.second,
            artifactSha256 = artifactSha,
            sourceModelSha256 = sourceSha,
            architecture = string(root, "architecture"),
            mode = mode,
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

/**
 * ★ 2026-10-07（DIST-NEXT-6）：单条层段 manifest 的**结构化判定**。
 *
 * `listLayerArtifacts` 保留 fail-closed（不可用工件绝不返回），但不再把它静默丢掉：
 * 每条 manifest 都留下稳定 `errorCode` 与已解析出的身份字段，使「读不到 manifest」
 * 「manifest 非法」「工件文件不存在」「摘要不符」「不属于本次请求的源模型」可区分 ——
 * 此前这些都被压成同一个「没有该区间」。
 *
 * **路径纪律**：本地路径/URI 只留在日志与本地诊断 API；[toAdvertisement] 是能力
 * 广告用的那一行，键集固定且不含路径（主仓 `_validate_capabilities` 精确校验它）。
 */
data class LayerArtifactDiagnostic(
    val manifestName: String,
    val errorCode: String,
    val detail: String = "",
    val artifactName: String = "",
    val startLayer: Int? = null,
    val endLayerExclusive: Int? = null,
    val architecture: String = "",
    val mode: String = "",
    val sourceModelSha256: String = "",
    val artifactSha256: String = "",
    val artifactPresent: Boolean = false,
    val artifactSizeBytes: Long = 0L,
) {
    val usable: Boolean get() = errorCode == ERROR_OK

    /** 能力广告用的一行（无本地路径）。 */
    fun toAdvertisement(): Map<String, Any?> = mapOf(
        "error_code" to errorCode,
        "manifest" to manifestName,
        "architecture" to architecture,
        "mode" to mode,
        "layer_range" to if (startLayer != null && endLayerExclusive != null) {
            listOf(startLayer, endLayerExclusive)
        } else {
            null
        },
        "artifact_present" to artifactPresent,
    )

    companion object {
        const val ERROR_OK = ""
        const val MANIFEST_UNREADABLE = "manifest_unreadable"
        const val MANIFEST_INVALID = "manifest_invalid"
        const val SOURCE_DIGEST_MISMATCH = "source_digest_mismatch"
        const val ARTIFACT_MISSING = "artifact_missing"
        const val ARTIFACT_DIGEST_MISMATCH = "artifact_digest_mismatch"
    }
}

/** ★ 2026-10-07（DIST-NEXT-6）：一次层段扫描的结果 —— 可用工件 + 全部判定。 */
data class LayerArtifactInventory(
    val artifacts: List<LayerArtifact>,
    val diagnostics: List<LayerArtifactDiagnostic>,
    /** 扫描根：`saf`（SAF tree）或 `internal`（`filesDir/models`）。 */
    val root: String,
    val manifestCount: Int,
    val artifactFileCount: Int,
) {
    val failures: List<LayerArtifactDiagnostic> get() = diagnostics.filterNot { it.usable }

    fun failureCounts(): Map<String, Int> =
        failures.groupingBy { it.errorCode }.eachCount()
}

/** ★ 2026-10-07（DIST-NEXT-6）：IO 层对工件文件的探测结果（判定逻辑的输入）。 */
data class LayerArtifactProbe(
    val sizeBytes: Long,
    val digestMatches: Boolean,
)

/** ★ 2026-10-07（DIST-NEXT-6）：一条 manifest 的判定（诊断 + 可用时的描述符）。 */
data class LayerArtifactClassification(
    val diagnostic: LayerArtifactDiagnostic,
    val descriptor: LayerArtifactDescriptor? = null,
) {
    val usable: Boolean get() = diagnostic.usable
}

/**
 * ★ 2026-10-07（DIST-NEXT-6）：**纯函数**判定一条 manifest 是否可用（fail-closed）。
 *
 * 判定顺序即原因优先级，六种结果互斥且各有稳定 error code：
 * 读不到 → `manifest_unreadable`；解析/字段非法 → `manifest_invalid`；
 * 不属于本次请求的源模型 → `source_digest_mismatch`；
 * 工件文件不在根目录 → `artifact_missing`；摘要不符 → `artifact_digest_mismatch`；
 * 全部通过 → 可用（[LayerArtifactDiagnostic.ERROR_OK]）。
 *
 * 判定逻辑集中在这里（而不是散落在扫描循环）才能被单测锁住；IO 与摘要计算由
 * 调用方通过 [probeArtifact] 注入。
 */
fun classifyLayerArtifact(
    manifestName: String,
    raw: String?,
    expectedModelSha256: String = "",
    probeArtifact: (LayerArtifactDescriptor) -> LayerArtifactProbe?,
): LayerArtifactClassification {
    if (raw == null) {
        return LayerArtifactClassification(
            LayerArtifactDiagnostic(
                manifestName = manifestName,
                errorCode = LayerArtifactDiagnostic.MANIFEST_UNREADABLE,
                detail = "manifest could not be read",
            ),
        )
    }
    val parsed = LayerArtifactManifestParser.parse(raw, manifestName)
    val descriptor = parsed.getOrNull()
    if (descriptor == null) {
        return LayerArtifactClassification(
            LayerArtifactDiagnostic(
                manifestName = manifestName,
                errorCode = LayerArtifactDiagnostic.MANIFEST_INVALID,
                detail = parsed.exceptionOrNull()?.message ?: "manifest did not parse",
            ),
        )
    }
    val identity = LayerArtifactDiagnostic(
        manifestName = manifestName,
        errorCode = LayerArtifactDiagnostic.ERROR_OK,
        artifactName = descriptor.artifactName,
        startLayer = descriptor.startLayer,
        endLayerExclusive = descriptor.endLayerExclusive,
        architecture = descriptor.architecture,
        mode = descriptor.mode,
        sourceModelSha256 = descriptor.sourceModelSha256,
        artifactSha256 = descriptor.artifactSha256,
    )
    val expected = expectedModelSha256.trim().lowercase()
    if (
        expected.isNotEmpty() &&
        descriptor.sourceModelSha256 != expected &&
        descriptor.artifactSha256 != expected
    ) {
        return LayerArtifactClassification(
            identity.copy(
                errorCode = LayerArtifactDiagnostic.SOURCE_DIGEST_MISMATCH,
                detail = "manifest belongs to another source model",
            ),
            descriptor,
        )
    }
    val probe = probeArtifact(descriptor)
    if (probe == null) {
        return LayerArtifactClassification(
            identity.copy(
                errorCode = LayerArtifactDiagnostic.ARTIFACT_MISSING,
                detail = "artifact file is not present in the model root",
            ),
            descriptor,
        )
    }
    if (!probe.digestMatches) {
        return LayerArtifactClassification(
            identity.copy(
                errorCode = LayerArtifactDiagnostic.ARTIFACT_DIGEST_MISMATCH,
                detail = "artifact digest does not match the manifest",
                artifactPresent = true,
                artifactSizeBytes = probe.sizeBytes,
            ),
            descriptor,
        )
    }
    return LayerArtifactClassification(
        identity.copy(
            artifactPresent = true,
            artifactSizeBytes = probe.sizeBytes,
        ),
        descriptor,
    )
}
