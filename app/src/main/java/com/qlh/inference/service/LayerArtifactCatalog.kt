package com.qlh.inference.service

import com.google.gson.JsonArray
import com.google.gson.JsonObject
import com.google.gson.JsonParser
import com.qlh.inference.service.ModelManager.LayerArtifact
import com.qlh.inference.worker.AndroidWorkerCapabilities
import java.io.File

/** Metadata required to bind one GGUF artifact to a source-model layer range. */
data class LayerArtifactDescriptor(
    val artifactName: String,
    val startLayer: Int,
    val endLayerExclusive: Int,
    val artifactSha256: String,
    val sourceModelSha256: String,
    val sourceModelId: String?,
    val hiddenSize: Int?,
    val tokenizerSha256: String?,
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
object LayerArtifactManifestParser {
    private val sha256Pattern = Regex("[0-9a-fA-F]{64}")
    private val safeIdPattern = Regex("^[A-Za-z0-9_.:-]{1,128}$")
    private val segmentModes = setOf("head", "middle", "tail")

    fun parse(raw: String, manifestName: String): Result<LayerArtifactDescriptor> = runCatching {
        val root = JsonParser.parseString(raw).asJsonObject
        val artifact = string(root, "artifact")
            .ifBlank { File(manifestName).nameWithoutExtension.removeSuffix(".manifest") + ".gguf" }
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
        val sourceModelId = string(root, "source_model_id")
        val tokenizerSha256 = string(root, "tokenizer_sha256").lowercase()
        val hiddenSize = positiveInt(root, "hidden_size")
        val modelPreflightContractPresent =
            sourceModelId.isNotBlank() || tokenizerSha256.isNotBlank()
        if (modelPreflightContractPresent) {
            require(safeIdPattern.matches(sourceModelId)) {
                "manifest source_model_id is invalid"
            }
            require(hiddenSize != null) {
                "manifest hidden_size must be a positive integer"
            }
            require(sha256Pattern.matches(tokenizerSha256)) {
                "manifest tokenizer_sha256 must be a 64-char hex digest"
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
            sourceModelId = sourceModelId.takeIf { modelPreflightContractPresent },
            hiddenSize = hiddenSize,
            tokenizerSha256 = tokenizerSha256.takeIf { modelPreflightContractPresent },
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

    private fun positiveInt(root: JsonObject, name: String): Int? {
        val item = root.get(name)?.takeUnless { it.isJsonNull } ?: return null
        if (!item.isJsonPrimitive || !item.asJsonPrimitive.isNumber) return null
        val number = runCatching { item.asDouble }.getOrNull() ?: return null
        if (!number.isFinite() || number % 1.0 != 0.0 || number < 1 || number > Int.MAX_VALUE) {
            return null
        }
        return number.toInt()
    }
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
        const val FALLBACK_UNAVAILABLE = "fallback_storage_unavailable"
        const val ARTIFACT_AMBIGUOUS = "artifact_ambiguous"
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

/** Local-only reference; never include it in a task-worker capability advertisement. */
data class LayerArtifactCandidate(
    val descriptor: LayerArtifactDescriptor,
    val reference: String,
    val manifestName: String,
    val root: Root,
) {
    enum class Root { SAF, FALLBACK, INTERNAL }
}

/** Local-only candidate metadata; callers must verify bytes before advertising it. */
fun LayerArtifact.toCandidate(
    root: LayerArtifactCandidate.Root = when (document.source) {
        ModelManager.ModelSource.SAF -> LayerArtifactCandidate.Root.SAF
        ModelManager.ModelSource.FALLBACK -> LayerArtifactCandidate.Root.FALLBACK
        ModelManager.ModelSource.INTERNAL -> LayerArtifactCandidate.Root.INTERNAL
    },
): LayerArtifactCandidate = LayerArtifactCandidate(
    descriptor = LayerArtifactDescriptor(
        artifactName = document.name,
        startLayer = startLayer,
        endLayerExclusive = endLayerExclusive,
        artifactSha256 = artifactSha256,
        sourceModelSha256 = sourceModelSha256,
        sourceModelId = sourceModelId,
        hiddenSize = hiddenSize,
        tokenizerSha256 = tokenizerSha256,
        architecture = architecture,
        mode = mode,
    ),
    reference = document.uri.toString(),
    manifestName = manifest.name,
    root = root,
)

const val FALLBACK_LAYER_ARTIFACT_DIRECTORY = "/sdcard/Download/QLH/models"

private fun LayerArtifactCandidate.identityKey(): List<Any?> = listOf(
    descriptor.sourceModelSha256.lowercase(),
    descriptor.sourceModelId,
    descriptor.startLayer,
    descriptor.endLayerExclusive,
    descriptor.artifactSha256.lowercase(),
    descriptor.artifactName,
    descriptor.mode,
    descriptor.architecture,
    descriptor.hiddenSize,
    descriptor.tokenizerSha256,
)

/** Keep alternative digests and source identities until the master's request is known. */
fun mergeLayerArtifactCandidates(
    saf: List<LayerArtifactCandidate>,
    fallback: List<LayerArtifactCandidate>,
): List<LayerArtifactCandidate> = (saf + fallback)
    .sortedBy { it.root.ordinal }
    .distinctBy { it.identityKey() }

fun unambiguousLayerArtifactCandidates(
    candidates: List<LayerArtifactCandidate>,
): List<LayerArtifactCandidate> = candidates
    .groupBy { it.descriptor.startLayer to it.descriptor.endLayerExclusive }
    .values.filter { it.size == 1 }.map { it.single() }

sealed interface LayerArtifactChoice {
    data class Selected(val artifact: LayerArtifactCandidate) : LayerArtifactChoice
    data class ManualRequired(val reason: String, val prompt: String) : LayerArtifactChoice
}

/**
 * Require a master source digest and [start, end); the offer carries a separate artifact digest.
 * storageUnavailable is local-only: use it when neither SAF nor direct fallback can be read.
 */
fun chooseLayerArtifact(
    candidates: List<LayerArtifactCandidate>,
    masterModelId: String,
    masterSourceSha256: String,
    startLayer: Int,
    endLayerExclusive: Int,
    offeredArtifactSha256: String = "",
    storageUnavailable: Boolean = false,
    offeredArtifactModelId: String = "",
): LayerArtifactChoice {
    val sha256 = Regex("[0-9a-fA-F]{64}")
    if (masterModelId.isBlank() || !sha256.matches(masterSourceSha256) ||
        startLayer < 0 || endLayerExclusive <= startLayer ||
        (offeredArtifactSha256.isNotBlank() && !sha256.matches(offeredArtifactSha256))
    ) {
        return LayerArtifactChoice.ManualRequired(
            "model_identity_missing", "请确认主节点模型身份、摘要和层区间后重试。",
        )
    }
    val matches = candidates.filter { candidate ->
        val artifact = candidate.descriptor
        artifact.sourceModelSha256.equals(masterSourceSha256, ignoreCase = true) &&
            (artifact.sourceModelId == null || artifact.sourceModelId == masterModelId) &&
            artifact.startLayer == startLayer && artifact.endLayerExclusive == endLayerExclusive &&
            (offeredArtifactSha256.isBlank() || artifact.artifactSha256.equals(
                offeredArtifactSha256, ignoreCase = true,
            )) &&
            (offeredArtifactModelId.isBlank() || AndroidWorkerCapabilities.artifactModelId(
                artifact.artifactName, artifact.artifactSha256,
            ) == offeredArtifactModelId)
    }
    val distinct = matches.sortedBy { it.root.ordinal }.distinctBy { it.identityKey() }
    if (distinct.size == 1) return LayerArtifactChoice.Selected(distinct.single())
    val reason = when {
        distinct.size > 1 -> "layer_artifact_ambiguous"
        storageUnavailable -> "artifact_storage_unavailable"
        else -> "layer_artifact_not_found"
    }
    return LayerArtifactChoice.ManualRequired(
        reason = reason,
        prompt = "请在设置中用 SAF 授权 Download/QLH/models 子目录，不要选择 Download 根目录；" +
            "其他应用创建的下载工件不能仅凭路径 $FALLBACK_LAYER_ARTIFACT_DIRECTORY 访问。" +
            "确认 GGUF 与 manifest 的源摘要、工件摘要及层区间后重试。",
    )
}
