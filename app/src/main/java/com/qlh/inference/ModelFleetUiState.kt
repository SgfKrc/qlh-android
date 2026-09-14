package com.qlh.inference

import com.qlh.inference.network.GgufModelInfo
import com.qlh.inference.network.LocalModelAssetSummary
import com.qlh.inference.network.MasterModelFleetData
import com.qlh.inference.network.ServerModelSummary

/** Read-only, path-free Android projection of the main-node model fleet. */
data class ModelFleetUiState(
    val snapshot: ModelFleetSnapshot? = null,
    val loading: Boolean = false,
    val error: String? = null,
)

data class ModelFleetSnapshot(
    val currentModelId: String? = null,
    val currentModelName: String = "",
    val currentLoaded: Boolean = false,
    val pipelinePrepared: Boolean = false,
    val currentEngine: String = "",
    val currentQuantType: String? = null,
    val androidSelectedModelName: String = "",
    val androidSelectedModelSizeBytes: Long = 0L,
    val registryCount: Int = 0,
    val localAssetCount: Int = 0,
    val verifiedGgufCount: Int = 0,
    val entries: List<ModelFleetEntry> = emptyList(),
)

data class ModelFleetEntry(
    val modelId: String,
    val name: String,
    val modelType: String,
    val status: ModelFleetStatus,
    val sources: List<String>,
    val formats: List<String>,
    val totalBytes: Long = 0L,
)

enum class ModelFleetStatus {
    ACTIVE,
    AVAILABLE,
    MISSING,
    UNVERIFIED,
}

/**
 * Merge the four established server inventories without retaining absolute paths,
 * download URLs, proxy settings, runtime hints, or mutation handles in Android UI state.
 */
fun toModelFleetSnapshot(
    data: MasterModelFleetData,
    androidSelectedModelName: String = "",
    androidSelectedModelSizeBytes: Long = 0L,
): ModelFleetSnapshot {
    val currentId = data.current.modelId?.trim().orEmpty()
    val assetsById = data.localAssets.assets
        .filter { it.modelId.isNotBlank() }
        .groupBy { canonicalFleetId(it.modelId) }
    val entries = linkedMapOf<String, ModelFleetEntry>()

    data.registry.models
        .filter { it.modelId.isNotBlank() }
        .forEach { model ->
            val modelId = model.modelId.trim()
            val matchingAssets = assetsById[canonicalFleetId(modelId)].orEmpty()
            entries["registry:$modelId"] = model.toFleetEntry(
                isActive = data.current.loaded && canonicalFleetId(modelId) == canonicalFleetId(currentId),
                matchingAssets = matchingAssets,
            )
        }

    data.localAssets.assets
        .filter { asset -> asset.modelId.isNotBlank() }
        .filter { asset ->
            data.registry.models.none {
                canonicalFleetId(it.modelId) == canonicalFleetId(asset.modelId)
            }
        }
        .forEach { asset ->
            entries["asset:${asset.modelId}"] = asset.toFleetEntry()
        }

    data.verifiedGguf
        .filter { it.filename.isNotBlank() }
        .forEach { gguf ->
            entries["gguf:${gguf.filename}"] = gguf.toFleetEntry()
        }

    if (currentId.isNotBlank() && entries.values.none {
            canonicalFleetId(it.modelId) == canonicalFleetId(currentId)
        }
    ) {
        entries["runtime:$currentId"] = ModelFleetEntry(
            modelId = currentId,
            name = data.current.modelName.ifBlank { currentId },
            modelType = "runtime",
            status = if (data.current.loaded) ModelFleetStatus.ACTIVE else ModelFleetStatus.AVAILABLE,
            sources = listOf("主节点运行时"),
            formats = emptyList(),
        )
    }

    return ModelFleetSnapshot(
        currentModelId = currentId.ifBlank { null },
        currentModelName = data.current.modelName,
        currentLoaded = data.current.loaded,
        pipelinePrepared = data.current.pipelinePrepared,
        currentEngine = data.current.engine,
        currentQuantType = data.current.quantType,
        androidSelectedModelName = androidSelectedModelName.trim(),
        androidSelectedModelSizeBytes = androidSelectedModelSizeBytes.coerceAtLeast(0L),
        registryCount = data.registry.models.count { it.modelId.isNotBlank() },
        localAssetCount = data.localAssets.assets.count { it.modelId.isNotBlank() },
        verifiedGgufCount = data.verifiedGguf.size,
        entries = entries.values.sortedWith(
            compareBy<ModelFleetEntry> { fleetStatusOrder(it.status) }
                .thenBy { it.name.lowercase() }
                .thenBy { it.modelId },
        ),
    )
}

private fun ServerModelSummary.toFleetEntry(
    isActive: Boolean,
    matchingAssets: List<LocalModelAssetSummary>,
): ModelFleetEntry {
    val verifiedAsset = matchingAssets.any { it.integrity == "manifest_verified" }
    val discoveredAsset = matchingAssets.isNotEmpty()
    val status = when {
        isActive -> ModelFleetStatus.ACTIVE
        isAvailable || verifiedAsset -> ModelFleetStatus.AVAILABLE
        discoveredAsset -> ModelFleetStatus.UNVERIFIED
        else -> ModelFleetStatus.MISSING
    }
    val sources = buildList {
        add("主节点注册表")
        if (discoveredAsset) add(if (verifiedAsset) "已验证资产" else "待验证资产")
        if (hasGguf) add("主节点 GGUF")
        if (hasSafetensors) add("主节点 PyTorch")
    }.distinct()
    val assetBytes = matchingAssets.sumOf { it.totalBytes.coerceAtLeast(0L) }
    return ModelFleetEntry(
        modelId = modelId.trim(),
        name = name.ifBlank { modelId.trim() },
        modelType = modelType.ifBlank { "unknown" },
        status = status,
        sources = sources,
        formats = availableFormats.filter { it.isNotBlank() }.distinct(),
        totalBytes = assetBytes,
    )
}

private fun LocalModelAssetSummary.toFleetEntry(): ModelFleetEntry = ModelFleetEntry(
    modelId = modelId.trim(),
    name = name.ifBlank { modelId.trim() },
    modelType = modelType.ifBlank { "unknown" },
    status = if (integrity == "manifest_verified") ModelFleetStatus.AVAILABLE else ModelFleetStatus.UNVERIFIED,
    sources = listOf(if (integrity == "manifest_verified") "已验证资产" else "待验证资产"),
    formats = availableFormats.filter { it.isNotBlank() }.distinct(),
    totalBytes = totalBytes.coerceAtLeast(0L),
)

private fun GgufModelInfo.toFleetEntry(): ModelFleetEntry = ModelFleetEntry(
    modelId = "gguf:${filename.trim()}",
    name = filename.trim(),
    modelType = "gguf",
    status = ModelFleetStatus.AVAILABLE,
    sources = listOf("已验证 GGUF 下载目录"),
    formats = listOf("gguf"),
    totalBytes = sizeBytes.coerceAtLeast(0L),
)

private fun canonicalFleetId(value: String): String = value.trim().lowercase()
    .replace('_', '-')
    .removeSuffix("-gguf")

private fun fleetStatusOrder(status: ModelFleetStatus): Int = when (status) {
    ModelFleetStatus.ACTIVE -> 0
    ModelFleetStatus.AVAILABLE -> 1
    ModelFleetStatus.UNVERIFIED -> 2
    ModelFleetStatus.MISSING -> 3
}
