package com.qlh.inference.service

import java.io.File
import java.io.IOException
import java.nio.file.Files

internal fun scanModelFiles(root: File, extensions: Set<String>): List<File> {
    if (!root.exists()) return emptyList()
    if (!root.isDirectory) throw IOException("model storage is not a directory")
    val files = root.listFiles() ?: throw IOException("model storage is not accessible")
    val canonicalRoot = root.canonicalPath + File.separator
    return files.filter { file ->
        file.isFile && !Files.isSymbolicLink(file.toPath()) &&
            file.canonicalPath.startsWith(canonicalRoot) &&
            extensions.any { file.name.endsWith(it, ignoreCase = true) }
    }.sortedBy { it.name.lowercase() }
}

internal fun <T> pairLayerArtifact(
    manifestParentId: String,
    artifactName: String,
    candidates: List<T>,
    parentId: (T) -> String,
    name: (T) -> String,
    verify: (T) -> Boolean,
): Pair<T, Boolean>? {
    var firstMatch: Pair<T, Boolean>? = null
    for (candidate in candidates) {
        if (parentId(candidate) != manifestParentId ||
            !name(candidate).equals(artifactName, ignoreCase = true)
        ) continue
        val match = candidate to verify(candidate)
        if (match.second) return match
        if (firstMatch == null) firstMatch = match
    }
    return firstMatch
}

internal suspend fun <T> pairLayerArtifactVerified(
    manifestParentId: String,
    artifactName: String,
    candidates: List<T>,
    parentId: (T) -> String,
    name: (T) -> String,
    verifyArtifactDigest: Boolean,
    verify: suspend (T) -> Boolean,
): Pair<T, Boolean>? {
    var firstMatch: Pair<T, Boolean>? = null
    for (candidate in candidates) {
        if (parentId(candidate) != manifestParentId ||
            !name(candidate).equals(artifactName, ignoreCase = true)
        ) continue
        val match = candidate to (!verifyArtifactDigest || verify(candidate))
        if (match.second) return match
        if (firstMatch == null) firstMatch = match
    }
    return firstMatch
}

/** Published private directories provide reuse bytes; only the current registration enters the catalog. */
internal fun scanSuppliedSegmentRoots(root: File): List<Pair<File, ModelSupplyManifest>> {
    if (!root.isDirectory || Files.isSymbolicLink(root.toPath())) return emptyList()
    return root.listFiles().orEmpty().mapNotNull { directory ->
        if (!directory.isDirectory || !Regex("segment-[0-9a-f]{64}").matches(directory.name) ||
            Files.isSymbolicLink(directory.toPath()) || directory.canonicalFile.parentFile != root.canonicalFile) return@mapNotNull null
        val marker = File(directory, ModelSupplyManifest.MARKER)
        if (!marker.isFile || Files.isSymbolicLink(marker.toPath()) || marker.length() > 4 * 1024 * 1024) return@mapNotNull null
        runCatching { ModelSupplyManifest.parse(marker.readText(Charsets.UTF_8)) }.getOrNull()
            ?.takeIf { directory.name == "segment-${it.manifestSha256}" }?.let { directory to it }
    }.sortedBy { it.first.name }
}