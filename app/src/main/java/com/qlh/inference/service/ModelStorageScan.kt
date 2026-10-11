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
