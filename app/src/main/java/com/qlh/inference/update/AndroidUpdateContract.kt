package com.qlh.inference.update

import com.google.gson.JsonParser
import com.qlh.inference.network.formatUrlHost
import java.net.URI

const val MAX_UPDATE_MANIFEST_BYTES = 1 * 1024 * 1024L
const val MAX_ANDROID_APK_BYTES = 512L * 1024L * 1024L

data class AndroidUpdateAsset(
    val version: String,
    val name: String,
    val url: String,
    val sizeBytes: Long,
    val sha256: String,
)

/**
 * Returns an installable APK for the current flavour, or null when the manifest
 * has no newer compatible package. Parsing is deliberately fail-closed.
 */
fun selectAndroidUpdate(
    manifestJson: String,
    currentVersion: String,
    variant: String,
): Result<AndroidUpdateAsset?> = runCatching {
    require(variant in setOf("full", "lite")) { "未知 Android 变体" }
    val root = JsonParser.parseString(manifestJson).asJsonObject
    require(root["schema_version"]?.asInt == 1) { "不支持的更新清单版本" }
    val version = root["tag"]?.asString?.trim().orEmpty()
    require(parseVersion(version) != null) { "更新清单版本无效" }
    val candidateVersion = requireNotNull(parseVersion(version))
    val installedVersion = parseVersion(currentVersion)
        ?: error("当前应用版本无效")
    if (compareVersions(candidateVersion, installedVersion) <= 0) {
        return@runCatching null
    }
    val assets = root["assets"]?.takeIf { it.isJsonArray }?.asJsonArray
        ?: error("更新清单缺少 assets")
    val matches = assets.mapNotNull { item ->
        if (!item.isJsonObject) return@mapNotNull null
        val asset = item.asJsonObject
        val platform = asset["platform"]?.asString?.lowercase()
        val assetVariant = asset["variant"]?.asString?.lowercase()
        val kind = asset["kind"]?.asString?.lowercase()
        if (platform != "android" || assetVariant != variant || kind != "installer") {
            return@mapNotNull null
        }
        val name = asset["name"]?.asString?.trim().orEmpty()
        val url = asset["url"]?.asString?.trim().orEmpty()
        val size = asset["size"]?.asLong ?: -1L
        val sha256 = asset["sha256"]?.asString?.trim()?.lowercase().orEmpty()
        require(name.endsWith(".apk", ignoreCase = true)) { "Android 更新必须为 APK" }
        require(isSupportedUpdateUrl(url)) { "更新下载地址无效" }
        require(size in 1..MAX_ANDROID_APK_BYTES) { "APK 大小超出安全上限" }
        require(sha256.matches(Regex("[0-9a-f]{64}"))) { "APK SHA-256 无效" }
        AndroidUpdateAsset(version, name, url, size, sha256)
    }
    require(matches.size <= 1) { "更新清单存在多个匹配 APK" }
    matches.singleOrNull()
}

fun defaultAndroidUpdateSources(host: String): List<String> = listOf(
    "http://${formatUrlHost(host)}:9090/latest.json",
    "https://github.com/SgfKrc/LEDS_BJTU/releases/latest/download/latest.json",
)

fun resolveUpdateAssetUrl(manifestUrl: String, assetUrl: String): String? = runCatching {
    val asset = URI(assetUrl)
    val resolved = if (asset.isAbsolute) asset else URI(manifestUrl).resolve(asset)
    require(resolved.scheme in setOf("http", "https")) { "不支持的下载协议" }
    require(!resolved.host.isNullOrBlank()) { "下载地址缺少主机名" }
    resolved.toString()
}.getOrNull()

private fun isSupportedUpdateUrl(value: String): Boolean {
    if (value.startsWith("/") && !value.startsWith("//")) return true
    return runCatching {
        val parsed = URI(value)
        parsed.scheme in setOf("http", "https") && !parsed.host.isNullOrBlank()
    }.getOrDefault(false)
}

private fun parseVersion(value: String): List<Int>? {
    val match = Regex("^v?(\\d+(?:\\.\\d+){0,3})(?:[-+].*)?$")
        .matchEntire(value.trim()) ?: return null
    return match.groupValues[1].split('.').map { part ->
        part.toIntOrNull() ?: return null
    }
}

private fun compareVersions(left: List<Int>, right: List<Int>): Int {
    val length = maxOf(left.size, right.size)
    for (index in 0 until length) {
        val result = (left.getOrElse(index) { 0 }).compareTo(right.getOrElse(index) { 0 })
        if (result != 0) return result
    }
    return 0
}
