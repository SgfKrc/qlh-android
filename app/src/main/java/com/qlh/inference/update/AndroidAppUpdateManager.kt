package com.qlh.inference.update

import android.content.Context
import android.content.Intent
import android.content.pm.PackageInfo
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Build
import android.provider.Settings
import androidx.core.content.FileProvider
import com.qlh.inference.BuildConfig
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.OkHttpClient
import okhttp3.Request
import java.io.ByteArrayOutputStream
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.security.MessageDigest
import java.util.concurrent.TimeUnit

data class AndroidUpdateCandidate(
    val asset: AndroidUpdateAsset,
    val downloadUrl: String,
)

data class UpdateDownloadProgress(
    val downloadedBytes: Long,
    val totalBytes: Long,
) {
    val percent: Int
        get() = if (totalBytes <= 0L) 0 else
            ((downloadedBytes * 100L) / totalBytes).toInt().coerceIn(0, 100)
}

class AndroidAppUpdateManager(context: Context) {
    private val appContext = context.applicationContext
    private val http = OkHttpClient.Builder()
        .connectTimeout(10, TimeUnit.SECONDS)
        .readTimeout(120, TimeUnit.SECONDS)
        .followRedirects(true)
        .build()

    suspend fun checkForUpdate(
        host: String,
        currentVersion: String = BuildConfig.VERSION_NAME,
        variant: String = if (BuildConfig.IS_LITE) "lite" else "full",
    ): Result<AndroidUpdateCandidate?> = withContext(Dispatchers.IO) {
        var validManifestSeen = false
        val errors = mutableListOf<String>()
        for (source in defaultAndroidUpdateSources(host)) {
            try {
                val manifest = fetchManifest(source)
                val selectedResult = selectAndroidUpdate(manifest, currentVersion, variant)
                if (selectedResult.isFailure) {
                    errors += "${source.substringBefore('?')}: ${selectedResult.exceptionOrNull()?.message}"
                    continue
                }
                val selected = selectedResult.getOrNull()
                validManifestSeen = true
                if (selected != null) {
                    val resolved = resolveUpdateAssetUrl(source, selected.url)
                        ?: error("更新下载地址无法解析")
                    return@withContext Result.success(AndroidUpdateCandidate(selected, resolved))
                }
            } catch (error: CancellationException) {
                throw error
            } catch (error: Exception) {
                errors += "${source.substringBefore('?')}: ${error.message ?: error.javaClass.simpleName}"
            }
        }
        if (validManifestSeen) {
            Result.success(null)
        } else {
            Result.failure(IOException(errors.joinToString("; ").take(600).ifBlank { "没有可用更新源" }))
        }
    }

    suspend fun downloadAndVerify(
        candidate: AndroidUpdateCandidate,
        onProgress: suspend (UpdateDownloadProgress) -> Unit = {},
    ): Result<File> = withContext(Dispatchers.IO) {
        val updateDir = File(appContext.cacheDir, "updates").also { it.mkdirs() }
        val output = File(updateDir, "qlh-update-${candidate.asset.sha256.take(16)}.apk")
        val partial = File(updateDir, output.name + ".part")
        try {
            updateDir.listFiles()?.filter { it != output && it != partial }?.forEach { it.delete() }
            partial.delete()
            val request = Request.Builder().url(candidate.downloadUrl).get().build()
            http.newCall(request).execute().use { response ->
                if (!response.isSuccessful) throw IOException("APK 下载 HTTP ${response.code}")
                val body = response.body ?: throw IOException("APK 下载响应为空")
                val declaredLength = body.contentLength()
                if (declaredLength > 0L && declaredLength != candidate.asset.sizeBytes) {
                    throw IOException("APK 响应大小与清单不一致")
                }
                val digest = MessageDigest.getInstance("SHA-256")
                var downloaded = 0L
                var lastReported = -1
                FileOutputStream(partial).use { sink ->
                    body.byteStream().use { input ->
                        val buffer = ByteArray(DEFAULT_BUFFER_SIZE)
                        while (true) {
                            val count = input.read(buffer)
                            if (count < 0) break
                            downloaded += count
                            if (downloaded > candidate.asset.sizeBytes || downloaded > MAX_ANDROID_APK_BYTES) {
                                throw IOException("APK 下载超过清单大小")
                            }
                            digest.update(buffer, 0, count)
                            sink.write(buffer, 0, count)
                            val percent = ((downloaded * 100L) / candidate.asset.sizeBytes).toInt()
                            if (percent != lastReported) {
                                lastReported = percent
                                withContext(Dispatchers.Main) {
                                    onProgress(UpdateDownloadProgress(downloaded, candidate.asset.sizeBytes))
                                }
                            }
                        }
                        sink.fd.sync()
                    }
                }
                if (downloaded != candidate.asset.sizeBytes) throw IOException("APK 下载不完整")
                val actualHash = digest.digest().joinToString("") { "%02x".format(it) }
                if (actualHash != candidate.asset.sha256) throw IOException("APK SHA-256 校验失败")
            }
            if (!partial.renameTo(output)) {
                partial.copyTo(output, overwrite = true)
                partial.delete()
            }
            verifyPackageIdentity(output)
            Result.success(output)
        } catch (error: CancellationException) {
            partial.delete()
            throw error
        } catch (error: Exception) {
            partial.delete()
            output.delete()
            Result.failure(error)
        }
    }

    fun canRequestPackageInstalls(): Boolean =
        Build.VERSION.SDK_INT < Build.VERSION_CODES.O || appContext.packageManager.canRequestPackageInstalls()

    fun openInstallPermissionSettings(): Result<Unit> = runCatching {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            appContext.startActivity(
                Intent(
                    Settings.ACTION_MANAGE_UNKNOWN_APP_SOURCES,
                    Uri.parse("package:${appContext.packageName}"),
                ).addFlags(Intent.FLAG_ACTIVITY_NEW_TASK),
            )
        }
    }

    fun launchInstaller(apk: File): Result<Unit> = runCatching {
        require(canRequestPackageInstalls()) { "尚未允许此应用安装更新" }
        verifyPackageIdentity(apk)
        val uri = FileProvider.getUriForFile(
            appContext,
            "${appContext.packageName}.update-files",
            apk,
        )
        val intent = Intent(Intent.ACTION_VIEW).apply {
            setDataAndType(uri, "application/vnd.android.package-archive")
            addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION or Intent.FLAG_ACTIVITY_NEW_TASK)
        }
        require(intent.resolveActivity(appContext.packageManager) != null) { "系统没有可用的 APK 安装器" }
        appContext.startActivity(intent)
    }

    @Suppress("DEPRECATION")
    private fun verifyPackageIdentity(apk: File) {
        val flags = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
            PackageManager.GET_SIGNING_CERTIFICATES
        } else {
            PackageManager.GET_SIGNATURES
        }
        val packageManager = appContext.packageManager
        val archive = packageManager.getPackageArchiveInfo(apk.absolutePath, flags)
            ?: error("无法解析 APK")
        require(archive.packageName == appContext.packageName) { "APK 包名与当前应用不一致" }
        val installed = packageManager.getPackageInfo(appContext.packageName, flags)
        val installedVersionCode = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
            installed.longVersionCode
        } else {
            installed.versionCode.toLong()
        }
        val archiveVersionCode = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
            archive.longVersionCode
        } else {
            archive.versionCode.toLong()
        }
        require(archiveVersionCode > installedVersionCode) { "APK versionCode 未高于当前应用" }
        val trustedSigners = signerDigests(installed)
        val archiveSigners = signerDigests(archive)
        require(trustedSigners.isNotEmpty() && archiveSigners.isNotEmpty()) { "无法读取 APK 签名" }
        if (hasMultipleSigners(installed) || hasMultipleSigners(archive)) {
            require(trustedSigners == archiveSigners) { "APK 多签名集合与当前应用不一致" }
        } else {
            require(trustedSigners.intersect(archiveSigners).isNotEmpty()) { "APK 签名与当前应用不一致" }
        }
    }

    @Suppress("DEPRECATION")
    private fun signerDigests(info: PackageInfo): Set<String> {
        val signatures = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
            val signingInfo = info.signingInfo ?: return emptySet()
            if (signingInfo.hasMultipleSigners()) {
                signingInfo.apkContentsSigners
            } else {
                signingInfo.signingCertificateHistory
            }
        } else {
            info.signatures
        }
        return signatures.orEmpty().mapTo(linkedSetOf()) { signature ->
            MessageDigest.getInstance("SHA-256")
                .digest(signature.toByteArray())
                .joinToString("") { "%02x".format(it) }
        }
    }

    private fun hasMultipleSigners(info: PackageInfo): Boolean =
        Build.VERSION.SDK_INT >= Build.VERSION_CODES.P && info.signingInfo?.hasMultipleSigners() == true

    private fun fetchManifest(url: String): String {
        val request = Request.Builder().url(url).get().build()
        return http.newCall(request).execute().use { response ->
            if (!response.isSuccessful) throw IOException("清单 HTTP ${response.code}")
            val body = response.body ?: throw IOException("更新清单响应为空")
            val contentLength = body.contentLength()
            if (contentLength > MAX_UPDATE_MANIFEST_BYTES) throw IOException("更新清单过大")
            val bytes = ByteArrayOutputStream()
            body.byteStream().use { input ->
                val buffer = ByteArray(DEFAULT_BUFFER_SIZE)
                var total = 0L
                while (true) {
                    val count = input.read(buffer)
                    if (count < 0) break
                    total += count
                    if (total > MAX_UPDATE_MANIFEST_BYTES) throw IOException("更新清单过大")
                    bytes.write(buffer, 0, count)
                }
            }
            bytes.toString(Charsets.UTF_8.name())
        }
    }
}
