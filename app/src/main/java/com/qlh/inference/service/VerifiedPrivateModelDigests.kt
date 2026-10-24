package com.qlh.inference.service

import android.os.Build
import android.system.Os
import android.system.OsConstants
import java.io.File
import java.io.IOException
import java.nio.file.Files
import java.nio.file.LinkOption
import java.nio.file.StandardOpenOption
import java.nio.file.StandardCopyOption
import java.nio.file.attribute.BasicFileAttributes
import java.security.MessageDigest
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.ensureActive

internal class VerifiedPrivateModelDigests(
    private val cache: VerifiedModelDigestCache = VerifiedModelDigestCache({ System.nanoTime() / 1_000_000 }),
    private val fingerprint: (File) -> ModelFileFingerprint? = ::privateModelFingerprint,
    private val hash: suspend (File, suspend () -> Unit) -> String = ::hashPrivateModelFile,
) {
    internal data class Verification(val digest: String, val fingerprint: ModelFileFingerprint?)

    suspend fun verify(
        file: File,
        fresh: Boolean = false,
        onLookup: (Boolean) -> Unit = {},
        checkCurrent: suspend () -> Unit = {},
    ): Verification {
        currentCoroutineContext().ensureActive()
        checkCurrent()
        ModelSupplyPaths.check(file)
        val before = attributes(file)
        if (!before.isRegularFile) throw IOException("model artifact is not a regular file")
        val version = fingerprint(file)
        val actual = Files.newInputStream(file.toPath(), StandardOpenOption.READ, LinkOption.NOFOLLOW_LINKS).use {
            cache.digest(version?.key() ?: "private-unversioned:${file.absolutePath}",
                { fingerprint(file) }, onLookup, fresh = fresh,
            ) {
                hash(file) {
                    currentCoroutineContext().ensureActive()
                    checkCurrent()
                }
            }
        }
        currentCoroutineContext().ensureActive()
        checkCurrent()
        ModelSupplyPaths.check(file)
        val after = attributes(file)
        val observed = fingerprint(file)
        if (!after.isRegularFile || before.size() != after.size() || before.fileKey() != after.fileKey() ||
            before.lastModifiedTime() != after.lastModifiedTime() || version != observed
        ) throw IOException("model artifact changed during SHA-256 verification")
        return Verification(actual, observed)
    }

    suspend fun linkVerified(
        source: File,
        target: File,
        verified: Verification,
        createLink: (File, File) -> Unit,
    ): Boolean {
        currentCoroutineContext().ensureActive()
        ModelSupplyPaths.check(source)
        ModelSupplyPaths.check(target)
        if (verified.fingerprint != fingerprint(source)) {
            throw IOException("model artifact changed before linking")
        }
        createLink(source, target)
        val before = verified.fingerprint ?: return false
        val after = fingerprint(target) ?: return false
        if (fingerprint(source) != after || !Files.isSameFile(source.toPath(), target.toPath())) return false
        return cache.rebind(before.key(), before, after)
    }

    suspend fun moveVerified(source: File, target: File, verified: Verification): Boolean {
        currentCoroutineContext().ensureActive()
        ModelSupplyPaths.check(source)
        ModelSupplyPaths.check(target)
        if (verified.fingerprint != fingerprint(source)) {
            throw IOException("model artifact changed before renaming")
        }
        Files.move(source.toPath(), target.toPath(), StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING)
        val before = verified.fingerprint ?: return false
        val after = fingerprint(target) ?: return false
        return cache.rebind(before.key(), before, after)
    }

    private fun attributes(file: File): BasicFileAttributes =
        Files.readAttributes(file.toPath(), BasicFileAttributes::class.java, LinkOption.NOFOLLOW_LINKS)

    private fun ModelFileFingerprint.key(): String = "private:$device:$inode"

    companion object {
        val shared = VerifiedPrivateModelDigests()
    }
}

private fun privateModelFingerprint(file: File): ModelFileFingerprint? = runCatching {
    if (Build.VERSION.SDK_INT < 27) return@runCatching null
    val stat = Os.lstat(file.absolutePath)
    if (!OsConstants.S_ISREG(stat.st_mode) || stat.st_dev < 0L || stat.st_ino <= 0L || stat.st_size < 0L ||
        stat.st_ctim.tv_sec <= 0L || stat.st_mtim.tv_nsec !in 0L..999_999_999L ||
        stat.st_ctim.tv_nsec !in 0L..999_999_999L
    ) return@runCatching null
    ModelFileFingerprint(stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtim.tv_sec, stat.st_ctim.tv_sec,
        stat.st_mtim.tv_nsec, stat.st_ctim.tv_nsec)
}.getOrNull()

internal suspend fun hashPrivateModelFile(file: File, checkCurrent: suspend () -> Unit): String {
    val digest = MessageDigest.getInstance("SHA-256")
    Files.newInputStream(file.toPath(), StandardOpenOption.READ, LinkOption.NOFOLLOW_LINKS).use { input ->
        val buffer = ByteArray(1024 * 1024)
        while (true) {
            currentCoroutineContext().ensureActive()
            checkCurrent()
            val count = input.read(buffer)
            if (count < 0) break
            digest.update(buffer, 0, count)
        }
    }
    return digest.digest().joinToString("") { "%02x".format(it.toInt() and 255) }
}
