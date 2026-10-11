package com.qlh.inference.service

import java.io.IOException
import java.util.LinkedHashMap
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.sync.Mutex

/** Process-local, bounded reuse of a full-file digest; never persisted or trusted by itself. */
internal class VerifiedModelDigestCache(
    private val clockMs: () -> Long,
    private val maxAgeMs: Long = 5 * 60 * 1000L,
    private val maxEntries: Int = 64,
) {
    init {
        require(maxAgeMs > 0L && maxEntries > 0)
    }

    private class Entry {
        val mutex = Mutex()
        var fingerprint: ModelFileFingerprint? = null
        var digest: String? = null
        var verifiedAtMs: Long = 0
    }

    private val entries = LinkedHashMap<String, Entry>(16, 0.75f, true)

    suspend fun digest(
        key: String,
        fingerprint: () -> ModelFileFingerprint?,
        onLookup: (reused: Boolean) -> Unit = {},
        hash: suspend () -> String,
    ): String {
        currentCoroutineContext().ensureActive()
        val entry = synchronized(entries) {
            entries.getOrPut(key) { Entry() }.also {
                if (entries.size > maxEntries) entries.remove(entries.keys.first())
            }
        }
        entry.mutex.lock()
        try {
            currentCoroutineContext().ensureActive()
            val before = fingerprint()
            val now = clockMs()
            val cached = entry.digest
            if (before != null && cached != null && entry.fingerprint == before &&
                now >= entry.verifiedAtMs && now - entry.verifiedAtMs < maxAgeMs
            ) {
                onLookup(true)
                return cached
            }
            entry.digest = null
            entry.fingerprint = null
            onLookup(false)
            val actual = hash()
            currentCoroutineContext().ensureActive()
            val after = fingerprint()
            if (before != null && before != after) {
                throw IOException("model artifact changed during SHA-256 verification")
            }
            if (before != null) {
                entry.fingerprint = before
                entry.digest = actual
                entry.verifiedAtMs = clockMs()
            }
            return actual
        } finally {
            entry.mutex.unlock()
        }
    }
}

/** Inode and change time supplement size/mtime; a short lease still bounds spoofed metadata. */
internal data class ModelFileFingerprint(
    val device: Long,
    val inode: Long,
    val size: Long,
    val modifiedSeconds: Long,
    val changedSeconds: Long,
)
