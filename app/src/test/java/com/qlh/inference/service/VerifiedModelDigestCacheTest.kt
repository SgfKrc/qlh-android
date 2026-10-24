package com.qlh.inference.service

import java.io.IOException
import java.util.concurrent.atomic.AtomicInteger
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.async
import kotlinx.coroutines.awaitAll
import kotlinx.coroutines.cancelAndJoin
import kotlinx.coroutines.delay
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withTimeout
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Test

class VerifiedModelDigestCacheTest {
    private val original = ModelFileFingerprint(1, 2, 1024, 10, 10)

    @Test
    fun `nanosecond changes invalidate a digest within the same second`() = runBlocking {
        val cache = VerifiedModelDigestCache({ 0L })
        var fingerprint = original
        var reads = 0
        suspend fun digest(): String = cache.digest("artifact", { fingerprint }) { (++reads).toString() }
        assertEquals("1", digest())
        fingerprint = original.copy(modifiedNanoseconds = 100)
        assertEquals("2", digest())
        fingerprint = fingerprint.copy(changedNanoseconds = 200)
        assertEquals("3", digest())
        assertEquals("3", digest())
    }

    @Test
    fun `controlled ctime migration keeps the original verification lease and rejects changed content versions`() = runBlocking {
        var time = 0L
        val cache = VerifiedModelDigestCache({ time }, maxAgeMs = 100)
        assertEquals("verified", cache.digest("artifact", { original }) { "verified" })
        val linked = original.copy(changedNanoseconds = 100)
        time = 90
        assertTrue(cache.rebind("artifact", original, linked))
        assertEquals("verified", cache.digest("artifact", { linked }) { "unexpected read" })
        assertFalse(cache.rebind("artifact", original, linked))
        assertFalse(cache.rebind("artifact", linked, linked.copy(inode = 3)))
        assertFalse(cache.rebind("artifact", linked, linked.copy(modifiedNanoseconds = 100)))
        time = 100
        assertEquals("fresh", cache.digest("artifact", { linked }) { "fresh" })
    }

    @Test
    fun `explicit fresh verification invalidates a hot entry even when verification then fails`() = runBlocking {
        val cache = VerifiedModelDigestCache({ 0L })
        assertEquals("old", cache.digest("artifact", { original }) { "old" })
        assertThrows(IOException::class.java) {
            runBlocking { cache.digest("artifact", { original }, fresh = true) { throw IOException("unreadable") } }
        }
        assertFalse(cache.rebind("artifact", original, original.copy(changedNanoseconds = 1)))
        assertEquals("new", cache.digest("artifact", { original }) { "new" })
    }

    @Test
    fun `unchanged reconnect reuses digest but file identity and expiry force a rehash`() = runBlocking {
        var time = 0L
        var fingerprint = original
        val calls = AtomicInteger()
        val lookups = mutableListOf<Boolean>()
        val cache = VerifiedModelDigestCache({ time }, maxAgeMs = 300_000L)
        suspend fun digest(scope: String = "tree-A"): String = cache.digest(
            "$scope|content://models/segment.gguf", { fingerprint }, { lookups += it },
        ) { calls.incrementAndGet().toString() }

        assertEquals("1", digest())
        assertEquals("1", digest())
        assertEquals(1, calls.get())

        fingerprint = original.copy(changedSeconds = 11)
        assertEquals("2", digest())
        fingerprint = original.copy(inode = 3, changedSeconds = 10)
        assertEquals("3", digest())
        assertEquals("4", digest(scope = "tree-B"))
        time = 300_000L
        assertEquals("5", digest(scope = "tree-B"))
        assertEquals(listOf(false, true, false, false, false, false), lookups)
    }

    @Test
    fun `unreliable fingerprint is never cached and changed file during hash fails closed`() = runBlocking {
        val cache = VerifiedModelDigestCache({ 0L })
        val calls = AtomicInteger()
        var fingerprint: ModelFileFingerprint? = null
        suspend fun digest(): String = cache.digest("artifact", { fingerprint }) {
            calls.incrementAndGet().toString()
        }

        assertEquals("1", digest())
        assertEquals("2", digest())
        fingerprint = original
        val changed = assertThrows(IOException::class.java) {
            runBlocking {
                cache.digest("artifact", { fingerprint }) {
                    fingerprint = original.copy(size = 99)
                    "bad"
                }
            }
        }
        assertEquals("model artifact changed during SHA-256 verification", changed.message)
        assertEquals("3", digest())
        assertEquals("3", digest())
    }

    @Test
    fun `failed verification cannot become a trusted cache hit`() = runBlocking {
        val cache = VerifiedModelDigestCache({ 0L })
        val calls = AtomicInteger()
        assertThrows(IOException::class.java) {
            runBlocking {
                cache.digest("artifact", { original }) {
                    calls.incrementAndGet()
                    throw IOException("unreadable")
                }
            }
        }
        assertEquals("verified", cache.digest("artifact", { original }) {
            calls.incrementAndGet()
            "verified"
        })
        assertEquals(2, calls.get())
    }

    @Test
    fun `concurrent readers of same artifact share one completed verification`() = runBlocking {
        val cache = VerifiedModelDigestCache({ 0L })
        val started = CompletableDeferred<Unit>()
        val release = CompletableDeferred<Unit>()
        val calls = AtomicInteger()
        val requests = List(4) {
            async(Dispatchers.Default) {
                cache.digest("artifact", { original }) {
                    calls.incrementAndGet()
                    started.complete(Unit)
                    release.await()
                    "verified"
                }
            }
        }
        started.await()
        release.complete(Unit)
        assertEquals(List(4) { "verified" }, requests.awaitAll())
        assertEquals(1, calls.get())
    }

    @Test
    fun `hashing another artifact is not blocked by an in-flight digest`() = runBlocking {
        val cache = VerifiedModelDigestCache({ 0L })
        val started = CompletableDeferred<Unit>()
        val release = CompletableDeferred<Unit>()
        val first = async(Dispatchers.Default) {
            cache.digest("first", { original }) {
                started.complete(Unit)
                release.await()
                "first-sha"
            }
        }
        try {
            started.await()
            assertEquals("second-sha", withTimeout(5_000L) {
                cache.digest("second", { original }) { "second-sha" }
            })
        } finally {
            release.complete(Unit)
        }
        assertEquals("first-sha", first.await())
    }

    @Test
    fun `cancelled verification releases its lock without caching partial work`() = runBlocking {
        val cache = VerifiedModelDigestCache({ 0L })
        val started = CompletableDeferred<Unit>()
        val interrupted = async(Dispatchers.Default) {
            cache.digest("artifact", { original }) {
                started.complete(Unit)
                delay(Long.MAX_VALUE)
                "partial"
            }
        }
        withTimeout(5_000L) { started.await() }
        withTimeout(5_000L) { interrupted.cancelAndJoin() }
        assertEquals("complete", cache.digest("artifact", { original }) { "complete" })
    }
}
