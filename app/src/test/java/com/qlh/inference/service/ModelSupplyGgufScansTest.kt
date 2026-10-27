package com.qlh.inference.service

import com.qlh.inference.worker.GgufLayerWeightReader
import java.io.ByteArrayInputStream
import java.io.IOException
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicInteger
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.CoroutineStart
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.async
import kotlinx.coroutines.awaitAll
import kotlinx.coroutines.cancelAndJoin
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.supervisorScope
import kotlinx.coroutines.withTimeout
import org.junit.Assert.*
import org.junit.Test

class ModelSupplyGgufScansTest {
    private val bytes = CutFixture.bytes("hybrid/tail.gguf.fixture")

    @Test fun `warm metadata reads no bytes but still checks access and current version`() = runBlocking {
        val cache = ModelSupplyGgufScans()
        var version: String? = "inode1-size1-mtime1-ctime1"
        var reads = 0
        var opens = 0
        var readable = true
        suspend fun scan() = cache.scan("artifact", 2, { version }, {
            opens++
            if (!readable) throw IOException("read access revoked")
            object : ByteArrayInputStream(bytes) {
                override fun read(buffer: ByteArray, offset: Int, length: Int): Int =
                    super.read(buffer, offset, length).also { if (it > 0) reads += it }
            }
        })
        val original = scan()
        assertNotNull(original)
        val coldReads = reads
        assertEquals(original, scan())
        assertEquals(coldReads, reads)
        assertEquals(2, opens)
        for (changed in listOf("new-inode", "same-size-new-ctime", "new-mtime")) {
            version = changed
            assertEquals(original, scan())
        }
        assertEquals(4 * coldReads, reads)
        readable = false
        assertThrows(IOException::class.java) { runBlocking { scan() } }
        readable = true
        version = null
        scan(); scan()
        assertEquals(6 * coldReads, reads)
    }

    @Test fun `expiry and layer count require another scan and malformed results are not cached`() = runBlocking {
        var time = 0L
        var calls = 0
        val cache = ModelSupplyGgufScans(clockMs = { time }, maxAgeMs = 100,
            scanner = { input, count -> calls++; GgufLayerWeightReader.scan(input, count) })
        suspend fun scan(layers: Int = 2) = cache.scan("artifact", layers, { "same-version" }, { bytes.inputStream() })
        assertNotNull(scan())
        time = 99
        assertNotNull(scan())
        assertEquals(1, calls)
        time = 100
        assertNotNull(scan())
        assertEquals(2, calls)
        repeat(2) { assertNull(scan(1)) }
        assertEquals(4, calls)
    }

    @Test fun `mutation or cancellation during metadata read never seeds reusable results`(): Unit = runBlocking {
        var version = "before"
        var mutate = true
        var calls = 0
        val cache = ModelSupplyGgufScans(scanner = { input, count ->
            calls++
            if (mutate) version = "after"
            GgufLayerWeightReader.scan(input, count)
        })
        suspend fun scan() = cache.scan("artifact", 2, { version }, { bytes.inputStream() })
        assertThrows(IOException::class.java) { runBlocking { scan() } }
        mutate = false
        assertNotNull(scan())
        assertNotNull(scan())
        assertEquals(2, calls)
        assertThrows(CancellationException::class.java) {
            runBlocking { cache.scan("artifact", 2, { version }, { bytes.inputStream() },
                checkCurrent = { throw CancellationException("retired") }) }
        }
    }

    @Test fun `concurrent scans share work even when another entry exceeds the cache bound`() = runBlocking {
        val entered = CountDownLatch(1)
        val release = CountDownLatch(1)
        val calls = AtomicInteger()
        val cache = ModelSupplyGgufScans(maxEntries = 1, scanner = { input, count ->
            if (calls.incrementAndGet() == 1) {
                entered.countDown()
                check(release.await(10, TimeUnit.SECONDS))
            }
            GgufLayerWeightReader.scan(input, count)
        })
        suspend fun scan(key: String) = cache.scan(key, 2, { "same-version" }, { bytes.inputStream() })
        val first = async(Dispatchers.IO) { scan("artifact") }
        try {
            assertTrue(entered.await(10, TimeUnit.SECONDS))
            assertNotNull(scan("other"))
            val followers = List(4) { async(start = CoroutineStart.UNDISPATCHED) { scan("artifact") } }
            release.countDown()
            assertTrue(followers.awaitAll().all { it == first.await() })
        } finally { release.countDown() }
        first.await()
        assertEquals(2, calls.get())
    }

    @Test fun `cold metadata parsing uses buffered bulk IO instead of scalar file reads`() = runBlocking {
        var bulkReads = 0
        val cache = ModelSupplyGgufScans()
        val result = cache.scan("artifact", 2, { "stable" }, {
            object : ByteArrayInputStream(bytes) {
                override fun read(): Int = error("unbuffered scalar read")
                override fun read(buffer: ByteArray, offset: Int, length: Int): Int {
                    bulkReads++
                    return super.read(buffer, offset, length)
                }
            }
        })
        assertNotNull(result)
        assertEquals(1, bulkReads)
    }

    @Test fun `inventory flights coalesce overlapping requests and later calls scan again`() = runBlocking {
        val flights = ModelSupplyScanFlights<String, Int>()
        val entered = CompletableDeferred<Unit>()
        val release = CompletableDeferred<Unit>()
        var calls = 0
        val first = async { flights.run("same") { calls++; entered.complete(Unit); release.await(); calls } }
        entered.await()
        val second = async(start = CoroutineStart.UNDISPATCHED) { flights.run("same") { error("duplicate scan") } }
        val cancelled = async(start = CoroutineStart.UNDISPATCHED) { flights.run("same") { error("duplicate scan") } }
        cancelled.cancelAndJoin()
        assertEquals(10, flights.run("different-selection") { 10 })
        release.complete(Unit)
        assertEquals(1, first.await())
        assertEquals(1, second.await())
        assertEquals(2, flights.run("same") { ++calls })
    }

    @Test fun `cancelled inventory owner releases the flight for a new scan`() = runBlocking {
        val flights = ModelSupplyScanFlights<String, Int>()
        val entered = CompletableDeferred<Unit>()
        val owner = async { flights.run("same") { entered.complete(Unit); CompletableDeferred<Int>().await() } }
        entered.await()
        val follower = async(start = CoroutineStart.UNDISPATCHED) { flights.run("same") { error("duplicate scan") } }
        owner.cancelAndJoin()
        assertThrows(CancellationException::class.java) { runBlocking { follower.await() } }
        assertEquals(2, withTimeout(3000) { flights.run("same") { 2 } })
    }

    @Test fun `failed inventory never leaves a reusable flight`() = runBlocking {
        val flights = ModelSupplyScanFlights<String, Int>()
        supervisorScope {
            val failed = async { flights.run("same") { throw IOException("unreadable") } }
            assertThrows(IOException::class.java) { runBlocking { failed.await() } }
        }
        assertEquals(2, flights.run("same") { 2 })
    }
}
