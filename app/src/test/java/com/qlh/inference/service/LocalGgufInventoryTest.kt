package com.qlh.inference.service

import com.google.gson.JsonParser
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withTimeout
import org.junit.Assert.*
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import java.io.ByteArrayInputStream
import java.io.IOException
import java.util.concurrent.atomic.AtomicBoolean

class LocalGgufInventoryTest {
    @get:Rule val temp = TemporaryFolder()

    private inner class Source(padding: Int = 0) {
        val directory = temp.newFolder()
        var bytes = CutFixture.bytes("hybrid/source.gguf.fixture") + ByteArray(padding)
        var version: String? = "1"
        var available = true
        var readable = true
        var opens = 0
        var closes = 0
        var readBytes = 0L
        var onOpen: () -> Unit = {}
        var onRead: () -> Unit = {}
        fun registry() = LocalGgufSourceRegistry(directory) {
            if (!available) emptyList() else listOf(
                LocalGgufSourceRegistry.Document("content://test/golden.gguf", bytes.size.toLong(), { version }) {
                    opens++
                    if (!readable) throw IOException("read grant revoked")
                    onOpen()
                    object : ByteArrayInputStream(bytes) {
                        override fun read(): Int = super.read().also { if (it >= 0) consumed(1) }
                        override fun read(buffer: ByteArray, offset: Int, length: Int): Int =
                            super.read(buffer, offset, length).also { if (it > 0) consumed(it) }
                        override fun skip(count: Long): Long = error("SAF pipe cannot seek")
                        override fun close() { closes++; super.close() }
                        private fun consumed(count: Int) { readBytes += count; onRead() }
                    }
                },
            )
        }
        fun hasCache() = directory.resolve("local-cut-digests.json").exists()
    }

    @Test fun `cold inventory reads once and verified identity survives inventory resolve registration and restart`() = runBlocking {
        val source = Source()
        val registry = source.registry()
        val identity = registry.inventory("golden", "", { true }).single()
        assertEquals(supplySha256(source.bytes), identity.sha256)
        assertEquals(CutFixture.source().tokenizerSha256, identity.tokenizerSha256)
        assertEquals(1, source.opens)
        assertEquals(source.bytes.size.toLong(), source.readBytes)
        assertEquals(identity.sha256, registry.resolve(CutFixture.manifest(), { true }).sha256)
        assertEquals(identity.sha256, source.registry().inventory("golden", "", { true }).single().sha256)
        assertEquals(identity.sha256, registry.register("content://test/golden.gguf", "golden").sha256)
        assertEquals(identity.sha256, source.registry().inventory("golden", "", { true }).single().sha256)
        assertEquals(source.bytes.size.toLong(), source.readBytes)
        assertEquals(source.opens, source.closes)
    }

    @Test fun `selected model filter opens no unrelated source`() = runBlocking {
        val bytes = CutFixture.bytes("hybrid/source.gguf.fixture")
        var reads = 0
        val registry = LocalGgufSourceRegistry(temp.newFolder()) {
            listOf(
                LocalGgufSourceRegistry.Document("content://test/other.gguf", bytes.size.toLong()) { throw AssertionError("unselected source opened") },
                LocalGgufSourceRegistry.Document("content://test/golden.gguf", bytes.size.toLong()) { reads++; bytes.inputStream() },
            )
        }
        assertEquals(CutFixture.manifest().sourceSha256, registry.inventory("golden", "", { true }).single().sha256)
        assertEquals(1, reads)
    }

    @Test fun `cold registration reads once and registered inventory shares its verified identity`() = runBlocking {
        val source = Source()
        val registered = source.registry().register("content://test/golden.gguf", "golden")
        assertEquals(supplySha256(source.bytes), registered.sha256)
        assertEquals(1, source.opens)
        assertEquals(registered.sha256, source.registry().inventory("golden", "", { true }).single().sha256)
        assertEquals(source.bytes.size.toLong(), source.readBytes)
        assertEquals(source.opens, source.closes)
    }

    @Test fun `file size change invalidates the persisted result`() = runBlocking {
        val source = Source()
        val original = source.registry().inventory("golden", "", { true }).single().sha256
        val before = source.readBytes
        source.bytes += byteArrayOf(1)
        assertTrue(source.registry().inventory("golden", original, { true }).isEmpty())
        assertEquals(source.bytes.size.toLong(), source.readBytes - before)
        assertEquals(supplySha256(source.bytes), source.registry().inventory("golden", "", { true }).single().sha256)
        assertEquals(source.bytes.size.toLong(), source.readBytes - before)
    }

    @Test fun `invalid header is not cached or followed by a full payload read`() = runBlocking {
        val source = Source(1024 * 1024)
        source.bytes[0] = 0
        repeat(2) { assertTrue(source.registry().inventory("golden", "", { true }).isEmpty()) }
        assertEquals(2, source.opens)
        assertEquals(2, source.closes)
        assertTrue(source.readBytes < source.bytes.size)
        assertFalse(source.hasCache())
    }

    @Test fun `same size edits and replacements invalidate persisted identities`() = runBlocking {
        val source = Source()
        val registry = source.registry()
        val original = registry.inventory("golden", "", { true }).single().sha256
        for (version in listOf("changed-mtime", "new-inode", "changed-ctime")) {
            source.bytes = source.bytes.copyOf().apply { this[lastIndex] = (this[lastIndex].toInt() xor 1).toByte() }
            source.version = version
            val before = source.readBytes
            val changed = source.registry().inventory("golden", "", { true }).single()
            assertEquals(supplySha256(source.bytes), changed.sha256)
            assertEquals(source.bytes.size.toLong(), source.readBytes - before)
        }
        assertTrue(registry.inventory("golden", original, { true }).isEmpty())
        assertEquals(source.opens, source.closes)
    }

    @Test fun `registered identity cannot survive a same size source edit`() = runBlocking {
        val source = Source()
        val registry = source.registry()
        registry.register("content://test/golden.gguf", "golden")
        source.bytes[source.bytes.lastIndex] = (source.bytes.last().toInt() xor 1).toByte()
        source.version = "2"
        assertTrue(registry.inventory("golden", "", { true }).isEmpty())
        assertTrue(source.registry().inventory("golden", "", { true }).isEmpty())
        assertEquals(2L * source.bytes.size, source.readBytes)
    }

    @Test fun `unversioned providers always read once per inventory and weak legacy cache is ignored`() = runBlocking {
        val source = Source()
        source.version = null
        source.directory.resolve("local-cut-digests.json").writeText("""{
            "schema_version":1,"entries":{"content://test/golden.gguf|${source.bytes.size}":{
                "sha256":"${"a".repeat(64)}","tokenizer_sha256":"${"b".repeat(64)}"}}
        }""")
        repeat(2) {
            assertEquals(supplySha256(source.bytes), source.registry().inventory("golden", "", { true }).single().sha256)
        }
        assertEquals(2, source.opens)
        assertEquals(2L * source.bytes.size, source.readBytes)
        source.version = "1"
        source.registry().inventory("golden", "", { true })
        val cache = JsonParser.parseString(source.directory.resolve("local-cut-digests.json").readText()).asJsonObject
        assertEquals(2, cache["schema_version"].asInt)
    }

    @Test fun `cached sources still require current enumeration and read access`() = runBlocking {
        val source = Source()
        val registry = source.registry()
        registry.inventory("golden", "", { true })
        source.available = false
        assertTrue(registry.inventory("golden", "", { true }).isEmpty())
        source.available = true
        source.readable = false
        assertTrue(registry.inventory("golden", "", { true }).isEmpty())
        source.readable = true
        assertEquals(1, registry.inventory("golden", "", { true }).size)
        assertEquals(source.bytes.size.toLong(), source.readBytes)
    }

    @Test fun `source version changing during read is never published or persisted`() = runBlocking {
        val source = Source()
        source.onRead = { source.version = "2" }
        assertTrue(source.registry().inventory("golden", "", { true }).isEmpty())
        assertFalse(source.hasCache())
        source.onRead = {}
        assertEquals(1, source.registry().inventory("golden", "", { true }).size)
        assertEquals(2L * source.bytes.size, source.readBytes)
    }

    @Test fun `retirement before reading during payload and on cache hit prevents publication`() = runBlocking {
        val source = Source(1024 * 1024)
        val registry = source.registry()
        expectCancelled { registry.inventory("golden", "", { false }) }
        assertEquals(0, source.opens)
        var current = true
        source.onRead = { current = false }
        expectCancelled { registry.inventory("golden", "", { current }) }
        assertTrue(source.readBytes < source.bytes.size)
        assertEquals(source.opens, source.closes)
        assertFalse(source.hasCache())
        source.onRead = {}
        registry.inventory("golden", "", { true })
        val before = source.readBytes
        current = true
        source.onOpen = { current = false }
        expectCancelled { registry.inventory("golden", "", { current }) }
        assertEquals(before, source.readBytes)
        assertEquals(source.opens, source.closes)
    }

    @Test fun `coroutine cancellation during metadata closes stream and never persists partial identity`() = runBlocking {
        val source = Source(1024 * 1024)
        val scope = CoroutineScope(SupervisorJob() + Dispatchers.Unconfined)
        val cancelled = AtomicBoolean(false)
        source.onRead = { scope.cancel() }
        val job = scope.launch {
            try {
                source.registry().inventory("golden", "", { true })
                fail("cancelled inventory returned")
            } catch (error: CancellationException) {
                cancelled.set(true)
                throw error
            }
        }
        withTimeout(3000) { job.join() }
        assertTrue(cancelled.get())
        assertTrue(source.readBytes < source.bytes.size)
        assertEquals(1, source.closes)
        assertFalse(source.hasCache())
    }

    private suspend fun expectCancelled(block: suspend () -> Unit) {
        try { block(); fail("retired inventory returned") }
        catch (error: ModelSupplyException) { assertEquals("SUPPLY_CANCELLED", error.reasonCode) }
    }
}
