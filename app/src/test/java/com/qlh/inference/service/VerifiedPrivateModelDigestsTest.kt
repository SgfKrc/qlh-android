package com.qlh.inference.service

import java.io.File
import java.io.IOException
import java.nio.file.Files
import java.nio.file.LinkOption
import java.nio.file.StandardCopyOption
import java.nio.file.attribute.BasicFileAttributes
import java.util.concurrent.TimeUnit
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.runBlocking
import org.junit.Assert.*
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder

internal class PrivateDigestTestFiles(private val anchors: File) {
    private data class Identity(val anchor: File, val inode: Long, var changed: Long = 0)
    private val identities = mutableListOf<Identity>()
    val reads = mutableListOf<File>()
    var reliable = true
    var beforeHash: (File) -> Unit = {}
    val digests = VerifiedPrivateModelDigests(fingerprint = ::fingerprint, hash = { file, checkCurrent ->
        reads += file
        beforeHash(file)
        hashPrivateModelFile(file, checkCurrent)
    })

    private fun identity(file: File): Identity = identities.firstOrNull {
        Files.isSameFile(it.anchor.toPath(), file.toPath())
    } ?: Identity(File(anchors, "inode-${identities.size}"), identities.size + 1L).also {
        Files.createLink(it.anchor.toPath(), file.toPath())
        identities += it
    }

    fun fingerprint(file: File): ModelFileFingerprint? {
        if (!reliable || !file.isFile) return null
        val observed = Files.readAttributes(file.toPath(), BasicFileAttributes::class.java, LinkOption.NOFOLLOW_LINKS)
        val identity = identity(file)
        val modified = observed.lastModifiedTime().to(TimeUnit.NANOSECONDS)
        return ModelFileFingerprint(1, identity.inode, observed.size(), modified / 1_000_000_000, 1,
            modified % 1_000_000_000, identity.changed)
    }

    fun changed(file: File) { identity(file).changed++ }

    fun link(source: File, target: File) {
        Files.createLink(target.toPath(), source.toPath())
        changed(source)
    }
}

class VerifiedPrivateModelDigestsTest {
    @get:Rule val temp = TemporaryFolder()

    private fun files() = PrivateDigestTestFiles(temp.newFolder())

    @Test
    fun `a verified inode shares one full read across hardlinks ctime changes and atomic rename`() = runBlocking {
        val files = files()
        val source = temp.newFile("source.gguf").apply { writeText("verified model bytes") }
        val original = files.digests.verify(source)
        val linked = File(temp.root, "linked.gguf")
        assertTrue(files.digests.linkVerified(source, linked, original, files::link))
        assertEquals(original.digest, files.digests.verify(linked).digest)
        assertEquals(original.digest, files.digests.verify(source).digest)
        val renamed = File(temp.root, "renamed.gguf")
        assertTrue(files.digests.moveVerified(linked, renamed, files.digests.verify(linked)))
        assertEquals(original.digest, files.digests.verify(renamed).digest)
        assertEquals(1, files.reads.size)
    }

    @Test
    fun `preserved mtime tampering still rehashes on ctime change and inode replacement`() = runBlocking {
        val files = files()
        val source = temp.newFile("source.gguf").apply { writeText("original") }
        val modified = Files.getLastModifiedTime(source.toPath())
        val original = files.digests.verify(source)
        source.writeText("modified")
        Files.setLastModifiedTime(source.toPath(), modified)
        files.changed(source)
        val changed = files.digests.verify(source)
        assertNotEquals(original.digest, changed.digest)
        val replacement = temp.newFile("replacement.gguf").apply { writeText("replaced") }
        Files.setLastModifiedTime(replacement.toPath(), modified)
        Files.move(replacement.toPath(), source.toPath(), StandardCopyOption.REPLACE_EXISTING)
        assertNotEquals(changed.digest, files.digests.verify(source).digest)
        assertEquals(3, files.reads.size)
    }

    @Test
    fun `runtime stamp refresh reuses a verified hardlink but rejects same size same mtime tampering`() = runBlocking {
        val files = files()
        val source = temp.newFile("source.gguf").apply { writeText("original") }
        val verified = files.digests.verify(source)
        val previous = requireNotNull(verified.fingerprint)
        assertTrue(files.digests.linkVerified(source, File(temp.root, "next.gguf"), verified, files::link))
        val linked = files.digests.reuseContentVersion(source, previous, verified.digest)
        assertNotNull(linked)
        assertNotEquals(previous, linked)
        assertEquals(1, files.reads.size)
        val modified = Files.getLastModifiedTime(source.toPath())
        source.writeText("modified")
        Files.setLastModifiedTime(source.toPath(), modified)
        files.changed(source)
        assertNull(files.digests.reuseContentVersion(source, requireNotNull(linked), verified.digest))
        assertEquals(2, files.reads.size)
    }

    @Test
    fun `source changed before link cannot migrate its old verification`() = runBlocking {
        val files = files()
        val source = temp.newFile("source.gguf").apply { writeText("original") }
        val verified = files.digests.verify(source)
        files.changed(source)
        val target = File(temp.root, "linked.gguf")
        assertThrows(IOException::class.java) {
            runBlocking { files.digests.linkVerified(source, target, verified, files::link) }
        }
        assertFalse(target.exists())
    }

    @Test
    fun `unknown file version always reads content again`() = runBlocking {
        val files = files().apply { reliable = false }
        val source = temp.newFile("source.gguf").apply { writeText("original") }
        assertEquals(files.digests.verify(source).digest, files.digests.verify(source).digest)
        assertEquals(2, files.reads.size)
    }

    @Test
    fun `hot verification still checks cancellation and regular file access`() = runBlocking {
        val files = files()
        val source = temp.newFile("source.gguf").apply { writeText("original") }
        files.digests.verify(source)
        assertThrows(CancellationException::class.java) {
            runBlocking { files.digests.verify(source, checkCurrent = { throw CancellationException("cancelled") }) }
        }
        assertTrue(source.delete())
        assertTrue(source.mkdir())
        assertThrows(IOException::class.java) { runBlocking { files.digests.verify(source) } }
        assertEquals(1, files.reads.size)
    }

    @Test
    fun `mutation during a full read never seeds a reusable digest`() = runBlocking {
        val files = files()
        val source = temp.newFile("source.gguf").apply { writeText("original") }
        files.beforeHash = { file -> file.writeText("modified"); files.changed(file) }
        assertThrows(IOException::class.java) { runBlocking { files.digests.verify(source) } }
        files.beforeHash = {}
        assertEquals(supplySha256("modified".toByteArray()), files.digests.verify(source).digest)
        files.digests.verify(source)
        assertEquals(2, files.reads.size)
    }

    @Test
    fun `forced fresh verification reads the bytes even with a deliberately unchanged version`() = runBlocking {
        val files = files()
        val source = temp.newFile("source.gguf").apply { writeText("original") }
        val modified = Files.getLastModifiedTime(source.toPath())
        val original = files.digests.verify(source)
        source.writeText("modified")
        Files.setLastModifiedTime(source.toPath(), modified)
        assertEquals(original.digest, files.digests.verify(source).digest)
        assertNotEquals(original.digest, files.digests.verify(source, fresh = true).digest)
        assertEquals(2, files.reads.size)
    }
}
