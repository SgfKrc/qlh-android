package com.qlh.inference.service

import java.io.IOException
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Assert.assertThrows
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder

class ModelStorageScanTest {
    @get:Rule val temporaryFolder = TemporaryFolder()

    private data class Candidate(val parent: String, val name: String, val digest: String)

    @Test
    fun pairsOnlyWithinTheManifestDirectoryAndTriesAllSameNameCandidates() {
        val candidates = listOf(
            Candidate("root/B", "mid.gguf", "correct"),
            Candidate("root/A", "mid.gguf", "wrong"),
            Candidate("root/A", "MID.GGUF", "correct"),
        )
        val attempted = mutableListOf<Candidate>()
        val selected = pairLayerArtifact(
            manifestParentId = "root/A",
            artifactName = "mid.gguf",
            candidates = candidates,
            parentId = { it.parent },
            name = { it.name },
            verify = { attempted += it; it.digest == "correct" },
        )
        assertEquals(candidates[2] to true, selected)
        assertEquals(candidates.drop(1), attempted)
        assertEquals(null, pairLayerArtifact(
            "root/C", "mid.gguf", candidates, { it.parent }, { it.name }, { true },
        ))
        assertEquals(candidates[1] to false, pairLayerArtifact(
            "root/A", "mid.gguf", candidates, { it.parent }, { it.name }, { false },
        ))
    }

    @Test
    fun suspendPairingPreservesDirectoryAndDigestMatching() = runBlocking {
        val candidates = listOf(
            Candidate("A", "crop.gguf", "wrong"),
            Candidate("B", "crop.gguf", "correct"),
            Candidate("A", "CROP.GGUF", "correct"),
        )
        val selected = pairLayerArtifactVerified(
            "A", "crop.gguf", candidates, { it.parent }, { it.name },
            verifyArtifactDigest = true,
            verify = { it.digest == "correct" },
        )
        assertEquals(candidates.last() to true, selected)
    }

    @Test
    fun metadataOnlyPreflightDoesNotReadArtifactBytes() = runBlocking {
        val candidate = Candidate("A", "crop.gguf", "correct")
        val preliminary = pairLayerArtifactVerified(
            "A", "crop.gguf", listOf(candidate), { it.parent }, { it.name },
            verifyArtifactDigest = false,
            verify = { throw AssertionError("metadata preflight must not hash") },
        )
        val verified = pairLayerArtifactVerified(
            "A", "crop.gguf", listOf(candidate), { it.parent }, { it.name },
            verifyArtifactDigest = true,
            verify = { false },
        )
        assertEquals(candidate to true, preliminary)
        assertEquals(candidate to false, verified)
    }

    @Test
    fun scansVisibleArtifactsAndManifestsIndependently() {
        val directory = temporaryFolder.newFolder("models")
        directory.resolve("a.GGUF").writeText("artifact")
        directory.resolve("a.manifest.JSON").writeText("{}")
        directory.resolve("a.gguf.part").writeText("partial")
        directory.resolve("subdir").mkdir()
        directory.resolve("subdir/b.gguf").writeText("not directly in the root")

        assertEquals(listOf("a.GGUF"), scanModelFiles(directory, setOf(".gguf")).map { it.name })
        assertEquals(
            listOf("a.manifest.JSON"),
            scanModelFiles(directory, setOf(".json")).map { it.name },
        )
    }

    @Test
    fun missingRootIsEmptyButAFileRootIsAnError() {
        val missing = temporaryFolder.root.resolve("missing")
        assertEquals(emptyList<String>(), scanModelFiles(missing, setOf(".gguf")).map { it.name })
        val file = temporaryFolder.newFile("not-a-directory")
        assertThrows(IOException::class.java) { scanModelFiles(file, setOf(".gguf")) }
    }
}
