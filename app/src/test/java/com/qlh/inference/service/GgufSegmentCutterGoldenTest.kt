package com.qlh.inference.service

import com.qlh.inference.worker.GgufLayerWeightReader
import kotlinx.coroutines.runBlocking
import org.junit.Assert.*
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import org.junit.runner.RunWith
import org.junit.runners.Parameterized
import java.io.FilterInputStream
import java.io.InputStream

/** Golden bytes come from the real Python build_gguf_segment / cut_layers.py, not this writer. */
@RunWith(Parameterized::class)
class GgufSegmentCutterGoldenTest(private val kind: String, private val mode: String) {
    @get:Rule val temp = TemporaryFolder()
    companion object {
        @JvmStatic @Parameterized.Parameters(name = "{0}-{1}")
        fun cases(): List<Array<String>> = listOf("plain", "hybrid", "quantized").flatMap { kind ->
            listOf("head", "middle", "tail").map { arrayOf(kind, it) }
        }
    }
    @Test fun `streaming cut matches Python bytes and receipt metadata exactly including SAF pipes`() = runBlocking {
        val manifest = CutFixture.manifest(kind, mode)
        val source = CutFixture.source(kind, mode) { input -> object : FilterInputStream(input) {
            override fun skip(n: Long): Long = error("SAF pipe cannot seek")
            override fun read(b: ByteArray, off: Int, len: Int): Int {
                assertTrue("payload buffer must stay bounded", len <= 256 * 1024)
                return super.read(b, off, len)
            }
        } }
        val directory = temp.newFolder()
        val output = directory.resolve(manifest.weight.path)
        val progress = mutableListOf<Long>()
        GgufSegmentCutter.cut(source, manifest, output, { true }, { progress += it })
        val expected = CutFixture.bytes("$kind/$mode.gguf.fixture")
        assertArrayEquals("actual Python artifact bytes", expected, output.readBytes())
        assertEquals(manifest.artifactSha256, supplySha256(output.readBytes()))
        assertEquals(manifest.weight.size, output.length())
        assertTrue(progress.isNotEmpty())
        assertEquals(manifest.weight.size, progress.last())
        assertTrue(progress.zipWithNext().all { (a, b) -> a <= b })
        directory.resolve(manifest.native.path).writeBytes(CutFixture.bytes("$kind/$mode.native.json"))
        ModelSupplyReceiver(directory.parentFile!!, "unused").validateFiles(directory, manifest)
        val layout = output.inputStream().use { GgufSegmentCutter.inspect(it) }
        assertEquals(manifest.end - manifest.start, layout.layers)
        assertEquals(0, layout.nextn)
        assertFalse(layout.tensors.any { it.name.contains("nextn") })
        assertTrue("unknown non-block tensors must survive", layout.tensors.any { it.name == "unknown.support.weight" })
        val scan = output.inputStream().use { GgufLayerWeightReader.scan(it, manifest.end - manifest.start) }!!
        assertEquals(source.tokenizerSha256, scan.tokenizerSha256)
    }
}

internal object CutFixture {
    fun bytes(path: String): ByteArray = requireNotNull(javaClass.classLoader!!.getResourceAsStream("local-cut/$path")) {
        "missing Python golden fixture: $path"
    }.use { it.readBytes() }
    fun manifest(kind: String = "hybrid", mode: String = "tail") =
        ModelSupplyManifest.parse(bytes("$kind/$mode.supply.json").toString(Charsets.UTF_8))
    fun source(kind: String = "hybrid", mode: String = "tail", wrap: (InputStream) -> InputStream = { it }): LocalGgufSource {
        val manifest = manifest(kind, mode)
        val bytes = bytes("$kind/source.gguf.fixture")
        return LocalGgufSource("golden", manifest.sourceSha256, manifest.json["tokenizer_sha256"].asString,
            bytes.size.toLong()) { wrap(bytes.inputStream()) }
    }
}
