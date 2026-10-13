package com.qlh.inference.worker

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.ByteArrayInputStream
import java.io.ByteArrayOutputStream
import java.io.DataOutputStream
import java.io.IOException

class GgufLayerWeightReaderTest {
    private data class Tensor(val name: String, val dimensions: List<Long>, val type: Int)

    private fun gguf(
        tensors: List<Tensor>,
        firstToken: String = "foo",
        withTokenizer: Boolean = true,
    ): ByteArray = ByteArrayOutputStream().also { bytes ->
        DataOutputStream(bytes).use { data ->
            data.write("GGUF".toByteArray(Charsets.US_ASCII))
            data.intLE(3)
            data.longLE(tensors.size.toLong())
            data.longLE(if (withTokenizer) 3L else 1L)
            data.string("general.architecture")
            data.intLE(8)
            data.string("qwen35")
            if (withTokenizer) {
                data.string("tokenizer.ggml.tokens")
                data.intLE(9)
                data.intLE(8)
                data.longLE(2)
                data.string(firstToken)
                data.string("bar")
                data.string("tokenizer.ggml.bos_token_id")
                data.intLE(4)
                data.intLE(151643)
            }
            tensors.forEach { tensor ->
                data.string(tensor.name)
                data.intLE(tensor.dimensions.size)
                tensor.dimensions.forEach { data.longLE(it) }
                data.intLE(tensor.type)
                data.longLE(0)
            }
        }
    }.toByteArray()

    private fun DataOutputStream.intLE(value: Int) = writeInt(Integer.reverseBytes(value))
    private fun DataOutputStream.longLE(value: Long) = writeLong(java.lang.Long.reverseBytes(value))
    private fun DataOutputStream.string(value: String) {
        val bytes = value.toByteArray(Charsets.UTF_8)
        longLE(bytes.size.toLong())
        write(bytes)
    }

    @Test
    fun `quantized tensor bytes use type size without offset padding`() {
        val input = gguf(listOf(
            Tensor("token_embd.weight", listOf(32), 2),
            Tensor("blk.0.attn_q.weight", listOf(32), 2),
            Tensor("blk.0.ffn_up.weight", listOf(2, 32), 2),
            Tensor("blk.1.attn_q.weight", listOf(32), 1),
            Tensor("blk.1.attn_k.weight", listOf(256), 10),
        ))
        assertEquals(
            listOf(54L, 148L),
            GgufLayerWeightReader.read(ByteArrayInputStream(input), 2),
        )
    }

    @Test
    fun `missing truncated malformed or unknown tensor data never fabricates layer sizes`() {
        val valid = gguf(listOf(Tensor("blk.0.attn_q.weight", listOf(32), 2)))
        assertNull(GgufLayerWeightReader.read(ByteArrayInputStream(valid), 2))
        assertNull(GgufLayerWeightReader.read(ByteArrayInputStream(valid.dropLast(2).toByteArray()), 1))
        assertNull(GgufLayerWeightReader.read(ByteArrayInputStream(byteArrayOf(1, 2, 3)), 1))
        assertNull(GgufLayerWeightReader.read(ByteArrayInputStream(gguf(listOf(
            Tensor("blk.0.attn_q.weight", listOf(32), 99),
        ))), 1))
        assertNull(GgufLayerWeightReader.read(ByteArrayInputStream(gguf(listOf(
            Tensor("blk.0.attn_q.weight", listOf(31), 2),
        ))), 1))
        assertNull(GgufLayerWeightReader.read(ByteArrayInputStream(gguf(listOf(
            Tensor("blk.0.attn_q.weight", listOf(Long.MAX_VALUE, 2), 0),
        ))), 1))
    }

    @Test
    fun `tokenizer digest matches canonical python bytes not metadata insertion order`() {
        val tensor = Tensor("blk.0.attn_q.weight", listOf(32), 2)
        val scan = GgufLayerWeightReader.scan(ByteArrayInputStream(gguf(listOf(tensor))), 1)
        val expected = "e91e8882e1b84d172788b3f4bfe54af0e9d83cc95c8037a25342618212911800"
        assertEquals(expected, scan?.tokenizerSha256)
        assertEquals(listOf(18L), scan?.layerWeightBytes)
        assertTrue(GgufLayerWeightReader.matchesTokenizerSha256(expected.uppercase(), scan))
        assertFalse(GgufLayerWeightReader.matchesTokenizerSha256("a".repeat(64), scan))
        val changed = GgufLayerWeightReader.scan(
            ByteArrayInputStream(gguf(listOf(tensor), firstToken = "fob")), 1,
        )
        assertFalse(GgufLayerWeightReader.matchesTokenizerSha256(expected, changed))
        assertFalse(GgufLayerWeightReader.matchesTokenizerSha256(expected, null))
    }

    @Test
    fun `legacy manifest needs no tokenizer digest but new manifest fails closed`() {
        val tensor = Tensor("blk.0.attn_q.weight", listOf(32), 2)
        val noTokenizer = GgufLayerWeightReader.scan(
            ByteArrayInputStream(gguf(listOf(tensor), withTokenizer = false)), 1,
        )
        assertEquals(listOf(18L), noTokenizer?.layerWeightBytes)
        assertNull(noTokenizer?.tokenizerSha256)
        assertTrue(GgufLayerWeightReader.matchesTokenizerSha256(null, noTokenizer))
        assertTrue(GgufLayerWeightReader.matchesTokenizerSha256(null, null))
        assertFalse(GgufLayerWeightReader.matchesTokenizerSha256("a".repeat(64), noTokenizer))
    }

    @Test
    fun `whole model or overflow block index cannot masquerade as a cropped artifact`() {
        val full = (0..3).map { Tensor("blk.$it.attn_q.weight", listOf(32), 2) }
        assertNull(GgufLayerWeightReader.scan(ByteArrayInputStream(gguf(full)), 2))
        assertEquals(4, GgufLayerWeightReader.scan(ByteArrayInputStream(gguf(full)), 4)?.layerWeightBytes?.size)
        assertNull(GgufLayerWeightReader.scan(ByteArrayInputStream(gguf(listOf(
            Tensor("blk.0.attn_q.weight", listOf(32), 2),
            Tensor("blk.9999999999999999999999.attn_q.weight", listOf(32), 2),
        ))), 1))
    }

    @Test
    fun `tokenizer scan stops before model weight payload`() {
        val header = gguf(listOf(Tensor("blk.0.attn_q.weight", listOf(32), 2)))
        val guarded = object : ByteArrayInputStream(header + ByteArray(8192)) {
            override fun read(buffer: ByteArray, offset: Int, length: Int): Int {
                if (pos + length > header.size) throw IOException("tensor data was read")
                return super.read(buffer, offset, length)
            }

            override fun read(): Int {
                if (pos >= header.size) throw IOException("tensor data was read")
                return super.read()
            }

            override fun skip(amount: Long): Long {
                if (pos + amount > header.size) throw IOException("tensor data was skipped")
                return super.skip(amount)
            }
        }
        assertEquals(listOf(18L), GgufLayerWeightReader.scan(guarded, 1)?.layerWeightBytes)
    }
}
