package com.qlh.inference.worker

import com.google.gson.JsonParser

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.ByteArrayInputStream
import java.io.ByteArrayOutputStream
import java.io.DataOutputStream
import java.io.IOException
import java.io.InputStream
import java.io.SequenceInputStream

class GgufLayerWeightReaderTest {
    private data class Metadata(val key: String, val type: Int, val write: DataOutputStream.() -> Unit)

    private data class Tensor(val name: String, val dimensions: List<Long>, val type: Int)

    private fun gguf(
        tensors: List<Tensor>,
        firstToken: String = "foo",
        withTokenizer: Boolean = true,
        metadata: List<Metadata> = emptyList(),
        architectureLast: Boolean = false,
    ): ByteArray = ByteArrayOutputStream().also { bytes ->
        DataOutputStream(bytes).use { data ->
            data.write("GGUF".toByteArray(Charsets.US_ASCII))
            data.intLE(3)
            data.longLE(tensors.size.toLong())
            data.longLE((if (withTokenizer) 3L else 1L) + metadata.size)
            fun architecture() {
                data.string("general.architecture")
                data.intLE(8)
                data.string("qwen35")
            }
            if (!architectureLast) architecture()
            metadata.forEach { field ->
                data.string(field.key)
                data.intLE(field.type)
                field.write(data)
            }
            if (architectureLast) architecture()
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
        val header = gguf(listOf(Tensor("blk.0.attn_q.weight", listOf(32), 2)), metadata = listOf(
            Metadata("qwen35.block_count", 4) { intLE(1) },
            Metadata("qwen35.attention.head_count_kv", 4) { intLE(2) },
            Metadata("qwen35.attention.key_length", 4) { intLE(128) },
            Metadata("qwen35.full_attention_interval", 4) { intLE(4) },
        ))
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

    @Test
    fun `contract metadata is normalized by architecture even when architecture is last`() {
        val header = gguf(listOf(Tensor("blk.0.attn_q.weight", listOf(32), 2)), metadata = listOf(
            Metadata("qwen35.block_count", 4) { intLE(1) },
            Metadata("qwen35.attention.head_count_kv", 9) { intLE(5); longLE(2); intLE(2); intLE(4) },
            Metadata("qwen35.attention.key_length", 10) { longLE(-1) },
            Metadata("other.full_attention_interval", 4) { intLE(99) },
            Metadata("qwen35.embedding_length", 4) { intLE(2048) },
            Metadata("qwen35.nextn_predict_layers", 4) { intLE(0) },
        ), architectureLast = true)
        val scan = GgufLayerWeightReader.scan(header.inputStream(), 1)!!
        val expected = JsonParser.parseString("""{
            "architecture":"qwen35","block_count":1,"head_count_kv":[2,4],
            "key_length":18446744073709551615,"full_attention_interval":null
        }""").asJsonObject
        assertEquals(expected.keySet(), scan.ggufMetadata.keys)
        expected.entrySet().forEach { (key, value) -> assertEquals(value.toString(), scan.ggufMetadata.getValue(key).toString()) }
        assertEquals("e91e8882e1b84d172788b3f4bfe54af0e9d83cc95c8037a25342618212911800", scan.tokenizerSha256)
    }

    @Test
    fun `all signed and unsigned GGUF integer widths preserve exact contract values`() {
        val fields = listOf(
            Metadata("qwen35.attention.head_count_kv", 0) { writeByte(255) } to "255",
            Metadata("qwen35.attention.head_count_kv", 1) { writeByte(-128) } to "-128",
            Metadata("qwen35.attention.head_count_kv", 2) { writeShort(java.lang.Short.reverseBytes((-1).toShort()).toInt()) } to "65535",
            Metadata("qwen35.attention.head_count_kv", 3) { writeShort(java.lang.Short.reverseBytes(Short.MIN_VALUE).toInt()) } to "-32768",
            Metadata("qwen35.attention.head_count_kv", 4) { intLE(-1) } to "4294967295",
            Metadata("qwen35.attention.head_count_kv", 5) { intLE(Int.MIN_VALUE) } to "-2147483648",
            Metadata("qwen35.attention.head_count_kv", 10) { longLE(-1) } to "18446744073709551615",
            Metadata("qwen35.attention.head_count_kv", 11) { longLE(Long.MIN_VALUE) } to "-9223372036854775808",
        )
        for ((field, expected) in fields) {
            val scan = GgufLayerWeightReader.scan(gguf(listOf(Tensor("blk.0.attn_q.weight", listOf(32), 2)),
                metadata = listOf(field)).inputStream(), 1)!!
            assertEquals(expected, scan.ggufMetadata.getValue("head_count_kv").toString())
        }
    }

    @Test
    fun `contract booleans floats nested arrays oversized arrays and duplicate KVs fail closed`() {
        val invalid = listOf(
            listOf(Metadata("qwen35.block_count", 7) { writeByte(1) }),
            listOf(Metadata("qwen35.block_count", 6) { intLE(1.0f.toRawBits()) }),
            listOf(Metadata("qwen35.block_count", 12) { longLE(1.0.toRawBits()) }),
            listOf(Metadata("qwen35.block_count", 8) { string("1") }),
            listOf(Metadata("qwen35.attention.key_length", 9) { intLE(7); longLE(1); writeByte(1) }),
            listOf(Metadata("qwen35.attention.key_length", 9) { intLE(6); longLE(1); intLE(1.0f.toRawBits()) }),
            listOf(Metadata("qwen35.attention.key_length", 9) { intLE(9); longLE(0) }),
            listOf(Metadata("qwen35.attention.key_length", 9) { intLE(4); longLE(1025) }),
            listOf(Metadata("qwen35.block_count", 4) { intLE(1) }, Metadata("qwen35.block_count", 4) { intLE(1) }),
            listOf(Metadata("general.architecture", 8) { string("qwen35") }),
        )
        for (fields in invalid) assertNull(GgufLayerWeightReader.scan(
            gguf(listOf(Tensor("blk.0.attn_q.weight", listOf(32), 2)), metadata = fields).inputStream(), 1))
    }

    /** Generate tokenizer strings lazily so the test itself does not retain their array. */
    private fun streamedTokenizer(count: Int, tokenLength: Int): Pair<InputStream, Long> {
        val prefix = ByteArrayOutputStream().also { bytes -> DataOutputStream(bytes).use { data ->
            data.write("GGUF".toByteArray()); data.intLE(3); data.longLE(1); data.longLE(3)
            data.string("general.architecture"); data.intLE(8); data.string("qwen35")
            data.string("qwen35.block_count"); data.intLE(4); data.intLE(1)
            data.string("tokenizer.ggml.tokens"); data.intLE(9); data.intLE(8); data.longLE(count.toLong())
        } }.toByteArray()
        val token = ByteArrayOutputStream().also { bytes -> DataOutputStream(bytes).use { data ->
            data.string("x".repeat(tokenLength))
        } }.toByteArray()
        val tokens = object : InputStream() {
            var position = 0L
            val total = count.toLong() * token.size
            override fun read(): Int = if (position == total) -1 else token[(position++ % token.size).toInt()].toInt() and 255
            override fun read(buffer: ByteArray, offset: Int, length: Int): Int {
                if (position == total) return -1
                val size = minOf(length.toLong(), total - position).toInt()
                repeat(size) { buffer[offset + it] = token[(position++ % token.size).toInt()] }
                return size
            }
        }
        val tensor = ByteArrayOutputStream().also { bytes -> DataOutputStream(bytes).use { data ->
            data.string("blk.0.attn_q.weight"); data.intLE(1); data.longLE(32); data.intLE(2); data.longLE(0)
        } }.toByteArray()
        return SequenceInputStream(SequenceInputStream(prefix.inputStream(), tokens), tensor.inputStream()) to
            (prefix.size + count.toLong() * token.size + tensor.size)
    }

    @Test
    fun `large tokenizer array stays streamed and only five normalized metadata values survive`() {
        val (input, size) = streamedTokenizer(100_000, 128)
        assertTrue(size > 12 * 1024 * 1024)
        val scan = input.use { GgufLayerWeightReader.scan(it, 1) }!!
        assertNotNull(scan.tokenizerSha256)
        assertEquals(setOf("architecture", "block_count", "head_count_kv", "key_length", "full_attention_interval"), scan.ggufMetadata.keys)
        assertEquals(listOf(18L), scan.layerWeightBytes)
    }

    @Test
    fun `total header byte budget also bounds streamed tokenizer values`() {
        val (input, size) = streamedTokenizer(600_000, 128)
        assertTrue(size > 64L * 1024 * 1024)
        assertNull(input.use { GgufLayerWeightReader.scan(it, 1) })
    }

}
