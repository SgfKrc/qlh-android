package com.qlh.inference.worker

import com.google.gson.JsonArray
import com.google.gson.JsonElement
import com.google.gson.JsonNull
import com.google.gson.JsonPrimitive
import java.io.DataInputStream
import java.io.FilterInputStream
import java.io.IOException
import java.io.InputStream
import java.math.BigInteger
import java.security.DigestInputStream
import java.security.MessageDigest

/** Streams the GGUF header and tensor directory; never reads model tensor data. */
internal object GgufLayerWeightReader {
    data class ScanResult(
        val layerWeightBytes: List<Long>,
        val tokenizerSha256: String?,
        val ggufMetadata: Map<String, JsonElement> = emptyMap(),
    )

    private val metadataSuffixes = mapOf(
        "block_count" to "block_count",
        "head_count_kv" to "attention.head_count_kv",
        "key_length" to "attention.key_length",
        "full_attention_interval" to "full_attention_interval",
    )

    private val blockTypes = mapOf(
        0 to (1 to 4), 1 to (1 to 2), 2 to (32 to 18), 3 to (32 to 20),
        6 to (32 to 22), 7 to (32 to 24), 8 to (32 to 34), 9 to (32 to 36),
        10 to (256 to 84), 11 to (256 to 110), 12 to (256 to 144),
        13 to (256 to 176), 14 to (256 to 210), 15 to (256 to 292),
        16 to (256 to 66), 17 to (256 to 74), 18 to (256 to 98),
        19 to (256 to 50), 20 to (32 to 18), 21 to (256 to 110),
        22 to (256 to 82), 23 to (256 to 136), 24 to (1 to 1),
        25 to (1 to 2), 26 to (1 to 4), 27 to (1 to 8),
        28 to (1 to 8), 29 to (256 to 56), 30 to (1 to 2),
    )
    private val blockName = Regex("^blk\\.(\\d+)\\.")

    fun read(input: InputStream, layerCount: Int): List<Long>? =
        scan(input, layerCount)?.layerWeightBytes

    fun matchesTokenizerSha256(declared: String?, scan: ScanResult?): Boolean =
        declared == null || scan?.tokenizerSha256?.equals(declared, ignoreCase = true) == true

    fun scan(input: InputStream, layerCount: Int): ScanResult? {
        if (layerCount <= 0 || layerCount > 1024) return null
        return try {
            val data = DataInputStream(HeaderInputStream(input))
            val magic = ByteArray(4)
            data.readFully(magic)
            if (!magic.contentEquals("GGUF".toByteArray(Charsets.US_ASCII))) return null
            if (data.intLE() !in 2..3) return null
            val tensorCount = data.longLE()
            val metadataCount = data.longLE()
            if (tensorCount !in 1L..100_000L || metadataCount !in 0L..100_000L) return null
            val scratch = ByteArray(8192)
            val tokenizerFields = mutableMapOf<String, ByteArray>()
            // Keep only contract KVs, never tokenizer values or arbitrary metadata arrays.
            // Architecture may occur after its qualified fields in a valid GGUF header.
            val metadataFields = mutableMapOf<String, JsonElement>()
            val metadataNames = HashSet<String>()
            repeat(metadataCount.toInt()) {
                val key = data.readString()
                if (!metadataNames.add(key)) throw IOException("duplicate GGUF metadata key")
                if (key == "general.architecture") {
                    if (data.intLE() != 8) throw IOException("invalid GGUF architecture type")
                    val architecture = data.readString()
                    if (architecture.isEmpty() || architecture.length > 128) throw IOException("invalid GGUF architecture")
                    metadataFields[key] = JsonPrimitive(architecture)
                } else if (!key.startsWith("tokenizer.") && metadataSuffixes.values.any { key.endsWith(".$it") }) {
                    if (metadataFields.size >= 128) throw IOException("too many GGUF contract fields")
                    metadataFields[key] = data.readMetadataInteger(data.intLE())
                } else if (key.startsWith("tokenizer.")) {
                    if (tokenizerFields.containsKey(key)) throw IOException("duplicate GGUF tokenizer key")
                    val digest = MessageDigest.getInstance("SHA-256")
                    val hashed = DataInputStream(DigestInputStream(data, digest))
                    hashed.skipValue(hashed.intLE(), true, scratch)
                    tokenizerFields[key] = digest.digest()
                } else {
                    data.skipValue(data.intLE(), false, scratch)
                }
            }
            val tokenizerSha256 = tokenizerFields.takeIf { it.isNotEmpty() }?.let { fields ->
                val digest = MessageDigest.getInstance("SHA-256")
                fields.keys.sorted().forEach { key ->
                    digest.update(key.toByteArray(Charsets.UTF_8))
                    digest.update(0.toByte())
                    digest.update(requireNotNull(fields[key]))
                }
                digest.digest().joinToString("") { byte ->
                    "%02x".format(byte.toInt() and 0xff)
                }
            }
            val metadata = linkedMapOf<String, JsonElement>()
            metadataFields["general.architecture"]?.let { architecture ->
                metadata["architecture"] = architecture
                for ((key, suffix) in metadataSuffixes) {
                    // Explicit null is absence in this architecture's header, never a wildcard.
                    metadata[key] = metadataFields["${architecture.asString}.$suffix"] ?: JsonNull.INSTANCE
                }
            }
            val totals = LongArray(layerCount)
            val names = HashSet<String>()
            repeat(tensorCount.toInt()) {
                val name = data.readString()
                if (!names.add(name)) throw IOException("duplicate GGUF tensor")
                val dimensionCount = data.intLE()
                if (dimensionCount !in 1..4) throw IOException("invalid GGUF dimensions")
                var elements = 1L
                repeat(dimensionCount) {
                    val dimension = data.longLE()
                    if (dimension <= 0) throw IOException("invalid GGUF dimension")
                    elements = Math.multiplyExact(elements, dimension)
                }
                val type = blockTypes[data.intLE()] ?: throw IOException("unknown GGUF tensor type")
                if (data.longLE() < 0 || elements % type.first != 0L) {
                    throw IOException("invalid GGUF tensor")
                }
                val bytes = Math.multiplyExact(elements / type.first, type.second.toLong())
                val block = blockName.find(name)
                val index = block?.groupValues?.get(1)?.toIntOrNull()
                if (block != null && (index == null || index !in 0 until layerCount)) {
                    throw IOException("GGUF contains decoder blocks outside the requested segment")
                }
                if (index != null) {
                    totals[index] = Math.addExact(totals[index], bytes)
                }
            }
            totals.takeIf { it.all { size -> size > 0L } }?.let { sizes ->
                ScanResult(sizes.toList(), tokenizerSha256, metadata)
            }
        } catch (_: IOException) {
            null
        } catch (_: ArithmeticException) {
            null
        } catch (_: IllegalArgumentException) {
            null
        }
    }

    /** Bound reads and skips together, including streamed tokenizer bytes and tensor names. */
    private class HeaderInputStream(input: InputStream) : FilterInputStream(input) {
        private var remaining = 64L * 1024 * 1024

        private fun requireBytes(size: Long) {
            if (size < 0 || size > remaining) throw IOException("GGUF header exceeds limit")
        }

        override fun read(): Int {
            requireBytes(1)
            return `in`.read().also { if (it >= 0) remaining-- }
        }

        override fun read(buffer: ByteArray, offset: Int, length: Int): Int {
            requireBytes(length.toLong())
            return `in`.read(buffer, offset, length).also { if (it > 0) remaining -= it }
        }

        override fun skip(count: Long): Long {
            requireBytes(count)
            return `in`.skip(count).also { remaining -= it }
        }
    }

    private fun DataInputStream.readMetadataInteger(type: Int): JsonElement = when (type) {
        0 -> JsonPrimitive(readUnsignedByte())
        1 -> JsonPrimitive(readByte())
        2 -> JsonPrimitive(java.lang.Short.reverseBytes(readShort()).toInt() and 0xffff)
        3 -> JsonPrimitive(java.lang.Short.reverseBytes(readShort()))
        4 -> JsonPrimitive(intLE().toLong() and 0xffffffffL)
        5 -> JsonPrimitive(intLE())
        10 -> JsonPrimitive(BigInteger(java.lang.Long.toUnsignedString(longLE())))
        11 -> JsonPrimitive(longLE())
        9 -> {
            val elementType = intLE()
            val count = longLE()
            if (count !in 0L..1024L || elementType !in setOf(0, 1, 2, 3, 4, 5, 10, 11)) {
                throw IOException("invalid GGUF contract integer array")
            }
            JsonArray().apply { repeat(count.toInt()) { add(readMetadataInteger(elementType)) } }
        }
        else -> throw IOException("GGUF contract value must be an integer or integer array")
    }

    private fun DataInputStream.intLE(): Int = Integer.reverseBytes(readInt())
    private fun DataInputStream.longLE(): Long = java.lang.Long.reverseBytes(readLong())

    private fun DataInputStream.readString(): String {
        val size = longLE()
        if (size !in 0L..1_048_576L) throw IOException("invalid GGUF string size")
        return ByteArray(size.toInt()).also { readFully(it) }.toString(Charsets.UTF_8)
    }

    private fun DataInputStream.skipString(hashed: Boolean, scratch: ByteArray) {
        val size = longLE()
        if (size !in 0L..16_777_216L) throw IOException("invalid GGUF string size")
        skipFully(size, hashed, scratch)
    }

    private fun DataInputStream.skipValue(type: Int, hashed: Boolean, scratch: ByteArray) {
        when (type) {
            0, 1, 7 -> skipFully(1, hashed, scratch)
            2, 3 -> skipFully(2, hashed, scratch)
            4, 5, 6 -> skipFully(4, hashed, scratch)
            10, 11, 12 -> skipFully(8, hashed, scratch)
            8 -> skipString(hashed, scratch)
            9 -> {
                val elementType = intLE()
                val count = longLE()
                if (count !in 0L..1_000_000L || elementType == 9) {
                    throw IOException("invalid GGUF array")
                }
                repeat(count.toInt()) { skipValue(elementType, hashed, scratch) }
            }
            else -> throw IOException("unknown GGUF metadata type")
        }
    }

    private fun DataInputStream.skipFully(size: Long, hashed: Boolean, scratch: ByteArray) {
        var remaining = size
        while (remaining > 0) {
            val consumed = if (hashed) {
                read(scratch, 0, minOf(remaining, scratch.size.toLong()).toInt()).toLong()
            } else {
                skip(remaining)
            }
            if (consumed > 0) {
                remaining -= consumed
            } else if (read() == -1) {
                throw IOException("truncated GGUF metadata")
            } else {
                remaining--
            }
        }
    }
}
