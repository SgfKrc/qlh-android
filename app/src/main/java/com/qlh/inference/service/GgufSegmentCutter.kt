package com.qlh.inference.service

import com.google.gson.JsonArray
import com.google.gson.JsonElement
import com.google.gson.JsonPrimitive
import com.qlh.inference.worker.GgufLayerWeightReader
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.ensureActive
import java.io.DataInputStream
import java.io.EOFException
import java.io.File
import java.io.FilterInputStream
import java.io.InputStream
import java.io.RandomAccessFile
import java.security.DigestInputStream
import java.security.MessageDigest

/** Reopenable read-only source, including SAF providers that return a pipe. Never passed to JNI. */
internal data class LocalGgufSource(
    val modelId: String,
    val sha256: String,
    val tokenizerSha256: String,
    val size: Long,
    val open: () -> InputStream,
)

/** Bounded header inventory plus a 256 KiB payload buffer. No mmap, tensor decoding or model loader. */
internal object GgufSegmentCutter {
    const val TOOL = "android_gguf_stream_v1"
    private const val CHUNK = 256 * 1024
    private const val HEADER_LIMIT = 64L * 1024 * 1024
    private val layerArraySuffixes = listOf("attention.recurrent_layers", "attention.head_count",
        "attention.head_count_kv", "attention.key_length", "full_attention_interval")
    private val blockName = Regex("^blk\\.(0|[1-9][0-9]*)\\.(.+)$")

    data class Metadata(val name: String, val offset: Long, val size: Long, val type: Int,
        val value: Long? = null, val text: String? = null, val arrayType: Int? = null,
        val arrayCount: Long = 0, val arrayOffset: Long = 0)
    data class Tensor(val name: String, val dimensions: List<Long>, val type: Int, val offset: Long, val size: Long)
    data class Layout(val metadata: List<Metadata>, val tensors: List<Tensor>, val dataOffset: Long,
        val architecture: String, val blocks: Int, val nextn: Int, val hidden: Int,
        val scan: GgufLayerWeightReader.ScanResult?) {
        val layers get() = blocks - nextn
    }
    data class Inspection(val layout: Layout, val sha256: String, val tokenizerSha256: String)

    private class Header(input: InputStream, val check: () -> Unit) : FilterInputStream(input) {
        var position = 0L
        private val skipBuffer = ByteArray(8192)
        override fun read(): Int {
            check(); requireHeader(1)
            return `in`.read().also { if (it >= 0) position++ }
        }
        override fun read(b: ByteArray, off: Int, len: Int): Int {
            check(); requireHeader(len.toLong())
            return `in`.read(b, off, len).also { if (it > 0) position += it }
        }
        override fun skip(n: Long): Long {
            check(); requireHeader(n)
            var left = n
            while (left > 0) {
                val consumed = read(skipBuffer, 0, minOf(left, skipBuffer.size.toLong()).toInt())
                if (consumed < 0) break
                left -= consumed
            }
            return n - left
        }
        private fun requireHeader(n: Long) = checkSupply(n >= 0 && position + n <= HEADER_LIMIT,
            "SUPPLY_LOCAL_CUT_UNSUPPORTED", "GGUF header exceeds limit")
    }
    private fun DataInputStream.i32() = Integer.reverseBytes(readInt())
    private fun DataInputStream.i64() = java.lang.Long.reverseBytes(readLong())
    private fun DataInputStream.string(): String {
        val n = i64()
        checkSupply(n in 0..1_048_576, "SUPPLY_LOCAL_CUT_UNSUPPORTED", "GGUF string length")
        return ByteArray(n.toInt()).also { readFully(it) }.toString(Charsets.UTF_8)
    }
    private fun skip(input: InputStream, count: Long) {
        checkSupply(count >= 0, "SUPPLY_LOCAL_CUT_UNSUPPORTED", "negative GGUF span")
        var left = count
        while (left > 0) {
            val n = input.skip(minOf(left, CHUNK.toLong()))
            if (n > 0) left -= n else { if (input.read() < 0) throw EOFException(); left-- }
        }
    }
    private fun width(type: Int): Int = when (type) {
        0, 1, 7 -> 1; 2, 3 -> 2; 4, 5, 6 -> 4; 10, 11, 12 -> 8
        else -> throw ModelSupplyException("SUPPLY_LOCAL_CUT_UNSUPPORTED", "GGUF scalar type")
    }
    private fun DataInputStream.integer(type: Int): Long? = when (type) {
        0 -> readUnsignedByte().toLong(); 1 -> readByte().toLong()
        2 -> (java.lang.Short.reverseBytes(readShort()).toInt() and 65535).toLong()
        3 -> java.lang.Short.reverseBytes(readShort()).toLong()
        4 -> i32().toLong() and 0xffffffffL; 5 -> i32().toLong()
        10, 11 -> i64().also { checkSupply(it >= 0 || type == 11, "SUPPLY_LOCAL_CUT_UNSUPPORTED", "GGUF uint64") }
        else -> { skip(this, width(type).toLong()); null }
    }
    private fun DataInputStream.skipValue(type: Int) {
        when (type) {
            8 -> { val n = i64(); checkSupply(n in 0..16_777_216, "SUPPLY_LOCAL_CUT_UNSUPPORTED", "GGUF string"); skip(this, n) }
            9 -> {
                val sub = i32(); val n = i64()
                checkSupply(n in 0..1_000_000 && sub != 9, "SUPPLY_LOCAL_CUT_UNSUPPORTED", "GGUF array")
                repeat(n.toInt()) { skipValue(sub) }
            }
            else -> skip(this, width(type).toLong())
        }
    }

    fun inspect(input: InputStream, check: () -> Unit = {}): Layout {
        val header = Header(input, check)
        val data = DataInputStream(header)
        checkSupply(data.i32() == 0x46554747 && data.i32() in 2..3, "SUPPLY_LOCAL_CUT_UNSUPPORTED", "GGUF version/endian")
        val tensors = data.i64(); val fields = data.i64()
        checkSupply(tensors in 1..100_000 && fields in 1..100_000, "SUPPLY_LOCAL_CUT_UNSUPPORTED", "GGUF counts")
        val metadata = ArrayList<Metadata>()
        val contractFields = mutableMapOf<String, JsonElement>()
        val tokenizerFields = mutableMapOf<String, ByteArray>()
        val keys = HashSet<String>()
        repeat(fields.toInt()) {
            val start = header.position
            val name = data.string()
            checkSupply(keys.add(name) && !name.startsWith("split."), "SUPPLY_LOCAL_CUT_UNSUPPORTED", "duplicate/split GGUF")
            val type = data.i32()
            val tokenizerDigest = if (name.startsWith("tokenizer.")) MessageDigest.getInstance("SHA-256").apply {
                update(ByteArray(4) { index -> (type ushr (index * 8)).toByte() })
            } else null
            val valueInput = tokenizerDigest?.let { digest ->
                DataInputStream(object : DigestInputStream(data, digest) {
                    private val buffer = ByteArray(8192)
                    override fun skip(count: Long): Long =
                        read(buffer, 0, minOf(count, buffer.size.toLong()).toInt()).coerceAtLeast(0).toLong()
                })
            } ?: data
            val contract = GgufLayerWeightReader.isContractMetadata(name)
            if (contract) checkSupply(contractFields.size < 128, "SUPPLY_LOCAL_CUT_UNSUPPORTED", "too many GGUF contract fields")
            var value: Long? = null; var text: String? = null
            var arrayType: Int? = null; var arrayCount = 0L; var arrayOffset = 0L
            if (name == "general.architecture") {
                checkSupply(type == 8, "SUPPLY_LOCAL_CUT_UNSUPPORTED", "GGUF architecture type")
                text = valueInput.string()
                checkSupply(text.isNotBlank() && text.length <= 128, "SUPPLY_LOCAL_CUT_UNSUPPORTED", "GGUF architecture")
                contractFields[name] = JsonPrimitive(text)
            } else if (type == 9) {
                arrayType = valueInput.i32(); arrayCount = valueInput.i64(); arrayOffset = header.position
                checkSupply(arrayCount in 0..1_000_000 && arrayType != 9, "SUPPLY_LOCAL_CUT_UNSUPPORTED", "GGUF array")
                if (contract) {
                    checkSupply(arrayCount <= 1024 && arrayType in setOf(0, 1, 2, 3, 4, 5, 10, 11),
                        "SUPPLY_GGUF_METADATA_MISMATCH", "GGUF contract integer array")
                    contractFields[name] = JsonArray().apply {
                        repeat(arrayCount.toInt()) { add(with(GgufLayerWeightReader) { valueInput.readMetadataInteger(arrayType) }) }
                    }
                } else repeat(arrayCount.toInt()) { valueInput.skipValue(arrayType) }
            } else if (contract) {
                val field = with(GgufLayerWeightReader) { valueInput.readMetadataInteger(type) }
                contractFields[name] = field
                value = field.asLong
            } else if (type != 8) value = valueInput.integer(type)
            else valueInput.skipValue(type)
            tokenizerDigest?.let { tokenizerFields[name] = it.digest() }
            metadata += Metadata(name, start, header.position - start, type, value, text, arrayType, arrayCount, arrayOffset)
        }
        val arch = metadata.singleOrNull { it.name == "general.architecture" }?.text
            ?: throw ModelSupplyException("SUPPLY_LOCAL_CUT_UNSUPPORTED", "GGUF architecture missing")
        fun number(suffix: String, default: Long? = null): Int {
            val n = metadata.singleOrNull { it.name == "$arch.$suffix" }?.value ?: default
            checkSupply(n != null && n in 0..Int.MAX_VALUE, "SUPPLY_LOCAL_CUT_UNSUPPORTED", "GGUF $suffix")
            return n!!.toInt()
        }
        val blocks = number("block_count"); val nextn = number("nextn_predict_layers", 0); val hidden = number("embedding_length")
        checkSupply(blocks in 1..1024 && nextn in 0 until blocks && hidden > 0,
            "SUPPLY_LOCAL_CUT_UNSUPPORTED", "GGUF decoder metadata")
        val alignment = metadata.singleOrNull { it.name == "general.alignment" }?.value ?: 32L
        // Existing Python writer always emits 32-byte alignment; reject other layouts truthfully.
        checkSupply(alignment == 32L, "SUPPLY_LOCAL_CUT_UNSUPPORTED", "GGUF alignment is not supported by supply writer")
        val entries = ArrayList<Tensor>()
        val names = HashSet<String>(); val totals = LongArray(blocks)
        repeat(tensors.toInt()) {
            val name = data.string(); val dimensions = data.i32()
            checkSupply(names.add(name) && dimensions in 1..4, "SUPPLY_LOCAL_CUT_UNSUPPORTED", "GGUF tensor directory")
            val shape = List(dimensions) { data.i64() }
            val type = data.i32(); val offset = data.i64()
            val size = GgufLayerWeightReader.tensorByteSize(type, shape)
            checkSupply(offset >= 0 && offset % alignment == 0L && size > 0, "SUPPLY_LOCAL_CUT_UNSUPPORTED", "GGUF tensor offset")
            if (name.startsWith("blk.")) {
                val match = blockName.matchEntire(name)
                val index = match?.groupValues?.get(1)?.toIntOrNull()
                checkSupply(index != null && index in 0 until blocks, "SUPPLY_LOCAL_CUT_UNSUPPORTED", "GGUF block name")
                totals[index!!] = Math.addExact(totals[index], size)
            }
            entries += Tensor(name, shape, type, offset, size)
        }
        checkSupply(totals.take(blocks - nextn).all { it > 0 }, "SUPPLY_LOCAL_CUT_UNSUPPORTED", "missing decoder block")
        var previousEnd = 0L
        for (entry in entries.sortedBy { it.offset }) {
            checkSupply(entry.offset >= previousEnd, "SUPPLY_LOCAL_CUT_UNSUPPORTED", "overlapping tensor payloads")
            previousEnd = Math.addExact(entry.offset, entry.size)
        }
        for (field in metadata.filter { layerArraySuffixes.any(it.name::endsWith) }) {
            if (field.type == 9) checkSupply(field.arrayCount == (if (field.name.endsWith("attention.recurrent_layers")) blocks - nextn else blocks).toLong()
                && field.arrayType in setOf(0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 12), "SUPPLY_LOCAL_CUT_UNSUPPORTED", "GGUF per-layer array")
            else checkSupply(!field.name.endsWith("attention.recurrent_layers"), "SUPPLY_LOCAL_CUT_UNSUPPORTED", "GGUF recurrent array")
        }
        val scan = totals.takeIf { it.all { size -> size > 0 } }?.let {
            GgufLayerWeightReader.ScanResult(it.toList(), GgufLayerWeightReader.tokenizerSha256(tokenizerFields),
                GgufLayerWeightReader.normalizeMetadata(contractFields))
        }
        return Layout(metadata, entries, pad(header.position), arch, blocks, nextn, hidden, scan)
    }

    suspend fun inspectSource(source: LocalGgufSource, stillCurrent: suspend () -> Boolean): Inspection {
        checkCurrent(stillCurrent)
        val job = currentCoroutineContext()
        val digest = MessageDigest.getInstance("SHA-256")
        var size = 0L
        val counted = object : FilterInputStream(DigestInputStream(source.open(), digest)) {
            override fun read(): Int = `in`.read().also { if (it >= 0) consumed(1) }
            override fun read(buffer: ByteArray, offset: Int, length: Int): Int =
                `in`.read(buffer, offset, length).also { if (it > 0) consumed(it) }
            private fun consumed(count: Int) {
                job.ensureActive()
                size = Math.addExact(size, count.toLong())
                checkSupply(size <= source.size, "SUPPLY_SOURCE_DIGEST_MISMATCH", "source grew")
            }
        }
        val layout = counted.buffered(64 * 1024).use { input ->
            val inspected = inspect(input) { job.ensureActive() }
            val maximum = inspected.tensors.maxOf { Math.addExact(inspected.dataOffset, Math.addExact(it.offset, it.size)) }
            checkSupply(maximum <= source.size, "SUPPLY_LOCAL_CUT_UNSUPPORTED", "truncated source GGUF")
            val scan = inspected.scan
            checkSupply(scan != null && scan.tokenizerSha256 != null &&
                GgufMetadataContract.matches(scan.ggufMetadata, com.google.gson.JsonObject().apply {
                    scan.ggufMetadata.forEach { (key, value) -> add(key, value) }
                }, inspected.blocks), "SUPPLY_GGUF_METADATA_MISMATCH", "source metadata/tokenizer")
            val buffer = ByteArray(CHUNK)
            while (true) {
                checkCurrent(stillCurrent)
                if (input.read(buffer) < 0) break
            }
            inspected
        }
        checkCurrent(stillCurrent)
        checkSupply(size == source.size, "SUPPLY_SOURCE_DIGEST_MISMATCH", "source size changed")
        return Inspection(layout, digest.digest().joinToString("") { "%02x".format(it.toInt() and 255) }, layout.scan!!.tokenizerSha256!!)
    }

    suspend fun verify(source: LocalGgufSource, stillCurrent: suspend () -> Boolean): Layout {
        val inspected = inspectSource(source, stillCurrent)
        checkSupply(inspected.tokenizerSha256 == source.tokenizerSha256, "SUPPLY_TOKENIZER_DIGEST_MISMATCH", "source tokenizer")
        checkSupply(inspected.sha256 == source.sha256, "SUPPLY_SOURCE_DIGEST_MISMATCH", "source changed")
        return inspected.layout
    }

    suspend fun hash(source: LocalGgufSource, stillCurrent: suspend () -> Boolean): String {
        val digest = MessageDigest.getInstance("SHA-256")
        var size = 0L
        source.open().use { input ->
            val buffer = ByteArray(CHUNK)
            while (true) {
                checkCurrent(stillCurrent)
                val n = input.read(buffer); if (n < 0) break
                size = Math.addExact(size, n.toLong())
                checkSupply(size <= source.size, "SUPPLY_SOURCE_DIGEST_MISMATCH", "source grew")
                digest.update(buffer, 0, n)
            }
        }
        checkSupply(size == source.size, "SUPPLY_SOURCE_DIGEST_MISMATCH", "source size changed")
        return digest.digest().joinToString("") { "%02x".format(it.toInt() and 255) }
    }

    suspend fun cut(source: LocalGgufSource, manifest: ModelSupplyManifest, output: File,
        stillCurrent: suspend () -> Boolean, progress: suspend (Long) -> Unit) {
        checkSupply(source.sha256 == manifest.sourceSha256 && source.tokenizerSha256 == manifest.json["tokenizer_sha256"].asString,
            "SUPPLY_LOCAL_SOURCE_UNAVAILABLE", "source identity")
        val layout = verify(source, stillCurrent)
        checkSupply(layout.layers == manifest.json["total_layers"].asInt && layout.hidden == manifest.json["hidden_size"].asInt,
            "SUPPLY_SOURCE_METADATA_MISMATCH", "source shape")
        val start = manifest.start; val end = manifest.end
        checkSupply(start < end && end <= layout.layers && end - start < layout.layers,
            "SUPPLY_FULL_MODEL_FORBIDDEN", "proper decoder segment required")
        val selected = layout.tensors.mapNotNull { tensor ->
            if (layout.architecture == "qwen35" && end < layout.layers && tensor.name == "output.weight") {
                return@mapNotNull null
            }
            val match = blockName.matchEntire(tensor.name)
            if (match == null) tensor else {
                val index = match.groupValues[1].toInt()
                if (index in start until end) tensor.copy(name = "blk.${index - start}.${match.groupValues[2]}") else null
            }
        }
        // The output keeps every non-block tensor, and excludes all out-of-range decoder/MTP blocks.
        val totals = LongArray(end - start)
        selected.forEach { tensor -> blockName.matchEntire(tensor.name)?.let {
            val index = it.groupValues[1].toInt(); totals[index] = Math.addExact(totals[index], tensor.size)
        } }
        checkSupply(totals.toList() == manifest.localLayerBytes, "SUPPLY_LAYER_BYTES_MISMATCH", "selected source layer bytes")
        val fields = listOf(layout.metadata.single { it.name == "general.architecture" }) +
            layout.metadata.filter { it.name != "general.architecture" && !it.name.startsWith("GGUF.") }
        val buffer = ByteArray(CHUNK)
        RandomAccessFile(output, "rw").use { out ->
            out.setLength(0)
            out.write("GGUF".toByteArray(Charsets.US_ASCII)); out.i32(3); out.i64(selected.size.toLong()); out.i64(fields.size.toLong())
            // Stream raw tokenizer and unrelated KVs; only rewrite the same fields as cut_layers.py.
            val architecture = fields.first()
            out.string(architecture.name); out.i32(8); out.string(layout.architecture)
            source.open().use { input ->
                var position = 0L
                for (field in fields.drop(1)) {
                    checkCurrent(stillCurrent)
                    skipChecked(input, field.offset - position, stillCurrent, buffer)
                    val scalar = when (field.name) {
                        "${layout.architecture}.block_count" -> (end - start).toLong()
                        "${layout.architecture}.nextn_predict_layers" -> if (layout.nextn > 0) 0L else null
                        else -> null
                    }
                    if (scalar != null) {
                        out.string(field.name); out.i32(field.type); out.integer(field.type, scalar)
                        skipChecked(input, field.size, stillCurrent, buffer)
                    } else if (field.type == 9 && layerArraySuffixes.any(field.name::endsWith)) {
                        out.string(field.name); out.i32(9); out.i32(field.arrayType!!); out.i64((end - start).toLong())
                        val sliceOffset = field.arrayOffset + start.toLong() * width(field.arrayType)
                        val sliceSize = (end - start).toLong() * width(field.arrayType)
                        skipChecked(input, sliceOffset - field.offset, stillCurrent, buffer)
                        copy(input, out, sliceSize, buffer, stillCurrent)
                        skipChecked(input, field.offset + field.size - sliceOffset - sliceSize, stillCurrent, buffer)
                    } else copy(input, out, field.size, buffer, stillCurrent)
                    position = field.offset + field.size
                }
            }
            var offset = 0L
            val destinations = selected.map { tensor ->
                out.string(tensor.name); out.i32(tensor.dimensions.size)
                tensor.dimensions.forEach { out.i64(it) }; out.i32(tensor.type); out.i64(offset)
                offset.also { offset = Math.addExact(offset, pad(tensor.size)) }
            }
            val dataStart = pad(out.filePointer)
            zeros(out, dataStart - out.filePointer)
            val outputSize = Math.addExact(dataStart, offset)
            checkSupply(outputSize == manifest.weight.size, "SUPPLY_ARTIFACT_DIGEST_MISMATCH", "cut layout differs from master's writer")
            checkSupply(output.parentFile!!.usableSpace >= outputSize, "SUPPLY_DISK_INSUFFICIENT", "insufficient cut storage")
            out.setLength(outputSize)
            var built = dataStart
            // Read payloads in source offset order, writing their directory-order output positions.
            // This also works with non-seekable SAF streams; skip by reading bounded chunks.
            source.open().use { input ->
                var sourcePosition = 0L
                for ((index, tensor) in selected.withIndex().sortedBy { it.value.offset }) {
                    val absolute = layout.dataOffset + tensor.offset
                    skipChecked(input, absolute - sourcePosition, stillCurrent, buffer)
                    out.seek(dataStart + destinations[index])
                    copy(input, out, tensor.size, buffer, stillCurrent) { copied -> progress(built + copied) }
                    zeros(out, pad(tensor.size) - tensor.size)
                    sourcePosition = absolute + tensor.size
                    built += pad(tensor.size)
                    progress(built)
                }
            }
            out.fd.sync()
        }
        // A mutable external provider must still contain the registered full source after cutting.
        checkSupply(hash(source, stillCurrent) == source.sha256, "SUPPLY_SOURCE_DIGEST_MISMATCH", "source changed during cut")
        checkCurrent(stillCurrent)
    }
    private suspend fun checkCurrent(current: suspend () -> Boolean) {
        currentCoroutineContext().ensureActive()
        checkSupply(current(), "SUPPLY_CANCELLED", "source operation retired")
    }
    private suspend fun skipChecked(input: InputStream, count: Long, current: suspend () -> Boolean, buffer: ByteArray) {
        checkSupply(count >= 0, "SUPPLY_LOCAL_CUT_UNSUPPORTED", "source ordering")
        var left = count
        while (left > 0) {
            checkCurrent(current)
            // Read rather than InputStream.skip: SAF pipes need no seek support.
            val n = input.read(buffer, 0, minOf(left, buffer.size.toLong()).toInt())
            if (n < 0) throw EOFException()
            left -= n
        }
    }
    private suspend fun copy(input: InputStream, output: RandomAccessFile, length: Long, buffer: ByteArray,
        current: suspend () -> Boolean, progress: suspend (Long) -> Unit = {}) {
        var copied = 0L
        while (copied < length) {
            checkCurrent(current)
            val n = input.read(buffer, 0, minOf(length - copied, buffer.size.toLong()).toInt())
            if (n < 0) throw EOFException()
            output.write(buffer, 0, n); copied += n; progress(copied)
        }
    }
    private fun pad(n: Long): Long = Math.addExact(n, 31) / 32 * 32
    private fun zeros(out: RandomAccessFile, n: Long) { repeat(n.toInt()) { out.write(0) } }
    private fun RandomAccessFile.i32(n: Int) = writeInt(Integer.reverseBytes(n))
    private fun RandomAccessFile.i64(n: Long) = writeLong(java.lang.Long.reverseBytes(n))
    private fun RandomAccessFile.string(s: String) { val bytes = s.toByteArray(Charsets.UTF_8); i64(bytes.size.toLong()); write(bytes) }
    private fun RandomAccessFile.integer(type: Int, n: Long) {
        when (width(type)) { 1 -> write(n.toInt()); 2 -> writeShort(java.lang.Short.reverseBytes(n.toShort()).toInt()); 4 -> i32(n.toInt()); 8 -> i64(n) }
    }
}
