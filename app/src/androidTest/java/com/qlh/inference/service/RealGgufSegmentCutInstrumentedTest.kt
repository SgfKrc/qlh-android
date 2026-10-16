package com.qlh.inference.service

import android.os.Bundle
import android.os.Debug
import android.os.Process
import android.os.SystemClock
import android.util.Base64
import android.util.Log
import androidx.test.platform.app.InstrumentationRegistry
import com.google.gson.JsonArray
import com.google.gson.JsonObject
import com.qlh.inference.worker.GgufLayerWeightReader
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withTimeout
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File
import java.io.FilterInputStream
import java.io.InputStream
import java.security.MessageDigest
import java.util.UUID
import java.util.concurrent.atomic.AtomicLong

/** Explicit device acceptance: no Activity, service, receiver, model loader or supply publication. */
class RealGgufSegmentCutInstrumentedTest {
    @Test
    fun registeredNineBSourceProducesExactMasterMiddle() = runBlocking(Dispatchers.IO) {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val context = instrumentation.targetContext
        val arguments = InstrumentationRegistry.getArguments()
        val encoded = requireNotNull(arguments.getString(MANIFEST_ARGUMENT)) {
            "Pass the real master supply manifest as $MANIFEST_ARGUMENT (base64 UTF-8); no fixture or skip is allowed"
        }
        require(encoded.length in 1..131072) { "Manifest argument exceeds test bound" }
        val manifest = ModelSupplyManifest.parse(Base64.decode(encoded, Base64.DEFAULT).toString(Charsets.UTF_8))
        assertEquals("qwen3-5-9b", manifest.sourceModelId)
        assertEquals(SOURCE_SHA, manifest.sourceSha256)
        assertEquals(MANIFEST_SHA, manifest.manifestSha256)
        assertEquals(WEIGHT_SHA, manifest.weight.sha256)
        assertEquals(WEIGHT_BYTES, manifest.weight.size)
        assertEquals(8, manifest.start)
        assertEquals(12, manifest.end)
        assertEquals(32, manifest.json["total_layers"].asInt)
        assertEquals("middle", manifest.json["segment_mode"].asString)
        val timeoutMs = arguments.getString("ticket13_timeout_ms")?.toLongOrNull() ?: 1_800_000L
        require(timeoutMs in 60_000L..3_600_000L) { "Invalid acceptance timeout" }

        val cacheRoot = File(context.cacheDir.canonicalFile, "ticket13-test")
        assertTrue("Cannot create test-only cache root", cacheRoot.isDirectory || cacheRoot.mkdir())
        assertEquals(context.cacheDir.canonicalFile, cacheRoot.canonicalFile.parentFile)
        val runDirectory = File(cacheRoot, "middle-${UUID.randomUUID()}")
        assertTrue("Fresh output directory required", runDirectory.mkdir())
        val output = File(runDirectory, "segment.gguf")
        assertFalse("Never reuse a prior output", output.exists())
        val reportFile = File(runDirectory, "result.json")
        val report = JsonObject().apply {
            addProperty("schema_version", 1)
            addProperty("kind", "ticket13.real_android_kotlin_cut.v1")
            addProperty("status", "running")
            addProperty("package", context.packageName)
            addProperty("pid", Process.myPid())
            addProperty("tool", GgufSegmentCutter.TOOL)
            addProperty("manifest_sha256", manifest.manifestSha256)
            addProperty("source_model_sha256", manifest.sourceSha256)
            addProperty("expected_weight_sha256", manifest.weight.sha256)
            addProperty("expected_weight_bytes", manifest.weight.size)
            addProperty("output_path", output.absolutePath)
            addProperty("production_publication_requested", false)
            addProperty("native_loader_requested", false)
            addProperty("cache_reuse_allowed", false)
            add("layer_range", JsonArray().apply { add(manifest.start); add(manifest.end) })
        }
        val started = SystemClock.elapsedRealtime()
        fun event(phase: String) {
            report.addProperty("phase", phase)
            report.addProperty("elapsed_ms", SystemClock.elapsedRealtime() - started)
            reportFile.writeText(report.toString(), Charsets.UTF_8)
            Log.i(TAG, "$phase elapsed_ms=${SystemClock.elapsedRealtime() - started}")
            instrumentation.sendStatus(2, Bundle().apply { putString("stream", "$TAG $phase\n") })
        }
        val registryFile = File(context.filesDir, "local-cut/local-cut-sources.json")
        assertTrue("Existing registered source required", registryFile.isFile)
        val registryBefore = digest(registryFile)
        val grantsBefore = context.contentResolver.persistedUriPermissions.map {
            "${it.uri}|${it.isReadPermission}|${it.isWritePermission}|${it.persistedTime}"
        }.sorted()
        val supplyRoot = File(context.filesDir, "models/supply")
        val supplyBefore = supplySnapshot(supplyRoot)
        val settingsFile = File(context.filesDir, "datastore/qlh_settings.preferences_pb")
        val settingsBefore = settingsFile.takeIf { it.isFile }?.let { digest(it) }
        val nativeBefore = modelMappings()
        val readBytes = AtomicLong()
        val sourceOpens = AtomicLong()
        var peakJavaBytes = 0L
        var peakNativeBytes = 0L
        fun sampleMemory() {
            val runtime = Runtime.getRuntime()
            peakJavaBytes = maxOf(peakJavaBytes, runtime.totalMemory() - runtime.freeMemory())
            peakNativeBytes = maxOf(peakNativeBytes, Debug.getNativeHeapAllocatedSize())
        }
        try {
            assertTrue("Run only after model retirement; model/native mappings found: $nativeBefore", nativeBefore.isEmpty())
            event("resolve_registered_read_only_source")
            withTimeout(timeoutMs) {
                val source = ModelManager(context).localCutSources.resolve(manifest) { true }
                assertEquals(SOURCE_SHA, source.sha256)
                assertEquals(manifest.json["tokenizer_sha256"].asString, source.tokenizerSha256)
                report.addProperty("source_size_bytes", source.size)
                val registry = com.google.gson.JsonParser.parseString(registryFile.readText(Charsets.UTF_8)).asJsonObject
                val registrations = registry.getAsJsonArray("sources").filter {
                    val entry = it.asJsonObject
                    entry["source_model_id"].asString == manifest.sourceModelId &&
                        entry["source_model_sha256"].asString == SOURCE_SHA
                }
                assertTrue("Expected existing SAF registration", registrations.any {
                    it.asJsonObject["reference"].asString.startsWith("content://")
                })
                assertTrue("Existing SAF read grant required", grantsBefore.any { it.contains("|true|") })
                val tracked = source.copy(open = {
                    sourceOpens.incrementAndGet()
                    CountingInput(source.open(), readBytes)
                })
                val cutStarted = SystemClock.elapsedRealtime()
                var lastProgress = cutStarted
                var builtBytes = 0L
                event("cutting_fresh_output")
                GgufSegmentCutter.cut(tracked, manifest, output, { true }) { built ->
                    builtBytes = maxOf(builtBytes, built)
                    sampleMemory()
                    val now = SystemClock.elapsedRealtime()
                    if (now - lastProgress >= 5_000L) {
                        lastProgress = now
                        report.addProperty("built_bytes", builtBytes)
                        event("cutting_fresh_output")
                    }
                }
                report.addProperty("cut_elapsed_ms", SystemClock.elapsedRealtime() - cutStarted)
                report.addProperty("built_bytes", builtBytes)
                assertEquals("Cutter must generate the complete master weight", WEIGHT_BYTES, builtBytes)
                assertEquals(WEIGHT_BYTES, output.length())
                event("hashing_and_inspecting_output")
                val actualSha = digest(output)
                report.addProperty("actual_weight_sha256", actualSha)
                report.addProperty("actual_weight_bytes", output.length())
                assertEquals("Exact real Python/master bytes required", WEIGHT_SHA, actualSha)
                val layout = output.inputStream().buffered(64 * 1024).use { GgufSegmentCutter.inspect(it) }
                assertEquals("Only four decoder blocks", 4, layout.blocks)
                assertEquals("No MTP blocks", 0, layout.nextn)
                assertEquals(4, layout.layers)
                assertEquals(manifest.json["hidden_size"].asInt, layout.hidden)
                val blockIndices = layout.tensors.mapNotNull {
                    Regex("^blk\\.(\\d+)\\.").find(it.name)?.groupValues?.get(1)?.toInt()
                }.toSet()
                assertEquals(setOf(0, 1, 2, 3), blockIndices)
                val scan = requireNotNull(output.inputStream().buffered(64 * 1024).use {
                    GgufLayerWeightReader.scan(it, 4)
                }) { "Output GGUF reader rejected header" }
                assertEquals(manifest.localLayerBytes, scan.layerWeightBytes)
                assertEquals(source.tokenizerSha256, scan.tokenizerSha256)
                assertEquals(4, scan.ggufMetadata.getValue("block_count").asInt)
                report.addProperty("decoder_layers", layout.layers)
                report.addProperty("mtp_layers", layout.nextn)
                report.addProperty("tokenizer_sha256", scan.tokenizerSha256)
                report.add("gguf_metadata", JsonObject().apply { scan.ggufMetadata.forEach { (key, value) -> add(key, value) } })
                assertTrue("Real source payload must have been streamed", readBytes.get() >= source.size * 2)
            }
            sampleMemory()
            val nativeAfter = modelMappings()
            report.add("model_native_mappings_before", JsonArray().apply { nativeBefore.forEach { add(it) } })
            report.add("model_native_mappings_after", JsonArray().apply { nativeAfter.forEach { add(it) } })
            assertTrue("Test must not load any GGUF or model JNI library", nativeAfter.isEmpty())
            assertEquals("Source registry changed", registryBefore, digest(registryFile))
            assertEquals("SAF grants changed", grantsBefore, context.contentResolver.persistedUriPermissions.map {
                "${it.uri}|${it.isReadPermission}|${it.isWritePermission}|${it.persistedTime}"
            }.sorted())
            assertEquals("Production supply changed", supplyBefore, supplySnapshot(supplyRoot))
            assertEquals("Settings/identity metadata changed", settingsBefore, settingsFile.takeIf { it.isFile }?.let { digest(it) })
            report.addProperty("registry_unchanged", true)
            report.addProperty("saf_grants_unchanged", true)
            report.addProperty("production_supply_unchanged", true)
            report.addProperty("settings_unchanged", true)
            report.addProperty("source_stream_open_count", sourceOpens.get())
            report.addProperty("source_stream_read_bytes", readBytes.get())
            report.addProperty("sampled_peak_java_heap_bytes", peakJavaBytes)
            report.addProperty("sampled_peak_native_heap_bytes", peakNativeBytes)
            report.addProperty("status", "passed")
            event("complete")
            instrumentation.sendStatus(2, Bundle().apply {
                putString("stream", "$TAG RESULT ${report}\n")
            })
        } catch (failure: Throwable) {
            report.addProperty("status", "failed")
            report.addProperty("failure_type", failure.javaClass.name)
            report.addProperty("failure", failure.message.orEmpty())
            event("failed")
            throw failure
        }
        Unit
    }

    private suspend fun digest(file: File): String {
        val digest = MessageDigest.getInstance("SHA-256")
        val buffer = ByteArray(256 * 1024)
        file.inputStream().use { input ->
            while (true) {
                currentCoroutineContext().ensureActive()
                val count = input.read(buffer)
                if (count < 0) break
                digest.update(buffer, 0, count)
            }
        }
        return digest.digest().joinToString("") { "%02x".format(it.toInt() and 255) }
    }

    private fun supplySnapshot(root: File): List<String> = if (!root.exists()) emptyList() else
        root.walkTopDown().map {
            "${it.relativeTo(root).path}|${it.isDirectory}|${it.length()}|${it.lastModified()}"
        }.sorted().toList()

    private fun modelMappings(): List<String> = File("/proc/self/maps").useLines { lines ->
        lines.filter {
            it.contains(".gguf", true) || it.contains("libqlh_llama_jni.so") ||
                it.contains("libllama.so") || it.contains("libggml")
        }.toList()
    }

    private class CountingInput(input: InputStream, private val bytes: AtomicLong) : FilterInputStream(input) {
        override fun read(): Int = `in`.read().also { if (it >= 0) bytes.incrementAndGet() }
        override fun read(buffer: ByteArray, offset: Int, length: Int): Int =
            `in`.read(buffer, offset, length).also { if (it > 0) bytes.addAndGet(it.toLong()) }
    }

    private companion object {
        const val TAG = "Ticket13RealCut"
        const val MANIFEST_ARGUMENT = "ticket13_supply_manifest_base64"
        const val SOURCE_SHA = "ada47e14d93f7ca03e30bac750c330fd96b81a29fbe3333e0d0dcc5ebc14b82c"
        const val MANIFEST_SHA = "e4e9ad9bdb227210699a8f73bd4f93bf1db2f4f40f0dce8e6b1e296ac2949b21"
        const val WEIGHT_SHA = "089944ca61e8a801420cc021c9775be03c38da0208d9fc823878a089995fc20d"
        const val WEIGHT_BYTES = 1_926_059_296L
    }
}