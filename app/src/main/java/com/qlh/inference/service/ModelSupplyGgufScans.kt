package com.qlh.inference.service

import com.qlh.inference.worker.GgufLayerWeightReader
import java.io.File
import java.io.IOException
import java.io.InputStream
import java.nio.file.Files
import java.nio.file.LinkOption
import java.nio.file.StandardOpenOption
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.sync.Mutex

internal class ModelSupplyGgufScans(
    private val clockMs: () -> Long = { System.nanoTime() / 1_000_000 },
    private val maxAgeMs: Long = 300_000L,
    private val maxEntries: Int = 64,
    private val fingerprint: (File) -> ModelFileFingerprint? = ::privateModelFingerprint,
    private val scanner: (InputStream, Int) -> GgufLayerWeightReader.ScanResult? = GgufLayerWeightReader::scan,
) {
    init { require(maxAgeMs > 0 && maxEntries > 0) }

    private class Entry {
        val mutex = Mutex()
        var users = 0
        var version: String? = null
        var result: GgufLayerWeightReader.ScanResult? = null
        var scannedAtMs = 0L
    }

    private val entries = LinkedHashMap<Pair<String, Int>, Entry>(16, 0.75f, true)

    suspend fun scan(
        file: File,
        layerCount: Int,
        checkCurrent: suspend () -> Unit = {},
    ): GgufLayerWeightReader.ScanResult? {
        ModelSupplyPaths.check(file)
        val identity = fingerprint(file)
        val key = identity?.let { "private:${it.device}:${it.inode}" } ?: "private:${file.absolutePath}"
        return scan(key, layerCount, version = {
            ModelSupplyPaths.check(file)
            fingerprint(file)?.toString()
        }, open = {
            Files.newInputStream(file.toPath(), StandardOpenOption.READ, LinkOption.NOFOLLOW_LINKS)
        }, checkCurrent = checkCurrent)
    }

    suspend fun scan(
        key: String,
        layerCount: Int,
        version: () -> String?,
        open: () -> InputStream,
        checkCurrent: suspend () -> Unit = {},
    ): GgufLayerWeightReader.ScanResult? {
        currentCoroutineContext().ensureActive()
        checkCurrent()
        val entry = synchronized(entries) {
            entries.getOrPut(key to layerCount) { Entry() }.also { it.users++ }
        }
        try {
            entry.mutex.lock()
            try {
                currentCoroutineContext().ensureActive()
                checkCurrent()
                val before = version()
                val now = clockMs()
                val cached = entry.result
                val reuse = before != null && cached != null && entry.version == before &&
                    now >= entry.scannedAtMs && now - entry.scannedAtMs < maxAgeMs
                if (!reuse) {
                    entry.result = null
                    entry.version = null
                }
                val result = open().buffered(64 * 1024).use { input ->
                    if (reuse) cached else scanner(input, layerCount)
                }
                currentCoroutineContext().ensureActive()
                checkCurrent()
                if (before != null && before != version()) {
                    entry.result = null
                    entry.version = null
                    throw IOException("model artifact changed during GGUF metadata scan")
                }
                if (!reuse && before != null && result != null) {
                    entry.version = before
                    entry.result = result
                    entry.scannedAtMs = clockMs()
                }
                return result
            } finally {
                entry.mutex.unlock()
            }
        } finally {
            synchronized(entries) {
                entry.users--
                val iterator = entries.entries.iterator()
                while (entries.size > maxEntries && iterator.hasNext()) {
                    if (iterator.next().value.users == 0) iterator.remove()
                }
            }
        }
    }

    companion object {
        val shared = ModelSupplyGgufScans()
    }
}

internal class ModelSupplyScanFlights<Key, Value> {
    private val flights = mutableMapOf<Key, CompletableDeferred<Value>>()

    suspend fun run(key: Key, scan: suspend () -> Value): Value {
        currentCoroutineContext().ensureActive()
        val (flight, owner) = synchronized(flights) {
            flights[key]?.let { it to false } ?: CompletableDeferred<Value>().let {
                flights[key] = it
                it to true
            }
        }
        if (!owner) return flight.await()
        try {
            val result = scan()
            currentCoroutineContext().ensureActive()
            flight.complete(result)
            return result
        } catch (error: Throwable) {
            flight.completeExceptionally(error)
            throw error
        } finally {
            synchronized(flights) { if (flights[key] === flight) flights.remove(key) }
        }
    }
}
