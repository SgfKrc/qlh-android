package com.qlh.inference.service

import com.google.gson.JsonArray
import com.google.gson.JsonObject
import com.google.gson.JsonParser
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import java.io.File
import java.io.FileOutputStream
import java.io.InputStream
import java.nio.file.Files
import java.nio.file.StandardCopyOption
import java.util.concurrent.ConcurrentHashMap

internal interface LocalGgufSources {
    suspend fun inventory(modelId: String, sourceSha256: String, current: suspend () -> Boolean): List<LocalGgufSource>
    suspend fun resolve(manifest: ModelSupplyManifest, current: suspend () -> Boolean): LocalGgufSource
}

/** Explicit source registration is separate from cluster identity, selected inference model and SAF grants. */
internal class LocalGgufSourceRegistry(
    private val directory: File,
    private val documents: suspend () -> List<Document>,
) : LocalGgufSources {
    data class Document(val reference: String, val size: Long, val version: () -> String? = { null }, val open: () -> InputStream)
    private data class Registration(val modelId: String, val reference: String, val sha: String, val tokenizer: String, val size: Long)
    private val mutex get() = registryMutexes.computeIfAbsent(directory.canonicalPath) { Mutex() }
    // Shared across ModelManager instances in the UI and service; writes are tiny atomic replacements.
    private companion object {
        val diskLock = Any()
        val registryMutexes = ConcurrentHashMap<String, Mutex>()
        val idPattern = Regex("[A-Za-z0-9_.:-]{1,128}")
        val shaPattern = Regex("[0-9a-f]{64}")
        const val MARKER = "local-cut-sources.json"
        const val DIGEST_MARKER = "local-cut-digests.json"
        const val DIGEST_SCHEMA = 2
    }

    private fun readDigestCache(): MutableMap<String, Pair<String, String>> = synchronized(diskLock) {
        val file = ModelSupplyPaths.child(directory, DIGEST_MARKER)
        if (!file.isFile || file.length() > 8L * 1024 * 1024) return@synchronized mutableMapOf()
        try {
            val json = JsonParser.parseString(file.readText(Charsets.UTF_8)).asJsonObject
            if (json["schema_version"]?.asInt != DIGEST_SCHEMA || json["entries"]?.isJsonObject != true) {
                return@synchronized mutableMapOf()
            }
            val entries = json.getAsJsonObject("entries")
            val out = mutableMapOf<String, Pair<String, String>>()
            for ((key, value) in entries.entrySet()) {
                if (!value.isJsonObject) continue
                val entry = value.asJsonObject
                val sha = entry["sha256"]?.takeIf { it.isJsonPrimitive }?.asString ?: continue
                val tokenizer = entry["tokenizer_sha256"]?.takeIf { it.isJsonPrimitive }?.asString ?: continue
                if (shaPattern.matches(sha) && shaPattern.matches(tokenizer)) out[key] = sha to tokenizer
            }
            out
        } catch (error: Exception) {
            // A broken cache is never a capability; fall back to rehashing.
            mutableMapOf()
        }
    }

    private fun writeDigestCache(entries: Map<String, Pair<String, String>>) = synchronized(diskLock) {
        try {
            checkSupply(directory.isDirectory || directory.mkdirs(), "SUPPLY_STORAGE_UNAVAILABLE", "source cache directory")
            val payload = JsonObject().apply {
                addProperty("schema_version", DIGEST_SCHEMA)
                add("entries", JsonObject().apply {
                    for ((key, value) in entries) {
                        add(key, JsonObject().apply {
                            addProperty("sha256", value.first)
                            addProperty("tokenizer_sha256", value.second)
                        })
                    }
                })
            }
            val temp = ModelSupplyPaths.child(directory, DIGEST_MARKER + ".tmp")
            temp.writeText(payload.toString(), Charsets.UTF_8)
            Files.move(temp.toPath(), ModelSupplyPaths.child(directory, DIGEST_MARKER).toPath(),
                StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING)
        } catch (error: Exception) {
            // The cache is an optimization; failing to persist must not break inventory.
        }
    }

    private suspend fun checkCurrent(current: suspend () -> Boolean) {
        currentCoroutineContext().ensureActive()
        checkSupply(current(), "SUPPLY_CANCELLED", "source inventory retired")
    }

    private suspend fun inspectDocument(document: Document, current: suspend () -> Boolean,
        cache: MutableMap<String, Pair<String, String>>): Pair<String, String> {
        checkCurrent(current)
        val version = document.version()
        val key = version?.let { document.reference + "|" + document.size + "|" + it }
        val cached = key?.let(cache::get)
        val identity = if (cached != null) {
            document.open().use { }
            cached
        } else {
            val inspected = GgufSegmentCutter.inspectSource(LocalGgufSource("", "", "", document.size, document.open), current)
            inspected.sha256 to inspected.tokenizerSha256
        }
        checkCurrent(current)
        checkSupply(version == null || document.version() == version,
            "SUPPLY_SOURCE_DIGEST_MISMATCH", "source changed during inventory")
        if (key != null) cache[key] = identity
        return identity
    }
    private fun read(): List<Registration> = synchronized(diskLock) {
        ModelSupplyPaths.check(directory)
        val file = ModelSupplyPaths.child(directory, MARKER)
        if (!file.exists()) return@synchronized emptyList()
        checkSupply(file.isFile && file.length() <= 1024 * 1024, "SUPPLY_LOCAL_SOURCE_UNAVAILABLE", "source registry size")
        val json = JsonParser.parseString(file.readText(Charsets.UTF_8)).asJsonObject
        checkSupply(json["schema_version"]?.asInt == 1 && json["sources"]?.isJsonArray == true,
            "SUPPLY_LOCAL_SOURCE_UNAVAILABLE", "source registry schema")
        val entries = json.getAsJsonArray("sources")
        checkSupply(entries.size() <= 128, "SUPPLY_LOCAL_SOURCE_UNAVAILABLE", "source registry count")
        entries.map { value ->
            val entry = value.asJsonObject
            val id = ModelSupplyManifest.text(entry, "source_model_id")
            val sha = ModelSupplyManifest.text(entry, "source_model_sha256")
            val tokenizer = ModelSupplyManifest.text(entry, "tokenizer_sha256")
            val reference = ModelSupplyManifest.text(entry, "reference")
            checkSupply(idPattern.matches(id) && id !in setOf(".", "..") && shaPattern.matches(sha) && shaPattern.matches(tokenizer)
                && reference.length in 1..8192, "SUPPLY_LOCAL_SOURCE_UNAVAILABLE", "source registry entry")
            Registration(id, reference, sha, tokenizer, ModelSupplyManifest.integer(entry["size_bytes"], 1))
        }
    }
    private fun save(registration: Registration) = synchronized(diskLock) {
        ModelSupplyPaths.check(directory)
        checkSupply(directory.isDirectory || directory.mkdirs(), "SUPPLY_STORAGE_UNAVAILABLE", "source registry directory")
        val entries = read().filter { it.reference != registration.reference || it.modelId != registration.modelId } + registration
        checkSupply(entries.size <= 128, "SUPPLY_LOCAL_SOURCE_UNAVAILABLE", "source registry full")
        val json = JsonObject().apply {
            addProperty("schema_version", 1)
            add("sources", JsonArray().apply { entries.forEach { entry -> add(JsonObject().apply {
                addProperty("source_model_id", entry.modelId); addProperty("reference", entry.reference)
                addProperty("source_model_sha256", entry.sha); addProperty("tokenizer_sha256", entry.tokenizer)
                addProperty("size_bytes", entry.size)
            }) } })
        }
        val marker = ModelSupplyPaths.child(directory, MARKER)
        val temporary = File.createTempFile(".local-cut-sources-", ".tmp", directory)
        try {
            FileOutputStream(temporary).use { it.write(canonicalSupplyJson(json).toByteArray(Charsets.UTF_8)); it.fd.sync() }
            ModelSupplyPaths.child(directory, temporary.name)
            ModelSupplyPaths.child(directory, MARKER)
            Files.move(temporary.toPath(), marker.toPath(), StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING)
        } finally { Files.deleteIfExists(temporary.toPath()) }
    }
    fun summary(reference: String): String = read().filter { it.reference == reference }.joinToString("\n") {
        "已登记 ${it.modelId} · SHA ${it.sha}。当前可用性在供给前重新核验。"
    }
    suspend fun register(reference: String, modelId: String, current: suspend () -> Boolean = { true }): LocalGgufSource = mutex.withLock {
        checkSupply(idPattern.matches(modelId) && modelId !in setOf(".", ".."), "SUPPLY_INVALID_SOURCE_ID", "use master model ID")
        val document = documents().singleOrNull { it.reference == reference }
            ?: throw ModelSupplyException("SUPPLY_LOCAL_SOURCE_UNAVAILABLE", "select a full GGUF without a segment manifest")
        val cache = readDigestCache()
        val (digest, tokenizer) = inspectDocument(document, current, cache)
        val source = LocalGgufSource(modelId, digest, tokenizer, document.size, document.open)
        checkCurrent(current)
        save(Registration(modelId, reference, digest, tokenizer, document.size))
        writeDigestCache(cache)
        source
    }
    override suspend fun inventory(modelId: String, sourceSha256: String, current: suspend () -> Boolean): List<LocalGgufSource> = mutex.withLock {
        checkCurrent(current)
        val registrations = read().filter { it.modelId == modelId || (sourceSha256.isNotEmpty() && it.sha == sourceSha256) }
        val available = documents().associateBy { it.reference }
        val sources = mutableListOf<LocalGgufSource>()
        val cache = readDigestCache()
        val previousCache = cache.toMap()
        if (registrations.isEmpty()) {
            val modelKey = modelId.lowercase()
            for (document in available.values) {
                checkCurrent(current)
                if (document.size <= 0) continue
                if (modelKey.isNotEmpty()) {
                    val normalized = document.reference.lowercase().replace('.', '-').replace('_', '-')
                    if (!normalized.contains(modelKey)) continue
                }
                try {
                    val (digest, tokenizer) = inspectDocument(document, current, cache)
                    if (sourceSha256.isNotEmpty() && digest != sourceSha256) continue
                    val source = LocalGgufSource(modelId, digest, tokenizer, document.size, document.open)
                    sources += source
                } catch (cancelled: CancellationException) { throw cancelled }
                catch (error: Exception) {
                    if (error is ModelSupplyException && error.reasonCode == "SUPPLY_CANCELLED") throw error
                    // A candidate that does not verify is not a capability.
                }
            }
            checkCurrent(current)
            if (cache != previousCache) writeDigestCache(cache)
            return@withLock sources.distinctBy { Triple(it.modelId, it.sha256, it.tokenizerSha256) }
        }
        for (entry in registrations) {
            checkCurrent(current)
            val document = available[entry.reference] ?: continue
            if (entry.size != document.size || (sourceSha256.isNotEmpty() && entry.sha != sourceSha256)) continue
            val source = LocalGgufSource(entry.modelId, entry.sha, entry.tokenizer, entry.size, document.open)
            try {
                val (digest, tokenizer) = inspectDocument(document, current, cache)
                if (digest != entry.sha || tokenizer != entry.tokenizer) continue
                sources += source
            } catch (cancelled: CancellationException) { throw cancelled }
            catch (error: Exception) {
                if (error is ModelSupplyException && error.reasonCode == "SUPPLY_CANCELLED") throw error
                // A stale registration is not a capability; leave the explicit record for UI re-registration.
            }
        }
        checkCurrent(current)
        if (cache != previousCache) writeDigestCache(cache)
        sources.distinctBy { Triple(it.modelId, it.sha256, it.tokenizerSha256) }
    }
    override suspend fun resolve(manifest: ModelSupplyManifest, current: suspend () -> Boolean): LocalGgufSource =
        inventory(manifest.sourceModelId, manifest.sourceSha256, current).firstOrNull {
            it.sha256 == manifest.sourceSha256 && it.tokenizerSha256 == manifest.json["tokenizer_sha256"].asString
        } ?: throw ModelSupplyException("SUPPLY_LOCAL_SOURCE_UNAVAILABLE", "registered full source is unavailable or changed")
}
