package com.qlh.inference.service

import com.google.gson.JsonArray
import com.google.gson.JsonObject
import com.google.gson.JsonParser
import com.qlh.inference.worker.GgufLayerWeightReader
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

internal interface LocalGgufSources {
    suspend fun inventory(modelId: String, sourceSha256: String, current: suspend () -> Boolean): List<LocalGgufSource>
    suspend fun resolve(manifest: ModelSupplyManifest, current: suspend () -> Boolean): LocalGgufSource
}

/** Explicit source registration is separate from cluster identity, selected inference model and SAF grants. */
internal class LocalGgufSourceRegistry(
    private val directory: File,
    private val documents: suspend () -> List<Document>,
) : LocalGgufSources {
    data class Document(val reference: String, val size: Long, val open: () -> InputStream)
    private data class Registration(val modelId: String, val reference: String, val sha: String, val tokenizer: String, val size: Long)
    private val mutex = Mutex()
    // Shared across ModelManager instances in the UI and service; writes are tiny atomic replacements.
    private companion object {
        val diskLock = Any()
        val idPattern = Regex("[A-Za-z0-9_.:-]{1,128}")
        val shaPattern = Regex("[0-9a-f]{64}")
        const val MARKER = "local-cut-sources.json"
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
        val job = currentCoroutineContext()
        val layout = document.open().buffered(64 * 1024).use { GgufSegmentCutter.inspect(it) { job.ensureActive() } }
        val scan = document.open().buffered(64 * 1024).use { GgufLayerWeightReader.scan(it, layout.blocks) }
        val tokenizer = scan?.tokenizerSha256
            ?: throw ModelSupplyException("SUPPLY_TOKENIZER_DIGEST_MISMATCH", "source tokenizer unavailable")
        val candidate = LocalGgufSource(modelId, "", tokenizer, document.size, document.open)
        val digest = GgufSegmentCutter.hash(candidate, current)
        val source = candidate.copy(sha256 = digest)
        GgufSegmentCutter.verify(source, current)
        job.ensureActive()
        checkSupply(current(), "SUPPLY_CANCELLED", "source registration cancelled")
        save(Registration(modelId, reference, digest, tokenizer, document.size))
        source
    }
    override suspend fun inventory(modelId: String, sourceSha256: String, current: suspend () -> Boolean): List<LocalGgufSource> = mutex.withLock {
        val registrations = read().filter { it.modelId == modelId || (sourceSha256.isNotEmpty() && it.sha == sourceSha256) }
        val available = documents().associateBy { it.reference }
        val sources = mutableListOf<LocalGgufSource>()
        // ★ 2026-10-24（#117）：注册表为空时**不能直接放弃**。本地整模 GGUF
        //   （adb 推入或本地工具直接放入 `files/local-cut`）确实存在，只是没有经过
        //   显式 `register()`。此前 `if (registrations.isEmpty()) return emptyList()`
        //   会让这类源永不可见 ⇒ master 判 `SUPPLY_LOCAL_SOURCE_UNAVAILABLE` ⇒ 该节点
        //   只能从网络重下 GB 级段（实测要下 2.96 GB），而它本可本地秒级自裁。
        //   这里退化为"枚举候选 + 逐个核对身份（大小与 sha256）"，与显式登记走同一套
        //   校验（`GgufSegmentCutter.verify`），安全边界不变。
        if (registrations.isEmpty()) {
            for (document in available.values) {
                if (document.size <= 0) continue
                val candidate = LocalGgufSource(modelId, "", "", document.size, document.open)
                try {
                    val digest = GgufSegmentCutter.hash(candidate, current)
                    val source = candidate.copy(sha256 = digest)
                    if (sourceSha256.isNotEmpty() && digest != sourceSha256) continue
                    GgufSegmentCutter.verify(source, current)
                    sources += source
                } catch (cancelled: CancellationException) { throw cancelled }
                catch (error: Exception) {
                    if (error is ModelSupplyException && error.reasonCode == "SUPPLY_CANCELLED") throw error
                    // A candidate that does not verify is not a capability.
                }
            }
            return@withLock sources.distinctBy { Triple(it.modelId, it.sha256, it.tokenizerSha256) }
        }
        for (entry in registrations) {
            val document = available[entry.reference] ?: continue
            if (entry.size != document.size || (sourceSha256.isNotEmpty() && entry.sha != sourceSha256)) continue
            val source = LocalGgufSource(entry.modelId, entry.sha, entry.tokenizer, entry.size, document.open)
            try {
                GgufSegmentCutter.verify(source, current)
                sources += source
            } catch (cancelled: CancellationException) { throw cancelled }
            catch (error: Exception) {
                if (error is ModelSupplyException && error.reasonCode == "SUPPLY_CANCELLED") throw error
                // A stale registration is not a capability; leave the explicit record for UI re-registration.
            }
        }
        sources.distinctBy { Triple(it.modelId, it.sha256, it.tokenizerSha256) }
    }
    override suspend fun resolve(manifest: ModelSupplyManifest, current: suspend () -> Boolean): LocalGgufSource =
        inventory(manifest.sourceModelId, manifest.sourceSha256, current).firstOrNull {
            it.sha256 == manifest.sourceSha256 && it.tokenizerSha256 == manifest.json["tokenizer_sha256"].asString
        } ?: throw ModelSupplyException("SUPPLY_LOCAL_SOURCE_UNAVAILABLE", "registered full source is unavailable or changed")
}
