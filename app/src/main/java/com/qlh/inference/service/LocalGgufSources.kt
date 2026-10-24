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
    // ★ 2026-10-24（#117）：未登记候选的**全量 sha256 代价很高**（整模 5.78 GB，
    //   实测 70~120 s），而 `inventory()` 会被反复调用（每次 hello、每次供给）⇒
    //   不缓存的话 master 侧直接 `SUPPLY_WORKER_TIMEOUT`（实测 112 s 超时）。
    //   键含 reference 与 size：文件被替换（size 变化）即自动失效，语义安全。
    private val unregisteredDigests = java.util.concurrent.ConcurrentHashMap<String, Pair<String, String>>()
    // Shared across ModelManager instances in the UI and service; writes are tiny atomic replacements.
    private companion object {
        val diskLock = Any()
        val idPattern = Regex("[A-Za-z0-9_.:-]{1,128}")
        val shaPattern = Regex("[0-9a-f]{64}")
        const val MARKER = "local-cut-sources.json"
        // ★ 2026-10-24（#125）：把未登记候选的摘要**持久化**。实测 android 侧
        //   `inventory` 耗时 72.7 s（返回 10 个源）—— 它对本地每个 GGUF 都做整模全量
        //   sha256，而进程内缓存**重启即失效**。落盘后跨重启、跨供给复用，摘要本身
        //   仍是完整的源身份证明，只是不再重算。
        const val DIGEST_MARKER = "local-cut-digests.json"
        const val DIGEST_SCHEMA = 1
    }

    private fun readDigestCache(): MutableMap<String, Pair<String, String>> {
        val file = ModelSupplyPaths.child(directory, DIGEST_MARKER)
        if (!file.isFile || file.length() > 8L * 1024 * 1024) return mutableMapOf()
        return try {
            val json = JsonParser.parseString(file.readText(Charsets.UTF_8)).asJsonObject
            if (json["schema_version"]?.asInt != DIGEST_SCHEMA || json["entries"]?.isJsonObject != true) {
                return mutableMapOf()
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

    private fun writeDigestCache(entries: Map<String, Pair<String, String>>) {
        try {
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
                StandardCopyOption.REPLACE_EXISTING)
        } catch (error: Exception) {
            // The cache is an optimization; failing to persist must not break inventory.
        }
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
            // ★ 2026-10-24（#125）：**先按 modelId 过滤**。此前这里遍历本地**全部**
            //   GGUF 并逐个做整模全量 sha256，而 android 上存着 10 个不同模型的整模
            //   ⇒ 实测 `supply_inventory_done` 记录 72.7 s（rog 仅 0.046 s）。文件名
            //   规范化后不含本次 modelId 的候选，本来也不可能通过下面的摘要校验，
            //   直接跳过不损失任何能力。
            val modelKey = modelId.lowercase()
            val persisted = readDigestCache()
            var cacheDirty = false
            for (document in available.values) {
                if (document.size <= 0) continue
                if (modelKey.isNotEmpty()) {
                    val normalized = document.reference.lowercase().replace('.', '-').replace('_', '-')
                    if (!normalized.contains(modelKey)) continue
                }
                try {
                    val cacheKey = document.reference + "|" + document.size
                    val cached = unregisteredDigests[cacheKey] ?: persisted[cacheKey]
                    val tokenizer: String
                    val digest: String
                    if (cached != null) {
                        digest = cached.first
                        tokenizer = cached.second
                        unregisteredDigests[cacheKey] = cached
                    } else {
                        // 与 `register()` 同一套身份核对：先读 GGUF 布局与 tokenizer 摘要
                        // （`verify()` 要求 `source.tokenizerSha256` 与实测一致，传空串会
                        // 直接抛 SUPPLY_TOKENIZER_DIGEST_MISMATCH 并被下面静默吞掉 ——
                        // 这正是"能自裁却报无本地源"的那处断点）。
                        // `use {}` 不是挂起上下文，这里与 `register()` 一致用
                        // `ensureActive()` 表达取消。
                        val job = currentCoroutineContext()
                        val layout = document.open().buffered(64 * 1024).use {
                            GgufSegmentCutter.inspect(it) { job.ensureActive() }
                        }
                        val scan = document.open().buffered(64 * 1024).use {
                            GgufLayerWeightReader.scan(it, layout.blocks)
                        }
                        tokenizer = scan?.tokenizerSha256 ?: continue
                        val candidate = LocalGgufSource(modelId, "", tokenizer, document.size, document.open)
                        digest = GgufSegmentCutter.hash(candidate, current)
                        unregisteredDigests[cacheKey] = digest to tokenizer
                        persisted[cacheKey] = digest to tokenizer
                        cacheDirty = true
                    }
                    if (sourceSha256.isNotEmpty() && digest != sourceSha256) continue
                    val source = LocalGgufSource(modelId, digest, tokenizer, document.size, document.open)
                    GgufSegmentCutter.verify(source, current)
                    sources += source
                } catch (cancelled: CancellationException) { throw cancelled }
                catch (error: Exception) {
                    if (error is ModelSupplyException && error.reasonCode == "SUPPLY_CANCELLED") throw error
                    // A candidate that does not verify is not a capability.
                }
            }
            if (cacheDirty) writeDigestCache(persisted)
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
