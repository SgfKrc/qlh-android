package com.qlh.inference.worker

import com.google.gson.JsonArray
import com.google.gson.JsonObject
import com.qlh.inference.logging.QlhLogger
import com.qlh.inference.service.ModelSupplyException
import com.qlh.inference.service.ModelSupplyManifest
import com.qlh.inference.service.ModelSupplyReceiver
import com.qlh.inference.service.ModelSupplyCurrent
import com.qlh.inference.service.ModelSupplyGc
import com.qlh.inference.service.ModelSupplyPaths
import com.qlh.inference.service.ModelSupplyProgress
import com.qlh.inference.service.supplyBindingSha256
import com.qlh.inference.service.checkSupply
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.CoroutineStart
import kotlinx.coroutines.Job
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.launch
import java.io.File

interface TaskWorkerModelSupplyHandler {
    suspend fun command(data: JsonObject, scope: CoroutineScope, reply: suspend (JsonObject) -> Unit, refreshHello: suspend () -> Unit, authorize: () -> Unit = {})
    fun cancelAll()
}

/** Single supply owner; connection retirement invalidates callbacks before closing HTTP. */
internal class ModelSupplyController(
    private val nodeId: String,
    private val boundHost: String,
    private val root: File,
    private val distributed: suspend () -> Boolean,
    private val bindingCurrent: suspend () -> Boolean,
    private val memoryAvailable: () -> Long,
    private val publicationGate: suspend (() -> Unit) -> Unit,
    private val available: (ModelSupplyManifest) -> Unit,
    private val bindingSha256: String = supplyBindingSha256("", boundHost, 0, 0),
    private val participationCurrent: suspend () -> Boolean = { distributed() && bindingCurrent() },
    private val localSources: com.qlh.inference.service.LocalGgufSources? = null,
) : TaskWorkerModelSupplyHandler {
    private data class Active(
        val operation: String, val manifest: ModelSupplyManifest, val receiver: ModelSupplyReceiver, val job: Job,
        var progress: ModelSupplyProgress = ModelSupplyProgress("preparing", 0,
            manifest.files.fold(0L) { n, file -> Math.addExact(n, file.size) }, 0, 0, 0),
        var registration: ModelSupplyCurrent.Registration? = null,
        var completed: Boolean = false,
    )
    private val currentSupply = ModelSupplyCurrent(root)
    private val lock = Any()
    private var generation = 0L
    private var active: Active? = null
    private val retiring = mutableSetOf<Job>()
    private val inventoryJobs = mutableMapOf<Job, String>()
    // Connection-owned memory only; replacing the controller starts a new scope.
    private val cancelledOperations = linkedSetOf<String>()

    private fun rememberCancelledLocked(operation: String) {
        cancelledOperations.remove(operation)
        cancelledOperations.add(operation)
        while (cancelledOperations.size > MAX_CANCELLED_OPERATIONS) {
            cancelledOperations.remove(cancelledOperations.first())
        }
    }

    /**
     * #135：报出「**此刻磁盘上真实存在**」的已注册段，供 master 判定能否复用。
     *
     * master 侧曾只用 hello 上报的 capabilities 快照做复用判定。运维清掉段缓存后
     * （磁盘满、迁移、手工回收——都是正常操作），快照在下一次重扫前仍持有旧清单 ⇒
     * master 误判「可复用」⇒ 改判 `distribute` ⇒ 接收端找不到文件 ⇒ 退化成真下载，
     * 反而比本地裁层更慢（Python 侧实测卡在 11.5 MB / 1.98 GB 超过 7 分钟）。
     *
     * 策略：指针与磁盘每次都重新读，**只有全部段文件都真实存在**才报告；任一项缺失
     * （或被删除、或被换成链接）就报空数组，master 会立刻回落到 `local_cut`。
     */
    private fun registeredArtifacts(): JsonArray {
        val result = JsonArray()
        val manifest = runCatching { currentSupply.read(bindingSha256) }.getOrNull() ?: return result
        val directory = runCatching {
            ModelSupplyPaths.child(root, "segment-${manifest.manifestSha256}")
        }.getOrNull() ?: return result
        val present = manifest.files.all { file ->
            runCatching { ModelSupplyPaths.child(directory, file.path).isFile }.getOrDefault(false)
        }
        if (!present) return result
        result.add(JsonObject().apply {
            add("layer_range", JsonArray().apply { add(manifest.start); add(manifest.end) })
            addProperty("artifact_sha256", manifest.artifactSha256)
            addProperty("source_model_id", manifest.sourceModelId)
            addProperty("source_model_sha256", manifest.sourceSha256)
            addProperty("tokenizer_sha256", manifest.json["tokenizer_sha256"]?.asString ?: "")
        })
        return result
    }

    private fun retireLocked(): Active? {
        generation++
        return active.also { active = null; it?.job?.let(retiring::add) }
    }

    override fun cancelAll() {
        val (previous, inventories) = synchronized(lock) { retireLocked() to inventoryJobs.keys.toList() }
        inventories.forEach { it.cancel() }
        cancelRetired(previous)
    }

    private fun cancelRetired(previous: Active?) {
        try {
            synchronized(lock) {
                if (previous != null && !previous.completed) {
                    previous.registration?.let { currentSupply.rollback(it, bindingSha256) }
                    previous.registration = null
                }
            }
        } finally {
            previous?.receiver?.cancel()
            previous?.job?.invokeOnCompletion { synchronized(lock) { retiring.remove(previous.job) } }
            previous?.job?.cancel()
        }
    }

    override suspend fun command(data: JsonObject, scope: CoroutineScope, reply: suspend (JsonObject) -> Unit, refreshHello: suspend () -> Unit, authorize: () -> Unit) {
        val operation = data["operation_id"]?.takeIf { it.isJsonPrimitive && it.asJsonPrimitive.isString }?.asString.orEmpty()
        var manifest: ModelSupplyManifest? = null
        val commandGeneration = synchronized(lock) { generation }
        try {
            checkSupply(ModelSupplyManifest.OPERATION.matches(operation), "SUPPLY_INVALID_OPERATION", "operation id")
            authorize()
            checkSupply(bindingCurrent(), "SUPPLY_MASTER_BINDING_CHANGED", "master binding changed")
            when (ModelSupplyManifest.text(data, "action")) {
                "inventory" -> {
                    val modelId = ModelSupplyManifest.text(data, "model_id")
                    val sourceSha = ModelSupplyManifest.text(data, "source_model_sha256")
                    checkSupply(sourceSha.isEmpty() || Regex("[0-9a-f]{64}").matches(sourceSha), "SUPPLY_INVALID_REQUEST", "source sha")
                    suspend fun reportInventory() {
                        currentCoroutineContext().ensureActive()
                        val sources = if (modelId.isNotEmpty() || sourceSha.isNotEmpty()) {
                            localSources?.inventory(modelId, sourceSha) {
                                currentCoroutineContext().ensureActive()
                                authorize()
                                bindingCurrent() && synchronized(lock) { generation == commandGeneration }
                            }.orEmpty()
                        } else emptyList()
                        val response = JsonObject().apply {
                            addProperty("action", "inventory_result"); addProperty("operation_id", operation); addProperty("node_id", nodeId)
                            addProperty("distributed", distributed()); addProperty("disk_free_bytes", root.parentFile?.usableSpace ?: 0L)
                            addProperty("memory_available_bytes", memoryAvailable().coerceAtLeast(0)); addProperty("local_cut", localSources != null)
                            addProperty("local_cut_tool", if (localSources != null) com.qlh.inference.service.GgufSegmentCutter.TOOL else "")
                            addProperty("local_cut_reason", if (localSources == null) "SUPPLY_LOCAL_CUT_UNAVAILABLE" else if (sources.isEmpty()) "SUPPLY_LOCAL_SOURCE_UNAVAILABLE" else "")
                            add("engines", com.google.gson.JsonArray().apply { add("llama_cpp") })
                            // #135：只有段文件此刻真实存在时才报告，缺失即报空数组。
                            add("artifacts", registeredArtifacts())
                            add("sources", com.google.gson.JsonArray().apply { sources.forEach { source -> add(JsonObject().apply {
                                addProperty("source_model_id", source.modelId); addProperty("source_model_sha256", source.sha256)
                                addProperty("tokenizer_sha256", source.tokenizerSha256); addProperty("size_bytes", source.size)
                            }) } })
                        }
                        authorize()
                        val bound = bindingCurrent()
                        currentCoroutineContext().ensureActive()
                        checkSupply(bound && synchronized(lock) { generation == commandGeneration },
                            "SUPPLY_CANCELLED", "inventory retired")
                        reply(response)
                    }
                    if (localSources == null) reportInventory() else {
                        val job = scope.launch(start = CoroutineStart.LAZY) {
                            try { reportInventory() }
                            catch (cancelled: CancellationException) { throw cancelled }
                            catch (error: Exception) {
                                currentCoroutineContext().ensureActive()
                                authorize()
                                if (bindingCurrent() && synchronized(lock) { generation == commandGeneration }) {
                                    reply(failed(operation, (error as? ModelSupplyException)?.reasonCode ?: "SUPPLY_INVENTORY_FAILED"))
                                }
                            }
                        }
                        synchronized(lock) {
                            checkSupply(generation == commandGeneration, "SUPPLY_CANCELLED", "inventory retired")
                            inventoryJobs[job] = operation
                        }
                        job.invokeOnCompletion { synchronized(lock) { inventoryJobs.remove(job) } }
                        job.start()
                    }
                }
                "cancel" -> {
                    val current = synchronized(lock) {
                        rememberCancelledLocked(operation)
                        if (active?.operation == operation) retireLocked() else null
                    }
                    synchronized(lock) { inventoryJobs.filterValues { it == operation }.keys.toList() }.forEach { it.cancel() }
                    if (current != null) { manifest = current.manifest; cancelRetired(current) }
                    reply(current?.let { result(it, "failed", "SUPPLY_CANCELLED") } ?: failed(operation, "SUPPLY_CANCELLED"))
                }
                "ensure" -> {
                    synchronized(lock) {
                        checkSupply(operation !in cancelledOperations && generation == commandGeneration,
                            "SUPPLY_CANCELLED", "supply retired")
                    }
                    checkSupply(distributed(), "SUPPLY_MODE_DISABLED", "worker mode disabled")
                    checkSupply(ModelSupplyManifest.text(data, "node_id") == nodeId, "SUPPLY_NODE_MISMATCH", "wrong target")
                    val parsed = ModelSupplyManifest.parse(data.getAsJsonObject("manifest"))
                    manifest = parsed
                    checkSupply(parsed.operationId == operation && parsed.nodeId == nodeId, "SUPPLY_IDENTITY_MISMATCH", "manifest target differs")
                    val strategy = ModelSupplyManifest.text(data, "strategy")
                    val allowFallback = data["allow_fallback"]?.let {
                        checkSupply(it.isJsonPrimitive && it.asJsonPrimitive.isBoolean,
                            "SUPPLY_INVALID_REQUEST", "allow_fallback must be boolean")
                        it.asBoolean
                    } ?: false
                    checkSupply(strategy != "local_cut" || localSources != null, "SUPPLY_LOCAL_CUT_UNAVAILABLE", "local cutter unavailable")
                    checkSupply(strategy in setOf("distribute", "local_cut"), "SUPPLY_INVALID_REQUEST", "unknown strategy")
                    val port = ModelSupplyManifest.integer(data["api_port"], 1, 65535).toInt()
                    val token = ModelSupplyManifest.text(data, "token")
                    val cutTimeoutMillis = data["local_cut_timeout_seconds"]?.let {
                        ModelSupplyManifest.integer(it, 1, 300) * 1000
                    } ?: 300_000L
                    checkSupply(participationCurrent(), "SUPPLY_CANCELLED", "worker participation changed")
                    val (previous, epoch) = synchronized(lock) {
                        checkSupply(operation !in cancelledOperations && generation == commandGeneration,
                            "SUPPLY_CANCELLED", "supply retired")
                        retireLocked() to generation
                    }
                    cancelRetired(previous)
                    val receiver = ModelSupplyReceiver(root, boundHost)
                    suspend fun current(): Boolean {
                        if (!synchronized(lock) { generation == epoch && operation !in cancelledOperations }
                            || !participationCurrent()) return false
                        authorize()
                        return synchronized(lock) { generation == epoch && operation !in cancelledOperations }
                    }
                    val job = scope.launch(start = CoroutineStart.LAZY) {
                        var lastProgress = 0L
                        var lastPhase = ""
                        val owner = synchronized(lock) { active?.takeIf { generation == epoch && it.operation == operation } }
                            ?: return@launch
                        // Keep donors and the newly published directory until registration/terminal ACK.
                        val pendingReceive = com.qlh.inference.service.ModelSupplyStorage.pending(root)
                        try {
                            synchronized(lock) { retiring.toList() }.forEach { it.join() }
                            checkSupply(current(), "SUPPLY_CANCELLED", "supply retired")
                            var cutSource: com.qlh.inference.service.LocalGgufSource? = null
                            val localWeight: (suspend (File, suspend (Long) -> Unit) -> Unit)? = if (strategy == "local_cut") {
                                { output, progress ->
                                    try {
                                        kotlinx.coroutines.withTimeout(cutTimeoutMillis) {
                                            val source = cutSource ?: localSources!!.resolve(parsed, ::current).also { cutSource = it }
                                            // Identity failures must not become recoverable source unavailability.
                                            checkSupply(source.sha256 == parsed.sourceSha256,
                                                "SUPPLY_SOURCE_DIGEST_MISMATCH", "local source identity differs")
                                            checkSupply(source.tokenizerSha256 == parsed.json["tokenizer_sha256"].asString,
                                                "SUPPLY_TOKENIZER_DIGEST_MISMATCH", "local source tokenizer differs")
                                            com.qlh.inference.service.GgufSegmentCutter.cut(source, parsed, output, ::current, progress)
                                        }
                                    } catch (timeout: kotlinx.coroutines.TimeoutCancellationException) {
                                        currentCoroutineContext().ensureActive()
                                        throw ModelSupplyException("SUPPLY_LOCAL_CUT_TIMEOUT", "local cutter budget exhausted")
                                    }
                                }
                            } else null
                            receiver.receive(parsed, port, token, ::current,
                                progress = { progress ->
                                    synchronized(lock) { owner.progress = progress }
                                    val phase = progress.phase
                                    val downloaded = progress.downloadedBytes
                                    val total = progress.totalBytes
                                    val now = System.nanoTime()
                                    if (phase != lastPhase || now - lastProgress >= 200_000_000L || downloaded == total) {
                                        currentCoroutineContext().ensureActive()
                                        checkSupply(current(), "SUPPLY_CANCELLED", "supply retired")
                                        reply(JsonObject().apply {
                                            addProperty("action", "progress"); addProperty("operation_id", operation); addProperty("node_id", nodeId)
                                            addProperty("phase", phase); addProperty("downloaded_bytes", downloaded); addProperty("total_bytes", total)
                                            addProperty("actual_downloaded_bytes", progress.actualDownloadedBytes)
                                            addProperty("reused_bytes", progress.reusedBytes); addProperty("resumed_bytes", progress.resumedBytes)
                                        })
                                        lastProgress = now; lastPhase = phase
                                    }
                                },
                                publish = { publish ->
                                    publicationGate {
                                        synchronized(lock) {
                                            checkSupply(generation == epoch && operation !in cancelledOperations, "SUPPLY_CANCELLED", "supply retired")
                                            checkSupply(active?.job?.isActive == true, "SUPPLY_CANCELLED", "supply cancelled")
                                            authorize()
                                            publish()
                                        }
                                    }
                                },
                                diagnostics = { QlhLogger.i("ModelSupply", it.logMessage()) },
                                allowLocalCutFallback = allowFallback, localWeight = localWeight)
                            currentCoroutineContext().ensureActive()
                            checkSupply(current(), "SUPPLY_CANCELLED", "supply retired")
                            publicationGate {
                                synchronized(lock) {
                                    checkSupply(generation == epoch && operation !in cancelledOperations && active?.job?.isActive == true, "SUPPLY_CANCELLED", "supply retired")
                                    authorize()
                                    owner.registration = currentSupply.select(parsed, bindingSha256)
                                    available(parsed)
                                }
                            }
                            refreshHello()
                            currentCoroutineContext().ensureActive()
                            checkSupply(current(), "SUPPLY_CANCELLED", "supply retired")
                            reply(result(owner, "available"))
                            synchronized(lock) {
                                owner.completed = true
                                owner.registration?.let { currentSupply.complete(it) }
                                owner.registration = null
                            }
                            pendingReceive.close()
                            // Only a completed current operation can authorize covered partial cleanup.
                            val gc = ModelSupplyGc(root).collect(nodeId, ::current, commit = { cleanup ->
                                publicationGate {
                                    synchronized(lock) {
                                        checkSupply(generation == epoch && active?.job?.isActive == true,
                                            "SUPPLY_CANCELLED", "supply retired during cleanup")
                                        authorize()
                                        cleanup()
                                    }
                                }
                            }, completedManifest = parsed)
                            QlhLogger.i("ModelSupply", gc.logMessage())
                        } catch (cancelled: CancellationException) { throw cancelled }
                        catch (error: Exception) {
                            synchronized(lock) { owner.registration?.let { currentSupply.rollback(it, bindingSha256) }; owner.registration = null }
                            if (current()) reply(result(owner, "failed", (error as? ModelSupplyException)?.reasonCode ?: "SUPPLY_RECEIVE_FAILED"))
                        } finally {
                            pendingReceive.close()
                            receiver.cancel()
                            synchronized(lock) {
                                if (!owner.completed) owner.registration?.let { currentSupply.rollback(it, bindingSha256) }
                                if (generation == epoch) active = null
                            }
                        }
                    }
                    try {
                        synchronized(lock) {
                            checkSupply(generation == epoch && operation !in cancelledOperations,
                                "SUPPLY_CANCELLED", "supply retired")
                            active = Active(operation, parsed, receiver, job)
                        }
                        job.start()
                    } catch (error: Exception) {
                        receiver.cancel()
                        job.cancel()
                        throw error
                    }
                }
                else -> throw ModelSupplyException("SUPPLY_INVALID_REQUEST", "unknown action")
            }
        } catch (cancelled: CancellationException) { throw cancelled }
        catch (error: Exception) {
            val reason = (error as? ModelSupplyException)?.reasonCode ?: "SUPPLY_INVALID_REQUEST"
            reply(manifest?.let { parsed ->
                withProgress(parsed.result("failed", reason), ModelSupplyProgress("preparing", 0,
                    parsed.files.sumOf { it.size }, 0, 0, 0))
            } ?: failed(operation, reason))
        }
    }
    private companion object { const val MAX_CANCELLED_OPERATIONS = 256 }

    private fun result(owner: Active, status: String, reason: String = ""): JsonObject = synchronized(lock) {
        withProgress(owner.manifest.result(status, reason), owner.progress)
    }
    private fun withProgress(result: JsonObject, progress: ModelSupplyProgress) = result.apply {
        addProperty("phase", progress.phase)
        addProperty("downloaded_bytes", progress.downloadedBytes); addProperty("total_bytes", progress.totalBytes)
        addProperty("actual_downloaded_bytes", progress.actualDownloadedBytes)
        addProperty("reused_bytes", progress.reusedBytes); addProperty("resumed_bytes", progress.resumedBytes)
    }
    private fun failed(operation: String, reason: String) = JsonObject().apply {
        addProperty("action", "result"); addProperty("operation_id", operation); addProperty("node_id", nodeId)
        addProperty("status", "failed"); addProperty("reason_code", reason)
        withProgress(this, ModelSupplyProgress("preparing", 0, 0, 0, 0, 0))
    }
}
