package com.qlh.inference.worker

import com.google.gson.JsonObject
import com.qlh.inference.logging.QlhLogger
import com.qlh.inference.service.ModelSupplyException
import com.qlh.inference.service.ModelSupplyManifest
import com.qlh.inference.service.ModelSupplyReceiver
import com.qlh.inference.service.ModelSupplyCurrent
import com.qlh.inference.service.ModelSupplyGc
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
    // Connection-owned memory only; replacing the controller starts a new scope.
    private val cancelledOperations = linkedSetOf<String>()

    private fun rememberCancelledLocked(operation: String) {
        cancelledOperations.remove(operation)
        cancelledOperations.add(operation)
        while (cancelledOperations.size > MAX_CANCELLED_OPERATIONS) {
            cancelledOperations.remove(cancelledOperations.first())
        }
    }

    private fun retireLocked(): Active? {
        generation++
        return active.also { active = null; it?.job?.let(retiring::add) }
    }

    override fun cancelAll() {
        cancelRetired(synchronized(lock) { retireLocked() })
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
                    ModelSupplyManifest.text(data, "model_id")
                    val sourceSha = ModelSupplyManifest.text(data, "source_model_sha256")
                    checkSupply(sourceSha.isEmpty() || Regex("[0-9a-f]{64}").matches(sourceSha), "SUPPLY_INVALID_REQUEST", "source sha")
                    reply(JsonObject().apply {
                        addProperty("action", "inventory_result"); addProperty("operation_id", operation); addProperty("node_id", nodeId)
                        addProperty("distributed", distributed()); addProperty("disk_free_bytes", root.parentFile?.usableSpace ?: 0L)
                        addProperty("memory_available_bytes", memoryAvailable().coerceAtLeast(0)); addProperty("local_cut", false)
                        add("engines", com.google.gson.JsonArray().apply { add("llama_cpp") }); add("sources", com.google.gson.JsonArray())
                    })
                }
                "cancel" -> {
                    val current = synchronized(lock) {
                        rememberCancelledLocked(operation)
                        if (active?.operation == operation) retireLocked() else null
                    }
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
                    checkSupply(strategy != "local_cut", "SUPPLY_LOCAL_CUT_UNAVAILABLE", "Android has no GGUF cutting tool")
                    checkSupply(strategy == "distribute", "SUPPLY_INVALID_REQUEST", "unknown strategy")
                    val port = ModelSupplyManifest.integer(data["api_port"], 1, 65535).toInt()
                    val token = ModelSupplyManifest.text(data, "token")
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
                                diagnostics = { QlhLogger.i("ModelSupply", it.logMessage()) })
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
