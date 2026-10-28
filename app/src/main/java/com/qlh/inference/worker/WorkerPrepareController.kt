package com.qlh.inference.worker

import com.google.gson.JsonObject
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.CoroutineStart
import kotlinx.coroutines.Job
import kotlinx.coroutines.NonCancellable
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import java.util.concurrent.atomic.AtomicBoolean

interface TaskWorkerModelPrepareHandler {
    suspend fun command(data: JsonObject, scope: CoroutineScope, reply: suspend (JsonObject) -> Unit, authorize: () -> Unit)
    fun cancelAll()
    fun cancelPending(): Boolean = false
    suspend fun retireAndJoin()
    fun validateStage(offer: TaskWorkerEnvelope): String?
    suspend fun executeStage(offer: TaskWorkerEnvelope, action: suspend () -> TaskWorkerStageExecution): TaskWorkerStageExecution
}

/** load must verify the exact operation and return only after JNI succeeds; unload must join JNI. */
internal interface WorkerPrepareRuntime {
    suspend fun load(identity: WorkerPrepareIdentity)
    suspend fun loadWithSpec(identity: WorkerPrepareIdentity, spec: JsonObject) { throw WorkerPrepareException("prepare_load_spec_unsupported", "runtime policy unsupported") }
    suspend fun receipt(identity: WorkerPrepareIdentity, spec: JsonObject): JsonObject { throw WorkerPrepareException("prepare_runtime_receipt_missing", "runtime observation unavailable") }
    suspend fun unload()
    fun loaded(identity: WorkerPrepareIdentity): Boolean
    fun hasRuntime(): Boolean
    fun currentPreparedIdentity(): WorkerPrepareIdentity? = null
}

/** One native slot. Replacement fences the old generation until global commit or abort cleanup. */
internal class WorkerPrepareController(
    private val nodeId: String,
    private val runtime: WorkerPrepareRuntime,
    private val participationCurrent: suspend () -> Boolean = { true },
) : TaskWorkerModelPrepareHandler {
    private val gate = Mutex()
    private val lifecycle = Any()
    private val jobs = linkedMapOf<Job, WorkerPrepareRequest>()
    private var epoch = 0L
    @Volatile private var visible: WorkerPrepareIdentity? = null
    @Volatile private var candidate: WorkerPrepareIdentity? = null
    @Volatile private var committed: WorkerPrepareIdentity? = null
    private var previous: WorkerPrepareIdentity? = null
    @Volatile private var finalized = true
    private var highestGeneration = 0L
    private val policies = linkedMapOf<WorkerPrepareIdentity, JsonObject>()
    private val retired = linkedSetOf<WorkerPrepareIdentity>()
    // Cancellation while awaiting the stage gate never entered native loading.
    private val neverLoaded = linkedSetOf<WorkerPrepareIdentity>()

    fun committedIdentity(): WorkerPrepareIdentity? = visible

    override suspend fun command(data: JsonObject, scope: CoroutineScope, reply: suspend (JsonObject) -> Unit, authorize: () -> Unit) {
        val request = WorkerPrepareRequest.parse(data)
        authorize()
        if (request.identity.text("node_id") != nodeId) {
            authorize(); reply(request.ack("error", false, "prepare_node_mismatch")); return
        }
        val pending: List<Job>
        val commandEpoch: Long
        synchronized(lifecycle) {
            checkPrepare(jobs.size < 128, "prepare_queue_full", "too many commands")
            pending = jobs.keys.toList()
            commandEpoch = epoch
            if (request.action in setOf("abort", "release")) {
                if (request.action == "release" || committed != request.identity || !finalized) {
                    jobs.filterValues { it.action == "prepare" && it.identity == request.identity }.keys.forEach { it.cancel() }
                }
                // Fence matching stages immediately, before the pending native load returns.
                if (visible == request.identity && (request.action == "release" || !finalized)) visible = null
            }
        }
        val entered = AtomicBoolean(false)
        val job = scope.launch(start = CoroutineStart.LAZY) {
            pending.forEach { it.join() }
            gate.withLock {
                entered.set(true)
                process(request, commandEpoch, reply, authorize)
            }
        }
        synchronized(lifecycle) {
            if (epoch != commandEpoch) { job.cancel(); return }
            jobs[job] = request
        }
        job.invokeOnCompletion {
            synchronized(lifecycle) {
                if (job.isCancelled && !entered.get() && request.action == "prepare" &&
                    candidate != request.identity && committed != request.identity) {
                    remember(neverLoaded, request.identity)
                }
                jobs.remove(job)
            }
        }
        job.start()
    }

    private suspend fun current(commandEpoch: Long, authorize: () -> Unit) {
        currentCoroutineContext().ensureActive()
        if (synchronized(lifecycle) { epoch != commandEpoch }) throw CancellationException("prepare connection retired")
        if (!participationCurrent()) throw CancellationException("prepare participation retired")
        try { authorize() }
        catch (error: Exception) { throw CancellationException("prepare authority retired").also { it.initCause(error) } }
    }

    private suspend fun process(request: WorkerPrepareRequest, commandEpoch: Long, reply: suspend (JsonObject) -> Unit, authorize: () -> Unit) {
        val identity = request.identity
        var touched = false
        try {
            current(commandEpoch, authorize)
            val response = when (request.action) {
                "prepare" -> {
                    checkPrepare(identity !in retired && !synchronized(lifecycle) { identity in neverLoaded }, "prepare_generation_retired", "retired generation")
                    if (candidate == identity || (committed == identity && candidate == null)) {
                        checkPrepare(runtime.loaded(identity), "prepare_runtime_lost", "native runtime missing")
                    } else {
                        checkPrepare(candidate == null && finalized, "prepare_candidate_busy", "unfinished transaction")
                        checkPrepare(identity.generation > highestGeneration, "prepare_stale_generation", "generation must advance")
                        highestGeneration = identity.generation
                        candidate = identity
                        previous = committed
                        // One-slot JNI cannot retain two contexts. The previous identity is metadata only.
                        visible = null
                        touched = true
                        if (request.loadSpec != null) {
                            runtime.loadWithSpec(identity, request.loadSpec)
                            policies[identity] = request.loadSpec.deepCopy()
                        } else runtime.load(identity)
                        current(commandEpoch, authorize)
                        checkPrepare(runtime.loaded(identity), "prepare_native_load_failed", "native runtime not loaded")
                    }
                    request.ack("prepared", true).apply {
                        policies[identity]?.let { add("runtime_receipt", runtime.receipt(identity, it)) }
                    }
                }
                "commit" -> {
                    checkPrepare(candidate == identity || (committed == identity && candidate == null), "prepare_identity_mismatch", "no matching candidate")
                    checkPrepare(runtime.loaded(identity), "prepare_runtime_lost", "native runtime missing")
                    if (candidate == identity) {
                        previous = committed
                        committed = identity
                        candidate = null
                        finalized = false
                    }
                    touched = true
                    visible = identity
                    request.ack("ready", true).apply {
                        policies[identity]?.let { add("runtime_receipt", runtime.receipt(identity, it)) }
                    }
                }
                "abort", "release" -> {
                    if (!terminalCleanup(identity, request.action)) {
                        checkPrepare(candidate == identity || (candidate == null && committed == identity), "prepare_identity_mismatch", "no matching runtime")
                        // Aborting a finalized transaction would unload a published active generation.
                        checkPrepare(request.action != "abort" || candidate == identity || !finalized,
                            "prepare_already_finalized", "active generation already finalized")
                        touched = true
                        runtime.unload()
                        checkPrepare(!runtime.hasRuntime(), "prepare_unload_failed", "native runtime remains loaded")
                        // The replaced context cannot be restored without a new verified prepare.
                        candidate?.let { remember(retired, it) }
                        committed?.let { remember(retired, it) }
                        previous?.let { remember(retired, it) }
                        candidate = null; committed = null; previous = null; visible = null; finalized = true
                    }
                    request.ack("released", false)
                }
                "finalize" -> {
                    if (candidate == null && committed == identity && runtime.loaded(identity)) {
                        previous?.let { remember(retired, it) }
                        previous = null; finalized = true
                        request.ack("finalized", true)
                    } else {
                        checkPrepare(terminalCleanup(identity, request.action), "prepare_identity_mismatch", "no matching finalized or retired identity")
                        // A terminal proof only; this must never enable a stage or publish loaded-ready.
                        request.ack("finalized", false)
                    }
                }
                else -> error("validated action")
            }
            current(commandEpoch, authorize)
            reply(response)
        } catch (error: Exception) {
            var cleanupFailed = false
            val ownsSlot = candidate == identity || (candidate == null && committed == identity)
            if (request.action == "prepare" && !touched && !ownsSlot) remember(retired, identity)
            if (touched || (error is CancellationException && ownsSlot)) {
                try { withContext(NonCancellable) { fenceAndUnload() } }
                catch (cleanup: Exception) { cleanupFailed = true; error.addSuppressed(cleanup) }
            }
            if (error is CancellationException) throw error
            // Expired authority/binding must never authorize an error ACK either.
            try { current(commandEpoch, authorize) }
            catch (retired: CancellationException) {
                try { withContext(NonCancellable) { fenceAndUnload() } }
                catch (cleanup: Exception) { retired.addSuppressed(cleanup) }
                throw retired
            }
            reply(request.ack("error", (candidate ?: committed) == identity && runtime.loaded(identity),
                if (cleanupFailed) "prepare_unload_failed" else (error as? WorkerPrepareException)?.code ?: "prepare_native_load_failed"))
        }
    }

    /** Old identities never own the newer slot, including native SHA reuse across generations. */
    private fun terminalCleanup(identity: WorkerPrepareIdentity, action: String): Boolean {
        if (identity in retired) return true
        val owner = candidate ?: committed
        if (owner != identity && synchronized(lifecycle) { identity in neverLoaded }) {
            remember(retired, identity)
            return true
        }
        val runtimeIdentity = runtime.currentPreparedIdentity()
        if (action == "release" && runtimeIdentity != null &&
            runtimeIdentity.text("node_id") == identity.text("node_id") &&
            runtimeIdentity.generation > identity.generation && runtime.loaded(runtimeIdentity)) {
            remember(retired, identity)
            if (previous == identity) previous = null
            if (candidate == identity) candidate = null
            if (committed == identity) committed = null
            if (owner == identity) {
                visible = null
                finalized = true
            }
            return true
        }
        if (previous == identity && owner != identity) {
            remember(retired, identity)
            previous = null
            if (committed == identity) committed = null
            return true
        }
        if ((owner == null || owner == identity) && !runtime.hasRuntime()) {
            // Fresh process/worker or externally retired matching context: native absence is terminal.
            remember(retired, identity)
            if (owner == identity) {
                candidate = null; committed = null; previous = null; visible = null; finalized = true
            }
            return true
        }
        return false
    }

    private fun remember(set: MutableSet<WorkerPrepareIdentity>, identity: WorkerPrepareIdentity) {
        set.add(identity)
        while (set.size > 256) set.remove(set.first())
    }

    private suspend fun fenceAndUnload() {
        visible = null
        // Clear only after real native retirement. A failed unload leaves the controller fenced.
        runtime.unload()
        checkPrepare(!runtime.hasRuntime(), "prepare_unload_failed", "native runtime remains loaded")
        candidate?.let { remember(retired, it) }
        committed?.let { remember(retired, it) }
        previous?.let { remember(retired, it) }
        candidate = null; committed = null; previous = null; finalized = true
    }

    override fun cancelPending(): Boolean {
        val pending = synchronized(lifecycle) { candidate != null || jobs.values.any { it.action == "prepare" } }
        if (pending) cancelAll()
        return pending
    }

    override fun cancelAll() {
        synchronized(lifecycle) {
            epoch++
            visible = null
            jobs.keys.toList().forEach { it.cancel() }
        }
    }

    suspend fun awaitIdle() { synchronized(lifecycle) { jobs.keys.toList() }.forEach { it.join() } }

    override suspend fun retireAndJoin() = withContext(NonCancellable) {
        awaitIdle()
        // Called after the client's stage job has joined, also during a mode switch holding exclusive.
        // Runtime unload therefore must NOT acquire inferenceModes.worker/exclusive again.
        gate.withLock { fenceAndUnload() }
    }

    override fun validateStage(offer: TaskWorkerEnvelope): String? {
        val identity = visible ?: return "prepare_not_committed"
        val p = offer.payload
        if (p["stage_type"] != TaskWorkerProtocol.LAYER_FORWARD_STAGE ||
            !WorkerPrepareIdentity.tupleMatches(identity, p["prepare_identity"])) return "prepare_identity_mismatch"
        if (p["model_identity"] != identity.stageIdentity || (p["layer_range"] as? List<*>)?.map { (it as? Number)?.toInt() } != identity.range) return "prepare_identity_mismatch"
        val root = p["root_input"] as? Map<*, *> ?: return "prepare_stage_shape_mismatch"
        val spec = p["hidden_spec"] as? Map<*, *> ?: return "prepare_stage_shape_mismatch"
        if (root["want_hidden"] != (identity.mode != "tail") ||
            (spec["n_embd"] as? Number)?.toInt() != identity.hiddenSize ||
            p["middle_channel"] != "keep_head_layer_out") return "prepare_stage_shape_mismatch"
        return if (runtime.loaded(identity)) null else "prepare_runtime_lost"
    }

    override suspend fun executeStage(offer: TaskWorkerEnvelope, action: suspend () -> TaskWorkerStageExecution): TaskWorkerStageExecution = gate.withLock {
        validateStage(offer)?.let { throw AndroidFullWorkerStageException(it, "Stage differs from committed prepare") }
        action()
    }
}
