package com.qlh.inference.status

import android.os.ParcelFileDescriptor
import android.system.ErrnoException
import android.system.Os
import android.system.OsConstants
import com.google.gson.JsonObject
import com.google.gson.JsonParser
import com.google.gson.GsonBuilder
import com.qlh.inference.worker.WorkerPrepareIdentity
import java.io.File
import java.io.IOException
import java.io.InputStream
import java.net.InetAddress
import java.net.InetSocketAddress
import java.nio.ByteOrder
import java.security.MessageDigest
import com.qlh.inference.worker.TaskWorkerAttemptState
import com.qlh.inference.worker.TaskWorkerConnectionState
import com.qlh.inference.worker.TaskWorkerSnapshot
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withTimeout
import java.util.concurrent.Callable
import java.util.concurrent.Future
import java.util.concurrent.RejectedExecutionException
import java.util.concurrent.TimeoutException
import java.util.concurrent.SynchronousQueue
import java.util.concurrent.ThreadPoolExecutor
import java.util.concurrent.TimeUnit

/** Read-only release evidence. Only explicit map keys cross the R8/JSON boundary. */
internal object ReleaseRuntimeProbe {
    const val PREFIX = "QLH_RELEASE_PROBE: "
    private val noncePattern = Regex("[0-9a-f]{32}")
    private val gson = GsonBuilder().serializeNulls().create()

    enum class ErrorCode {
        PROBE_TIMEOUT, PROBE_BUSY, PROBE_INTERRUPTED, PROBE_FAILED, PROBE_PENDING,
        PACKAGE_INFO_INVALID, ENGINE_MISSING, ENGINE_CLOSED,
        NATIVE_LIBRARY_UNAVAILABLE, NATIVE_QUERY_FAILED, NATIVE_BACKEND_INVALID,
        NATIVE_STUB_BACKEND, MODEL_INFO_FAILED, MODEL_STATE_INVALID,
        WORKER_SNAPSHOT_CHANGED, WORKER_STATE_INVALID, INFERENCE_MODE_CHANGED,
        INFERENCE_MODE_UNSUPPORTED, RUNTIME_IDENTITY_CHANGED,
        RUNTIME_RECEIPT_UNAVAILABLE, RUNTIME_RECEIPT_INVALID, ARTIFACT_REHASH_FAILED,
    }

    data class PackageIdentity(
        val pid: Int,
        val packageName: String,
        val versionName: String? = null,
        val versionCode: Long? = null,
    )

    data class NativeState(
        val systemInfo: String = "",
        val backendDeviceCount: Int = 0,
        val backendDevices: String = "",
        val errorCode: ErrorCode? = null,
    ) {
        companion object {
            fun fromBackend(info: Map<String, String>): NativeState {
                val system = info["system_info"].orEmpty()
                val devices = info["backend_devices"].orEmpty()
                val count = info["backend_device_count"]?.toIntOrNull() ?: 0
                val error = when {
                    system.contains("stub", ignoreCase = true) ||
                        devices.contains("stub", ignoreCase = true) -> ErrorCode.NATIVE_STUB_BACKEND
                    system.isBlank() || devices.isBlank() || count <= 0 -> ErrorCode.NATIVE_BACKEND_INVALID
                    else -> null
                }
                return NativeState(system, count, devices, error)
            }
        }
    }

    data class ModelState(
        val loaded: Boolean = false,
        val layerRange: List<Int> = emptyList(),
        val nativeLayerCount: Int? = null,
    )

    data class RuntimeState(
        val native: NativeState,
        val model: ModelState = ModelState(),
        val errorCode: ErrorCode? = null,
        val process: Map<String, Any?>? = null,
        val receipt: Map<String, Any?>? = null,
    )

    data class Snapshot(
        val inferenceMode: String?,
        val worker: TaskWorkerSnapshot,
        val runtime: RuntimeState,
        val capturedAt: Double = unixSeconds(),
        val errorCode: ErrorCode? = null,
        val product: Boolean = false,
    )

    data class Request(val nonce: String, val product: Boolean)
    fun request(args: Array<out String>?): Request? {
        if (args == null || args.size != 2 || !noncePattern.matches(args[1])) return null
        return when (args[0]) {
            "--qlh-release-probe" -> Request(args[1], false)
            "--qlh-product-probe" -> Request(args[1], true)
            else -> null
        }
    }

    fun nonce(args: Array<out String>?): String? = args?.takeIf {
        it.size == 2 && it[0] == "--qlh-release-probe" && noncePattern.matches(it[1])
    }?.get(1)

    fun unixSeconds(): Double = System.currentTimeMillis() / 1000.0

    fun failure(code: ErrorCode, worker: TaskWorkerSnapshot = TaskWorkerSnapshot()): Snapshot = Snapshot(
        inferenceMode = null,
        worker = worker,
        runtime = RuntimeState(NativeState(errorCode = code)),
        errorCode = code,
    )

    /** The caller holds the model mutex throughout; runtime must call JNI directly, never a locking getter. */
    suspend fun capture(
        readMode: suspend () -> String,
        readWorker: () -> TaskWorkerSnapshot,
        readCommitted: () -> WorkerPrepareIdentity? = { null },
        readRuntime: () -> RuntimeState,
    ): Snapshot {
        val workerBefore = readWorker()
        val committedBefore = readCommitted()
        val modeBefore = readMode()
        currentCoroutineContext().ensureActive()
        val runtime = readRuntime()
        val modeAfter = readMode()
        val workerAfter = readWorker()
        val committedAfter = readCommitted()
        currentCoroutineContext().ensureActive()
        return Snapshot(
            inferenceMode = modeBefore,
            worker = workerBefore,
            runtime = runtime,
            errorCode = when {
                committedBefore != committedAfter -> ErrorCode.RUNTIME_IDENTITY_CHANGED
                workerBefore != workerAfter -> ErrorCode.WORKER_SNAPSHOT_CHANGED
                modeBefore != modeAfter -> ErrorCode.INFERENCE_MODE_CHANGED
                else -> null
            },
        )
    }

    private fun workerValid(mode: String?, worker: TaskWorkerSnapshot): Boolean {
        val connection = worker.connection
        if (worker.connectionAlive && (!worker.processRunning || connection != TaskWorkerConnectionState.READY)) return false
        if (connection == TaskWorkerConnectionState.READY && !worker.connectionAlive) return false
        if (worker.processRunning && connection == TaskWorkerConnectionState.STOPPED) return false
        if (!worker.processRunning && connection !in setOf(
                TaskWorkerConnectionState.STOPPED, TaskWorkerConnectionState.RETRY_EXHAUSTED,
            )) return false
        if (mode == "local" && worker.processRunning) return false
        if (worker.activeAttempt.state in setOf(
                TaskWorkerAttemptState.OFFERED, TaskWorkerAttemptState.RUNNING, TaskWorkerAttemptState.CANCELLING,
            ) && (!worker.processRunning || !worker.connectionAlive || connection != TaskWorkerConnectionState.READY)
        ) return false
        return true
    }

    private fun modelValid(mode: String?, worker: TaskWorkerSnapshot, model: ModelState): Boolean {
        if (!model.loaded) return model.layerRange.isEmpty() && model.nativeLayerCount == null
        if ((model.nativeLayerCount ?: 0) <= 0) return false
        if (mode == "distributed" && !worker.processRunning) return false
        val range = model.layerRange
        if (range.isEmpty()) return mode == "local"
        return mode == "distributed" && worker.processRunning && range.size == 2 &&
            range[0] >= 0 && range[1] > range[0] && range[1] - range[0] == model.nativeLayerCount
    }

    fun line(nonce: String, identity: PackageIdentity, snapshot: Snapshot, process: Map<String, Any?>? = snapshot.runtime.process): String {
        require(noncePattern.matches(nonce))
        val native = snapshot.runtime.native
        val model = snapshot.runtime.model
        val worker = snapshot.worker
        // Revalidate even constructed states, so no caller can accidentally bless an empty/stub backend.
        val nativeError = native.errorCode ?: NativeState.fromBackend(mapOf(
            "system_info" to native.systemInfo,
            "backend_device_count" to native.backendDeviceCount.toString(),
            "backend_devices" to native.backendDevices,
        )).errorCode
        val error = snapshot.errorCode ?: snapshot.runtime.errorCode ?: nativeError ?: when {
            identity.pid <= 0 || identity.packageName.isBlank() || identity.versionName.isNullOrBlank() ||
                (identity.versionCode ?: 0) <= 0 || !snapshot.capturedAt.isFinite() || snapshot.capturedAt <= 0 ->
                ErrorCode.PACKAGE_INFO_INVALID
            snapshot.inferenceMode !in setOf("local", "distributed") -> ErrorCode.INFERENCE_MODE_UNSUPPORTED
            !workerValid(snapshot.inferenceMode, worker) -> ErrorCode.WORKER_STATE_INVALID
            !modelValid(snapshot.inferenceMode, worker, model) -> ErrorCode.MODEL_STATE_INVALID
            else -> null
        }
        val role = when (snapshot.inferenceMode) {
            "distributed" -> if (worker.processRunning) "worker" else "client"
            "local" -> "local"
            else -> null
        }
        val fields = linkedMapOf<String, Any?>(
            "schema_version" to 1,
            "probe_kind" to if (snapshot.product) "product" else "release",
            "nonce" to nonce,
            "pid" to identity.pid,
            "captured_at" to snapshot.capturedAt,
            "package_name" to identity.packageName,
            "version_name" to identity.versionName,
            "version_code" to identity.versionCode,
            "service_running" to true,
            "inference_mode" to snapshot.inferenceMode,
            "role" to role,
            "worker" to linkedMapOf(
                "process_running" to worker.processRunning,
                "connection_alive" to worker.connectionAlive,
                "connection" to worker.connection.name,
                "active_attempt_state" to worker.activeAttempt.state.name,
            ),
            "native" to linkedMapOf(
                "available" to (error == null),
                "system_info" to native.systemInfo,
                "backend_device_count" to native.backendDeviceCount,
                "backend_devices" to native.backendDevices,
                "error_code" to (nativeError ?: error)?.name,
            ),
            "model" to linkedMapOf(
                "loaded" to model.loaded,
                "layer_range" to model.layerRange,
                "full_model_loaded" to (model.loaded && model.layerRange.isEmpty()),
                "native_layer_count" to model.nativeLayerCount,
            ),
            "runtime_receipt" to snapshot.runtime.receipt,
            "error_code" to error?.name,
        )
        if (process != null) fields["process"] = process
        return PREFIX + gson.toJson(fields)
    }

    /** Source range is read from the exact artifact's verified native manifest, never a plan. */
    data class PreparedMetadata(val range: List<Int>, val total: Int, val decoderLayers: Int,
        val sha256: String, val manifestSha256: String, val contract: JsonObject) {
        companion object {
            fun parse(identity: WorkerPrepareIdentity, raw: String, artifactName: String): PreparedMetadata {
                require(raw.toByteArray(Charsets.UTF_8).size <= 4 * 1024 * 1024)
                val value = JsonParser.parseString(raw).asJsonObject
                fun integer(key: String) = WorkerPrepareIdentity.integer(value[key], 1, Int.MAX_VALUE.toLong()).toInt()
                val span = value["source_layer_range"].asJsonArray
                require(span.size() == 2)
                val range = span.map { WorkerPrepareIdentity.integer(it, 0, Int.MAX_VALUE.toLong()).toInt() }
                val total = integer("n_layer")
                // Top-level nextn_predict_layers describes the SOURCE model. Supply's
                // kept_block_count already excludes MTP; subtracting it again rejects
                // valid Qwen3.5 decoder-only crops after a successful JNI load.
                val count = integer("kept_block_count")
                value["nextn_predict_layers"]?.let {
                    WorkerPrepareIdentity.integer(it, 0, Int.MAX_VALUE.toLong())
                }
                require(range == identity.range && total == identity.totalLayers && count == range[1] - range[0])
                value["gguf_metadata"]?.let { declared ->
                    require(declared.isJsonObject)
                    val metadata = declared.asJsonObject
                    require(WorkerPrepareIdentity.integer(metadata["block_count"], 1, Int.MAX_VALUE.toLong()).toInt() == count)
                    metadata["nextn_predict_layers"]?.takeUnless { it.isJsonNull }?.let {
                        require(WorkerPrepareIdentity.integer(it, 0, Int.MAX_VALUE.toLong()) == 0L)
                    }
                }
                for (key in listOf("artifact_sha256", "source_model_id", "source_model_sha256", "tokenizer_sha256")) {
                    require(value[key].asString == identity.text(key))
                }
                val artifact = value["artifact"].asString.substringAfterLast('/').substringAfterLast('\\')
                require(artifact == artifactName && value["mode"].asString == identity.mode)
                require(integer("hidden_size") == identity.hiddenSize)
                return PreparedMetadata(range, total, count, identity.text("artifact_sha256"),
                    sha256(raw.toByteArray(Charsets.UTF_8)), value.deepCopy())
            }
        }
    }

    fun sha256(bytes: ByteArray): String = MessageDigest.getInstance("SHA-256").digest(bytes)
        .joinToString("") { "%02x".format(it.toInt() and 255) }

    fun rehash(input: InputStream, expectedSize: Long, check: () -> Unit): String {
        require(expectedSize > 0)
        val digest = MessageDigest.getInstance("SHA-256")
        val buffer = ByteArray(1024 * 1024)
        var size = 0L
        while (true) {
            check()
            val n = input.read(buffer)
            if (n < 0) break
            require(n.toLong() <= expectedSize - size)
            size += n
            digest.update(buffer, 0, n)
        }
        require(size == expectedSize)
        check()
        return digest.digest().joinToString("") { "%02x".format(it.toInt() and 255) }
    }

    fun boundedText(input: InputStream, limit: Int): String = input.use {
        val bytes = it.readBytesBounded(limit)
        bytes.toString(Charsets.UTF_8)
    }

    private fun InputStream.readBytesBounded(limit: Int): ByteArray {
        val output = java.io.ByteArrayOutputStream()
        val buffer = ByteArray(8192)
        while (true) {
            if (Thread.currentThread().isInterrupted) throw InterruptedException()
            val n = read(buffer)
            if (n < 0) break
            require(output.size() + n <= limit) { "probe read exceeds limit" }
            output.write(buffer, 0, n)
        }
        return output.toByteArray()
    }

    /** Same-nonce polling keeps every Service.dump call below the OS pipe timeout.
     * One background slot and one cached immutable result; no pending request queue.
     */
    class ProductReader<T>(private val timeoutMillis: Long = 120_000) : AutoCloseable {
        private val executor = ThreadPoolExecutor(1, 1, 30, TimeUnit.SECONDS, SynchronousQueue(),
            { task -> Thread(task, "qlh-product-probe").apply { isDaemon = true } },
            ThreadPoolExecutor.AbortPolicy()).apply { allowCoreThreadTimeOut(true) }
        private var activeNonce: String? = null
        private var future: Future<T>? = null
        private var started = 0L

        @Synchronized fun poll(nonce: String, action: suspend () -> T): T? {
            val previous = future
            if (previous != null && !previous.isDone && System.nanoTime() - started >= TimeUnit.MILLISECONDS.toNanos(timeoutMillis)) {
                previous.cancel(true)
                throw TimeoutException("product probe timed out")
            }
            if (nonce == activeNonce && previous != null) {
                if (previous.isCancelled) throw TimeoutException("product probe timed out")
                return if (previous.isDone) previous.get() else null
            }
            if (previous != null && !previous.isDone) throw RejectedExecutionException("product probe busy")
            val submitted = executor.submit(Callable { runBlocking { withTimeout(timeoutMillis) { action() } } })
            activeNonce = nonce
            future = submitted
            started = System.nanoTime()
            return null
        }
        override fun close() { future?.cancel(true); executor.shutdownNow() }
    }

    /** JNI is not cancellable. A single daemon slot bounds both dump latency and abandoned work. */
    class BoundedReader(private val timeoutMillis: Long = 4_500) : AutoCloseable {
        private val executor = ThreadPoolExecutor(
            1, 1, 30, TimeUnit.SECONDS, SynchronousQueue(),
            { task -> Thread(task, "qlh-release-probe").apply { isDaemon = true } },
            ThreadPoolExecutor.AbortPolicy(),
        ).apply { allowCoreThreadTimeOut(true) }

        fun <T> read(action: suspend () -> T): T {
            val future = executor.submit(Callable {
                runBlocking { withTimeout((timeoutMillis - 250).coerceAtLeast(1)) { action() } }
            })
            try {
                return future.get(timeoutMillis, TimeUnit.MILLISECONDS)
            } finally {
                if (!future.isDone) future.cancel(true)
            }
        }

        override fun close() { executor.shutdownNow() }
    }
}


/** Full descendant closure. A missing/denied/racing proc read is evidence of incompleteness. */
internal object ReleaseProcessTree {
    data class Connections(val values: List<Map<String, Any?>>, val complete: Boolean,
        val errors: List<String> = emptyList())
    interface Reader {
        fun text(path: String, limit: Int): String
        fun list(path: String): List<String>
        fun link(path: String): String
        fun connections(pid: Int): Connections
        fun uid(pid: Int): Int = throw IOException("proc owner unavailable")
    }
    class MissingProcFile(val path: String) : IOException("proc file absent")
    private data class Identity(val pid: Int, val start: Long, val parent: Int) {
        fun wire(): Map<String, Any?> = linkedMapOf("pid" to pid, "created_at" to start)
    }
    private data class Children(val tids: List<Int>, val pids: List<Int>)
    private fun identity(pid: Int, raw: String): Identity {
        require(raw.startsWith("$pid ("))
        val end = raw.lastIndexOf(')')
        require(end > 0)
        val fields = raw.substring(end + 1).trim().split(Regex("\\s+"))
        require(fields.size >= 20)
        return Identity(pid, fields[19].toLong().also { require(it > 0) }, fields[1].toInt())
    }
    private data class UidScan(val uid: Int, val identities: Map<Int, Identity>,
        val owners: Map<Int, Int>, val visible: List<Int>, val outside: Map<Int, Identity>, val errors: List<String>) {
        fun wire(): Map<String, Any?> = linkedMapOf(
            "visible_pids" to visible,
            "owners" to owners.map { (pid, owner) -> mapOf("pid" to pid, "uid" to owner) },
            "roster" to identities.values.map { it.wire() + mapOf("parent_pid" to it.parent, "uid" to uid) },
            "outside_uid" to outside.values.map { it.wire() + mapOf("parent_pid" to it.parent, "uid" to owners[it.pid]) },
            "errors" to errors,
        )
    }
    private fun scanUid(reader: Reader, pid: Int): UidScan {
        val uid = reader.uid(pid).also { require(it >= 0) }
        val visible = numbers(reader.list("/proc").filter { Regex("[0-9]+").matches(it) })
        require(visible.size in 1..32768 && pid in visible)
        val owners = linkedMapOf<Int, Int>()
        val identities = linkedMapOf<Int, Identity>()
        val outside = linkedMapOf<Int, Identity>()
        val errors = mutableListOf<String>()
        for (candidate in visible) {
            if (Thread.currentThread().isInterrupted) throw InterruptedException()
            try {
                val owner = reader.uid(candidate).also { require(it >= 0) }
                owners[candidate] = owner
                val observed = identity(candidate, reader.text("/proc/$candidate/stat", 65536))
                require(reader.uid(candidate) == owner)
                if (owner == uid) identities[candidate] = observed else outside[candidate] = observed
            } catch (interrupted: InterruptedException) { throw interrupted }
              catch (_: Exception) { errors += "$candidate:uid_or_stat_unavailable_or_changed" }
        }
        if (pid !in identities) errors += "root_not_in_uid_roster"
        // A visible child that crossed UID cannot be silently excluded from the tree.
        if (outside.values.any { it.parent in identities }) errors += "cross_uid_descendant"
        return UidScan(uid, identities, owners, visible, outside, errors)
    }
    private fun taskIds(reader: Reader, pid: Int): List<Int> = numbers(reader.list("/proc/$pid/task"))
        .also { require(it.isNotEmpty() && it.size <= 4096) }
    private fun rosterChildren(reader: Reader, pid: Int, scan: UidScan?): Children {
        require(scan != null && scan.errors.isEmpty() && pid in scan.identities)
        val tasks = taskIds(reader, pid)
        val found = scan.identities.values.filter { it.parent == pid }.map { it.pid }.sorted()
        require(pid !in found && taskIds(reader, pid) == tasks)
        return Children(tasks, found)
    }
    private fun numbers(values: List<String>): List<Int> = values.map {
        require(Regex("[1-9][0-9]*").matches(it)); it.toInt()
    }.sorted().also { require(it.distinct().size == it.size) }
    private fun children(reader: Reader, pid: Int): Children {
        val tasks = numbers(reader.list("/proc/$pid/task"))
        require(tasks.isNotEmpty() && tasks.size <= 4096)
        val found = sortedSetOf<Int>()
        for (tid in tasks) {
            val raw = reader.text("/proc/$pid/task/$tid/children", 65536).trim()
            if (raw.isNotEmpty()) found.addAll(numbers(raw.split(Regex("\\s+"))))
        }
        require(tasks == numbers(reader.list("/proc/$pid/task")))
        require(pid !in found)
        return Children(tasks, found.toList())
    }

    fun captureCurrent(readTcpState: (Int) -> Int): Map<String, Any?> {
        val pid = android.os.Process.myPid()
        return capture(pid, AndroidReader(pid, readTcpState))
    }

    fun capture(pid: Int, reader: Reader): Map<String, Any?> = try {
        captureUsing(pid, reader, false)
    } catch (missing: MissingProcFile) {
        if (!missing.path.endsWith("/children")) throw missing
        captureUsing(pid, reader, true)
    }

    private fun captureUsing(pid: Int, reader: Reader, fallback: Boolean): Map<String, Any?> {
        val start = ReleaseRuntimeProbe.unixSeconds()
        val processes = linkedMapOf<Int, MutableMap<String, Any?>>()
        val observations = linkedMapOf<Int, Children>()
        val errors = mutableListOf<String>()
        fun <T> read(code: String, action: () -> T): T? = try {
            if (Thread.currentThread().isInterrupted) throw InterruptedException()
            action()
        } catch (interrupt: InterruptedException) { throw interrupt }
          catch (missing: MissingProcFile) {
              if (!fallback && missing.path.endsWith("/children")) throw missing
              errors += code; null
          } catch (_: Exception) { errors += code; null }
        val before = read("root_identity_unavailable") { identity(pid, reader.text("/proc/$pid/stat", 65536)) }
        val scanBefore = if (fallback) read("uid_scan_before_failed") { scanUid(reader, pid) } else null
        if (fallback) errors += scanBefore?.errors.orEmpty()
        val pending = java.util.ArrayDeque<Int>().apply { add(pid) }
        while (pending.isNotEmpty() && processes.size < 128) {
            val current = pending.removeFirst()
            if (current in processes) continue
            val localErrors = mutableListOf<String>()
            val entry = linkedMapOf<String, Any?>("pid" to current)
            processes[current] = entry
            val initial = read("$current:identity_unavailable") { identity(current, reader.text("/proc/$current/stat", 65536)) }
            entry["created_at"] = initial?.start
            entry["parent_pid"] = initial?.parent
            entry["command"] = read("$current:command_unavailable") {
                reader.text("/proc/$current/cmdline", 65536).split('\u0000').filter { it.isNotEmpty() }
                    .also { require(it.isNotEmpty()) }
            }
            entry["cwd"] = read("$current:cwd_unavailable") { reader.link("/proc/$current/cwd").also { require(it.startsWith('/')) } }
            val child = read("$current:children_unavailable") {
                if (fallback) rosterChildren(reader, current, scanBefore) else children(reader, current)
            }
            if (child != null) {
                observations[current] = child
                child.pids.forEach { if (it !in processes) pending.add(it) }
            }
            entry["child_pids"] = child?.pids.orEmpty()
            entry["task_ids"] = child?.tids.orEmpty()
            entry["children_complete"] = child != null
            val maps = read("$current:maps_unavailable") { reader.text("/proc/$current/maps", 4 * 1024 * 1024).also { require(it.isNotBlank()) } }
            val model = mutableListOf<String>()
            val native = mutableListOf<String>()
            if (maps != null) for (line in maps.lineSequence().filter { it.isNotBlank() }) {
                val fields = line.trim().split(Regex("\\s+"), limit = 6)
                if (fields.size < 5 || !Regex("[0-9a-f]+-[0-9a-f]+").matches(fields[0]) || fields[4].toLongOrNull() == null) {
                    localErrors += "maps_invalid"; continue
                }
                val path = fields.getOrNull(5) ?: continue
                val lower = path.lowercase()
                if (lower.contains(".gguf") || lower.contains(".safetensors")) model += path
                if (path.contains("libqlh_llama_jni.so") || path.contains("libllama.so") || path.contains("libggml")) native += path
            }
            entry["maps_complete"] = maps != null && localErrors.isEmpty()
            entry["maps_sha256"] = maps?.let { ReleaseRuntimeProbe.sha256(it.toByteArray(Charsets.UTF_8)) }
            entry["model_mappings"] = model.distinct()
            entry["native_mappings"] = native.distinct()
            val sockets = read("$current:connections_unavailable") { reader.connections(current) }
            entry["connections_kind"] = "tcp"
            entry["connections_complete"] = sockets?.complete == true
            entry["connections"] = sockets?.values.orEmpty()
            entry["connection_errors"] = sockets?.errors.orEmpty()
            if (sockets?.complete != true) localErrors += "connections_incomplete"
            entry["identity_after"] = read("$current:identity_after_unavailable") {
                identity(current, reader.text("/proc/$current/stat", 65536)).wire()
            }
            if (entry["identity_after"] != initial?.wire()) localErrors += "identity_changed"
            entry["errors"] = localErrors
        }
        if (pending.isNotEmpty()) errors += "process_limit_exceeded"
        val scanAfter = if (fallback) read("uid_scan_after_failed") { scanUid(reader, pid) } else null
        val scanStable = fallback && scanBefore != null && scanAfter != null &&
            scanBefore.uid == scanAfter.uid && scanBefore.errors.isEmpty() && scanAfter.errors.isEmpty() &&
            scanBefore.identities == scanAfter.identities
        if (fallback) {
            errors += scanAfter?.errors.orEmpty()
            if (!scanStable) errors += "uid_roster_changed_or_incomplete"
        }
        for ((current, entry) in processes) {
            val again = read("$current:children_recheck_failed") {
                if (fallback) rosterChildren(reader, current, scanAfter) else children(reader, current)
            }
            entry["children_complete"] = again != null && again == observations[current] && (!fallback || scanStable)
            if (again != observations[current]) errors += "$current:children_changed"
            val finalIdentity = read("$current:final_identity_unavailable") { identity(current, reader.text("/proc/$current/stat", 65536)).wire() }
            if (entry["identity_after"] != finalIdentity) errors += "$current:identity_changed"
            if (fallback && (scanBefore?.identities?.get(current)?.wire() != finalIdentity ||
                scanBefore?.identities?.get(current)?.parent != entry["parent_pid"])) errors += "$current:roster_identity_changed"
            entry["identity_after"] = finalIdentity
            for (child in observations[current]?.pids.orEmpty()) {
                if (processes[child]?.get("parent_pid") != current) errors += "$current:child_not_observed"
            }
        }
        val after = read("root_identity_after_unavailable") { identity(pid, reader.text("/proc/$pid/stat", 65536)) }
        val complete = before != null && before == after && errors.isEmpty() && processes.values.all { entry ->
            entry["created_at"] != null && entry["command"] != null && entry["cwd"] != null &&
                entry["children_complete"] == true && entry["maps_complete"] == true && entry["connections_complete"] == true &&
                (entry["errors"] as List<*>).isEmpty()
        }
        return linkedMapOf(
            "pid" to pid, "created_at" to before?.start, "identity_before" to before?.wire(), "identity_after" to after?.wire(),
            "started_at" to start, "finished_at" to ReleaseRuntimeProbe.unixSeconds(), "complete" to complete,
            "source" to if (fallback) "android_proc_uid_ppid" else "android_proc_task_children",
            "created_at_unit" to "proc_stat_start_ticks",
            "enumeration" to if (fallback) linkedMapOf("method" to "proc_uid_ppid", "scope" to "app_uid_descendants",
                "reason" to "task_children_enoent", "uid" to scanBefore?.uid, "complete" to scanStable,
                "before" to scanBefore?.wire(), "after" to scanAfter?.wire()) else null,
            "processes" to processes.values.toList(), "errors" to errors.distinct(),
        )
    }

    /** Own TCP states use TCP_INFO on duplicated FDs, without namespace network tables. */
    private class AndroidReader(private val self: Int, private val readTcpState: (Int) -> Int) : Reader {
        override fun text(path: String, limit: Int): String {
            val fd = try { Os.open(path, OsConstants.O_RDONLY or OsConstants.O_CLOEXEC, 0) }
                catch (error: ErrnoException) {
                    if (error.errno == OsConstants.ENOENT) throw MissingProcFile(path)
                    throw error
                }
            return ReleaseRuntimeProbe.boundedText(java.io.FileInputStream(fd), limit)
        }
        override fun uid(pid: Int): Int {
            // proc directory ownership can become root for a non-dumpable release
            // process. Read the kernel's real/effective/saved/fs UID facts instead.
            val lines = text("/proc/$pid/status", 256 * 1024).lineSequence().filter { it.startsWith("Uid:") }.toList()
            require(lines.size == 1)
            val values = lines.single().substringAfter(':').trim().split(Regex("\\s+")).map { it.toInt() }
            require(values.size == 4 && values.all { it >= 0 && it == values[0] })
            return values[0]
        }
        override fun list(path: String): List<String> = File(path).list()?.toList() ?: throw IOException("proc enumeration denied")
        override fun link(path: String): String = Os.readlink(path)
        private fun descriptors(pid: Int): Map<Int, String> {
            val sockets = linkedMapOf<Int, String>()
            for (name in list("/proc/$pid/fd")) {
                val fd = name.toIntOrNull() ?: throw IOException("invalid descriptor")
                try {
                    val target = link("/proc/$pid/fd/$fd")
                    if (target.startsWith("socket:[")) sockets[fd] = target.removePrefix("socket:[").removeSuffix("]")
                } catch (error: ErrnoException) {
                    // The directory FD used by enumeration is closed before readlink.
                    if (error.errno != OsConstants.ENOENT) throw error
                }
            }
            return sockets
        }
        override fun connections(pid: Int): Connections {
            val before = descriptors(pid)
            val values = mutableListOf<Map<String, Any?>>()
            val errors = mutableListOf<String>()
            if (pid == self) {
                for ((fd, inode) in before) try {
                    ParcelFileDescriptor.fromFd(fd).use { copy ->
                        require(link("/proc/$pid/fd/$fd") == "socket:[$inode]")
                        require(Os.fstat(copy.fileDescriptor).st_ino.toString() == inode)
                        // JNI first verifies SO_DOMAIN/SO_TYPE/SO_PROTOCOL. Zero is a
                        // positively classified non-TCP socket, never a guessed state.
                        val state = readTcpState(copy.fd)
                        if (state != 0) {
                            val local = Os.getsockname(copy.fileDescriptor) as? InetSocketAddress
                                ?: throw IOException("TCP local endpoint unavailable")
                            val remote = try { Os.getpeername(copy.fileDescriptor) as? InetSocketAddress }
                                catch (error: ErrnoException) { if (error.errno == OsConstants.ENOTCONN) null else throw error }
                            values += tcpInfoConnection(state, listOf(local.address.hostAddress!!, local.port),
                                remote?.let { listOf(it.address.hostAddress!!, it.port) }, inode) + mapOf("fd" to fd)
                        }
                    }
                } catch (_: Exception) { errors += "socket_inspection_failed" }
            } else if (before.isNotEmpty()) {
                // /proc/<pid>/net is namespace-wide; include only this PID's FD inodes.
                val rows = mutableMapOf<String, Map<String, Any?>>()
                for (family in listOf("tcp", "tcp6")) {
                    for ((inode, value) in parseTcpTable(text("/proc/$pid/net/$family", 4 * 1024 * 1024))) rows[inode] = value
                }
                val unix = text("/proc/$pid/net/unix", 4 * 1024 * 1024).lineSequence().drop(1)
                    .filter { it.isNotBlank() }.map { it.trim().split(Regex("\\s+"))[6] }.toSet()
                for (inode in before.values.toSet()) {
                    if (inode in rows) values += rows.getValue(inode)
                    else if (inode !in unix) errors += "socket_family_or_state_unavailable"
                }
            }
            val after = descriptors(pid)
            if (before != after) errors += "socket_descriptors_changed"
            return Connections(values, errors.isEmpty(), errors.distinct())
        }
    }
    internal fun tcpInfoConnection(state: Int, local: List<Any>, remote: List<Any>?, inode: String): Map<String, Any?> {
        require(state in 1..11 && (state != 1 || remote != null))
        return linkedMapOf("socket_inode" to inode, "status" to tcpState(state),
            "state_source" to "jni_tcp_info", "tcp_state" to state, "local" to local, "remote" to remote)
    }

    private fun tcpState(state: Int): String = listOf("UNKNOWN", "ESTABLISHED", "SYN_SENT", "SYN_RECV", "FIN_WAIT1",
        "FIN_WAIT2", "TIME_WAIT", "CLOSE", "CLOSE_WAIT", "LAST_ACK", "LISTEN", "CLOSING").getOrElse(state) { "UNKNOWN" }

    internal fun parseTcpTable(raw: String): Map<String, Map<String, Any?>> {
        require(raw.lineSequence().first().contains("local_address"))
        fun endpoint(raw: String): List<Any> {
            val (address, port) = raw.split(':').also { require(it.size == 2) }
            require(address.length in setOf(8, 32))
            val bytes = address.chunked(2).map { it.toInt(16).toByte() }.toByteArray()
            // Linux prints each native 32-bit word in host byte order.
            if (ByteOrder.nativeOrder() == ByteOrder.LITTLE_ENDIAN) for (i in bytes.indices step 4) {
                val word = bytes.copyOfRange(i, i + 4).reversedArray(); word.copyInto(bytes, i)
            }
            return listOf(InetAddress.getByAddress(bytes).hostAddress!!, port.toInt(16))
        }
        val rows = linkedMapOf<String, Map<String, Any?>>()
        for (line in raw.lineSequence().drop(1).filter { it.isNotBlank() }) {
            val fields = line.trim().split(Regex("\\s+"))
            require(fields.size >= 10)
            val inode = fields[9].also { require(it.toLong() >= 0) }
            rows[inode] = linkedMapOf("socket_inode" to inode, "status" to tcpState(fields[3].toInt(16)),
                "local" to endpoint(fields[1]), "remote" to endpoint(fields[2]))
        }
        return rows
    }
}
