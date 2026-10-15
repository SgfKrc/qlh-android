package com.qlh.inference.status

import com.google.gson.GsonBuilder
import com.qlh.inference.worker.TaskWorkerAttemptState
import com.qlh.inference.worker.TaskWorkerConnectionState
import com.qlh.inference.worker.TaskWorkerSnapshot
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withTimeout
import java.util.concurrent.Callable
import java.util.concurrent.SynchronousQueue
import java.util.concurrent.ThreadPoolExecutor
import java.util.concurrent.TimeUnit

/** Read-only release evidence. Only explicit map keys cross the R8/JSON boundary. */
internal object ReleaseRuntimeProbe {
    const val PREFIX = "QLH_RELEASE_PROBE: "
    private val noncePattern = Regex("[0-9a-f]{32}")
    private val gson = GsonBuilder().serializeNulls().create()

    enum class ErrorCode {
        PROBE_TIMEOUT, PROBE_BUSY, PROBE_INTERRUPTED, PROBE_FAILED,
        PACKAGE_INFO_INVALID, ENGINE_MISSING, ENGINE_CLOSED,
        NATIVE_LIBRARY_UNAVAILABLE, NATIVE_QUERY_FAILED, NATIVE_BACKEND_INVALID,
        NATIVE_STUB_BACKEND, MODEL_INFO_FAILED, MODEL_STATE_INVALID,
        WORKER_SNAPSHOT_CHANGED, WORKER_STATE_INVALID, INFERENCE_MODE_CHANGED,
        INFERENCE_MODE_UNSUPPORTED,
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
    )

    data class Snapshot(
        val inferenceMode: String?,
        val worker: TaskWorkerSnapshot,
        val runtime: RuntimeState,
        val capturedAt: Double = unixSeconds(),
        val errorCode: ErrorCode? = null,
    )

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
        readRuntime: () -> RuntimeState,
    ): Snapshot {
        val workerBefore = readWorker()
        val modeBefore = readMode()
        currentCoroutineContext().ensureActive()
        val runtime = readRuntime()
        val modeAfter = readMode()
        val workerAfter = readWorker()
        currentCoroutineContext().ensureActive()
        return Snapshot(
            inferenceMode = modeBefore,
            worker = workerBefore,
            runtime = runtime,
            errorCode = when {
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
            "error_code" to error?.name,
        )
        if (process != null) fields["process"] = process
        return PREFIX + gson.toJson(fields)
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
