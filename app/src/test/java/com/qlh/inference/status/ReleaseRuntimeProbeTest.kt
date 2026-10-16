package com.qlh.inference.status

import com.google.gson.JsonObject
import com.google.gson.JsonParser
import com.qlh.inference.status.ReleaseRuntimeProbe.ErrorCode
import com.qlh.inference.status.ReleaseRuntimeProbe.ModelState
import com.qlh.inference.status.ReleaseRuntimeProbe.NativeState
import com.qlh.inference.status.ReleaseRuntimeProbe.RuntimeState
import com.qlh.inference.status.ReleaseRuntimeProbe.Snapshot
import com.qlh.inference.worker.TaskWorkerAttemptSnapshot
import com.qlh.inference.worker.TaskWorkerAttemptState
import com.qlh.inference.worker.TaskWorkerConnectionState
import com.qlh.inference.worker.TaskWorkerSnapshot
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineStart
import kotlinx.coroutines.TimeoutCancellationException
import kotlinx.coroutines.async
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import org.junit.Assert.*
import org.junit.Test
import java.util.concurrent.CountDownLatch
import java.util.concurrent.ExecutionException
import java.util.concurrent.RejectedExecutionException
import java.util.concurrent.TimeUnit
import java.util.concurrent.TimeoutException

class ReleaseRuntimeProbeTest {
    private val nonce = "0123456789abcdef0123456789abcdef"
    private val identity = ReleaseRuntimeProbe.PackageIdentity(1234, "com.qlh.inference", "1.2.3", 123L)
    private val stopped = TaskWorkerSnapshot()
    private val ready = TaskWorkerSnapshot(
        connection = TaskWorkerConnectionState.READY, processRunning = true, connectionAlive = true,
    )
    private val backend = mapOf(
        "system_info" to "NEON = 1 | DOTPROD = 1", "backend_device_count" to "1", "backend_devices" to "CPU(cpu)",
    )
    private val native = NativeState.fromBackend(backend)

    private fun snapshot(
        mode: String = "distributed", worker: TaskWorkerSnapshot = stopped,
        model: ModelState = ModelState(), nativeState: NativeState = native,
    ) = Snapshot(mode, worker, RuntimeState(nativeState, model), capturedAt = 1_792_000_000.125)

    private fun json(snapshot: Snapshot, packageIdentity: ReleaseRuntimeProbe.PackageIdentity = identity): JsonObject {
        val line = ReleaseRuntimeProbe.line(nonce, packageIdentity, snapshot)
        assertTrue(line.startsWith(ReleaseRuntimeProbe.PREFIX))
        assertFalse(line.contains('\n'))
        assertFalse(line.contains('\r'))
        return JsonParser.parseString(line.removePrefix(ReleaseRuntimeProbe.PREFIX)).asJsonObject
    }

    private fun assertFailed(snapshot: Snapshot, code: ErrorCode) {
        val output = json(snapshot)
        assertEquals(code.name, output["error_code"].asString)
        assertFalse(output.getAsJsonObject("native")["available"].asBoolean)
        assertFalse(output.getAsJsonObject("native")["error_code"].isJsonNull)
        assertEquals(nonce, output["nonce"].asString)
        assertEquals(identity.pid, output["pid"].asInt)
    }

    @Test fun `only exact switch and lowercase 32 hex nonce are accepted`() {
        assertEquals(nonce, ReleaseRuntimeProbe.nonce(arrayOf("--qlh-release-probe", nonce)))
        listOf<Array<out String>?>(
            null, emptyArray(), arrayOf("--qlh-release-probe"), arrayOf(nonce), arrayOf("--other", nonce),
            arrayOf("--qlh-release-probe", nonce, "extra"), arrayOf("--qlh-release-probe", nonce.uppercase()),
            arrayOf("--qlh-release-probe", nonce.dropLast(1)), arrayOf("--qlh-release-probe", "g".repeat(32)),
            arrayOf("--qlh-release-probe", nonce + "\n"), arrayOf("--qlh-release-probe", " " + nonce),
        ).forEach { assertNull(ReleaseRuntimeProbe.nonce(it)) }
    }

    @Test fun `fresh distributed installation proves client role without joining a cluster`() {
        val output = json(snapshot())
        assertEquals(setOf(
            "schema_version", "nonce", "pid", "captured_at", "package_name", "version_name", "version_code",
            "service_running", "inference_mode", "role", "worker", "native", "model", "error_code", "runtime_receipt", "probe_kind",
        ), output.keySet())
        assertEquals(1, output["schema_version"].asInt)
        assertEquals("client", output["role"].asString)
        assertEquals("distributed", output["inference_mode"].asString)
        assertEquals("com.qlh.inference", output["package_name"].asString)
        assertEquals("1.2.3", output["version_name"].asString)
        assertEquals(123L, output["version_code"].asLong)
        assertEquals(1_792_000_000.125, output["captured_at"].asDouble, 0.0)
        assertTrue(output["service_running"].asBoolean)
        assertTrue(output["error_code"].isJsonNull)
        val worker = output.getAsJsonObject("worker")
        assertEquals(setOf("process_running", "connection_alive", "connection", "active_attempt_state"), worker.keySet())
        assertFalse(worker["process_running"].asBoolean)
        assertFalse(worker["connection_alive"].asBoolean)
        assertEquals("STOPPED", worker["connection"].asString)
        assertEquals("IDLE", worker["active_attempt_state"].asString)
        assertTrue(output.getAsJsonObject("native")["available"].asBoolean)
        assertTrue(output.getAsJsonObject("native")["error_code"].isJsonNull)
        val model = output.getAsJsonObject("model")
        assertFalse(model["loaded"].asBoolean)
        assertFalse(model["full_model_loaded"].asBoolean)
        assertEquals(0, model.getAsJsonArray("layer_range").size())
        assertTrue(model["native_layer_count"].isJsonNull)
    }

    @Test fun `package identity is dynamic and long version codes retain precision`() {
        val output = json(snapshot(), identity.copy(packageName = "com.qlh.inference.debug", versionCode = 5_000_000_000L))
        assertEquals("com.qlh.inference.debug", output["package_name"].asString)
        assertEquals(5_000_000_000L, output["version_code"].asLong)
    }

    @Test fun `local full model and worker crop report actual ranges and native layer counts`() {
        val local = json(snapshot(mode = "local", model = ModelState(true, emptyList(), 28)))
        assertEquals("local", local["role"].asString)
        assertTrue(local["error_code"].isJsonNull)
        assertTrue(local.getAsJsonObject("model")["full_model_loaded"].asBoolean)
        assertEquals(28, local.getAsJsonObject("model")["native_layer_count"].asInt)
        val worker = json(snapshot(worker = ready, model = ModelState(true, listOf(8, 12), 4)))
        assertEquals("worker", worker["role"].asString)
        assertTrue(worker["error_code"].isJsonNull)
        assertFalse(worker.getAsJsonObject("model")["full_model_loaded"].asBoolean)
        assertEquals(listOf(8, 12), worker.getAsJsonObject("model").getAsJsonArray("layer_range").map { it.asInt })
        assertEquals(4, worker.getAsJsonObject("model")["native_layer_count"].asInt)
    }

    @Test fun `distributed worker full model fails the release pipeline contract`() {
        val sampled = snapshot(worker = ready, model = ModelState(true, emptyList(), 28))
        assertFailed(sampled, ErrorCode.MODEL_STATE_INVALID)
        val output = json(sampled)
        assertEquals("worker", output["role"].asString)
        // Keep the actual model evidence even though the role/model combination is rejected.
        assertTrue(output.getAsJsonObject("model")["loaded"].asBoolean)
        assertTrue(output.getAsJsonObject("model")["full_model_loaded"].asBoolean)
        assertEquals(28, output.getAsJsonObject("model")["native_layer_count"].asInt)
    }

    @Test fun `stopped retry exhausted state accepts terminal attempts and rejects active attempts`() {
        TaskWorkerAttemptState.entries.forEach { state ->
            val worker = TaskWorkerSnapshot(
                connection = TaskWorkerConnectionState.RETRY_EXHAUSTED,
                activeAttempt = TaskWorkerAttemptSnapshot(state = state),
            )
            val sampled = snapshot(worker = worker)
            if (state in setOf(TaskWorkerAttemptState.OFFERED, TaskWorkerAttemptState.RUNNING, TaskWorkerAttemptState.CANCELLING)) {
                assertFailed(sampled, ErrorCode.WORKER_STATE_INVALID)
            } else {
                val output = json(sampled)
                assertTrue(output["error_code"].isJsonNull)
                assertEquals("client", output["role"].asString)
                assertEquals(state.name, output.getAsJsonObject("worker")["active_attempt_state"].asString)
            }
        }
    }

    @Test fun `connecting worker role comes from its process without claiming a live connection`() {
        val worker = TaskWorkerSnapshot(connection = TaskWorkerConnectionState.CONNECTING, processRunning = true)
        val output = json(snapshot(worker = worker))
        assertEquals("worker", output["role"].asString)
        assertFalse(output.getAsJsonObject("worker")["connection_alive"].asBoolean)
        assertTrue(output["error_code"].isJsonNull)
    }

    @Test fun `impossible worker states fail closed and active attempts emit actual enum names`() {
        listOf(
            ready.copy(processRunning = false), ready.copy(connectionAlive = false),
            stopped.copy(connectionAlive = true), stopped.copy(processRunning = true),
            stopped.copy(connection = TaskWorkerConnectionState.CONNECTING),
            stopped.copy(activeAttempt = TaskWorkerAttemptSnapshot(state = TaskWorkerAttemptState.RUNNING)),
        ).forEach { assertFailed(snapshot(worker = it), ErrorCode.WORKER_STATE_INVALID) }
        assertFailed(snapshot(mode = "local", worker = ready), ErrorCode.WORKER_STATE_INVALID)
        val worker = ready.copy(activeAttempt = TaskWorkerAttemptSnapshot(state = TaskWorkerAttemptState.CANCELLING))
        assertEquals("CANCELLING", json(snapshot(worker = worker)).getAsJsonObject("worker")["active_attempt_state"].asString)
    }

    @Test fun `empty malformed and stub backends cannot prove native readiness`() {
        listOf(
            emptyMap(), backend - "system_info", backend - "backend_devices", backend - "backend_device_count",
            backend + ("system_info" to " "), backend + ("backend_devices" to ""),
            backend + ("backend_device_count" to "0"), backend + ("backend_device_count" to "-1"),
            backend + ("backend_device_count" to "not-an-int"),
        ).forEach { assertFailed(snapshot(nativeState = NativeState.fromBackend(it)), ErrorCode.NATIVE_BACKEND_INVALID) }
        listOf(
            backend + ("system_info" to "QLH Lite STUB"), backend + ("backend_devices" to "stub_backend(cpu)"),
        ).forEach { assertFailed(snapshot(nativeState = NativeState.fromBackend(it)), ErrorCode.NATIVE_STUB_BACKEND) }
        assertFailed(snapshot(nativeState = NativeState("stub", 1, "CPU(cpu)")), ErrorCode.NATIVE_STUB_BACKEND)
    }

    @Test fun `invalid model metadata and ranges fail closed`() {
        listOf(
            ModelState(true, emptyList(), null), ModelState(true, emptyList(), 0),
            ModelState(false, listOf(8, 12), null), ModelState(false, emptyList(), 4),
            ModelState(true, listOf(8), 4), ModelState(true, listOf(-1, 3), 4),
            ModelState(true, listOf(8, 8), 4), ModelState(true, listOf(8, 12), 28),
        ).forEach { assertFailed(snapshot(worker = ready, model = it), ErrorCode.MODEL_STATE_INVALID) }
        assertFailed(snapshot(mode = "local", model = ModelState(true, listOf(8, 12), 4)), ErrorCode.MODEL_STATE_INVALID)
        assertFailed(snapshot(model = ModelState(true, emptyList(), 28)), ErrorCode.MODEL_STATE_INVALID)
    }

    @Test fun `unsupported modes and missing package metadata cannot produce green evidence`() {
        assertFailed(snapshot(mode = "fallback"), ErrorCode.INFERENCE_MODE_UNSUPPORTED)
        val output = json(snapshot(), identity.copy(versionName = null, versionCode = null))
        assertEquals(ErrorCode.PACKAGE_INFO_INVALID.name, output["error_code"].asString)
        assertFalse(output.getAsJsonObject("native")["available"].asBoolean)
    }

    @Test fun `closed missing and failed runtimes retain nonce and pid without green evidence`() {
        listOf(ErrorCode.ENGINE_CLOSED, ErrorCode.ENGINE_MISSING, ErrorCode.NATIVE_QUERY_FAILED,
            ErrorCode.NATIVE_LIBRARY_UNAVAILABLE, ErrorCode.MODEL_INFO_FAILED, ErrorCode.PROBE_TIMEOUT,
            ErrorCode.PROBE_BUSY, ErrorCode.PROBE_FAILED, ErrorCode.PROBE_INTERRUPTED,
        ).forEach { assertFailed(ReleaseRuntimeProbe.failure(it), it) }
        assertFailed(snapshot().copy(runtime = RuntimeState(native, errorCode = ErrorCode.MODEL_INFO_FAILED)), ErrorCode.MODEL_INFO_FAILED)
    }

    @Test fun `only native allowlist and worker enums cross JSON boundary and strings stay on one line`() {
        val secret = "private-uri-token-and-prompt"
        val worker = stopped.copy(lastErrorCode = secret, lastErrorMessage = secret,
            activeAttempt = TaskWorkerAttemptSnapshot(outputSha256 = secret, errorCode = secret))
        val state = NativeState.fromBackend(backend + mapOf("token" to secret, "name" to secret, "prompt" to secret))
        val line = ReleaseRuntimeProbe.line(nonce, identity, snapshot(worker = worker, nativeState = state))
        assertFalse(line.contains(secret))
        assertEquals(setOf("available", "system_info", "backend_device_count", "backend_devices", "error_code"),
            json(snapshot(nativeState = state)).getAsJsonObject("native").keySet())
        val escaped = json(snapshot(nativeState = NativeState("NEON=1\nDOT=1\r", 1, "CPU(cpu)\n")))
        assertEquals("NEON=1\nDOT=1\r", escaped.getAsJsonObject("native")["system_info"].asString)
    }

    @Test fun `any worker snapshot change during sampling fails closed`() = runBlocking {
        val changed = stopped.copy(reconnectAttempt = 1)
        var reads = 0
        val sampled = ReleaseRuntimeProbe.capture({ "distributed" }, { if (reads++ == 0) stopped else changed }) { RuntimeState(native) }
        assertFailed(sampled, ErrorCode.WORKER_SNAPSHOT_CHANGED)
    }

    @Test fun `mode changes during sampling fail closed`() = runBlocking {
        var reads = 0
        val sampled = ReleaseRuntimeProbe.capture({ if (reads++ == 0) "distributed" else "local" }, { stopped }) { RuntimeState(native) }
        assertFailed(sampled, ErrorCode.INFERENCE_MODE_CHANGED)
    }

    @Test fun `stable capture queries runtime once and stamps Unix seconds`() = runBlocking {
        var calls = 0
        val before = ReleaseRuntimeProbe.unixSeconds()
        val sampled = ReleaseRuntimeProbe.capture({ "distributed" }, { stopped }) { calls++; RuntimeState(native) }
        assertEquals(1, calls)
        assertTrue(sampled.capturedAt in before..ReleaseRuntimeProbe.unixSeconds())
        assertTrue(json(sampled)["error_code"].isJsonNull)
    }

    @Test fun `capture propagates cancellation without sampling native`() = runBlocking {
        var calls = 0
        try {
            ReleaseRuntimeProbe.capture({ throw CancellationException("private error") }, { stopped }) { calls++; RuntimeState(native) }
            fail("cancellation must escape")
        } catch (_: CancellationException) { assertEquals(0, calls) }
    }

    @Test fun `cancellation during native query cannot publish a successful sample`() = runBlocking {
        var emitted = false
        val read = async(start = CoroutineStart.UNDISPATCHED) {
            ReleaseRuntimeProbe.capture({ "distributed" }, { stopped }) {
                coroutineContext[kotlinx.coroutines.Job]!!.cancel()
                RuntimeState(native)
            }
            emitted = true
        }
        try { read.await(); fail("cancellation must escape") } catch (_: CancellationException) { }
        assertFalse(emitted)
    }

    @Test fun `bounded reader cancels model mutex wait without running native callback`() {
        val mutex = Mutex(locked = true)
        var queried = false
        ReleaseRuntimeProbe.BoundedReader(750).use { reader ->
            try {
                reader.read { mutex.withLock { queried = true } }
                fail("mutex wait must time out")
            } catch (error: ExecutionException) { assertTrue(error.cause is TimeoutCancellationException) }
            finally { mutex.unlock() }
        }
        assertFalse(queried)
    }

    @Test fun `bounded reader returns results and preserves cancellation`() {
        ReleaseRuntimeProbe.BoundedReader(2_000).use { assertEquals(7, it.read { 7 }) }
        ReleaseRuntimeProbe.BoundedReader(2_000).use { reader ->
            try { reader.read { throw CancellationException("private error") }; fail("cancellation must escape") }
            catch (error: ExecutionException) { assertTrue(error.cause is CancellationException) }
        }
    }

    @Test fun `stuck JNI times out and further requests cannot accumulate background work`() {
        val entered = CountDownLatch(1)
        val release = CountDownLatch(1)
        val exited = CountDownLatch(1)
        val reader = ReleaseRuntimeProbe.BoundedReader(750)
        try {
            val before = System.nanoTime()
            try {
                reader.read {
                    entered.countDown()
                    try {
                        while (release.count != 0L) {
                            try { release.await() } catch (_: InterruptedException) { /* Simulate noninterruptible JNI. */ }
                        }
                    } finally { exited.countDown() }
                    RuntimeState(native)
                }
                fail("stuck JNI must time out")
            } catch (_: TimeoutException) { }
            assertTrue(entered.await(1, TimeUnit.SECONDS))
            assertTrue(TimeUnit.NANOSECONDS.toMillis(System.nanoTime() - before) < 2_000)
            try { reader.read { fail("another JNI call must be rejected") }; fail("no task may be queued") }
            catch (_: RejectedExecutionException) { }
        } finally {
            release.countDown()
            assertTrue(exited.await(2, TimeUnit.SECONDS))
            reader.close()
        }
    }
}
