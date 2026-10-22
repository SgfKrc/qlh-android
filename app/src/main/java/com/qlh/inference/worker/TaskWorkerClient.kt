package com.qlh.inference.worker

import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.CoroutineStart
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.NonCancellable
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.TimeoutCancellationException
import kotlinx.coroutines.cancel
import kotlinx.coroutines.cancelAndJoin
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlin.coroutines.resumeWithException
import kotlinx.coroutines.withContext
import kotlinx.coroutines.withTimeout
import java.io.BufferedInputStream
import java.io.BufferedOutputStream
import java.io.DataInputStream
import java.io.DataOutputStream
import java.io.EOFException
import java.net.InetSocketAddress
import java.net.Socket
import java.nio.charset.StandardCharsets
import java.security.SecureRandom
import java.util.Base64
import java.util.Locale
import java.util.UUID
import javax.crypto.Mac
import javax.crypto.spec.SecretKeySpec
import com.google.gson.GsonBuilder
import com.google.gson.JsonParser
import com.qlh.inference.logging.QlhLogger

enum class TaskWorkerConnectionState {
    STOPPED,
    CONNECTING,
    HELLO_SENT,
    READY,
    BACKING_OFF,
    RETRY_EXHAUSTED,
}

enum class TaskWorkerAttemptState {
    IDLE,
    OFFERED,
    RUNNING,
    CANCELLING,
    SUCCEEDED,
    FAILED,
    CANCELLED,
    LOST,
}

data class TaskWorkerAttemptSnapshot(
    val identity: TaskWorkerAttemptIdentity? = null,
    val state: TaskWorkerAttemptState = TaskWorkerAttemptState.IDLE,
    val leaseExpiresAtMs: Long = 0L,
    val outputSha256: String? = null,
    val errorCode: String? = null,
    val retryable: Boolean = false,
)

data class TaskWorkerSnapshot(
    val connection: TaskWorkerConnectionState = TaskWorkerConnectionState.STOPPED,
    val reconnectAttempt: Int = 0,
    val nextRetryAtMs: Long = 0L,
    val lastErrorCode: String? = null,
    val lastErrorMessage: String? = null,
    val activeAttempt: TaskWorkerAttemptSnapshot = TaskWorkerAttemptSnapshot(),
    val processRunning: Boolean = false,
    val connectionAlive: Boolean = false,
) {
    val failureKind: String? get() = lastErrorCode?.let { code ->
        when {
            code == "registration_rejected" || code == "hello_rejected" ||
                code == "node_not_admitted" || code == "not_master" ||
                code.contains("identity") || code.contains("auth") ||
                code.contains("credential") || code.contains("secret_epoch") -> "identity"
            code.startsWith("transport_") || code.startsWith("heartbeat_") -> "network"
            else -> "protocol"
        }
    }
}

/**
 * Transport-independent lifecycle fencing for an Android Full Worker.
 *
 * A late result is accepted only for the current attempt identity and lease.
 * Disconnecting marks the active attempt LOST so a coordinator can safely
 * re-dispatch it without allowing an old socket to publish a result later.
 */
class TaskWorkerStateMachine(
    private val baseBackoffMs: Long = 1_000L,
    private val maxBackoffMs: Long = 30_000L,
    private val maxConsecutiveFailures: Int = 8,
    private val maxAttemptHistory: Int = 4096,
) {
    init { require(maxConsecutiveFailures > 0 && maxAttemptHistory > 0) }

    private var snapshot = TaskWorkerSnapshot()
    // Identity/deadline only: no hidden/result retention and no response replay promise.
    // Keep across reconnects; only expired leases may be removed to make room.
    private val admittedAttempts = LinkedHashMap<TaskWorkerAttemptIdentity, Long>()

    @Synchronized
    fun snapshot(): TaskWorkerSnapshot = snapshot

    @Synchronized
    fun start(): TaskWorkerSnapshot {
        if (snapshot.connection == TaskWorkerConnectionState.STOPPED) {
            snapshot = snapshot.copy(
                connection = TaskWorkerConnectionState.CONNECTING,
                lastErrorCode = null,
                lastErrorMessage = null,
            )
        }
        return snapshot
    }

    @Synchronized
    fun onConnected(): TaskWorkerSnapshot {
        if (snapshot.connection == TaskWorkerConnectionState.CONNECTING) {
            snapshot = snapshot.copy(connection = TaskWorkerConnectionState.HELLO_SENT)
        }
        return snapshot
    }

    @Synchronized
    fun onHelloAck(accepted: Boolean, reasonCode: String, reasonMessage: String, nowMs: Long): TaskWorkerSnapshot {
        if (snapshot.connection != TaskWorkerConnectionState.HELLO_SENT) return snapshot
        snapshot = if (accepted) {
            snapshot.copy(
                connection = TaskWorkerConnectionState.READY,
                reconnectAttempt = 0,
                nextRetryAtMs = 0L,
                lastErrorCode = null,
                lastErrorMessage = null,
            )
        } else {
            backoff(nowMs, reasonCode.ifEmpty { "hello_rejected" }, reasonMessage.ifEmpty { "coordinator rejected hello" })
        }
        return snapshot
    }

    @Synchronized
    fun onDisconnected(nowMs: Long, code: String, message: String): TaskWorkerSnapshot {
        if (snapshot.connection == TaskWorkerConnectionState.STOPPED) return snapshot
        if (snapshot.connection in setOf(TaskWorkerConnectionState.BACKING_OFF, TaskWorkerConnectionState.RETRY_EXHAUSTED)) return snapshot
        val attempt = snapshot.activeAttempt
        snapshot = backoff(nowMs, code.ifEmpty { "transport_disconnected" }, message).copy(
            activeAttempt = if (attempt.state in ACTIVE_ATTEMPT_STATES) {
                attempt.copy(state = TaskWorkerAttemptState.LOST)
            } else {
                attempt
            },
        )
        return snapshot
    }

    @Synchronized
    fun retryIfDue(nowMs: Long): TaskWorkerSnapshot {
        if (snapshot.connection == TaskWorkerConnectionState.BACKING_OFF &&
            nowMs >= snapshot.nextRetryAtMs
        ) {
            snapshot = snapshot.copy(connection = TaskWorkerConnectionState.CONNECTING)
        }
        return snapshot
    }

    @Synchronized
    fun stop(): TaskWorkerSnapshot {
        val current = snapshot.activeAttempt
        snapshot = snapshot.copy(
            connection = TaskWorkerConnectionState.STOPPED,
            nextRetryAtMs = 0L,
            activeAttempt = if (current.state in ACTIVE_ATTEMPT_STATES) {
                current.copy(state = TaskWorkerAttemptState.LOST)
            } else {
                current
            },
        )
        return snapshot
    }

    @Synchronized
    fun offer(identity: TaskWorkerAttemptIdentity, leaseExpiresAtMs: Long, nowMs: Long): Boolean =
        admitOffer(identity, leaseExpiresAtMs, nowMs) == null

    /** Atomically reserve both the logical slot and its bounded lease replay fence. */
    @Synchronized
    fun admitOffer(identity: TaskWorkerAttemptIdentity, leaseExpiresAtMs: Long, nowMs: Long): String? {
        expireActiveAttempt(nowMs)
        admittedAttempts.entries.removeAll { it.value <= nowMs }
        if (snapshot.connection != TaskWorkerConnectionState.READY || leaseExpiresAtMs <= nowMs) {
            return "worker_busy_or_lease_invalid"
        }
        if (identity in admittedAttempts) return "worker_attempt_already_admitted"
        if (snapshot.activeAttempt.state in ACTIVE_ATTEMPT_STATES) return "worker_busy_or_lease_invalid"
        if (admittedAttempts.size >= maxAttemptHistory) return "worker_attempt_history_full"
        admittedAttempts[identity] = leaseExpiresAtMs
        snapshot = snapshot.copy(
            activeAttempt = TaskWorkerAttemptSnapshot(
                identity = identity,
                state = TaskWorkerAttemptState.OFFERED,
                leaseExpiresAtMs = leaseExpiresAtMs,
            ),
        )
        return null
    }

    @Synchronized
    fun markRunning(identity: TaskWorkerAttemptIdentity, nowMs: Long): Boolean = transitionAttempt(
        identity,
        nowMs,
        allowed = setOf(TaskWorkerAttemptState.OFFERED),
        next = TaskWorkerAttemptState.RUNNING,
    )

    @Synchronized
    fun requestCancel(nowMs: Long, expectedIdentity: TaskWorkerAttemptIdentity? = null): TaskWorkerAttemptIdentity? {
        expireActiveAttempt(nowMs)
        val current = snapshot.activeAttempt
        if (current.identity == null || current.state !in setOf(
                TaskWorkerAttemptState.OFFERED,
                TaskWorkerAttemptState.RUNNING,
        ) || current.leaseExpiresAtMs <= nowMs
        ) return null
        if (expectedIdentity != null && current.identity != expectedIdentity) return null
        snapshot = snapshot.copy(activeAttempt = current.copy(state = TaskWorkerAttemptState.CANCELLING))
        return current.identity
    }

    @Synchronized
    fun renew(identity: TaskWorkerAttemptIdentity, leaseExpiresAtMs: Long, nowMs: Long): Boolean {
        expireActiveAttempt(nowMs)
        val current = snapshot.activeAttempt
        if (!sameIdentity(current.identity, identity) || current.state !in ACTIVE_ATTEMPT_STATES) return false
        if (leaseExpiresAtMs <= nowMs) return false
        admittedAttempts[identity] = maxOf(admittedAttempts[identity] ?: 0L, leaseExpiresAtMs)
        snapshot = snapshot.copy(activeAttempt = current.copy(leaseExpiresAtMs = leaseExpiresAtMs))
        return true
    }

    @Synchronized
    fun complete(identity: TaskWorkerAttemptIdentity, outputSha256: String, nowMs: Long): Boolean {
        expireActiveAttempt(nowMs)
        val current = snapshot.activeAttempt
        if (!sameIdentity(current.identity, identity) || current.state != TaskWorkerAttemptState.RUNNING) return false
        if (current.leaseExpiresAtMs <= nowMs) return false
        snapshot = snapshot.copy(
            activeAttempt = current.copy(
                state = TaskWorkerAttemptState.SUCCEEDED,
                outputSha256 = outputSha256,
            ),
        )
        return true
    }

    @Synchronized
    fun fail(identity: TaskWorkerAttemptIdentity, errorCode: String, retryable: Boolean): Boolean {
        val current = snapshot.activeAttempt
        if (!sameIdentity(current.identity, identity) || current.state !in setOf(
                TaskWorkerAttemptState.OFFERED, TaskWorkerAttemptState.RUNNING,
        )) return false
        snapshot = snapshot.copy(
            activeAttempt = current.copy(
                state = TaskWorkerAttemptState.FAILED,
                errorCode = errorCode,
                retryable = retryable,
            ),
        )
        return true
    }

    @Synchronized
    fun cancelled(identity: TaskWorkerAttemptIdentity): Boolean {
        val current = snapshot.activeAttempt
        if (!sameIdentity(current.identity, identity) || current.state != TaskWorkerAttemptState.CANCELLING) return false
        snapshot = snapshot.copy(activeAttempt = current.copy(state = TaskWorkerAttemptState.CANCELLED))
        return true
    }

    private fun transitionAttempt(
        identity: TaskWorkerAttemptIdentity,
        nowMs: Long,
        allowed: Set<TaskWorkerAttemptState>,
        next: TaskWorkerAttemptState,
    ): Boolean {
        expireActiveAttempt(nowMs)
        val current = snapshot.activeAttempt
        if (!sameIdentity(current.identity, identity) || current.state !in allowed) return false
        if (current.leaseExpiresAtMs <= nowMs) return false
        snapshot = snapshot.copy(activeAttempt = current.copy(state = next))
        return true
    }

    private fun expireActiveAttempt(nowMs: Long) {
        val current = snapshot.activeAttempt
        if (current.state in ACTIVE_ATTEMPT_STATES && current.leaseExpiresAtMs <= nowMs) {
            snapshot = snapshot.copy(activeAttempt = current.copy(state = TaskWorkerAttemptState.LOST))
        }
    }

    private fun backoff(nowMs: Long, code: String, message: String): TaskWorkerSnapshot {
        val attempt = (snapshot.reconnectAttempt + 1).coerceAtMost(maxConsecutiveFailures)
        val delay = (baseBackoffMs.coerceAtLeast(1L) * (1L shl (attempt - 1).coerceAtMost(5)))
            .coerceAtMost(maxBackoffMs.coerceAtLeast(baseBackoffMs))
        return snapshot.copy(
            connection = if (attempt >= maxConsecutiveFailures) {
                TaskWorkerConnectionState.RETRY_EXHAUSTED
            } else TaskWorkerConnectionState.BACKING_OFF,
            reconnectAttempt = attempt,
            nextRetryAtMs = if (attempt >= maxConsecutiveFailures) 0L else nowMs + delay,
            lastErrorCode = code,
            lastErrorMessage = message,
        )
    }

    private fun sameIdentity(left: TaskWorkerAttemptIdentity?, right: TaskWorkerAttemptIdentity): Boolean = left == right

    companion object {
        private val ACTIVE_ATTEMPT_STATES = setOf(
            TaskWorkerAttemptState.OFFERED,
            TaskWorkerAttemptState.RUNNING,
            TaskWorkerAttemptState.CANCELLING,
        )
    }
}

interface TaskWorkerTransport {
    val isBinaryHiddenNegotiated: Boolean get() = false
    suspend fun send(envelope: TaskWorkerEnvelope)
    suspend fun receive(): TaskWorkerEnvelope?
    suspend fun receiveEvent(): TaskWorkerInboundEvent =
        receive()?.let { TaskWorkerInboundEvent.Envelope(it) }
            ?: TaskWorkerInboundEvent.Closed
    suspend fun sendModelSupply(data: com.google.gson.JsonObject) { throw TaskWorkerProtocolException("supply transport unavailable", "supply_transport_unavailable") }
    suspend fun sendModelPrepare(data: com.google.gson.JsonObject) { throw TaskWorkerProtocolException("prepare transport unavailable", "prepare_transport_unavailable") }
    fun requireControlAuthority() = Unit
    suspend fun sendHeartbeat(nodeId: String, sentAtMs: Long) = Unit
    /** Configure local receive support; outbound binary also needs a positive master ACK. */
    fun setBinaryHiddenEnabled(enabled: Boolean) = Unit
    suspend fun close()
}

sealed class TaskWorkerInboundEvent {
    data class Envelope(val value: TaskWorkerEnvelope) : TaskWorkerInboundEvent()
    data class ModelSupply(val data: com.google.gson.JsonObject) : TaskWorkerInboundEvent()
    data class ModelPrepare(val data: com.google.gson.JsonObject) : TaskWorkerInboundEvent()
    data object HeartbeatAck : TaskWorkerInboundEvent()
    data object Closed : TaskWorkerInboundEvent()
}

data class TaskWorkerRegistration(
    val nodeId: String,
    val clusterSecret: String,
    val clusterSecretEpoch: Int = 1,
    val hostname: String,
    val networkType: String,
    val deviceInfo: Map<String, Any?>,
    val modelSha256: String = "",
)

fun interface TaskWorkerTransportFactory {
    suspend fun connect(host: String, port: Int): TaskWorkerTransport
}

/** Length-prefixed transport compatible with the existing PC TCP framing. */
internal class SocketTaskWorkerTransport(
    private val socket: Socket,
    private val maxFrameBytes: Int = TaskWorkerProtocol.MAX_BINARY_FRAME_BYTES,
    private val controlFence: TcpControlFence = TcpControlFence(),
) : TaskWorkerTransport {
    private val input = DataInputStream(BufferedInputStream(socket.getInputStream()))
    private val output = DataOutputStream(BufferedOutputStream(socket.getOutputStream()))
    private val sendLock = Any()
    private val gson = GsonBuilder().disableHtmlEscaping().create()

    private val binaryHiddenLock = Any()
    private var binaryHiddenEnabled: Boolean = false
    private var masterBinaryHiddenEnabled: Boolean = false
    private var binaryHiddenGeneration = 0L
    private var binaryHiddenClosed = false
    private val pendingHellos = java.util.ArrayDeque<Pair<Long, Boolean>>()
    private val helloAckReplay = TaskWorkerReplayCache()

    override val isBinaryHiddenNegotiated: Boolean
        get() = synchronized(binaryHiddenLock) { binaryHiddenEnabled && masterBinaryHiddenEnabled }

    override fun setBinaryHiddenEnabled(enabled: Boolean) {
        synchronized(binaryHiddenLock) {
            binaryHiddenEnabled = enabled && !binaryHiddenClosed
            masterBinaryHiddenEnabled = false
            binaryHiddenGeneration++
        }
    }

    private data class OuterFrame(val value: com.google.gson.JsonObject, val binaryHidden: Boolean = false)

    private suspend fun sendOuter(
        type: String,
        data: Any? = null,
        binaryBody: ByteArray? = null,
    ) = withContext(Dispatchers.IO) {
        synchronized(sendLock) { writeOuter(type, data, binaryBody) }
    }

    /** Caller holds sendLock so selection and the actual frame write share one order. */
    private fun writeOuter(
        type: String,
        data: Any?,
        binaryBody: ByteArray?,
        beforeWrite: () -> Unit = {},
    ) {
        val dataJson = data?.let { ",\"data\":${gson.toJson(it)}" } ?: ""
        // ★ 票 21a（COMM-WIRE-01）：二进制体独立成段，元数据里只留格式标记与长度。
        val format = if (binaryBody == null) "json" else "binary"
        val binaryJson = binaryBody?.let { ",\"binary_bytes\":${it.size}" } ?: ""
        val outer = "{\"type\":\"$type\",\"format\":\"$format\"$binaryJson$dataJson}"
            .toByteArray(StandardCharsets.UTF_8)
        val bodyBytes = binaryBody?.let { TaskWorkerBinaryHidden.HEADER_BYTES + it.size } ?: 0
        val wireLimit = if (binaryBody == null) maxFrameBytes else minOf(maxFrameBytes, TaskWorkerProtocol.MAX_BINARY_FRAME_BYTES)
        if (outer.size.toLong() + bodyBytes.toLong() > wireLimit.toLong()) throw TaskWorkerProtocolException(
            "task worker frame exceeds maximum size",
            "message_too_large",
            "message",
        )
        beforeWrite()
        output.writeInt(outer.size)
        output.write(outer)
        if (binaryBody != null) {
            output.writeInt(binaryBody.size)
            output.write(binaryBody)
        }
        output.flush()
    }

    override suspend fun send(envelope: TaskWorkerEnvelope) = withContext(Dispatchers.IO) {
        synchronized(sendLock) {
            // Validate the full envelope, but measure the wire representation after extraction.
            TaskWorkerProtocol.validate(envelope)
            val inner = JsonParser.parseString(
                TaskWorkerProtocol.canonicalBytes(envelope).toString(StandardCharsets.UTF_8),
            ).asJsonObject
            val (body, binaryBody) = TaskWorkerBinaryHidden.split(inner, isBinaryHiddenNegotiated)
            if (binaryBody == null) TaskWorkerProtocol.encode(envelope)
            writeOuter("task_worker", controlFence.decorate(body), binaryBody) {
                if (envelope.messageType == TaskWorkerProtocol.HELLO) {
                    val capabilities = envelope.payload["capabilities"] as? Map<*, *>
                    synchronized(binaryHiddenLock) {
                        masterBinaryHiddenEnabled = false
                        binaryHiddenGeneration++
                        pendingHellos.addLast(binaryHiddenGeneration to
                            (binaryHiddenEnabled && capabilities?.get("binary_hidden") == true))
                    }
                }
            }
        }
    }

    override fun requireControlAuthority() {
        try { controlFence.requireCurrent() } catch (error: TaskWorkerProtocolException) {
            runCatching { socket.close() }
            throw error
        }
    }
    override suspend fun sendModelSupply(data: com.google.gson.JsonObject) = sendOuter("model_supply", controlFence.decorate(data))

    override suspend fun sendModelPrepare(data: com.google.gson.JsonObject) = sendOuter("model_prepare", controlFence.decorate(data))

    private fun invalidFrame(message: String, field: String = "message"): Nothing =
        throw TaskWorkerProtocolException(message, "invalid_frame", field)

    private fun readFrameLength(allowCleanEof: Boolean = false): Int? {
        val first = input.read()
        if (first < 0) {
            if (allowCleanEof) return null
            invalidFrame("missing binary body length")
        }
        val rest = ByteArray(3)
        try {
            input.readFully(rest)
        } catch (_: EOFException) {
            invalidFrame("truncated frame length")
        }
        return (first shl 24) or ((rest[0].toInt() and 255) shl 16) or
            ((rest[1].toInt() and 255) shl 8) or (rest[2].toInt() and 255)
    }

    private fun readFrameBytes(size: Int): ByteArray {
        val bytes = ByteArray(size)
        try {
            input.readFully(bytes)
        } catch (_: EOFException) {
            invalidFrame("truncated frame body")
        }
        return bytes
    }

    private suspend fun receiveOuter(): OuterFrame? = withContext(Dispatchers.IO) {
        val size = readFrameLength(allowCleanEof = true) ?: return@withContext null
        if (size <= 0 || size > maxFrameBytes) invalidFrame("invalid task worker frame length")
        val bytes = readFrameBytes(size)
        val root = try {
            val text = StandardCharsets.UTF_8.newDecoder()
                .onMalformedInput(java.nio.charset.CodingErrorAction.REPORT)
                .onUnmappableCharacter(java.nio.charset.CodingErrorAction.REPORT)
                .decode(java.nio.ByteBuffer.wrap(bytes)).toString()
            JsonParser.parseString(text)
        } catch (error: Exception) {
            throw TaskWorkerProtocolException("task worker frame is not valid JSON", "invalid_frame", "message").also {
                it.initCause(error)
            }
        }
        if (!root.isJsonObject) invalidFrame("task worker frame must be an object")
        val objectValue = root.asJsonObject
        val formatValue = objectValue.get("format")
        if (formatValue == null || !formatValue.isJsonPrimitive || !formatValue.asJsonPrimitive.isString) {
            invalidFrame("task worker frame format must be a string", "format")
        }
        val binary = formatValue.asString == "binary"
        if (binary) {
            if (!isBinaryHiddenNegotiated) {
                invalidFrame("binary hidden was not negotiated on this connection", "format")
            }
            if (objectValue.get("type")?.let {
                    it.isJsonPrimitive && it.asJsonPrimitive.isString && it.asString == "task_worker"
                } != true) invalidFrame("binary hidden requires a task_worker frame", "type")
            val data = objectValue.get("data")?.takeIf { it.isJsonObject }?.asJsonObject
                ?: invalidFrame("binary hidden data must be an object", "data")
            val payload = data.get("payload")?.takeIf { it.isJsonObject }?.asJsonObject
                ?: invalidFrame("binary hidden payload must be an object", "data.payload")
            val messageType = data.get("message_type")?.takeIf {
                it.isJsonPrimitive && it.asJsonPrimitive.isString
            }?.asString
            TaskWorkerBinaryHidden.restoreTarget(messageType, payload)
            val declared = objectValue.get("binary_bytes")?.takeIf {
                it.isJsonPrimitive && it.asJsonPrimitive.isNumber
            }?.asString?.takeIf { it.matches(Regex("[0-9]+")) }?.toLongOrNull()
                ?: invalidFrame("binary_bytes must be an integer", "binary_bytes")
            if (declared <= 0 || declared > minOf(maxFrameBytes, TaskWorkerProtocol.MAX_BINARY_FRAME_BYTES).toLong() -
                size.toLong() - TaskWorkerBinaryHidden.HEADER_BYTES) {
                invalidFrame("binary frame exceeds maximum size", "binary_bytes")
            }
            val bodySize = requireNotNull(readFrameLength())
            if (bodySize.toLong() != declared) invalidFrame("binary body length does not match binary_bytes", "binary_bytes")
            TaskWorkerBinaryHidden.restore(readFrameBytes(bodySize), messageType, payload)
        } else if (formatValue.asString != "json") {
            invalidFrame("unexpected task worker frame", "format")
        }
        OuterFrame(objectValue, binary)
    }

    suspend fun register(registration: TaskWorkerRegistration) {
        require(registration.nodeId.isNotBlank()) { "node id must not be blank" }
        require(registration.clusterSecret.isNotBlank()) { "cluster secret must not be blank" }
        require(registration.clusterSecretEpoch >= 1) { "cluster secret epoch must be positive" }
        val timestamp = System.currentTimeMillis() / 1000.0
        val nonceBytes = ByteArray(18).also { SecureRandom().nextBytes(it) }
        val authNonce = Base64.getUrlEncoder().withoutPadding().encodeToString(nonceBytes)
        val authMessage = String.format(
            Locale.US,
            "v2\n%s\n%.6f\n%s\n%d",
            registration.nodeId,
            timestamp,
            authNonce,
            registration.clusterSecretEpoch,
        )
        val mac = Mac.getInstance("HmacSHA256")
        mac.init(SecretKeySpec(registration.clusterSecret.toByteArray(StandardCharsets.UTF_8), "HmacSHA256"))
        val signature = mac.doFinal(authMessage.toByteArray(StandardCharsets.UTF_8))
            .joinToString("") { "%02x".format(Locale.US, it) }
        sendOuter(
            "register",
            mapOf(
                "client_id" to registration.nodeId,
                "role" to "client",
                "node_type" to "android",
                "hostname" to registration.hostname,
                "network_type" to registration.networkType,
                "advertised_host" to "",
                "advertised_port" to 8888,
                "advertised_address" to "",
                "device_info" to registration.deviceInfo,
                "model_sha256" to registration.modelSha256,
                "auth" to mapOf(
                    "auth_version" to 2,
                    "auth_timestamp" to timestamp,
                    "auth_nonce" to authNonce,
                    "auth_secret_epoch" to registration.clusterSecretEpoch,
                    "auth_signature" to signature,
                ),
            ),
        )
        val ack = receiveOuter()?.value ?: throw EOFException("coordinator closed during register")
        if (ack.get("type")?.asString != "register") {
            throw TaskWorkerProtocolException("expected register acknowledgement", "unexpected_message_type", "type")
        }
        val data = ack.getAsJsonObject("data")
        if (data?.get("control_fence_required")?.asBoolean == true) controlFence.required = true
        if (data?.has("control_certificate") == true) controlFence.admit(data)
        if (data?.get("status")?.asString != "registered") {
            throw TaskWorkerProtocolException(
                data?.get("reason")?.asString ?: "coordinator rejected registration",
                data?.get("error_code")?.asString?.takeIf {
                    it.matches(Regex("[a-z][a-z0-9_]{0,63}"))
                } ?: "registration_rejected", "data.status",
            )
        }
    }

    override suspend fun receiveEvent(): TaskWorkerInboundEvent {
        while (true) {
            val frame = receiveOuter() ?: return TaskWorkerInboundEvent.Closed
            val objectValue = frame.value
            when (objectValue.get("type")?.asString) {
                "task_worker" -> {
                    val data = objectValue.get("data")
                    if (data == null || !data.isJsonObject) {
                        throw TaskWorkerProtocolException("task worker frame has no data object", "invalid_frame", "data")
                    }
                    controlFence.admit(data.asJsonObject)
                    val admittedData = data.asJsonObject.deepCopy().apply { remove("control_certificate") }
                    // ★ 2026-10-08（诊断，定位后降级）：帧的内层契约是
                    //   `{"protocol","version","message_type","message_id","sent_at_ms","payload"}`
                    //   （主仓 `task_worker_protocol.build_message`），**不是** `type`/`data`
                    //   —— 我第一次查 `data.type` 得到 `(null)` 就是问错了字段。这里打出
                    //   `message_type`，直接判定「到了几片、offer 到没到」。
                    val innerType = data.asJsonObject.get("message_type")?.asString
                    val innerPayload = data.asJsonObject.get("payload")
                    QlhLogger.d(
                        "TaskWorkerClient",
                        "task_worker frame: message_type=${innerType ?: "(null)"}" +
                            (if (innerType == TaskWorkerProtocol.STAGE_CHUNK &&
                                innerPayload != null && innerPayload.isJsonObject
                            ) {
                                " index=${innerPayload.asJsonObject.get("chunk_index")}" +
                                    " count=${innerPayload.asJsonObject.get("chunk_count")}"
                            } else {
                                ""
                            }),
                    )
                    val decoded = try {
                        if (frame.binaryHidden) TaskWorkerProtocol.decodeRestoredBinary(admittedData)
                        else TaskWorkerProtocol.decode(
                            gson.toJson(admittedData).toByteArray(StandardCharsets.UTF_8)
                        )
                    } catch (error: Exception) {
                        // ★ 2026-10-11（三机验收实测）：帧字节完整到达、`message_type` 已能读出，
                        //   但尾段 Stage 在本端**完全静默**就重连（主节点只看到
                        //   `remote worker disconnected`）。decode 是这条路径上唯一还未留痕的
                        //   一步，先把它暴露出来。
                        QlhLogger.w(
                            "TaskWorkerClient",
                            "TaskWorkerProtocol.decode failed message_type=$innerType " +
                                "payloadKeys=" +
                                (data.asJsonObject.get("payload") as? com.google.gson.JsonObject)?.keySet() +
                                " ${error::class.simpleName}: ${error.message ?: ""}",
                        )
                        throw error
                    }
                    if (decoded.messageType == TaskWorkerProtocol.HELLO_ACK) {
                        synchronized(binaryHiddenLock) {
                            // A replay cannot consume another HELLO's pending acknowledgement.
                            if (helloAckReplay.replay(decoded) == null) {
                                helloAckReplay.remember(decoded, decoded)
                                val pending = pendingHellos.pollFirst()
                                if (pending != null && pending.first == binaryHiddenGeneration &&
                                    pendingHellos.isEmpty() && !binaryHiddenClosed) {
                                    masterBinaryHiddenEnabled = binaryHiddenEnabled && pending.second &&
                                        decoded.payload["accepted"] == true && decoded.payload["binary_hidden"] == true
                                }
                            }
                        }
                    }
                    return TaskWorkerInboundEvent.Envelope(decoded)
                }
                "model_prepare" -> {
                    val data = objectValue.getAsJsonObject("data") ?: throw TaskWorkerProtocolException("prepare data missing", "invalid_frame")
                    controlFence.admit(data)
                    return TaskWorkerInboundEvent.ModelPrepare(data.deepCopy().apply { remove("control_certificate") })
                }
                "model_supply" -> {
                    val data = objectValue.getAsJsonObject("data") ?: throw TaskWorkerProtocolException("supply data missing", "invalid_frame")
                    controlFence.admit(data)
                    return TaskWorkerInboundEvent.ModelSupply(data.deepCopy().apply { remove("control_certificate") })
                }
                "heartbeat_ack" -> return TaskWorkerInboundEvent.HeartbeatAck
                "error" -> {
                    val data = objectValue.getAsJsonObject("data")
                    val errorCode = data?.get("error_code")
                        ?.takeIf { it.isJsonPrimitive && it.asJsonPrimitive.isString }
                        ?.asString
                        ?.takeIf { it.matches(Regex("[a-z][a-z0-9_]{0,63}")) }
                        ?: "coordinator_error"
                    throw TaskWorkerProtocolException(
                        "coordinator rejected task-worker frame ($errorCode)",
                        errorCode,
                        "data.error_code",
                    )
                }
                "stage_chunk" -> {
                    // ★ 2026-10-08（DIST-NEXT-2b 真机回归暴露）：`stage_chunk` 是**外层帧类型**
                    //   （主仓 `build_stage_chunk` 用 `build_message("stage_chunk", …)`）。此前这里
                    //   没有该分支 ⇒ 分片落到下面的 `else` 被当作未知帧**丢弃** ⇒ offer 到达时装配
                    //   不齐 ⇒ 主节点只看到 `remote Stage response timed out`（真机现象：master 侧
                    //   `task_worker_stage_chunks_sent chunks=18`，而设备侧 CPU 空闲、一条分片日志
                    //   都没有）。这里把它解码成 envelope（补上外层类型），交给上层已有的
                    //   `TaskWorkerProtocol.STAGE_CHUNK -> handleStageChunk(...)` 分支。
                    val data = objectValue.get("data")
                    if (data == null || !data.isJsonObject) {
                        QlhLogger.w(
                            "TaskWorkerClient",
                            "stage_chunk rejected: invalid_frame (no data object)",
                        )
                    } else {
                        controlFence.admit(data.asJsonObject)
                        val payload = data.asJsonObject.deepCopy().apply { remove("control_certificate") }
                        payload.addProperty("type", TaskWorkerProtocol.STAGE_CHUNK)
                        // ★ 2026-10-08（诊断，定位后降级）：逐片记录到达情况。真机上
                        //   `task_worker_stage_chunks_sent chunks=18` 之后 offer 始终未到，
                        //   必须先看清「到底收到几片、有无越界/重复」。
                        QlhLogger.d(
                            "TaskWorkerClient",
                            "stage_chunk in: index=${payload.get("chunk_index")} " +
                                "count=${payload.get("chunk_count")} " +
                                "bytes=${payload.get("total_bytes")}",
                        )
                        return TaskWorkerInboundEvent.Envelope(
                            TaskWorkerProtocol.decode(
                                gson.toJson(payload).toByteArray(StandardCharsets.UTF_8)
                            )
                        )
                    }
                }
                // 已知但本端不消费的帧：吞掉，继续读下一帧。
                "node_list_sync", "node_update", "layer_config" -> {
                    val data = objectValue.getAsJsonObject("data") ?: com.google.gson.JsonObject()
                    controlFence.admit(data)
                }
                else -> {
                    // ★ 2026-10-08（真机闸门根因）：**未知帧不得断连**。
                    //
                    //   主仓的节点级查询帧（例如日志聚合按 PC 的 `LOG_REQUEST` 协议下发）此前
                    //   会落到 `else -> throw` ⇒ 这里的接收循环抛异常 ⇒ **连接被拆**。实测
                    //   `GET /api/cluster/nodes/log-aggregate` 每调一次 Y700 就断一次（约 1 秒后
                    //   重连），而 TUI 启动正好会调它 ⇒ 紧随其后的 `POST /api/chat/stream`
                    //   落进断开窗口被判 `pipeline workers not ready`。
                    //   协议前向兼容：记一条 WARN 后跳过该帧，连接保持不变。
                    QlhLogger.w(
                        "TaskWorkerClient",
                        "ignoring unknown coordinator frame: " +
                            (objectValue.get("type")?.asString ?: "(null)"),
                    )
                }
            }
        }
    }

    override suspend fun receive(): TaskWorkerEnvelope? = when (val event = receiveEvent()) {
        is TaskWorkerInboundEvent.Envelope -> event.value
        is TaskWorkerInboundEvent.ModelPrepare -> throw TaskWorkerProtocolException("prepare before hello acknowledgement", "unexpected_message_type")
        is TaskWorkerInboundEvent.ModelSupply -> throw TaskWorkerProtocolException("supply before hello acknowledgement", "unexpected_message_type")
        TaskWorkerInboundEvent.Closed -> null
        TaskWorkerInboundEvent.HeartbeatAck -> throw TaskWorkerProtocolException(
            "heartbeat acknowledgement is not a task worker envelope", "unexpected_message_type", "type",
        )
    }

    override suspend fun sendHeartbeat(nodeId: String, sentAtMs: Long) {
        sendOuter("heartbeat", mapOf("client_id" to nodeId, "t_send" to sentAtMs))
    }

    override suspend fun close() {
        synchronized(binaryHiddenLock) {
            binaryHiddenClosed = true
            binaryHiddenEnabled = false
            masterBinaryHiddenEnabled = false
            binaryHiddenGeneration++
            pendingHellos.clear()
        }
        withContext(Dispatchers.IO) {
            runCatching { socket.close() }
        }
    }
}

class SocketTaskWorkerTransportFactory(
    private val connectTimeoutMs: Int = 8_000,
    private val readTimeoutMs: Int = 45_000,
    private val registration: TaskWorkerRegistration? = null,
    private val controlFenceRequired: Boolean = false,
    private val controlCertificateValidator: ((com.google.gson.JsonObject) -> Unit)? = null,
) : TaskWorkerTransportFactory {
    override suspend fun connect(host: String, port: Int): TaskWorkerTransport {
        require(port in 1..65535) { "port must be between 1 and 65535" }
        val normalizedHost = host.trim().removePrefix("[").removeSuffix("]")
        val socket = Socket()
        return suspendCancellableCoroutine { continuation ->
            val opening = CoroutineScope(continuation.context).launch(Dispatchers.IO) {
                try {
                    socket.connect(InetSocketAddress(normalizedHost, port), connectTimeoutMs)
                    socket.soTimeout = readTimeoutMs
                    val transport = SocketTaskWorkerTransport(socket, controlFence = TcpControlFence(controlFenceRequired, controlCertificateValidator))
                    registration?.let { transport.register(it) }
                    continuation.resume(transport) { _, _, _ -> runCatching { socket.close() } }
                } catch (error: Exception) {
                    runCatching { socket.close() }
                    if (continuation.isActive) continuation.resumeWithException(error)
                }
            }
            // Cancellation must unblock connect/REGISTER reads before service retirement joins the owner.
            continuation.invokeOnCancellation {
                runCatching { socket.close() }
                opening.cancel()
            }
        }
    }
}

fun interface TaskWorkerStageHandler {
    /** Set by the owning connection immediately before execution; defaults to JSON budget. */
    fun setBinaryHiddenEnabled(enabled: Boolean) = Unit
    suspend fun execute(offer: TaskWorkerEnvelope): TaskWorkerStageExecution

    /**
     * ★ 2026-10-07（DIST-NEXT-1）：请求中止该 attempt 的执行。
     *
     * 默认空实现 ⇒ 未接线的执行器保持旧行为。已接线的实现（Android 层段执行器）
     * 会把请求传到 native：`llama_set_abort_callback` 在每个可分割的张量边界退出，
     * 使「取消」不再只能等到整个 stage 跑完。
     */
    fun requestCancel(identity: TaskWorkerAttemptIdentity) = Unit

    /**
     * ★ 2026-10-07（DIST-NEXT-1）：该 attempt 的执行**是否仍在进行**。
     *
     * 取消 ACK 用它如实区分 `execution_in_flight` 与 `execution_stopped`；
     * 默认 false ⇒ 未接线的执行器不会冒充「仍在执行」。
     */
    fun isExecutionInFlight(identity: TaskWorkerAttemptIdentity): Boolean = false
}

data class TaskWorkerStageExecution(
    val output: Map<String, Any?>,
    val metadata: Map<String, Any?> = emptyMap(),
)

/** Coroutine client used by TaskWorkerService; transport is injectable for B-03 contract tests. */
class TaskWorkerClient(
    private val host: String,
    private val port: Int,
    private val nodeId: String,
    /** Read at hello time so model/layer artifacts are never frozen at service start. */
    private val capabilities: suspend () -> Map<String, Any?>,
    private val registration: TaskWorkerRegistration? = null,
    private val transportFactory: TaskWorkerTransportFactory =
        SocketTaskWorkerTransportFactory(registration = registration),
    private val stageHandler: TaskWorkerStageHandler? = null,
    private val modelSupplyHandler: TaskWorkerModelSupplyHandler? = null,
    private val modelPrepareHandler: TaskWorkerModelPrepareHandler? = null,
    private val scope: CoroutineScope = CoroutineScope(SupervisorJob() + Dispatchers.IO),
    private val clockMs: () -> Long = { System.currentTimeMillis() },
    private val machine: TaskWorkerStateMachine = TaskWorkerStateMachine(),
    private val heartbeatIntervalMs: Long = HEARTBEAT_INTERVAL_MS,
    private val heartbeatTimeoutMs: Long = HEARTBEAT_TIMEOUT_MS,
    private val monotonicMs: () -> Long = { System.nanoTime() / 1_000_000L },
) {
    init {
        require(heartbeatIntervalMs > 0 && heartbeatTimeoutMs > heartbeatIntervalMs)
    }
    private val mutableSnapshot = MutableStateFlow(machine.snapshot())
    @Volatile private var loopJob: Job? = null
    @Volatile private var transport: TaskWorkerTransport? = null
    @Volatile private var lastHeartbeatAckMs: Long = 0L
    @Volatile private var executionJob: Job? = null
    private var heartbeatJob: Job? = null
    private val executionLock = Any()
    private class ExecutionSlot(val identity: TaskWorkerAttemptIdentity) {
        @Volatile var job: Job? = null
        @Volatile var terminalResponse: TaskWorkerEnvelope? = null
    }
    @Volatile private var executionSlot: ExecutionSlot? = null
    private data class CancellationRequest(
        val identity: TaskWorkerAttemptIdentity,
        val reasonCode: String,
        val providerId: String,
    )
    @Volatile private var cancellationRequest: CancellationRequest? = null
    private val convergenceLock = Any()
    private var convergenceIdentity: TaskWorkerAttemptIdentity? = null
    private var convergenceJob: Job? = null

    // ★ 2026-10-07（DIST-NEXT-1）：取消 ACK 的单一事实源。
    //   `stage_cancelled` 可能发两次：先是 ACK（可能 `execution_in_flight`），
    //   执行真正停止后再补一条终态。这里记录「已发出的状态」，保证每条状态
    //   只发一次，且不会把 ACK 冒充成终态。
    private val cancelAckMutex = Mutex()
    private var cancelAckIdentity: TaskWorkerAttemptIdentity? = null
    private var cancelAckState: String? = null

    /** offer 里的 `provider_id`（主节点口径）；本地取消的回程必须回显它。 */
    @Volatile
    private var activeProviderId: String? = null

    /** ★ 2026-10-07（DIST-NEXT-2b）：大 payload 分片的装配器（按 attempt 分区）。 */
    private val stageChunks = StageChunkAssembler()

    val snapshot: StateFlow<TaskWorkerSnapshot> = mutableSnapshot.asStateFlow()

    val isRunning: Boolean get() = loopJob?.isActive == true && scope.isActive &&
        machine.snapshot().connection != TaskWorkerConnectionState.STOPPED

    val isConnectionLive: Boolean get() = isRunning && transport != null &&
        machine.snapshot().connection == TaskWorkerConnectionState.READY &&
        monotonicMs() - lastHeartbeatAckMs < heartbeatTimeoutMs

    @Synchronized
    fun start() {
        if (isRunning) return
        machine.start()
        val job = scope.launch(start = CoroutineStart.LAZY) { runLoop() }
        loopJob = job
        job.invokeOnCompletion {
            machine.onDisconnected(clockMs(), "transport_loop_stopped", "worker connection loop stopped")
            publish()
        }
        job.start()
        publish()
    }

    fun cancelActive(reasonCode: String = "user_cancelled"): Boolean {
        val code = reasonCode.takeIf { it.matches(Regex("^[a-z][a-z0-9_]{0,63}$")) }
            ?: "worker_cancelled"
        val request = requestCancellation(code) ?: run {
            if (modelPrepareHandler?.cancelPending() != true) return false
            val current = transport
            scope.launch { runCatching { current?.close() } }
            return true
        }
        val identity = request.identity
        val providerId = request.providerId
        publish()
        scope.launch {
            val current = transport ?: return@launch
            try {
                sendCancelledAck(current, identity, providerId, code,
                    if (isExecutionInFlight(identity)) TaskWorkerProtocol.EXECUTION_IN_FLIGHT
                    else TaskWorkerProtocol.EXECUTION_STOPPED)
            } catch (error: Exception) {
                machine.onDisconnected(clockMs(), "cancel_send_failed", error.message ?: "cancel send failed")
                publish()
            } finally {
                convergeCancellation(current, identity, providerId)
            }
        }
        return true
    }

    private fun requestCancellation(reasonCode: String, expectedIdentity: TaskWorkerAttemptIdentity? = null): CancellationRequest? =
        synchronized(executionLock) {
            val identity = machine.requestCancel(clockMs(), expectedIdentity) ?: return@synchronized null
            val request = CancellationRequest(identity, reasonCode, activeProviderId ?: nodeId)
            cancellationRequest = request
            stageHandler?.requestCancel(identity)
            executionJob?.cancel()
            request
        }

    /** Covers queued/pre-entry execution as well as a native call that outlives Job cancellation. */
    private fun isExecutionInFlight(identity: TaskWorkerAttemptIdentity): Boolean =
        executionSlot?.let { it.identity == identity && it.job?.isCompleted != true } == true ||
            stageHandler?.isExecutionInFlight(identity) == true

    /** Non-blocking stop used by Android Service lifecycle callbacks. */
    fun stop() {
        modelPrepareHandler?.cancelAll()
        modelSupplyHandler?.cancelAll()
        machine.snapshot().activeAttempt.identity?.let { stageHandler?.requestCancel(it) }
        machine.stop()
        publish()
        executionJob?.cancel()
        heartbeatJob?.cancel()
        loopJob?.cancel()
        val current = transport
        if (current == null) {
            scope.cancel()
            return
        }
        scope.launch {
            runCatching { current.close() }
            transport = null
            scope.cancel()
        }
    }

    suspend fun stopAndJoin() = withContext(NonCancellable) {
        modelPrepareHandler?.cancelAll()
        modelSupplyHandler?.cancelAll()
        machine.snapshot().activeAttempt.identity?.let { stageHandler?.requestCancel(it) }
        machine.stop()
        publish()
        executionJob?.cancel()
        heartbeatJob?.cancel()
        loopJob?.cancel()
        runCatching { transport?.close() }
        // Cancelling a Job does not interrupt JNI. Join the owning scope, including
        // execution and cancellation-convergence jobs, before releasing the model.
        scope.coroutineContext[Job]?.cancelAndJoin()
        modelPrepareHandler?.retireAndJoin()
        transport = null
        executionJob = null
        heartbeatJob = null
        loopJob = null
    }

    private suspend fun runLoop() {
        while (scope.isActive && machine.snapshot().connection !in setOf(
                TaskWorkerConnectionState.STOPPED, TaskWorkerConnectionState.RETRY_EXHAUSTED,
        )) {
            val current = machine.snapshot()
            if (current.connection == TaskWorkerConnectionState.BACKING_OFF) {
                val waitMs = (current.nextRetryAtMs - clockMs()).coerceAtLeast(0L)
                delay(waitMs)
                machine.retryIfDue(clockMs())
                publish()
                continue
            }
            try {
                machine.start()
                publish()
                val opened = transportFactory.connect(host, port)
                transport = opened
                machine.onConnected()
                publish()
                // Local receive support and the master's explicit ACK are independent.
                // A new connection or refreshed hello starts with outbound binary disabled.
                val helloCapabilities = capabilities()
                opened.setBinaryHiddenEnabled(
                    helloCapabilities["binary_hidden"] == true,
                )
                opened.send(TaskWorkerProtocol.buildHello(
                    nodeId = nodeId,
                    capabilities = helloCapabilities,
                    messageId = newMessageId("hello"),
                    sentAtMs = clockMs(),
                ))
                val ack = opened.receive() ?: throw EOFException("coordinator closed during hello")
                if (ack.messageType != TaskWorkerProtocol.HELLO_ACK) {
                    throw TaskWorkerProtocolException("expected hello_ack", "unexpected_message_type", "message_type")
                }
                val accepted = ack.payload["accepted"] == true
                lastHeartbeatAckMs = monotonicMs()
                machine.onHelloAck(
                    accepted = accepted,
                    reasonCode = ack.payload["reason_code"] as? String ?: "hello_rejected",
                    reasonMessage = "coordinator hello acknowledgement",
                    nowMs = clockMs(),
                )
                publish()
                if (!accepted) throw TaskWorkerProtocolException("coordinator rejected hello", "hello_rejected", "payload.accepted")
                heartbeatJob?.cancel()
                heartbeatJob = scope.launch {
                    while (isActive && transport === opened &&
                        machine.snapshot().connection == TaskWorkerConnectionState.READY
                    ) {
                        val untilAckTimeout = (heartbeatTimeoutMs - (monotonicMs() - lastHeartbeatAckMs)).coerceAtLeast(1L)
                        delay(minOf(heartbeatIntervalMs, untilAckTimeout))
                        if (transport !== opened || machine.snapshot().connection != TaskWorkerConnectionState.READY) break
                        if (monotonicMs() - lastHeartbeatAckMs >= heartbeatTimeoutMs) {
                            disconnect(opened, "heartbeat_timeout", "coordinator heartbeat acknowledgement timed out")
                            break
                        }
                        try {
                            opened.sendHeartbeat(nodeId, clockMs())
                            QlhLogger.d("TaskWorkerClient", "heartbeat sent")
                        } catch (error: CancellationException) {
                            throw error
                        } catch (error: Exception) {
                            disconnect(opened, "heartbeat_send_failed", error.message ?: "heartbeat send failed")
                            break
                        }
                    }
                }
                receiveLoop(opened)
            } catch (error: CancellationException) {
                throw error
            } catch (error: Exception) {
                machine.onDisconnected(clockMs(), errorCode(error), error.message ?: error.javaClass.simpleName)
                publish()
            } finally {
                modelPrepareHandler?.cancelAll()
                modelSupplyHandler?.cancelAll()
                withContext(NonCancellable) {
                    heartbeatJob?.cancelAndJoin()
                    heartbeatJob = null
                    // Close before joining non-cancellable execution so the master observes offline immediately.
                    runCatching { transport?.close() }
                    transport = null
                    machine.snapshot().activeAttempt.identity?.let { stageHandler?.requestCancel(it) }
                    executionJob?.cancelAndJoin()
                    executionJob = null
                    modelPrepareHandler?.retireAndJoin()
                }
            }
        }
    }

    private suspend fun disconnect(connection: TaskWorkerTransport, code: String, message: String) {
        if (transport !== connection) return
        machine.onDisconnected(clockMs(), code, message)
        publish()
        runCatching { connection.close() }
    }

    private suspend fun receiveLoop(connection: TaskWorkerTransport) {
        while (scope.isActive && machine.snapshot().connection == TaskWorkerConnectionState.READY) {
            val event = connection.receiveEvent()
            if (transport !== connection || machine.snapshot().connection != TaskWorkerConnectionState.READY) return
            // ★ 2026-10-08（真机 P0 候选）：**分发层必须容错**。此前 `handleOffer` 里任何
            //   未预期的异常（例如分片装配抛出的类型不在那一个 `catch` 里）都会直接逃出本
            //   循环 ⇒ 读循环**静默死亡** ⇒ 之后 offer/分片全部无人处理，而心跳是另一个协程、
            //   仍在照常发送。真机现象与此完全一致：18 片 + offer 都送达、设备侧 CPU 空闲、
            //   连一条装配日志都没有，master 只看到 `remote Stage response timed out`。
            try {
                when (event) {
                    TaskWorkerInboundEvent.Closed -> throw EOFException("coordinator closed worker connection")
                    TaskWorkerInboundEvent.HeartbeatAck -> { lastHeartbeatAckMs = monotonicMs() }
                    is TaskWorkerInboundEvent.ModelPrepare -> {
                        val handler = modelPrepareHandler ?: throw TaskWorkerProtocolException("prepare handler unavailable", "prepare_unavailable")
                        handler.command(event.data, scope,
                            reply = { data ->
                                if (transport !== connection || machine.snapshot().connection != TaskWorkerConnectionState.READY) throw CancellationException("prepare connection retired")
                                connection.requireControlAuthority()
                                connection.sendModelPrepare(data)
                            }, authorize = {
                                if (transport !== connection || machine.snapshot().connection != TaskWorkerConnectionState.READY) throw CancellationException("prepare connection retired")
                                connection.requireControlAuthority()
                            })
                    }
                    is TaskWorkerInboundEvent.ModelSupply -> {
                        modelSupplyHandler?.command(event.data, scope,
                            reply = { data ->
                                if (transport !== connection || machine.snapshot().connection != TaskWorkerConnectionState.READY) throw CancellationException("supply connection retired")
                                connection.sendModelSupply(data)
                            },
                            refreshHello = {
                                if (transport !== connection || machine.snapshot().connection != TaskWorkerConnectionState.READY) throw CancellationException("supply connection retired")
                                // Refresh receive support and wait for a new explicit master ACK.
                                val refreshCapabilities = capabilities()
                                connection.setBinaryHiddenEnabled(
                                    refreshCapabilities["binary_hidden"] == true,
                                )
                                connection.send(TaskWorkerProtocol.buildHello(nodeId, refreshCapabilities, newMessageId("hello"), clockMs()))
                            }, authorize = {
                                if (transport !== connection || machine.snapshot().connection != TaskWorkerConnectionState.READY) throw CancellationException("supply connection retired")
                                connection.requireControlAuthority()
                            })
                    }
                    is TaskWorkerInboundEvent.Envelope -> when (event.value.messageType) {
                        TaskWorkerProtocol.HELLO_ACK -> {
                            if (event.value.payload["accepted"] != true) throw java.io.IOException("capability refresh rejected")
                        }
                        TaskWorkerProtocol.STAGE_OFFER -> handleOffer(connection, event.value)
                        TaskWorkerProtocol.STAGE_CANCEL -> handleCancel(connection, event.value)
                        TaskWorkerProtocol.LEASE_RENEW -> handleLeaseRenew(event.value)
                        // ★ 2026-10-07（DIST-NEXT-2b）：大 payload 分片 —— 只累积；装配在
                        //   随后的 `stage_offer` 路径完成（收到 offer 时分片应当已齐备）。
                        TaskWorkerProtocol.STAGE_CHUNK -> handleStageChunk(event.value)
                        else -> throw TaskWorkerProtocolException(
                            "unexpected coordinator message for Android worker",
                            "unexpected_message_type",
                            "message_type",
                        )
                    }
                }
            } catch (error: CancellationException) {
                throw error
            } catch (error: EOFException) {
                // 对端关闭 / 显式 EOF ⇒ 维持既有语义，交上层重连。
                throw error
            } catch (error: java.io.IOException) {
                // 传输层已坏 ⇒ 同样交上层重连，不要在这里吞掉。
                // ★ 2026-10-11：此前该分支**不留任何痕**，真机表现为"帧已收到、随后静默重连"，
                //   主节点只能看到 `remote worker disconnected`。先把失败类型/消息落盘。
                QlhLogger.w(
                    "TaskWorkerClient",
                    "transport read failed, connection will reconnect " +
                        "${error::class.simpleName}: ${error.message ?: ""}",
                )
                throw error
            } catch (error: Exception) {
                // 单条消息处理失败**不得**掐死整条连接：记名后继续读下一帧。
                QlhLogger.w(
                    "TaskWorkerClient",
                    "message handling failed (connection kept): " +
                        "${error::class.simpleName}: ${error.message ?: ""}",
                )
            }
        }
    }

    private suspend fun handleOffer(connection: TaskWorkerTransport, envelope: TaskWorkerEnvelope) {
        val payload = envelope.payload
        // ★ 2026-10-11（三机验收实测）：尾段 Stage 在设备侧**完全静默**就重连了 —— 帧已能解析出
        //   `message_type=stage_offer`，但本函数与读循环的任何日志都没出现 ⇒ 失败点落在
        //   `identityFrom`/lease 解析这些**未留痕**路径。这里先把入口与身份解析暴露出来，
        //   否则只能靠主节点侧的 `remote worker disconnected` 反推。
        QlhLogger.d(
            "TaskWorkerClient",
            "handleOffer entered payloadKeys=${payload.keys.take(24)}",
        )
        val identity = try {
            identityFrom(payload)
        } catch (error: Exception) {
            QlhLogger.w(
                "TaskWorkerClient",
                "handleOffer identityFrom failed ${error::class.simpleName}: " +
                    "${error.message ?: ""}",
            )
            throw error
        }
        // ★ 回程必须**原样回显** offer 里的 `provider_id` —— 它是主节点的 Provider 口径
        //   （`remote_<node_id>`），不是本节点的裸 nodeId。此前一律回裸 nodeId，被主节点
        //   以 `attempt_identity_mismatch` 拒掉，真机表现为 Stage 响应一直等到超时。
        val providerId = (payload["provider_id"] as? String)?.takeIf { it.isNotBlank() } ?: nodeId
        val occupied = executionSlot
        if (occupied != null && isExecutionInFlight(occupied.identity)) {
            stageChunks.discard(identity.attemptId)
            connection.send(TaskWorkerProtocol.buildStageAccept(
                identity, providerId, false, "worker_execution_in_flight", true,
                newMessageId("accept"), clockMs(),
            ))
            return
        }
        modelPrepareHandler?.validateStage(envelope)?.let { code ->
            stageChunks.discard(identity.attemptId)
            connection.send(TaskWorkerProtocol.buildStageAccept(
                identity, providerId, false, code, true, newMessageId("accept"), clockMs(),
            ))
            return
        }
        val expires = (payload["lease_expires_at_ms"] as Number).toLong()
        val rejection = machine.admitOffer(identity, expires, clockMs())
        if (rejection != null) {
            // ★ 2026-10-07（真机 P0）：拒绝必须留痕。此前设备侧完全没有日志，现场只能从
            //   主节点看到 `worker_busy_or_lease_invalid`，无法区分「上一个 attempt 未释放
            //   槽位」还是「lease 过期」。
            val state = machine.snapshot()
            QlhLogger.w(
                "TaskWorkerClient",
                "stage offer rejected: $rejection " +
                    "stage=${identity.stageId} attempt=${identity.attemptId} " +
                    "activeAttempt=${state.activeAttempt.state} " +
                    "connection=${state.connection}",
            )
            connection.send(TaskWorkerProtocol.buildStageAccept(
                identity, providerId, false, rejection, rejection != "worker_attempt_already_admitted",
                newMessageId("accept"), clockMs(),
            ))
            stageChunks.discard(identity.attemptId)
            return
        }
        activeProviderId = providerId
        val handler = stageHandler
        // ★ 2026-10-08（诊断，定位后降级）：真机卡点 —— 18 片分片与 offer 都完好送达，
        //   但之后**没有任何**装配/执行/拒绝痕迹、CPU 全程空闲。下面几个分支此前完全静默，
        //   是最大嫌疑；先把它们全部变可见，避免继续靠推断。
        val hasHiddenRef = (payload["root_input"] as? Map<*, *>)?.containsKey("hidden_ref") == true
        QlhLogger.d(
            "TaskWorkerClient",
            "stage offer handling: stage=${identity.stageId} attempt=${identity.attemptId} " +
                "handlerNull=${handler == null} hasHiddenRef=$hasHiddenRef",
        )
        if (handler == null) {
            QlhLogger.w(
                "TaskWorkerClient",
                "stage offer rejected: worker_execution_not_configured (stageHandler is null) " +
                    "stage=${identity.stageId}",
            )
            machine.fail(identity, "worker_execution_not_configured", retryable = true)
            publish()
            connection.send(TaskWorkerProtocol.buildStageAccept(
                identity, providerId, false, "worker_execution_not_configured", true,
                newMessageId("accept"), clockMs(),
            ))
            return
        }
        // ★ 2026-10-07（DIST-NEXT-2b）：分片输入在 accept **之前**装配 —— 装配不成功就
        //   别接这个 stage（用 accept 的具名拒绝码收敛，而不是执行阶段的笼统失败）。
        val effectiveOffer = try {
            assembleChunkedRootInput(envelope)
        } catch (error: AndroidFullWorkerStageException) {
            QlhLogger.w(
                "TaskWorkerClient",
                "stage offer rejected: chunked assembly failed ${error.code} " +
                    "${error.message ?: ""}",
            )
            machine.fail(identity, error.code, retryable = true)
            publish()
            connection.send(TaskWorkerProtocol.buildStageAccept(
                identity, providerId, false, error.code, true,
                newMessageId("accept"), clockMs(),
            ))
            return
        } ?: envelope
        connection.send(TaskWorkerProtocol.buildStageAccept(
            identity, providerId, true, "", false, newMessageId("accept"), clockMs(),
        ))
        val slot = ExecutionSlot(identity)
        val job = synchronized(executionLock) {
            if (!machine.markRunning(identity, clockMs())) null else {
                executionSlot = slot
                scope.launch(start = CoroutineStart.LAZY) {
                    executeOffer(connection, effectiveOffer, slot, providerId, handler)
                }.also { launched ->
                    slot.job = launched
                    executionJob = launched
                    launched.invokeOnCompletion {
                        // Retire compute ownership before a terminal write can become visible.
                        val response = slot.terminalResponse
                        slot.terminalResponse = null
                        if (response != null) {
                            scope.launch { sendTerminalResponse(connection, slot.identity, response) }
                        } else if (transport === connection && isAttemptCancellingOrDone(identity)) {
                            convergeCancellation(connection, identity, providerId)
                        }
                    }
                }
            }
        }
        publish()
        if (job == null) {
            if (isAttemptCancellingOrDone(identity)) {
                finishCancelledExecution(connection, identity, providerId)
            } else {
                val code = if (machine.snapshot().activeAttempt.leaseExpiresAtMs <= clockMs())
                    "worker_lease_expired_before_execution" else "worker_attempt_not_runnable"
                connection.send(TaskWorkerProtocol.buildStageError(
                    identity, providerId, code, true, newMessageId("error"), clockMs(),
                ))
            }
            return
        }
        job.start()
    }

    private suspend fun executeOffer(
        connection: TaskWorkerTransport,
        effectiveOffer: TaskWorkerEnvelope,
        slot: ExecutionSlot,
        providerId: String,
        handler: TaskWorkerStageHandler,
    ) {
        val identity = slot.identity
        try {
            handler.setBinaryHiddenEnabled(connection.isBinaryHiddenNegotiated)
            val action: suspend () -> TaskWorkerStageExecution = {
                currentCoroutineContext().ensureActive()
                handler.execute(effectiveOffer)
            }
            val execution = modelPrepareHandler?.executeStage(effectiveOffer, action) ?: action()
            if (isAttemptCancelling(identity)) {
                // ★ 2026-10-07（DIST-NEXT-1）：取消与结果竞态 —— 已进入取消的
                //   attempt 不得再发 `stage_result`（否则主节点会把「已取消」
                //   记成「已完成」）。迟到的结果由主节点吸收并单独计数。
                finishCancelledExecution(connection, identity, providerId)
            } else {
                val output = TaskWorkerProtocol.buildStageResult(
                    identity, providerId, execution.output, execution.metadata,
                    newMessageId("result"), clockMs(),
                )
                val digest = output.payload["output_sha256"] as String
                if (machine.complete(identity, digest, clockMs())) {
                    slot.terminalResponse = output
                    publish()
                } else if (isAttemptCancellingOrDone(identity)) {
                    // Cancellation may win while output hashing crosses the earlier check.
                    finishCancelledExecution(connection, identity, providerId)
                }
            }
        } catch (error: CancellationException) {
            finishCancelledExecution(connection, identity, providerId)
        } catch (error: Exception) {
            if (isAttemptCancelling(identity)) {
                // ★ 2026-10-07（DIST-NEXT-1）：native 层段前向被取消标志中断后
                //   返回的是普通失败。那属于取消，不能报成 `stage_error`
                //   （会把「用户取消」写成「执行失败」）。
                finishCancelledExecution(connection, identity, providerId)
            } else {
                val errorCode = (error as? AndroidFullWorkerStageException)?.code
                    ?: "worker_execution_failed"
                val response = TaskWorkerProtocol.buildStageError(
                    identity, providerId, errorCode, true, newMessageId("error"), clockMs(),
                )
                if (machine.fail(identity, errorCode, retryable = true)) {
                    slot.terminalResponse = response
                    // ★ 2026-10-07（真机 P0）：执行失败必须留设备侧证据（此前完全静默）。
                    QlhLogger.e(
                        "TaskWorkerClient",
                        "stage execution failed: stage=${identity.stageId} " +
                            "attempt=${identity.attemptId} code=$errorCode " +
                            "detail=${error.message ?: error.javaClass.simpleName}",
                    )
                    publish()
                } else if (isAttemptCancellingOrDone(identity)) {
                    finishCancelledExecution(connection, identity, providerId)
                }
            }
        }
    }

    /** Terminal I/O is a sibling of the completed compute owner, never its physical slot. */
    private suspend fun sendTerminalResponse(
        connection: TaskWorkerTransport,
        identity: TaskWorkerAttemptIdentity,
        response: TaskWorkerEnvelope,
    ) {
        // A successor may already be active; preserve the captured response and connection.
        if (transport !== connection) return
        try {
            withTimeout(heartbeatTimeoutMs) { connection.send(response) }
        } catch (error: TimeoutCancellationException) {
            QlhLogger.w("TaskWorkerClient", "stage terminal response timed out: " +
                "stage=${identity.stageId} attempt=${identity.attemptId}")
        } catch (error: CancellationException) {
            throw error
        } catch (error: Exception) {
            // A late send failure must not fail/disconnect a successor that now owns execution.
            QlhLogger.w("TaskWorkerClient", "stage terminal response send failed: " +
                "stage=${identity.stageId} attempt=${identity.attemptId} " +
                "detail=${error.message ?: error.javaClass.simpleName}")
        }
    }

    /**
     * ★ 2026-10-07（DIST-NEXT-2b）：接收一条 `stage_chunk`。
     *
     * 分片错误**只影响该 attempt**（不因此断开连接）：这里记录具名日志，随后的
     * `stage_offer` 会因为分片未齐备被拒（`incomplete_chunks`）。
     */
    private fun handleStageChunk(envelope: TaskWorkerEnvelope) {
        val payload = envelope.payload
        val chunk = try {
            java.util.Base64.getDecoder().decode(payload["payload_b64"] as String)
        } catch (error: Exception) {
            QlhLogger.w("TaskWorkerClient", "stage_chunk rejected: invalid_chunk_payload")
            return
        }
        try {
            stageChunks.add(
                attemptId = payload["attempt_id"] as String,
                chunkIndex = (payload["chunk_index"] as Number).toInt(),
                chunkCount = (payload["chunk_count"] as Number).toInt(),
                payload = chunk,
                payloadSha256 = payload["payload_sha256"] as String,
            )
        } catch (error: AndroidFullWorkerStageException) {
            QlhLogger.w(
                "TaskWorkerClient",
                "stage_chunk rejected: ${error.code} ${error.message ?: ""}",
            )
        }
    }

    /**
     * ★ 2026-10-07（DIST-NEXT-2b）：把 `hidden_ref` 的分片装配回 `hidden_f32`。
     *
     * 返回 `null` 表示**无需改写**（内联路径或非层段）—— 调用方原样使用收到的 envelope，
     * 与接线前逐字节一致。装配后以 offer 的 `hidden_sha256` 兜底校验；成功后丢弃分片状态。
     */
    private fun assembleChunkedRootInput(
        envelope: TaskWorkerEnvelope,
    ): TaskWorkerEnvelope? {
        val payload = envelope.payload
        if (payload["stage_type"] != TaskWorkerProtocol.LAYER_FORWARD_STAGE) return null
        val rootInput = payload["root_input"] as? Map<*, *> ?: return null
        if (rootInput["hidden_f32"] is String) return null
        if (rootInput["hidden_ref"] !is Map<*, *>) return null
        val identity = identityFrom(payload)
        val raw = try {
            val assembled = stageChunks.assemble(identity.attemptId)
            val declared = (payload["hidden_sha256"] as? String).orEmpty()
            val actual = StageChunkAssembler.sha256Hex(assembled)
            if (declared.isNotEmpty() && !actual.equals(declared, ignoreCase = true)) {
                throw AndroidFullWorkerStageException(
                    "chunk_digest_mismatch",
                    "assembled hidden digest does not match the offer",
                )
            }
            assembled
        } finally {
            stageChunks.discard(identity.attemptId)
        }
        // 改写后**移除** `hidden_ref`：执行器只应看到内联 `hidden_f32`（避免两个来源并存）。
        val merged = LinkedHashMap<String, Any?>(
            rootInput.entries
                .filterNot { it.key.toString() == "hidden_ref" }
                .associate { it.key.toString() to it.value },
        )
        merged["hidden_f32"] = java.util.Base64.getEncoder().encodeToString(raw)
        val rewritten = LinkedHashMap<String, Any?>(payload)
        rewritten["root_input"] = merged
        return envelope.copy(payload = rewritten)
    }

    private suspend fun handleCancel(connection: TaskWorkerTransport, envelope: TaskWorkerEnvelope) {
        val identity = identityFrom(envelope.payload)
        // ★ 2026-10-07（DIST-NEXT-1）：回程的 `provider_id` 必须是**主节点口径**
        //   （`remote_<node_id>`）。`stage_cancel` 的 payload 里**没有** `provider_id`
        //   （协议精确字段集如此），因此优先回显 offer 时记下的那个 ——
        //   回裸 `nodeId` 会被主节点以 `attempt_identity_mismatch` 拒绝，
        //   真机表现正是「`stage_cancel_acknowledged` 恒缺席」。
        val reasonCode = (envelope.payload["reason_code"] as? String)
            ?.takeIf { it.isNotBlank() } ?: "coordinator_cancelled"
        val request = requestCancellation(reasonCode, identity) ?: return
        val providerId = request.providerId
        publish()
        try {
            sendCancelledAck(connection, identity, providerId, reasonCode,
                if (isExecutionInFlight(identity)) TaskWorkerProtocol.EXECUTION_IN_FLIGHT
                else TaskWorkerProtocol.EXECUTION_STOPPED)
        } finally {
            convergeCancellation(connection, identity, providerId)
        }
    }

    /** Timeout is diagnostic only: physical execution must stop before ACK/slot release. */
    private fun convergeCancellation(
        connection: TaskWorkerTransport,
        identity: TaskWorkerAttemptIdentity,
        providerId: String,
    ) {
        val job = synchronized(convergenceLock) {
            if (!isAttemptCancellingOrDone(identity)) return
            if (convergenceIdentity == identity && convergenceJob?.isCompleted == false) return
            convergenceIdentity = identity
            scope.launch(start = CoroutineStart.LAZY) {
                val startedMs = monotonicMs()
                var warned = false
                // Keep waiting after 30s, but remain cancellable during connection/service retirement.
                while (isExecutionInFlight(identity)) {
                    if (transport !== connection || !isAttemptCancellingOrDone(identity)) return@launch
                    if (!warned && monotonicMs() - startedMs >= CANCEL_STOP_WAIT_MS) {
                        warned = true
                        runCatching { QlhLogger.w("TaskWorkerClient",
                            "cancel convergence timed out after ${CANCEL_STOP_WAIT_MS}ms " +
                                "attempt=${identity.attemptId}: retaining the execution slot") }
                    }
                    delay(CANCEL_STOP_POLL_MS)
                }
                finishCancelledExecution(connection, identity, providerId)
            }.also { convergenceJob = it }
        }
        job.start()
    }

    /** 该 attempt 是否已进入取消（`CANCELLING`）—— 决定是否还能发结果/错误。 */
    private fun isAttemptCancelling(identity: TaskWorkerAttemptIdentity): Boolean {
        val active = machine.snapshot().activeAttempt
        return active.identity == identity &&
            active.state == TaskWorkerAttemptState.CANCELLING
    }

    /**
     * ★ 2026-10-07（真机 P0）：该 attempt 是否处于**取消家族**状态。
     *
     * 取消收敛有两条可能路径（执行协程的取消分支、独立收敛协程），谁先到都可能把状态推进到
     * `CANCELLED`。若终态 ACK 仍以 `CANCELLING` 为门槛，后到的那条就会**跳过发送** ——
     * 主节点因此只看到 `execution_in_flight`（现场正是如此）。重复发送由
     * [sendCancelledAck] 的同状态去重负责，所以这里放宽到 `CANCELLED` 也允许。
     * 取消期间租约过期进入 LOST 后，同一身份仍可补真实停止回执；不能复活执行。
     */
    private fun isAttemptCancellingOrDone(identity: TaskWorkerAttemptIdentity): Boolean {
        val active = machine.snapshot().activeAttempt
        return active.identity == identity && (active.state in CANCELLING_STATES ||
            (active.state == TaskWorkerAttemptState.LOST && cancellationRequest?.identity == identity))
    }

    /**
     * ★ 2026-10-07（DIST-NEXT-1）：执行真正停止后把取消收敛成终态。
     *
     * 只有 `CANCELLING` 中的 attempt 才发 ACK；`stage_cancelled` 的
     * `execution_state` 由 [sendCancelledAck] 去重，因此这里补发的终态
     * 恰好把此前的 `execution_in_flight` 升级为 `execution_stopped`。
     */
    private suspend fun finishCancelledExecution(
        connection: TaskWorkerTransport,
        identity: TaskWorkerAttemptIdentity,
        providerId: String,
    ) = withContext(NonCancellable) {
        if (!isAttemptCancellingOrDone(identity) || isExecutionInFlight(identity)) return@withContext
        val reasonCode = cancellationRequest?.takeIf { it.identity == identity }?.reasonCode ?: "cancelled"
        try {
            sendCancelledAck(connection, identity, providerId, reasonCode, TaskWorkerProtocol.EXECUTION_STOPPED)
        } finally {
            // Sending may fail, but neither a cancelled Job nor a timeout proves native has stopped.
            if (!isExecutionInFlight(identity) && machine.cancelled(identity)) publish()
        }
    }

    /** Serialize check/send/record by full identity; STOPPED can never regress to IN_FLIGHT. */
    private suspend fun sendCancelledAck(
        connection: TaskWorkerTransport,
        identity: TaskWorkerAttemptIdentity,
        providerId: String,
        reasonCode: String,
        executionState: String,
    ) = cancelAckMutex.withLock {
        if (!isAttemptCancellingOrDone(identity)) return@withLock
        if (executionState == TaskWorkerProtocol.EXECUTION_STOPPED && isExecutionInFlight(identity)) return@withLock
        if (cancelAckIdentity == identity &&
            (cancelAckState == executionState || cancelAckState == TaskWorkerProtocol.EXECUTION_STOPPED)) return@withLock
        connection.send(TaskWorkerProtocol.buildStageCancelled(
            identity = identity,
            providerId = providerId,
            reasonCode = reasonCode,
            messageId = newMessageId("cancelled"),
            sentAtMs = clockMs(),
            executionState = executionState,
        ))
        // A failed/cancelled send must leave a later convergence attempt free to retry.
        cancelAckIdentity = identity
        cancelAckState = executionState
        publish()
    }

    private fun handleLeaseRenew(envelope: TaskWorkerEnvelope) {
        val identity = identityFrom(envelope.payload)
        val expires = (envelope.payload["lease_expires_at_ms"] as Number).toLong()
        machine.renew(identity, expires, clockMs())
        publish()
    }

    private fun identityFrom(payload: Map<String, Any?>): TaskWorkerAttemptIdentity = TaskWorkerAttemptIdentity(
        workflowId = payload["workflow_id"] as String,
        stageId = payload["stage_id"] as String,
        attemptId = payload["attempt_id"] as String,
        leaseId = payload["lease_id"] as String,
        leaseEpoch = (payload["lease_epoch"] as Number).toInt(),
    )

    private fun publish() {
        val next = machine.snapshot().copy(processRunning = isRunning, connectionAlive = isConnectionLive)
        val previous = mutableSnapshot.value
        mutableSnapshot.value = next
        if (previous.connection != next.connection || previous.reconnectAttempt != next.reconnectAttempt) {
            QlhLogger.i("TaskWorkerClient", "worker_connection=${next.connection} " +
                "process_running=${next.processRunning} connection_alive=${next.connectionAlive} " +
                "retry=${next.reconnectAttempt} next_retry_at_ms=${next.nextRetryAtMs} " +
                "error_kind=${next.failureKind} error_code=${next.lastErrorCode}")
        }
    }

    private fun errorCode(error: Throwable): String = when (error) {
        is TaskWorkerProtocolException -> error.code
        is java.net.SocketTimeoutException -> "transport_timeout"
        is java.net.ConnectException -> "transport_connect_failed"
        else -> "transport_disconnected"
    }

    private fun newMessageId(kind: String): String = "msg_worker_${kind}_${UUID.randomUUID().toString().replace("-", "").take(24)}"

    companion object {
        /**
         * ★ 2026-10-07（DIST-NEXT-4）：task-worker 控制面心跳间隔。
         *
         * 与主仓 `scheduler_types.ANDROID_TASK_WORKER_HEARTBEAT_INTERVAL_SECONDS`（15s）
         * **必须同值** —— 主仓据它推导容忍上限（`WORKER_HEARTBEAT_MAX_AGE`）与控制面
         * health 超时。此前这个数字只在这里硬编码，主仓侧没有任何记录。
         */
        const val HEARTBEAT_INTERVAL_MS: Long = 15_000L
        const val HEARTBEAT_TIMEOUT_MS: Long = 45_000L

        /**
         * ★ 2026-10-07（真机 P0）：取消收敛的等待上限。
         *
         * native 层段前向是原子调用，`executionJob.cancel()` 打断不了它；取消到达后我们
         * **必须**在没有执行在飞时释放 attempt 槽位，否则后续每个 `stage_offer` 都被拒
         * `worker_busy_or_lease_invalid`。到上限只记录告警，继续保留物理槽位并等待停止。
         */
        const val CANCEL_STOP_WAIT_MS: Long = 30_000L

        /** 取消收敛的轮询间隔。 */
        const val CANCEL_STOP_POLL_MS: Long = 50L

        /** ★ 2026-10-07（真机 P0）：取消家族状态 —— 终态 ACK 的允许集合。 */
        private val CANCELLING_STATES = setOf(
            TaskWorkerAttemptState.CANCELLING,
            TaskWorkerAttemptState.CANCELLED,
        )
    }
}
