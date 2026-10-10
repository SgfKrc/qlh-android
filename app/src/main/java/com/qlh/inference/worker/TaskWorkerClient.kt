package com.qlh.inference.worker

import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.NonCancellable
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.cancelAndJoin
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
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
)

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
) {
    private var snapshot = TaskWorkerSnapshot()

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
        if (snapshot.connection == TaskWorkerConnectionState.BACKING_OFF) return snapshot
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
    fun offer(identity: TaskWorkerAttemptIdentity, leaseExpiresAtMs: Long, nowMs: Long): Boolean {
        expireActiveAttempt(nowMs)
        if (snapshot.connection != TaskWorkerConnectionState.READY || leaseExpiresAtMs <= nowMs) return false
        val current = snapshot.activeAttempt
        if (current.state in ACTIVE_ATTEMPT_STATES) return false
        snapshot = snapshot.copy(
            activeAttempt = TaskWorkerAttemptSnapshot(
                identity = identity,
                state = TaskWorkerAttemptState.OFFERED,
                leaseExpiresAtMs = leaseExpiresAtMs,
            ),
        )
        return true
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
        snapshot = snapshot.copy(activeAttempt = current.copy(leaseExpiresAtMs = leaseExpiresAtMs))
        return true
    }

    @Synchronized
    fun complete(identity: TaskWorkerAttemptIdentity, outputSha256: String, nowMs: Long): Boolean {
        expireActiveAttempt(nowMs)
        val current = snapshot.activeAttempt
        if (!sameIdentity(current.identity, identity) || current.state !in ACTIVE_ATTEMPT_STATES) return false
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
        if (!sameIdentity(current.identity, identity) || current.state !in ACTIVE_ATTEMPT_STATES) return false
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
        val attempt = (snapshot.reconnectAttempt + 1).coerceAtMost(30)
        val delay = (baseBackoffMs.coerceAtLeast(1L) * (1L shl (attempt - 1).coerceAtMost(5)))
            .coerceAtMost(maxBackoffMs.coerceAtLeast(baseBackoffMs))
        return snapshot.copy(
            connection = TaskWorkerConnectionState.BACKING_OFF,
            reconnectAttempt = attempt,
            nextRetryAtMs = nowMs + delay,
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
    suspend fun send(envelope: TaskWorkerEnvelope)
    suspend fun receive(): TaskWorkerEnvelope?
    suspend fun receiveEvent(): TaskWorkerInboundEvent =
        receive()?.let { TaskWorkerInboundEvent.Envelope(it) }
            ?: TaskWorkerInboundEvent.Closed
    suspend fun sendHeartbeat(nodeId: String, sentAtMs: Long) = Unit
    suspend fun close()
}

sealed class TaskWorkerInboundEvent {
    data class Envelope(val value: TaskWorkerEnvelope) : TaskWorkerInboundEvent()
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
class SocketTaskWorkerTransport(
    private val socket: Socket,
    private val maxFrameBytes: Int = TaskWorkerProtocol.MAX_MESSAGE_BYTES + 1024,
) : TaskWorkerTransport {
    private val input = DataInputStream(BufferedInputStream(socket.getInputStream()))
    private val output = DataOutputStream(BufferedOutputStream(socket.getOutputStream()))
    private val sendLock = Any()
    private val gson = GsonBuilder().disableHtmlEscaping().create()

    private suspend fun sendOuter(type: String, data: Any? = null) = withContext(Dispatchers.IO) {
        val dataJson = data?.let { ",\"data\":${gson.toJson(it)}" } ?: ""
        val outer = "{\"type\":\"$type\",\"format\":\"json\"$dataJson}"
            .toByteArray(StandardCharsets.UTF_8)
        if (outer.size > maxFrameBytes) throw TaskWorkerProtocolException(
            "task worker frame exceeds maximum size",
            "message_too_large",
            "message",
        )
        synchronized(sendLock) {
            output.writeInt(outer.size)
            output.write(outer)
            output.flush()
        }
    }

    override suspend fun send(envelope: TaskWorkerEnvelope) = withContext(Dispatchers.IO) {
        val inner = TaskWorkerProtocol.encode(envelope).toString(StandardCharsets.UTF_8)
        sendOuterJson("{\"type\":\"task_worker\",\"format\":\"json\",\"data\":$inner}")
    }

    private suspend fun sendOuterJson(json: String) = withContext(Dispatchers.IO) {
        val outer = json.toByteArray(StandardCharsets.UTF_8)
        if (outer.size > maxFrameBytes) throw TaskWorkerProtocolException(
            "task worker frame exceeds maximum size", "message_too_large", "message",
        )
        synchronized(sendLock) {
            output.writeInt(outer.size)
            output.write(outer)
            output.flush()
        }
    }

    private suspend fun receiveOuter(): com.google.gson.JsonObject? = withContext(Dispatchers.IO) {
        val size = try {
            input.readInt()
        } catch (_: EOFException) {
            return@withContext null
        }
        if (size <= 0 || size > maxFrameBytes) {
            throw TaskWorkerProtocolException("invalid task worker frame length", "invalid_frame", "message")
        }
        val bytes = ByteArray(size)
        input.readFully(bytes)
        val root = try {
            JsonParser.parseString(String(bytes, StandardCharsets.UTF_8))
        } catch (error: Exception) {
            throw TaskWorkerProtocolException("task worker frame is not valid JSON", "invalid_frame", "message").also {
                it.initCause(error)
            }
        }
        if (!root.isJsonObject) throw TaskWorkerProtocolException("task worker frame must be an object", "invalid_frame", "message")
        val objectValue = root.asJsonObject
        if (objectValue.get("format")?.asString != "json") {
            throw TaskWorkerProtocolException("unexpected task worker frame", "invalid_frame", "format")
        }
        objectValue
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
        val ack = receiveOuter() ?: throw EOFException("coordinator closed during register")
        if (ack.get("type")?.asString != "register") {
            throw TaskWorkerProtocolException("expected register acknowledgement", "unexpected_message_type", "type")
        }
        val data = ack.getAsJsonObject("data")
        if (data?.get("status")?.asString != "registered") {
            throw TaskWorkerProtocolException(
                data?.get("reason")?.asString ?: "coordinator rejected registration",
                "registration_rejected", "data.status",
            )
        }
    }

    override suspend fun receiveEvent(): TaskWorkerInboundEvent {
        while (true) {
            val objectValue = receiveOuter() ?: return TaskWorkerInboundEvent.Closed
            when (objectValue.get("type")?.asString) {
                "task_worker" -> {
                    val data = objectValue.get("data")
                    if (data == null || !data.isJsonObject) {
                        throw TaskWorkerProtocolException("task worker frame has no data object", "invalid_frame", "data")
                    }
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
                    return TaskWorkerInboundEvent.Envelope(
                        TaskWorkerProtocol.decode(gson.toJson(data).toByteArray(StandardCharsets.UTF_8))
                    )
                }
                "heartbeat_ack" -> return TaskWorkerInboundEvent.HeartbeatAck
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
                        val payload = data.asJsonObject.deepCopy()
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
                "node_list_sync", "node_update", "layer_config" -> Unit
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
        TaskWorkerInboundEvent.Closed -> null
        TaskWorkerInboundEvent.HeartbeatAck -> throw TaskWorkerProtocolException(
            "heartbeat acknowledgement is not a task worker envelope", "unexpected_message_type", "type",
        )
    }

    override suspend fun sendHeartbeat(nodeId: String, sentAtMs: Long) {
        sendOuter("heartbeat", mapOf("client_id" to nodeId, "t_send" to sentAtMs))
    }

    override suspend fun close() {
        withContext(Dispatchers.IO) {
            runCatching { socket.close() }
        }
    }
}

class SocketTaskWorkerTransportFactory(
    private val connectTimeoutMs: Int = 8_000,
    private val readTimeoutMs: Int = 45_000,
    private val registration: TaskWorkerRegistration? = null,
) : TaskWorkerTransportFactory {
    override suspend fun connect(host: String, port: Int): TaskWorkerTransport = withContext(Dispatchers.IO) {
        require(port in 1..65535) { "port must be between 1 and 65535" }
        val normalizedHost = host.trim().removePrefix("[").removeSuffix("]")
        val socket = Socket()
        socket.connect(InetSocketAddress(normalizedHost, port), connectTimeoutMs)
        socket.soTimeout = readTimeoutMs
        val transport = SocketTaskWorkerTransport(socket)
        try {
            registration?.let { transport.register(it) }
            transport
        } catch (error: Exception) {
            transport.close()
            throw error
        }
    }
}

fun interface TaskWorkerStageHandler {
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
    private val scope: CoroutineScope = CoroutineScope(SupervisorJob() + Dispatchers.IO),
    private val clockMs: () -> Long = { System.currentTimeMillis() },
) {
    private val machine = TaskWorkerStateMachine()
    private val mutableSnapshot = MutableStateFlow(machine.snapshot())
    private var loopJob: Job? = null
    private var transport: TaskWorkerTransport? = null
    private var executionJob: Job? = null
    private var heartbeatJob: Job? = null

    // ★ 2026-10-07（DIST-NEXT-1）：取消 ACK 的单一事实源。
    //   `stage_cancelled` 可能发两次：先是 ACK（可能 `execution_in_flight`），
    //   执行真正停止后再补一条终态。这里记录「已发出的状态」，保证每条状态
    //   只发一次，且不会把 ACK 冒充成终态。
    private val cancelLock = Any()
    private var cancelAckAttemptId: String? = null
    private var cancelAckState: String? = null
    private var cancelAckReasonCode: String = ""

    /** offer 里的 `provider_id`（主节点口径）；本地取消的回程必须回显它。 */
    @Volatile
    private var activeProviderId: String? = null

    /** ★ 2026-10-07（DIST-NEXT-2b）：大 payload 分片的装配器（按 attempt 分区）。 */
    private val stageChunks = StageChunkAssembler()

    val snapshot: StateFlow<TaskWorkerSnapshot> = mutableSnapshot.asStateFlow()

    @Synchronized
    fun start() {
        if (loopJob?.isActive == true) return
        machine.start()
        publish()
        loopJob = scope.launch { runLoop() }
    }

    fun cancelActive(reasonCode: String = "user_cancelled"): Boolean {
        val identity = machine.requestCancel(clockMs()) ?: return false
        val code = reasonCode.takeIf { it.matches(Regex("^[a-z][a-z0-9_]{0,63}$")) }
            ?: "worker_cancelled"
        // ★ 2026-10-07（DIST-NEXT-1）：本地取消同样走**唯一取消状态机** ——
        //   回程是 `stage_cancelled`（而非 `stage_error`），并如实报告执行状态；
        //   主节点据此区分「用户取消」「执行失败」「对端已停止」。
        //   ⚠️ 不再取消心跳：心跳只取决于连接状态，与单个 attempt 无关；此前
        //      在这里停掉心跳会让 worker 在取消后 120s 内被判失联。
        stageHandler?.requestCancel(identity)
        executionJob?.cancel()
        publish()
        val inFlight = stageHandler?.isExecutionInFlight(identity) == true
        scope.launch {
            val current = transport ?: return@launch
            val providerId = activeProviderId ?: nodeId
            runCatching {
                sendCancelledAck(
                    current,
                    identity,
                    providerId,
                    code,
                    if (inFlight) TaskWorkerProtocol.EXECUTION_IN_FLIGHT
                    else TaskWorkerProtocol.EXECUTION_STOPPED,
                )
            }.onFailure { error ->
                machine.onDisconnected(clockMs(), "cancel_send_failed", error.message ?: "cancel send failed")
                publish()
            }
            if (!inFlight) {
                machine.cancelled(identity)
                publish()
            } else {
                // ★ 2026-10-07（真机 P0）：本地取消同样要收敛（此前只依赖执行协程的取消分支，
                //   而那条路径上的 `send` 会因协程已取消而抛 ⇒ 终态 ACK 与槽位释放一起丢失）。
                convergeCancellation(current, identity, providerId)
            }
        }
        return true
    }

    /** Non-blocking stop used by Android Service lifecycle callbacks. */
    fun stop() {
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

    suspend fun stopAndJoin() {
        machine.stop()
        publish()
        executionJob?.cancel()
        runCatching { transport?.close() }
        transport = null
        loopJob?.cancelAndJoin()
        loopJob = null
        scope.cancel()
    }

    private suspend fun runLoop() {
        while (scope.isActive && machine.snapshot().connection != TaskWorkerConnectionState.STOPPED) {
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
                opened.send(TaskWorkerProtocol.buildHello(
                    nodeId = nodeId,
                    capabilities = capabilities(),
                    messageId = newMessageId("hello"),
                    sentAtMs = clockMs(),
                ))
                val ack = opened.receive() ?: throw EOFException("coordinator closed during hello")
                if (ack.messageType != TaskWorkerProtocol.HELLO_ACK) {
                    throw TaskWorkerProtocolException("expected hello_ack", "unexpected_message_type", "message_type")
                }
                val accepted = ack.payload["accepted"] == true
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
                    while (isActive && machine.snapshot().connection == TaskWorkerConnectionState.READY) {
                        delay(HEARTBEAT_INTERVAL_MS)
                        // ★ 2026-10-08（真机根因）：单次发送失败**不得**让心跳循环静默退出。
                        //   此前 `sendHeartbeat` 抛异常会直接结束这个 coroutine（`while` 之外没有
                        //   catch）⇒ 该连接此后**一次心跳都不再发**；master 侧实测
                        //   `event=tcp_heartbeat_received` 收到 0 条，而 worker 的
                        //   `soTimeout = 45_000` 又要求 45 秒内必须有入站消息 ⇒ 连接只能被
                        //   断开重连（真机表现：Y700 每 ~37 秒断开一次）。这里改为：失败记日志并
                        //   继续下一轮；真需要重连时由 `receiveLoop` 的读失败驱动。
                        try {
                            opened.sendHeartbeat(nodeId, clockMs())
                            QlhLogger.d("TaskWorkerClient", "heartbeat sent")
                        } catch (error: CancellationException) {
                            throw error
                        } catch (error: Exception) {
                            QlhLogger.w(
                                "TaskWorkerClient",
                                "heartbeat send failed: " +
                                    (error.message ?: error.javaClass.simpleName),
                            )
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
                runCatching { transport?.close() }
                heartbeatJob?.cancel()
                heartbeatJob = null
                transport = null
            }
        }
    }

    private suspend fun receiveLoop(connection: TaskWorkerTransport) {
        while (scope.isActive && machine.snapshot().connection == TaskWorkerConnectionState.READY) {
            val event = connection.receiveEvent()
            // ★ 2026-10-08（真机 P0 候选）：**分发层必须容错**。此前 `handleOffer` 里任何
            //   未预期的异常（例如分片装配抛出的类型不在那一个 `catch` 里）都会直接逃出本
            //   循环 ⇒ 读循环**静默死亡** ⇒ 之后 offer/分片全部无人处理，而心跳是另一个协程、
            //   仍在照常发送。真机现象与此完全一致：18 片 + offer 都送达、设备侧 CPU 空闲、
            //   连一条装配日志都没有，master 只看到 `remote Stage response timed out`。
            try {
                when (event) {
                    TaskWorkerInboundEvent.Closed -> throw EOFException("coordinator closed worker connection")
                    TaskWorkerInboundEvent.HeartbeatAck -> Unit
                    is TaskWorkerInboundEvent.Envelope -> when (event.value.messageType) {
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
        val identity = identityFrom(payload)
        // ★ 回程必须**原样回显** offer 里的 `provider_id` —— 它是主节点的 Provider 口径
        //   （`remote_<node_id>`），不是本节点的裸 nodeId。此前一律回裸 nodeId，被主节点
        //   以 `attempt_identity_mismatch` 拒掉，真机表现为 Stage 响应一直等到超时。
        val providerId = (payload["provider_id"] as? String)?.takeIf { it.isNotBlank() } ?: nodeId
        activeProviderId = providerId
        val expires = (payload["lease_expires_at_ms"] as Number).toLong()
        if (!machine.offer(identity, expires, clockMs())) {
            // ★ 2026-10-07（真机 P0）：拒绝必须留痕。此前设备侧完全没有日志，现场只能从
            //   主节点看到 `worker_busy_or_lease_invalid`，无法区分「上一个 attempt 未释放
            //   槽位」还是「lease 过期」。
            val state = machine.snapshot()
            QlhLogger.w(
                "TaskWorkerClient",
                "stage offer rejected: worker_busy_or_lease_invalid " +
                    "stage=${identity.stageId} attempt=${identity.attemptId} " +
                    "activeAttempt=${state.activeAttempt.state} " +
                    "connection=${state.connection}",
            )
            connection.send(TaskWorkerProtocol.buildStageAccept(
                identity, providerId, false, "worker_busy_or_lease_invalid", true,
                newMessageId("accept"), clockMs(),
            ))
            return
        }
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
        machine.markRunning(identity, clockMs())
        publish()
        executionJob = scope.launch {
            try {
                val execution = handler.execute(effectiveOffer)
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
                        connection.send(output)
                        publish()
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
                    if (machine.fail(identity, errorCode, retryable = true)) {
                        // ★ 2026-10-07（真机 P0）：执行失败必须留设备侧证据（此前完全静默）。
                        QlhLogger.e(
                            "TaskWorkerClient",
                            "stage execution failed: stage=${identity.stageId} " +
                                "attempt=${identity.attemptId} code=$errorCode " +
                                "detail=${error.message ?: error.javaClass.simpleName}",
                        )
                        connection.send(TaskWorkerProtocol.buildStageError(
                            identity, providerId, errorCode, true,
                            newMessageId("error"), clockMs(),
                        ))
                        publish()
                    }
                }
            }
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
        val raw = stageChunks.assemble(identity.attemptId)
        val declared = (payload["hidden_sha256"] as? String).orEmpty()
        val actual = StageChunkAssembler.sha256Hex(raw)
        if (declared.isNotEmpty() && !actual.equals(declared, ignoreCase = true)) {
            throw AndroidFullWorkerStageException(
                "chunk_digest_mismatch",
                "assembled hidden digest does not match the offer",
            )
        }
        stageChunks.discard(identity.attemptId)
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
        val providerId = (envelope.payload["provider_id"] as? String)
            ?.takeIf { it.isNotBlank() }
            ?: activeProviderId
            ?: nodeId
        val reasonCode = (envelope.payload["reason_code"] as? String)
            ?.takeIf { it.isNotBlank() } ?: "coordinator_cancelled"
        if (machine.requestCancel(clockMs(), expectedIdentity = identity) != identity) return
        // ★ 2026-10-07（DIST-NEXT-1）：取消合同。
        //   ① 先让执行器知道要停（native 层段前向在下一个可分割边界退出）；
        //   ② 再取消协程；
        //   ③ ACK **立即**发出，与「执行是否已停止」解耦 —— 层段模式下
        //      `stageHandler != null` 也必须回 ACK，且状态如实（`execution_in_flight`
        //      / `execution_stopped`）。执行真正停止后由 finishCancelledExecution
        //      补发终态，主节点不再需要等到租约/步骤超时。
        val handler = stageHandler
        // ⚠️ 必须在 `executionJob.cancel()` **之前**求值：`Job.isActive` 在取消后立即
        //   变 false，用它判断会把「仍在跑的原子前向」误报成「已停止」。
        val inFlight = handler?.isExecutionInFlight(identity) == true
        handler?.requestCancel(identity)
        executionJob?.cancel()
        publish()
        sendCancelledAck(
            connection,
            identity,
            providerId,
            reasonCode,
            if (inFlight) TaskWorkerProtocol.EXECUTION_IN_FLIGHT
            else TaskWorkerProtocol.EXECUTION_STOPPED,
        )
        if (!inFlight && machine.cancelled(identity)) publish()
        if (inFlight) convergeCancellation(connection, identity, providerId)
    }

    /**
     * ★ 2026-10-07（真机 P0）：等执行真正停止后补发终态、释放槽位。
     *
     * 必须跑在**独立且不可取消**的协程里：`executionJob.cancel()` 之后，执行协程上的任何挂起
     * （包括 `send`）都会立刻抛 `CancellationException`，把「终态 ACK + 槽位释放」一起带走 ⇒
     * attempt 卡在 `CANCELLING`（`ACTIVE_ATTEMPT_STATES`）⇒ 之后每个 `stage_offer` 都被拒
     * `worker_busy_or_lease_invalid`，直到 lease 过期。协调方取消（`handleCancel`）与
     * worker 侧主动取消（`cancelActive`）共用这一条收敛路径。
     */
    private fun convergeCancellation(
        connection: TaskWorkerTransport,
        identity: TaskWorkerAttemptIdentity,
        providerId: String,
    ) {
        scope.launch {
            withContext(NonCancellable) {
                val deadline = clockMs() + CANCEL_STOP_WAIT_MS
                while (stageHandler?.isExecutionInFlight(identity) == true &&
                    clockMs() < deadline
                ) {
                    delay(CANCEL_STOP_POLL_MS)
                }
                val stopped = stageHandler?.isExecutionInFlight(identity) != true
                // ⚠️ 先收敛、后记日志：槽位释放是这条路径上**唯一**不可省的副作用。
                //   （日志走 `android.util.Log`，在某些环境会抛异常；顺序反了会让「取消后
                //   worker 假死」在那些环境里复现。）
                finishCancelledExecution(connection, identity, providerId)
                runCatching {
                    if (stopped) {
                        QlhLogger.i(
                            "TaskWorkerClient",
                            "cancel converged: attempt=${identity.attemptId} " +
                                "execution stopped, slot released",
                        )
                    } else {
                        // 超时也要释放槽位：卡在 `CANCELLING` 会让 worker 对后续所有 stage
                        // 假死，比「旧前向可能仍在飞」更严重（主节点侧仍有 lease/attempt 护栏）。
                        QlhLogger.w(
                            "TaskWorkerClient",
                            "cancel convergence timed out after ${CANCEL_STOP_WAIT_MS}ms " +
                                "attempt=${identity.attemptId}: releasing the attempt slot",
                        )
                    }
                }
            }
        }
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
     */
    private fun isAttemptCancellingOrDone(identity: TaskWorkerAttemptIdentity): Boolean {
        val active = machine.snapshot().activeAttempt
        return active.identity == identity &&
            active.state in CANCELLING_STATES
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
    ) {
        if (!isAttemptCancellingOrDone(identity)) return
        val reasonCode = synchronized(cancelLock) {
            if (cancelAckAttemptId == identity.attemptId) cancelAckReasonCode
            else "cancelled"
        }
        try {
            sendCancelledAck(
                connection, identity, providerId, reasonCode,
                TaskWorkerProtocol.EXECUTION_STOPPED,
            )
        } finally {
            // ★ 2026-10-07（真机 P0）：槽位释放**不得**依赖 ACK 是否送达 —— 连接可能已断、
            //   协程可能已被取消。否则 attempt 卡在 `CANCELLING` ⇒ 后续 offer 全被拒。
            if (machine.cancelled(identity)) publish()
        }
    }

    /**
     * 发送 `stage_cancelled` 的唯一入口（同一 attempt 的同一状态只发一次）。
     */
    private suspend fun sendCancelledAck(
        connection: TaskWorkerTransport,
        identity: TaskWorkerAttemptIdentity,
        providerId: String,
        reasonCode: String,
        executionState: String,
    ) {
        // ★ 2026-10-07（真机 P0）：去重标记必须代表「**已成功发送**」这个事实。
        //   此前是在发送**之前**写标记：执行协程里的那次发送会抛 `CancellationException`
        //   （协程已被 `executionJob.cancel()` 取消），标记却已置位 ⇒ 独立收敛协程被去重挡掉
        //   ⇒ 终态 `execution_stopped` ACK 永远发不出去（现场只看到 `execution_in_flight`）。
        val shouldSend = synchronized(cancelLock) {
            !(cancelAckAttemptId == identity.attemptId && cancelAckState == executionState)
        }
        if (!shouldSend) return
        connection.send(TaskWorkerProtocol.buildStageCancelled(
            identity = identity,
            providerId = providerId,
            reasonCode = reasonCode,
            messageId = newMessageId("cancelled"),
            sentAtMs = clockMs(),
            executionState = executionState,
        ))
        synchronized(cancelLock) {
            cancelAckAttemptId = identity.attemptId
            cancelAckState = executionState
            cancelAckReasonCode = reasonCode
        }
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
        mutableSnapshot.value = machine.snapshot()
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

        /**
         * ★ 2026-10-07（真机 P0）：取消收敛的等待上限。
         *
         * native 层段前向是原子调用，`executionJob.cancel()` 打断不了它；取消到达后我们
         * **必须**在没有执行在飞时释放 attempt 槽位，否则后续每个 `stage_offer` 都被拒
         * `worker_busy_or_lease_invalid`（真机实测：直到 lease 过期才恢复）。等待到上限仍
         * 未停止时也会释放槽位，并留 `QlhLogger.w` 证据。
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
