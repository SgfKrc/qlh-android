package com.qlh.inference.worker

import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
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
import java.util.UUID
import com.google.gson.GsonBuilder
import com.google.gson.JsonParser

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
    suspend fun close()
}

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

    override suspend fun send(envelope: TaskWorkerEnvelope) = withContext(Dispatchers.IO) {
        val inner = TaskWorkerProtocol.encode(envelope).toString(StandardCharsets.UTF_8)
        val outer = "{\"type\":\"task_worker\",\"format\":\"json\",\"data\":$inner}"
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

    override suspend fun receive(): TaskWorkerEnvelope? = withContext(Dispatchers.IO) {
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
        if (objectValue.get("type")?.asString != "task_worker" ||
            objectValue.get("format")?.asString != "json"
        ) {
            throw TaskWorkerProtocolException("unexpected task worker frame", "invalid_frame", "message")
        }
        val data = objectValue.get("data")
        if (data == null || !data.isJsonObject) {
            throw TaskWorkerProtocolException("task worker frame has no data object", "invalid_frame", "data")
        }
        TaskWorkerProtocol.decode(gson.toJson(data).toByteArray(StandardCharsets.UTF_8))
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
) : TaskWorkerTransportFactory {
    override suspend fun connect(host: String, port: Int): TaskWorkerTransport = withContext(Dispatchers.IO) {
        require(port in 1..65535) { "port must be between 1 and 65535" }
        val normalizedHost = host.trim().removePrefix("[").removeSuffix("]")
        val socket = Socket()
        socket.connect(InetSocketAddress(normalizedHost, port), connectTimeoutMs)
        socket.soTimeout = readTimeoutMs
        SocketTaskWorkerTransport(socket)
    }
}

fun interface TaskWorkerStageHandler {
    suspend fun execute(offer: TaskWorkerEnvelope): TaskWorkerStageExecution
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
    private val capabilities: () -> Map<String, Any?>,
    private val transportFactory: TaskWorkerTransportFactory = SocketTaskWorkerTransportFactory(),
    private val stageHandler: TaskWorkerStageHandler? = null,
    private val scope: CoroutineScope = CoroutineScope(SupervisorJob() + Dispatchers.IO),
    private val clockMs: () -> Long = { System.currentTimeMillis() },
) {
    private val machine = TaskWorkerStateMachine()
    private val mutableSnapshot = MutableStateFlow(machine.snapshot())
    private var loopJob: Job? = null
    private var transport: TaskWorkerTransport? = null
    private var executionJob: Job? = null

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
        val errorCode = reasonCode.takeIf { it.matches(Regex("^[a-z][a-z0-9_]{0,63}$")) }
            ?: "worker_cancelled"
        machine.fail(identity, errorCode, retryable = true)
        publish()
        executionJob?.cancel()
        scope.launch {
            val current = transport ?: return@launch
            runCatching {
                current.send(TaskWorkerProtocol.buildStageError(
                    identity = identity,
                    providerId = nodeId,
                    errorCode = errorCode,
                    retryable = true,
                    messageId = newMessageId("cancel"),
                    sentAtMs = clockMs(),
                ))
            }.onFailure { error ->
                machine.onDisconnected(clockMs(), "cancel_send_failed", error.message ?: "cancel send failed")
                publish()
            }
        }
        return true
    }

    /** Non-blocking stop used by Android Service lifecycle callbacks. */
    fun stop() {
        machine.stop()
        publish()
        executionJob?.cancel()
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
                receiveLoop(opened)
            } catch (error: CancellationException) {
                throw error
            } catch (error: Exception) {
                machine.onDisconnected(clockMs(), errorCode(error), error.message ?: error.javaClass.simpleName)
                publish()
            } finally {
                runCatching { transport?.close() }
                transport = null
            }
        }
    }

    private suspend fun receiveLoop(connection: TaskWorkerTransport) {
        while (scope.isActive && machine.snapshot().connection == TaskWorkerConnectionState.READY) {
            val envelope = connection.receive() ?: throw EOFException("coordinator closed worker connection")
            when (envelope.messageType) {
                TaskWorkerProtocol.STAGE_OFFER -> handleOffer(connection, envelope)
                TaskWorkerProtocol.STAGE_CANCEL -> handleCancel(connection, envelope)
                TaskWorkerProtocol.LEASE_RENEW -> handleLeaseRenew(envelope)
                else -> throw TaskWorkerProtocolException(
                    "unexpected coordinator message for Android worker",
                    "unexpected_message_type",
                    "message_type",
                )
            }
        }
    }

    private suspend fun handleOffer(connection: TaskWorkerTransport, envelope: TaskWorkerEnvelope) {
        val payload = envelope.payload
        val identity = identityFrom(payload)
        val expires = (payload["lease_expires_at_ms"] as Number).toLong()
        if (!machine.offer(identity, expires, clockMs())) {
            connection.send(TaskWorkerProtocol.buildStageAccept(
                identity, nodeId, false, "worker_busy_or_lease_invalid", true,
                newMessageId("accept"), clockMs(),
            ))
            return
        }
        val handler = stageHandler
        if (handler == null) {
            machine.fail(identity, "worker_execution_not_configured", retryable = true)
            publish()
            connection.send(TaskWorkerProtocol.buildStageAccept(
                identity, nodeId, false, "worker_execution_not_configured", true,
                newMessageId("accept"), clockMs(),
            ))
            return
        }
        connection.send(TaskWorkerProtocol.buildStageAccept(
            identity, nodeId, true, "", false, newMessageId("accept"), clockMs(),
        ))
        machine.markRunning(identity, clockMs())
        publish()
        executionJob = scope.launch {
            try {
                val execution = handler.execute(envelope)
                val output = TaskWorkerProtocol.buildStageResult(
                    identity, nodeId, execution.output, execution.metadata,
                    newMessageId("result"), clockMs(),
                )
                val digest = output.payload["output_sha256"] as String
                if (machine.complete(identity, digest, clockMs())) {
                    connection.send(output)
                    publish()
                }
            } catch (error: CancellationException) {
                if (machine.snapshot().activeAttempt.identity == identity && machine.snapshot().activeAttempt.state == TaskWorkerAttemptState.CANCELLING) {
                    machine.cancelled(identity)
                    connection.send(TaskWorkerProtocol.buildStageCancelled(
                        identity, nodeId, "cancelled", newMessageId("cancelled"), clockMs(),
                    ))
                    publish()
                }
            } catch (error: Exception) {
                val errorCode = (error as? AndroidFullWorkerStageException)?.code
                    ?: "worker_execution_failed"
                if (machine.fail(identity, errorCode, retryable = true)) {
                    connection.send(TaskWorkerProtocol.buildStageError(
                        identity, nodeId, errorCode, true,
                        newMessageId("error"), clockMs(),
                    ))
                    publish()
                }
            }
        }
    }

    private suspend fun handleCancel(connection: TaskWorkerTransport, envelope: TaskWorkerEnvelope) {
        val identity = identityFrom(envelope.payload)
        if (machine.requestCancel(clockMs(), expectedIdentity = identity) == identity) {
            executionJob?.cancel()
            if (stageHandler == null) {
                machine.cancelled(identity)
                connection.send(TaskWorkerProtocol.buildStageCancelled(
                    identity, nodeId, "cancelled", newMessageId("cancelled"), clockMs(),
                ))
                publish()
            }
        }
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
}
