package com.qlh.inference.network

/** Wire-independent lifecycle for Android HTTP presence. */
enum class AndroidPresenceState {
    STOPPED,
    REGISTERING,
    ONLINE,
    BACKING_OFF,
    OFFLINE,
}

data class AndroidPresenceSnapshot(
    val state: AndroidPresenceState = AndroidPresenceState.STOPPED,
    val generation: Long = 0L,
    val leaseId: String = "",
    val leaseExpiresAtMs: Long = 0L,
    val heartbeatIntervalMs: Long = 45_000L,
    val serverTimeOffsetMs: Long = 0L,
    val lastHeartbeatAtMs: Long = 0L,
    val reconnectAttempt: Int = 0,
    val nextRetryAtMs: Long = 0L,
    val lastErrorCode: String? = null,
    val lastErrorMessage: String? = null,
)

/**
 * Fences late register/heartbeat responses by lease generation and bounds retry
 * delay. The service owns I/O; this class stays deterministic and JVM-testable.
 */
class AndroidPresenceStateMachine(
    private val baseBackoffMs: Long = 1_000L,
    private val maxBackoffMs: Long = 30_000L,
) {
    private var snapshot = AndroidPresenceSnapshot()

    @Synchronized
    fun snapshot(): AndroidPresenceSnapshot = snapshot

    @Synchronized
    fun start(nowMs: Long): AndroidPresenceSnapshot {
        if (snapshot.state == AndroidPresenceState.STOPPED || snapshot.state == AndroidPresenceState.OFFLINE) {
            snapshot = snapshot.copy(
                state = AndroidPresenceState.REGISTERING,
                nextRetryAtMs = nowMs,
                lastErrorCode = null,
                lastErrorMessage = null,
            )
        }
        return snapshot
    }

    @Synchronized
    fun beginRegistration(nowMs: Long): Boolean {
        if (snapshot.state == AndroidPresenceState.STOPPED) start(nowMs)
        if (snapshot.state != AndroidPresenceState.REGISTERING) return false
        return nowMs >= snapshot.nextRetryAtMs
    }

    @Synchronized
    fun onRegistered(
        generation: Long,
        leaseId: String,
        leaseExpiresAtMs: Long,
        heartbeatIntervalSeconds: Int,
        nowMs: Long,
        serverTimeMs: Long = 0L,
    ): AndroidPresenceSnapshot {
        val offset = if (serverTimeMs > 0L) serverTimeMs - nowMs else snapshot.serverTimeOffsetMs
        if (generation <= 0L || leaseId.isBlank() || leaseExpiresAtMs <= nowMs + offset) {
            return onFailure("invalid_presence_lease", "主节点返回的 presence lease 无效", nowMs)
        }
        snapshot = snapshot.copy(
            state = AndroidPresenceState.ONLINE,
            generation = generation,
            leaseId = leaseId,
            leaseExpiresAtMs = leaseExpiresAtMs,
            heartbeatIntervalMs = heartbeatIntervalSeconds.coerceIn(5, 120) * 1_000L,
            serverTimeOffsetMs = offset,
            lastHeartbeatAtMs = nowMs,
            reconnectAttempt = 0,
            nextRetryAtMs = 0L,
            lastErrorCode = null,
            lastErrorMessage = null,
        )
        return snapshot
    }

    @Synchronized
    fun onHeartbeatSuccess(
        generation: Long,
        leaseId: String,
        leaseExpiresAtMs: Long,
        heartbeatIntervalSeconds: Int,
        nowMs: Long,
        serverTimeMs: Long = 0L,
    ): AndroidPresenceSnapshot {
        if (snapshot.state != AndroidPresenceState.ONLINE ||
            generation != snapshot.generation || leaseId != snapshot.leaseId
        ) return snapshot
        val offset = if (serverTimeMs > 0L) serverTimeMs - nowMs else snapshot.serverTimeOffsetMs
        if (leaseExpiresAtMs <= nowMs + offset) return onFailure("invalid_presence_lease", "心跳返回的租约已过期", nowMs)
        snapshot = snapshot.copy(
            leaseExpiresAtMs = leaseExpiresAtMs,
            heartbeatIntervalMs = heartbeatIntervalSeconds.coerceIn(5, 120) * 1_000L,
            serverTimeOffsetMs = offset,
            lastHeartbeatAtMs = nowMs,
            lastErrorCode = null,
            lastErrorMessage = null,
        )
        return snapshot
    }

    @Synchronized
    fun heartbeatDue(nowMs: Long): Boolean =
        snapshot.state == AndroidPresenceState.ONLINE &&
            (nowMs - snapshot.lastHeartbeatAtMs >= snapshot.heartbeatIntervalMs ||
                nowMs + snapshot.serverTimeOffsetMs >= snapshot.leaseExpiresAtMs - snapshot.heartbeatIntervalMs / 2)

    @Synchronized
    fun onFailure(
        code: String,
        message: String,
        nowMs: Long,
        retryable: Boolean = true,
    ): AndroidPresenceSnapshot {
        val requiresRegistration = code in setOf(
            "presence_not_registered",
            "presence_lease_required",
            "stale_generation",
            "stale_lease",
            "lease_expired",
            "invalid_presence_lease",
        )
        if (!retryable) {
            snapshot = snapshot.copy(
                state = AndroidPresenceState.OFFLINE,
                nextRetryAtMs = 0L,
                lastErrorCode = code,
                lastErrorMessage = message,
            )
            return snapshot
        }
        val attempt = (snapshot.reconnectAttempt + 1).coerceAtMost(30)
        val delayMs = (baseBackoffMs * (1L shl (attempt - 1).coerceAtMost(10)))
            .coerceAtMost(maxBackoffMs)
        snapshot = snapshot.copy(
            state = if (requiresRegistration) AndroidPresenceState.REGISTERING else AndroidPresenceState.BACKING_OFF,
            generation = if (requiresRegistration) 0L else snapshot.generation,
            leaseId = if (requiresRegistration) "" else snapshot.leaseId,
            leaseExpiresAtMs = if (requiresRegistration) 0L else snapshot.leaseExpiresAtMs,
            reconnectAttempt = attempt,
            nextRetryAtMs = nowMs + if (requiresRegistration) 0L else delayMs,
            lastErrorCode = code,
            lastErrorMessage = message,
        )
        return snapshot
    }

    @Synchronized
    fun retryIfDue(nowMs: Long): AndroidPresenceSnapshot {
        if (snapshot.state == AndroidPresenceState.BACKING_OFF && nowMs >= snapshot.nextRetryAtMs) {
            snapshot = snapshot.copy(state = AndroidPresenceState.REGISTERING)
        }
        return snapshot
    }

    @Synchronized
    fun stop(): AndroidPresenceSnapshot {
        snapshot = snapshot.copy(state = AndroidPresenceState.STOPPED, nextRetryAtMs = 0L)
        return snapshot
    }

    @Synchronized
    fun resetForEndpoint(nowMs: Long): AndroidPresenceSnapshot {
        snapshot = AndroidPresenceSnapshot(
            state = AndroidPresenceState.REGISTERING,
            nextRetryAtMs = nowMs,
        )
        return snapshot
    }
}
