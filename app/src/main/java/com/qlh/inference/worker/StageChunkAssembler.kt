package com.qlh.inference.worker

import java.security.MessageDigest
import java.util.concurrent.TimeUnit

internal class StageChunkBudget(
    private val maxBufferedBytes: Long,
    private val maxAttempts: Int,
    private val ttlNanos: Long,
    private val nanoTime: () -> Long = System::nanoTime,
) {
    private data class Key(val owner: Any, val attemptId: String)

    private class Attempt(val chunkCount: Int, var lastChunkAtNanos: Long) {
        val received = HashMap<Int, ByteArray>()
        var totalBytes: Int = 0
    }

    private val attempts = HashMap<Key, Attempt>()
    private var bufferedBytes = 0L

    init {
        require(maxBufferedBytes > 0 && maxAttempts > 0 && ttlNanos > 0)
    }

    private fun expire(now: Long) {
        val iterator = attempts.values.iterator()
        while (iterator.hasNext()) {
            val attempt = iterator.next()
            if (now - attempt.lastChunkAtNanos >= ttlNanos) {
                bufferedBytes -= attempt.totalBytes.toLong()
                iterator.remove()
            }
        }
    }

    @Synchronized
    fun add(
        owner: Any,
        attemptId: String,
        chunkIndex: Int,
        chunkCount: Int,
        payload: ByteArray,
        maxTotalBytes: Int,
    ): Int {
        val now = nanoTime()
        expire(now)
        val key = Key(owner, attemptId)
        val current = attempts[key]
        if (current != null && current.chunkCount != chunkCount) {
            throw AndroidFullWorkerStageException(
                "chunk_layout_mismatch",
                "chunk_count changed for an in-flight assembly",
            )
        }
        if (current?.received?.containsKey(chunkIndex) == true) {
            throw AndroidFullWorkerStageException(
                "duplicate_chunk",
                "chunk $chunkIndex was already received",
            )
        }
        if ((current?.totalBytes ?: 0).toLong() + payload.size > maxTotalBytes.toLong()) {
            throw AndroidFullWorkerStageException(
                "stage_payload_too_large",
                "assembled payload must not exceed $maxTotalBytes bytes",
            )
        }
        if ((current == null && attempts.size >= maxAttempts) ||
            payload.size.toLong() > maxBufferedBytes - bufferedBytes
        ) {
            throw AndroidFullWorkerStageException(
                "chunk_budget_exceeded",
                "in-flight stage chunks exceed the global budget",
            )
        }
        val stored = payload.copyOf()
        val attempt = current ?: Attempt(chunkCount, now).also { attempts[key] = it }
        attempt.received[chunkIndex] = stored
        attempt.totalBytes += stored.size
        attempt.lastChunkAtNanos = now
        bufferedBytes += stored.size
        return attempt.received.size
    }

    @Synchronized
    fun receivedCount(owner: Any, attemptId: String): Int {
        expire(nanoTime())
        return attempts[Key(owner, attemptId)]?.received?.size ?: 0
    }

    @Synchronized
    fun isComplete(owner: Any, attemptId: String): Boolean {
        expire(nanoTime())
        val attempt = attempts[Key(owner, attemptId)] ?: return false
        return attempt.received.size == attempt.chunkCount
    }

    @Synchronized
    fun missingIndices(owner: Any, attemptId: String): List<Int> {
        expire(nanoTime())
        val attempt = attempts[Key(owner, attemptId)] ?: return emptyList()
        return (0 until attempt.chunkCount).filterNot { attempt.received.containsKey(it) }
    }

    @Synchronized
    fun assemble(owner: Any, attemptId: String): ByteArray {
        expire(nanoTime())
        val attempt = attempts[Key(owner, attemptId)]
            ?: throw AndroidFullWorkerStageException(
                "incomplete_chunks",
                "no chunks were received for this attempt",
            )
        val missing = (0 until attempt.chunkCount)
            .filterNot { attempt.received.containsKey(it) }
        if (missing.isNotEmpty()) {
            throw AndroidFullWorkerStageException(
                "incomplete_chunks",
                "missing chunk indices: $missing",
            )
        }
        val out = ByteArray(attempt.totalBytes)
        var offset = 0
        for (index in 0 until attempt.chunkCount) {
            val chunk = attempt.received.getValue(index)
            System.arraycopy(chunk, 0, out, offset, chunk.size)
            offset += chunk.size
        }
        return out
    }

    @Synchronized
    fun discard(owner: Any, attemptId: String) {
        expire(nanoTime())
        val removed = attempts.remove(Key(owner, attemptId)) ?: return
        bufferedBytes -= removed.totalBytes.toLong()
    }

    @Synchronized
    fun pending(owner: Any): Map<String, Int> {
        expire(nanoTime())
        return attempts
            .filter { (key, attempt) ->
                key.owner === owner && attempt.received.size != attempt.chunkCount
            }
            .mapKeys { it.key.attemptId }
            .mapValues { it.value.received.size }
    }
}

/**
 * ★ 2026-10-07（DIST-NEXT-2b）：大 payload 的**有序分片装配**。
 *
 * 与主仓 `task_worker_chunks.StageChunkAssembler` 保持既有语义和错误码：按序号装配、
 * 重复/漂移/摘要不符/超限一律拒（fail-closed），集齐才产出，`discard` 复位。
 * 分片上限与主仓同值（`TaskWorkerProtocol.MAX_STAGE_CHUNKS` 等），因此分片不能用来
 * 绕过帧预算。跨实例共享 128 MiB / 128 attempt 的缓存上限；空闲五分钟的装配
 * 在下一次访问时清理，未过期的装配不会被挤掉。
 */
class StageChunkAssembler private constructor(
    private val maxChunks: Int,
    private val maxChunkBytes: Int,
    private val maxTotalBytes: Int,
    private val budget: StageChunkBudget,
) {
    constructor(
        maxChunks: Int = TaskWorkerProtocol.MAX_STAGE_CHUNKS,
        maxChunkBytes: Int = TaskWorkerProtocol.STAGE_CHUNK_BYTES,
        maxTotalBytes: Int = TaskWorkerProtocol.MAX_STAGE_PAYLOAD_BYTES,
    ) : this(maxChunks, maxChunkBytes, maxTotalBytes, globalBudget)

    internal constructor(
        budget: StageChunkBudget,
        maxChunks: Int = TaskWorkerProtocol.MAX_STAGE_CHUNKS,
        maxChunkBytes: Int = TaskWorkerProtocol.STAGE_CHUNK_BYTES,
        maxTotalBytes: Int = TaskWorkerProtocol.MAX_STAGE_PAYLOAD_BYTES,
    ) : this(maxChunks, maxChunkBytes, maxTotalBytes, budget)

    private val owner = Any()

    /** 接收一条分片；返回该 attempt 已收条数。不合法即抛错，不留下半条状态。 */
    fun add(
        attemptId: String,
        chunkIndex: Int,
        chunkCount: Int,
        payload: ByteArray,
        payloadSha256: String,
    ): Int {
        if (attemptId.isBlank()) {
            throw AndroidFullWorkerStageException("invalid_attempt", "attempt id is required")
        }
        if (attemptId.length > 100) {
            throw AndroidFullWorkerStageException(
                "invalid_attempt",
                "attempt id must not exceed 100 characters",
            )
        }
        if (chunkCount < 1 || chunkCount > maxChunks) {
            throw AndroidFullWorkerStageException(
                "invalid_chunk_count",
                "chunk_count must be in 1..$maxChunks",
            )
        }
        if (chunkIndex < 0 || chunkIndex >= chunkCount) {
            throw AndroidFullWorkerStageException(
                "invalid_chunk_index",
                "chunk_index must be less than chunk_count",
            )
        }
        if (payload.size > maxChunkBytes) {
            throw AndroidFullWorkerStageException(
                "chunk_too_large",
                "a single chunk must not exceed $maxChunkBytes bytes",
            )
        }
        if (!sha256Hex(payload).equals(payloadSha256.lowercase(), ignoreCase = true)) {
            throw AndroidFullWorkerStageException(
                "chunk_digest_mismatch",
                "chunk digest does not match its payload",
            )
        }
        return budget.add(owner, attemptId, chunkIndex, chunkCount, payload, maxTotalBytes)
    }

    fun receivedCount(attemptId: String): Int = budget.receivedCount(owner, attemptId)

    fun isComplete(attemptId: String): Boolean = budget.isComplete(owner, attemptId)

    fun missingIndices(attemptId: String): List<Int> = budget.missingIndices(owner, attemptId)

    /** 集齐则返回拼装后的字节；未集齐抛 `incomplete_chunks`（带缺失序号）。 */
    fun assemble(attemptId: String): ByteArray = budget.assemble(owner, attemptId)

    fun discard(attemptId: String) {
        budget.discard(owner, attemptId)
    }

    /** 诊断：`{attempt_id: 已收条数}`（只含未集齐的装配）。 */
    fun pending(): Map<String, Int> = budget.pending(owner)

    companion object {
        private val globalBudget = StageChunkBudget(
            maxBufferedBytes = 2L * TaskWorkerProtocol.MAX_STAGE_PAYLOAD_BYTES,
            maxAttempts = 128,
            ttlNanos = TimeUnit.MINUTES.toNanos(5),
        )

        fun sha256Hex(data: ByteArray): String =
            MessageDigest.getInstance("SHA-256").digest(data)
                .joinToString("") { "%02x".format(it) }
    }
}
