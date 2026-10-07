package com.qlh.inference.worker

import java.security.MessageDigest

/**
 * ★ 2026-10-07（DIST-NEXT-2b）：大 payload 的**有序分片装配**。
 *
 * 与主仓 `task_worker_chunks.StageChunkAssembler` **同语义、同错误码**：按序号装配、
 * 重复/漂移/摘要不符/超限一律拒（fail-closed），集齐才产出，`discard` 复位。
 * 分片上限与主仓同值（`TaskWorkerProtocol.MAX_STAGE_CHUNKS` 等），因此分片不能用来
 * 绕过帧预算。
 */
class StageChunkAssembler(
    private val maxChunks: Int = TaskWorkerProtocol.MAX_STAGE_CHUNKS,
    private val maxChunkBytes: Int = TaskWorkerProtocol.STAGE_CHUNK_BYTES,
    private val maxTotalBytes: Int = TaskWorkerProtocol.MAX_STAGE_PAYLOAD_BYTES,
) {
    private class Attempt(val chunkCount: Int) {
        val received = HashMap<Int, ByteArray>()
        var totalBytes: Int = 0
    }

    private val state = HashMap<String, Attempt>()

    /** 接收一条分片；返回该 attempt 已收条数。不合法即抛错，不留下半条状态。 */
    @Synchronized
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
        val attempt = state[attemptId]
        val current = attempt ?: Attempt(chunkCount).also { state[attemptId] = it }
        if (current.chunkCount != chunkCount) {
            throw AndroidFullWorkerStageException(
                "chunk_layout_mismatch",
                "chunk_count changed for an in-flight assembly",
            )
        }
        if (current.received.containsKey(chunkIndex)) {
            throw AndroidFullWorkerStageException(
                "duplicate_chunk",
                "chunk $chunkIndex was already received",
            )
        }
        if (current.totalBytes + payload.size > maxTotalBytes) {
            throw AndroidFullWorkerStageException(
                "stage_payload_too_large",
                "assembled payload must not exceed $maxTotalBytes bytes",
            )
        }
        current.received[chunkIndex] = payload
        current.totalBytes += payload.size
        return current.received.size
    }

    @Synchronized
    fun receivedCount(attemptId: String): Int = state[attemptId]?.received?.size ?: 0

    @Synchronized
    fun isComplete(attemptId: String): Boolean {
        val attempt = state[attemptId] ?: return false
        return attempt.received.size == attempt.chunkCount
    }

    @Synchronized
    fun missingIndices(attemptId: String): List<Int> {
        val attempt = state[attemptId] ?: return emptyList()
        return (0 until attempt.chunkCount).filterNot { attempt.received.containsKey(it) }
    }

    /** 集齐则返回拼装后的字节；未集齐抛 `incomplete_chunks`（带缺失序号）。 */
    @Synchronized
    fun assemble(attemptId: String): ByteArray {
        val attempt = state[attemptId]
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
        val total = attempt.received.values.sumOf { it.size }
        val out = ByteArray(total)
        var offset = 0
        for (index in 0 until attempt.chunkCount) {
            val chunk = attempt.received.getValue(index)
            System.arraycopy(chunk, 0, out, offset, chunk.size)
            offset += chunk.size
        }
        return out
    }

    @Synchronized
    fun discard(attemptId: String) {
        state.remove(attemptId)
    }

    /** 诊断：`{attempt_id: 已收条数}`（只含未集齐的装配）。 */
    @Synchronized
    fun pending(): Map<String, Int> = state
        .filterValues { it.received.size != it.chunkCount }
        .mapValues { it.value.received.size }

    companion object {
        fun sha256Hex(data: ByteArray): String =
            MessageDigest.getInstance("SHA-256").digest(data)
                .joinToString("") { "%02x".format(it) }
    }
}
