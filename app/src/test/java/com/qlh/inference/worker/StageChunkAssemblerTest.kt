package com.qlh.inference.worker

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Test
import java.util.concurrent.Callable
import java.util.concurrent.CountDownLatch
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit

/**
 * ★ 2026-10-07（DIST-NEXT-2b）：大 payload 分片装配（与主仓
 * `task_worker_chunks.StageChunkAssembler` 同语义的 Kotlin 版）。
 *
 * 判据：按序号装配、重复/漂移/摘要不符/超限一律拒、集齐才产出。
 */
class StageChunkAssemblerTest {
    private fun digest(chunk: ByteArray): String = StageChunkAssembler.sha256Hex(chunk)

    private fun rejects(code: String, action: () -> Unit) {
        try {
            action()
            fail("expected $code")
        } catch (error: AndroidFullWorkerStageException) {
            assertEquals(code, error.code)
        }
    }

    @Test
    fun `chunks are ordered by index and must be complete`() {
        val payload = ByteArray(2 * 1024 + 7) { (it % 251).toByte() }
        val chunks = listOf(
            payload.copyOfRange(0, 1024),
            payload.copyOfRange(1024, 2048),
            payload.copyOfRange(2048, payload.size),
        )
        val assembler = StageChunkAssembler()

        // 乱序到达：装配按序号而非到达顺序
        assembler.add("att_1", 2, 3, chunks[2], digest(chunks[2]))
        assembler.add("att_1", 0, 3, chunks[0], digest(chunks[0]))
        assertEquals(2, assembler.receivedCount("att_1"))
        assertFalse(assembler.isComplete("att_1"))
        assertEquals(listOf(1), assembler.missingIndices("att_1"))
        assertEquals(mapOf("att_1" to 2), assembler.pending())

        try {
            assembler.assemble("att_1")
            assertTrue("未集齐必须失败", false)
        } catch (error: AndroidFullWorkerStageException) {
            assertEquals("incomplete_chunks", error.code)
            assertTrue(error.message!!.contains("[1]"))
        }

        assembler.add("att_1", 1, 3, chunks[1], digest(chunks[1]))
        assertTrue(assembler.isComplete("att_1"))
        assertArrayEquals(payload, assembler.assemble("att_1"))
        assertEquals(emptyMap<String, Int>(), assembler.pending())
    }

    @Test
    fun `duplicates layout drift and bad digests are rejected`() {
        val chunk = ByteArray(40) { 1 }
        val other = ByteArray(40) { 2 }

        val duplicate = StageChunkAssembler()
        duplicate.add("att_2", 0, 3, chunk, digest(chunk))
        try {
            duplicate.add("att_2", 0, 3, chunk, digest(chunk))
            assertTrue("重复分片必须失败", false)
        } catch (error: AndroidFullWorkerStageException) {
            assertEquals("duplicate_chunk", error.code)
        }

        val badDigest = StageChunkAssembler()
        try {
            badDigest.add("att_3", 0, 1, chunk, "f".repeat(64))
            assertTrue("摘要不符必须失败", false)
        } catch (error: AndroidFullWorkerStageException) {
            assertEquals("chunk_digest_mismatch", error.code)
        }
        // 被拒的分片不得留下半条状态
        assertEquals(0, badDigest.receivedCount("att_3"))

        val drift = StageChunkAssembler()
        drift.add("att_4", 0, 3, chunk, digest(chunk))
        try {
            drift.add("att_4", 1, 5, other, digest(other))
            assertTrue("chunk_count 漂移必须失败", false)
        } catch (error: AndroidFullWorkerStageException) {
            assertEquals("chunk_layout_mismatch", error.code)
        }
    }

    @Test
    fun `hard limits are enforced before anything is stored`() {
        val assembler = StageChunkAssembler(
            maxChunks = 4, maxChunkBytes = 16, maxTotalBytes = 32,
        )
        val chunk = ByteArray(16) { 7 }

        try {
            assembler.add("att_5", 0, 9, chunk, digest(chunk))
            assertTrue("片数超限必须失败", false)
        } catch (error: AndroidFullWorkerStageException) {
            assertEquals("invalid_chunk_count", error.code)
        }
        try {
            assembler.add("att_5", 0, 4, ByteArray(17), digest(ByteArray(17)))
            assertTrue("单片超限必须失败", false)
        } catch (error: AndroidFullWorkerStageException) {
            assertEquals("chunk_too_large", error.code)
        }
        assembler.add("att_5", 0, 4, chunk, digest(chunk))
        assembler.add("att_5", 1, 4, chunk, digest(chunk))
        try {
            assembler.add("att_5", 2, 4, chunk, digest(chunk))
            assertTrue("装配总量超限必须失败", false)
        } catch (error: AndroidFullWorkerStageException) {
            assertEquals("stage_payload_too_large", error.code)
        }
        assertEquals(2, assembler.receivedCount("att_5"))
    }

    @Test
    fun `discard resets the accumulation`() {
        val assembler = StageChunkAssembler()
        val chunk = ByteArray(30) { 3 }
        assembler.add("att_6", 0, 2, chunk, digest(chunk))

        assembler.discard("att_6")

        assertEquals(0, assembler.receivedCount("att_6"))
        assertEquals(emptyMap<String, Int>(), assembler.pending())
        try {
            assembler.assemble("att_6")
            assertTrue("discard 后装配必须失败", false)
        } catch (error: AndroidFullWorkerStageException) {
            assertEquals("incomplete_chunks", error.code)
        }
    }

    @Test
    fun `global byte budget is shared without mixing same named attempts`() {
        val budget = StageChunkBudget(8, 4, TimeUnit.MINUTES.toNanos(1))
        val first = StageChunkAssembler(budget)
        val second = StageChunkAssembler(budget)
        val six = ByteArray(6) { 1 }
        val two = ByteArray(2) { 2 }
        assertEquals(1, first.add("same", 0, 2, six, digest(six)))
        assertEquals(0, second.receivedCount("same"))

        rejects("chunk_budget_exceeded") {
            second.add("same", 0, 1, ByteArray(3), digest(ByteArray(3)))
        }
        assertEquals(0, second.receivedCount("same"))
        assertEquals(1, second.add("same", 0, 1, two, digest(two)))
        assertArrayEquals(two, second.assemble("same"))
        assertEquals(mapOf("same" to 1), first.pending())
        assertEquals(emptyMap<String, Int>(), second.pending())

        first.discard("same")
        val replacement = ByteArray(6) { 3 }
        assertEquals(1, first.add("another", 0, 1, replacement, digest(replacement)))
        assertArrayEquals(replacement, first.assemble("another"))
        second.discard("same")
        first.discard("another")
    }

    @Test
    fun `chunk bytes cannot be changed after admission`() {
        val budget = StageChunkBudget(4, 2, TimeUnit.MINUTES.toNanos(1))
        val assembler = StageChunkAssembler(budget)
        val input = byteArrayOf(1, 2, 3, 4)
        assembler.add("mutated", 0, 1, input, digest(input))
        input.fill(0)
        assertArrayEquals(byteArrayOf(1, 2, 3, 4), assembler.assemble("mutated"))
        assembler.discard("mutated")
    }

    @Test
    fun `idle ttl expires across instances but successful chunks renew it`() {
        var now = 0L
        val budget = StageChunkBudget(8, 3, ttlNanos = 10) { now }
        val first = StageChunkAssembler(budget)
        val second = StageChunkAssembler(budget)
        val four = ByteArray(4) { 1 }
        val two = ByteArray(2) { 2 }
        first.add("old", 0, 3, four, digest(four))
        now = 9
        assertEquals(1, first.receivedCount("old"))
        first.add("old", 1, 3, two, digest(two))
        now = 18
        assertEquals(listOf(2), first.missingIndices("old"))
        second.add("live", 0, 1, two, digest(two))
        now = 19
        val six = ByteArray(6) { 3 }
        second.add("new", 0, 1, six, digest(six))
        assertEquals(0, first.receivedCount("old"))
        assertEquals(emptyMap<String, Int>(), first.pending())
        rejects("incomplete_chunks") { first.assemble("old") }
        assertArrayEquals(two, second.assemble("live"))
        assertArrayEquals(six, second.assemble("new"))
    }

    @Test
    fun `reads and rejected chunks do not renew idle ttl`() {
        var now = 0L
        val budget = StageChunkBudget(4, 1, ttlNanos = 10) { now }
        val assembler = StageChunkAssembler(budget)
        val payload = ByteArray(4) { 1 }
        assembler.add("old", 0, 1, payload, digest(payload))
        now = 9
        assertTrue(assembler.isComplete("old"))
        rejects("duplicate_chunk") { assembler.add("old", 0, 1, payload, digest(payload)) }
        now = 10
        assertEquals(0, assembler.receivedCount("old"))
        assertFalse(assembler.isComplete("old"))
        rejects("incomplete_chunks") { assembler.assemble("old") }
        assertEquals(1, assembler.add("again", 0, 1, payload, digest(payload)))
    }

    @Test
    fun `zero byte attempts are bounded and freed on discard or ttl`() {
        var now = 0L
        val budget = StageChunkBudget(4, 2, ttlNanos = 10) { now }
        val first = StageChunkAssembler(budget)
        val second = StageChunkAssembler(budget)
        val empty = byteArrayOf()
        first.add("one", 0, 1, empty, digest(empty))
        second.add("two", 0, 1, empty, digest(empty))
        rejects("chunk_budget_exceeded") { first.add("three", 0, 1, empty, digest(empty)) }
        assertEquals(0, first.receivedCount("three"))
        assertTrue(first.isComplete("one"))
        assertArrayEquals(empty, first.assemble("one"))

        first.discard("one")
        first.add("three", 0, 1, empty, digest(empty))
        rejects("invalid_attempt") {
            first.add("x".repeat(101), 0, 1, empty, digest(empty))
        }
        now = 10
        assertFalse(second.isComplete("two"))
        assertEquals(0, first.receivedCount("three"))
        second.add("fresh", 0, 1, empty, digest(empty))
    }

    @Test
    fun `maximum hidden payload still assembles at its protocol limit`() {
        val limit = TaskWorkerProtocol.MAX_STAGE_PAYLOAD_BYTES
        val budget = StageChunkBudget(limit.toLong(), 2, TimeUnit.MINUTES.toNanos(5))
        val assembler = StageChunkAssembler(budget)
        val chunk = ByteArray(TaskWorkerProtocol.STAGE_CHUNK_BYTES) { (it % 251).toByte() }
        val sha = digest(chunk)
        for (index in 0 until TaskWorkerProtocol.MAX_STAGE_CHUNKS) {
            assembler.add("hidden", index, TaskWorkerProtocol.MAX_STAGE_CHUNKS, chunk, sha)
        }
        assertTrue(assembler.isComplete("hidden"))
        val raw = assembler.assemble("hidden")
        assertEquals(limit, raw.size)
        assertArrayEquals(chunk, raw.copyOfRange(0, chunk.size))
        assertArrayEquals(chunk, raw.copyOfRange(limit - chunk.size, limit))

        val next = StageChunkAssembler(budget)
        rejects("chunk_budget_exceeded") { next.add("next", 0, 1, chunk, sha) }
        assembler.discard("hidden")
        assertEquals(1, next.add("next", 0, 1, chunk, sha))
        next.discard("next")
    }

    @Test
    fun `simultaneous instances cannot oversubscribe the global budget`() {
        val budget = StageChunkBudget(8, 2, TimeUnit.MINUTES.toNanos(1))
        val first = StageChunkAssembler(budget)
        val second = StageChunkAssembler(budget)
        val chunk = ByteArray(6) { 4 }
        val sha = digest(chunk)
        val started = CountDownLatch(1)
        val threads = Executors.newFixedThreadPool(2)
        try {
            val jobs = listOf(first, second).mapIndexed { index, assembler ->
                threads.submit(Callable {
                    started.await()
                    try {
                        assembler.add("attempt_$index", 0, 1, chunk, sha)
                        true
                    } catch (error: AndroidFullWorkerStageException) {
                        assertEquals("chunk_budget_exceeded", error.code)
                        false
                    }
                })
            }
            started.countDown()
            assertEquals(1, jobs.count { it.get(5, TimeUnit.SECONDS) })
            assertEquals(1, first.receivedCount("attempt_0") + second.receivedCount("attempt_1"))
        } finally {
            threads.shutdownNow()
        }
    }

    @Test
    fun `chunk limits match the master side`() {
        // 与主仓 `task_worker_protocol` 同值 —— 分片不能用来绕过帧预算
        assertEquals(64, TaskWorkerProtocol.MAX_STAGE_CHUNKS)
        assertEquals(1 shl 20, TaskWorkerProtocol.STAGE_CHUNK_BYTES)
        assertEquals(64 * (1 shl 20), TaskWorkerProtocol.MAX_STAGE_PAYLOAD_BYTES)
    }
}
