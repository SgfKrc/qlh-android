package com.qlh.inference.worker

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * ★ 2026-10-07（DIST-NEXT-2b）：大 payload 分片装配（与主仓
 * `task_worker_chunks.StageChunkAssembler` 同语义的 Kotlin 版）。
 *
 * 判据：按序号装配、重复/漂移/摘要不符/超限一律拒、集齐才产出。
 */
class StageChunkAssemblerTest {
    private fun digest(chunk: ByteArray): String = StageChunkAssembler.sha256Hex(chunk)

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
    fun `chunk limits match the master side`() {
        // 与主仓 `task_worker_protocol` 同值 —— 分片不能用来绕过帧预算
        assertEquals(64, TaskWorkerProtocol.MAX_STAGE_CHUNKS)
        assertEquals(1 shl 20, TaskWorkerProtocol.STAGE_CHUNK_BYTES)
        assertEquals(64 * (1 shl 20), TaskWorkerProtocol.MAX_STAGE_PAYLOAD_BYTES)
    }
}
