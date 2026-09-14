package com.qlh.inference.service

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Test

class ModelDownloadContractTest {
    @Test
    fun `range response appends only at exact manifest offset`() {
        val plan = planModelDownloadWrite(
            existingBytes = 4L,
            statusCode = 206,
            contentLength = 6L,
            contentRange = "bytes 4-9/10",
            expectedTotalBytes = 10L,
        )
        assertEquals(4L, plan.startBytes)
        assertEquals(10L, plan.totalBytes)
        assertTrue(plan.append)
    }

    @Test
    fun `full response restarts an existing partial file`() {
        val plan = planModelDownloadWrite(
            existingBytes = 4L,
            statusCode = 200,
            contentLength = 10L,
            contentRange = null,
            expectedTotalBytes = 10L,
        )
        assertEquals(0L, plan.startBytes)
        assertFalse(plan.append)
    }

    @Test
    fun `mismatched range and model size fail closed`() {
        assertThrows(IllegalArgumentException::class.java) {
            planModelDownloadWrite(4L, 206, 5L, "bytes 5-9/10", 10L)
        }
        assertThrows(IllegalArgumentException::class.java) {
            planModelDownloadWrite(0L, 200, 9L, null, 10L)
        }
        assertThrows(IllegalArgumentException::class.java) {
            planModelDownloadWrite(4L, 206, 7L, "bytes 4-10/10", 10L)
        }
    }

    // ---- T11：边界补齐（测试修复票排期） ----

    @Test
    fun `T11 non-2xx status and malformed content-range fail closed`() {
        assertThrows(IllegalArgumentException::class.java) {
            planModelDownloadWrite(0L, 301, 10L, null, 10L)
        }
        assertThrows(IllegalArgumentException::class.java) {
            planModelDownloadWrite(0L, 418, 10L, null, 10L)
        }
        // Content-Range 语法非法（非 bytes a-b/c；注意实现 IGNORE_CASE 接受
        // 大小写 bytes/BYTES，合法格式不在此列）
        for (bad in listOf(
            "content-range: bytes 4-9/10",
            "bytes -9/10",
            "bytes a-b/c",
            "bytes 4-9/",
            "bytes 4-9/10 extra",
        )) {
            assertThrows(
                "非法 Content-Range: $bad",
                IllegalArgumentException::class.java,
            ) {
                planModelDownloadWrite(4L, 206, 6L, bad, 10L)
            }
        }
    }

    @Test
    fun `T11 206 start at zero does not append`() {
        val plan = planModelDownloadWrite(
            existingBytes = 0L,
            statusCode = 206,
            contentLength = 10L,
            contentRange = "bytes 0-9/10",
            expectedTotalBytes = 10L,
        )
        assertEquals(0L, plan.startBytes)
        assertFalse("start=0 不应 append", plan.append)
        assertEquals(10L, plan.totalBytes)
    }

    @Test
    fun `T11 206 full single range is valid`() {
        val plan = planModelDownloadWrite(
            existingBytes = 0L,
            statusCode = 206,
            contentLength = 10L,
            contentRange = "bytes 0-9/10",
            expectedTotalBytes = 10L,
        )
        assertEquals(10L, plan.totalBytes)
    }

    @Test
    fun `T11 200 with zero contentLength is accepted (unknown size)`() {
        val plan = planModelDownloadWrite(
            existingBytes = 0L,
            statusCode = 200,
            contentLength = 0L,
            contentRange = null,
            expectedTotalBytes = 10L,
        )
        assertFalse(plan.append)
        assertEquals(10L, plan.totalBytes)
    }

    @Test
    fun `T11 preconditions reject negative and zero bounds`() {
        assertThrows(IllegalArgumentException::class.java) {
            planModelDownloadWrite(-1L, 200, 10L, null, 10L)
        }
        assertThrows(IllegalArgumentException::class.java) {
            planModelDownloadWrite(0L, 200, 10L, null, 0L)
        }
        assertThrows(IllegalArgumentException::class.java) {
            planModelDownloadWrite(0L, 200, 10L, null, -5L)
        }
    }
}
