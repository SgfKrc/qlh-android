package com.qlh.inference.worker

import com.google.gson.JsonObject
import com.google.gson.JsonParser
import java.util.Base64
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertNotNull
import org.junit.Test

/**
 * ★ 票 21a（COMM-WIRE-01）：android 侧二进制 hidden 的抽取/放回。
 *
 * 与主仓 `tests/test_task_worker_binary_hidden.py` 同构；两侧路径表必须一致，
 * 否则会出现「一边抽体、一边按 base64 解码」的隐性不对称。
 */
class TaskWorkerBinaryHiddenTest {

    private fun parse(text: String): JsonObject = JsonParser.parseString(text).asJsonObject

    private val hidden = ByteArray(32) { it.toByte() }
    private val encoded: String = Base64.getEncoder().encodeToString(hidden)

    private fun offer(): JsonObject = parse(
        """
        {"message_type":"stage_offer",
         "payload":{"root_input":{"hidden_f32":"$encoded","other":"keep"},
                    "input_sha256":"abc"}}
        """.trimIndent(),
    )

    @Test
    fun `disabled leaves the message untouched`() {
        val message = offer()
        val (written, body) = TaskWorkerBinaryHidden.split(message, enabled = false)

        assertNull(body)
        assertEquals(message, written)
    }

    @Test
    fun `split extracts the raw bytes and drops the base64 field`() {
        val (written, body) = TaskWorkerBinaryHidden.split(offer(), enabled = true)

        assertArrayEquals(hidden, body)
        val root = written.getAsJsonObject("payload").getAsJsonObject("root_input")
        assertEquals(false, root.has("hidden_f32"))
        assertEquals("keep", root.get("other").asString)
        assertEquals("abc", written.getAsJsonObject("payload").get("input_sha256").asString)
    }

    @Test
    fun `restore puts the bytes back so downstream sees the old shape`() {
        val original = offer()
        val (written, body) = TaskWorkerBinaryHidden.split(original, enabled = true)
        val payload = written.getAsJsonObject("payload")

        TaskWorkerBinaryHidden.restore(body, "stage_offer", payload)

        assertEquals(true, payload.getAsJsonObject("root_input").has("hidden_f32"))
        assertEquals(original, written)
    }

    @Test
    fun `restore with null body is a no-op`() {
        val (written, _) = TaskWorkerBinaryHidden.split(offer(), enabled = true)
        val before = written.deepCopy()

        TaskWorkerBinaryHidden.restore(null, "stage_offer", written.getAsJsonObject("payload"))

        assertEquals(before, written)
    }

    @Test
    fun `stage_result uses the output container`() {
        val message = parse(
            """
            {"message_type":"stage_result",
             "payload":{"output":{"hidden_out_f32":"$encoded"},"attempt_id":"a1"}}
            """.trimIndent(),
        )
        val (written, body) = TaskWorkerBinaryHidden.split(message, enabled = true)

        assertArrayEquals(hidden, body)
        assertEquals(false, written.getAsJsonObject("payload").getAsJsonObject("output").has("hidden_out_f32"))
    }

    @Test
    fun `chunk payloads sit at the top of payload for both directions`() {
        val chunk = ByteArray(64) { (it * 3).toByte() }
        val chunkB64 = Base64.getEncoder().encodeToString(chunk)

        for (type in listOf(TaskWorkerProtocol.STAGE_CHUNK, TaskWorkerBinaryHidden.STAGE_RESULT_CHUNK_TYPE)) {
            val message = parse(
                """{"message_type":"$type","payload":{"payload_b64":"$chunkB64","chunk_index":0}}""",
            )
            val (written, body) = TaskWorkerBinaryHidden.split(message, enabled = true)
            assertArrayEquals("type=$type", chunk, body)
            val payload = written.getAsJsonObject("payload")
            assertEquals("type=$type", false, payload.has("payload_b64"))
            assertEquals("type=$type", 0, payload.get("chunk_index").asInt)

            TaskWorkerBinaryHidden.restore(body, type, payload)
            assertEquals("type=$type", message, written)
        }
    }

    @Test
    fun `unusable payloads are left alone`() {
        // 已分片：root_input 只有 hidden_ref（它的载荷由 stage_chunk 自己承载）
        val chunked = parse(
            """
            {"message_type":"stage_offer",
             "payload":{"root_input":{"hidden_ref":{"total_bytes":4,"payload_sha256":"x"}}}}
            """.trimIndent(),
        )
        assertEquals(chunked, TaskWorkerBinaryHidden.split(chunked, enabled = true).first)

        // 非法 base64：交给既有协议校验报错，工具不改写
        val bad = parse(
            """{"message_type":"stage_offer","payload":{"root_input":{"hidden_f32":"not base64!!"}}}""",
        )
        assertEquals(bad, TaskWorkerBinaryHidden.split(bad, enabled = true).first)

        // 不承载内联 hidden 的消息类型
        val hello = parse(
            """{"message_type":"hello","payload":{"root_input":{"hidden_f32":"AAAA"}}}""",
        )
        assertEquals(hello, TaskWorkerBinaryHidden.split(hello, enabled = true).first)
    }

    @Test
    fun `split preserves the original envelope and nested payload`() {
        val original = offer()
        val before = original.deepCopy()
        val (split, raw) = TaskWorkerBinaryHidden.split(original, enabled = true)
        assertArrayEquals(hidden, raw)
        assertEquals(before, original)
        assertEquals(false, split == original)
    }

    @Test
    fun `noncanonical base64 and nonstring fields remain unchanged`() {
        for (value in listOf("null", "1234", "true", "[]", "{}", "\"AQ\"", "\"AB==\"", "\"\"")) {
            val message = parse("""{"message_type":"stage_offer","payload":{"root_input":{"hidden_f32":$value}}}""")
            val before = message.deepCopy()
            assertNull(TaskWorkerBinaryHidden.split(message, enabled = true).second)
            assertEquals(before, message)
        }
        for (payload in listOf("null", "[]", "123", "true", "{}", "{\"root_input\":[]}")) {
            val message = parse("""{"message_type":"stage_offer","payload":$payload}""")
            assertNull(TaskWorkerBinaryHidden.split(message, enabled = true).second)
        }
    }

    @Test
    fun `path table matches the python side`() {
        // 与 src/task_worker_protocol.BINARY_HIDDEN_PATHS 逐项对应
        assertEquals(Pair("root_input", "hidden_f32"), TaskWorkerBinaryHidden.PATHS["stage_offer"])
        assertEquals(Pair("output", "hidden_out_f32"), TaskWorkerBinaryHidden.PATHS["stage_result"])
        assertEquals(Pair<String?, String>(null, "payload_b64"), TaskWorkerBinaryHidden.PATHS["stage_chunk"])
        assertEquals(
            Pair<String?, String>(null, "payload_b64"),
            TaskWorkerBinaryHidden.PATHS[TaskWorkerBinaryHidden.STAGE_RESULT_CHUNK_TYPE],
        )
        assertEquals(4, TaskWorkerBinaryHidden.HEADER_BYTES)
        assertNotNull(TaskWorkerBinaryHidden.PATHS)
    }
}
