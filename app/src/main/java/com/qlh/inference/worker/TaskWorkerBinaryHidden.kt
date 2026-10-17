package com.qlh.inference.worker

import com.google.gson.JsonObject
import java.util.Base64

/**
 * ★ 票 21a（COMM-WIRE-01）：层段 hidden 的**二进制承载**。
 *
 * 与主仓 `src/task_worker_protocol.py` 的 `split_inline_hidden` /
 * `restore_inline_hidden` **同构**（同一张路径表、同一语义），两侧必须保持一致。
 *
 * 帧布局（见 `TaskWorkerClient.sendOuter` / `receiveOuter`）：
 * ```
 * JSON 帧   = [4B 长度][meta JSON]
 * 二进制帧 = [4B 长度][meta JSON] + [4B 长度][raw 字节]
 * ```
 *
 * **为什么不需要新协议字段**：raw 的位置由「帧 `format:"binary"` + 消息类型」共同推导
 * （见 [PATHS]），逐位完整性仍由既有的摘要字段承担（`hidden_sha256` /
 * `hidden_out_sha256` / 分片自己的 `payload_sha256`）。放回的是 base64 字符串 ⇒
 * 其后所有既有代码（含 `AndroidFullWorkerStageExecutor`）都不必知道二进制路径存在。
 *
 * 只有对端（主节点）通过能力位 `binary_hidden` 协商成功后才启用；未启用时行为与
 * 旧路径**逐字节一致**。
 */
object TaskWorkerBinaryHidden {

    /** 二进制体的长度头字节数（4 字节长度前缀）。 */
    const val HEADER_BYTES = 4

    /** Compatibility alias for the protocol's return-chunk message type. */
    const val STAGE_RESULT_CHUNK_TYPE = TaskWorkerProtocol.STAGE_RESULT_CHUNK

    /**
     * 内联 hidden 的落点表：`消息类型 -> (容器键, 字段名)`。
     *
     * 容器键为 `null` 表示载荷就在 `payload` 顶层（分片路径，`payload_b64` 自身即
     * 那一段 raw 的 base64）。入程 `stage_chunk` 与回程 `stage_result_chunk` 共用
     * 同一形状，所以一条定义覆盖两个方向。
     */
    val PATHS: Map<String, Pair<String?, String>> = mapOf(
        TaskWorkerProtocol.STAGE_OFFER to Pair("root_input", "hidden_f32"),
        TaskWorkerProtocol.STAGE_RESULT to Pair("output", "hidden_out_f32"),
        TaskWorkerProtocol.STAGE_CHUNK to Pair(null, "payload_b64"),
        STAGE_RESULT_CHUNK_TYPE to Pair(null, "payload_b64"),
    )

    /**
     * 把内联 base64 载荷抽成二进制体。返回 `(改写后的 inner, raw 或 null)`。
     *
     * `enabled == false`、消息类型不承载内联载荷、或字段不是合法 base64 ⇒
     * 原对象原样返回且 `raw == null`（零行为变化）。
     */
    fun split(inner: JsonObject, enabled: Boolean): Pair<JsonObject, ByteArray?> {
        if (!enabled) return Pair(inner, null)
        val messageType = inner.get("message_type")?.takeIf { it.isJsonPrimitive && it.asJsonPrimitive.isString }?.asString
        val path = messageType?.let { PATHS[it] } ?: return Pair(inner, null)
        val payload = inner.get("payload")?.takeIf { it.isJsonObject }?.asJsonObject
            ?: return Pair(inner, null)
        val (containerKey, fieldKey) = path

        val container = if (containerKey == null) payload else
            payload.get(containerKey)?.takeIf { it.isJsonObject }?.asJsonObject
                ?: return Pair(inner, null)
        val raw = decodeField(container, fieldKey) ?: return Pair(inner, null)
        val snapshot = inner.deepCopy()
        val newPayload = snapshot.getAsJsonObject("payload")
        val newContainer = if (containerKey == null) newPayload else newPayload.getAsJsonObject(containerKey)
        newContainer.remove(fieldKey)
        return Pair(snapshot, raw)
    }

    /**
     * `split` 的逆：把 raw 以 base64 放回原位。`raw == null` 时原样返回。
     */
    fun restore(raw: ByteArray?, messageType: String?, payload: JsonObject) {
        if (raw == null) return
        val container = restoreTarget(messageType, payload)
        val fieldKey = requireNotNull(PATHS[messageType]).second
        if (raw.isEmpty()) invalidBody("binary hidden body must not be empty")
        container.addProperty(fieldKey, Base64.getEncoder().encodeToString(raw))
    }

    /** Validate the destination before allocating or reading the binary body. */
    internal fun restoreTarget(messageType: String?, payload: JsonObject): JsonObject {
        val (containerKey, fieldKey) = messageType?.let { PATHS[it] }
            ?: invalidBody("message type does not carry binary hidden")
        val container = if (containerKey == null) payload else
            payload.get(containerKey)?.takeIf { it.isJsonObject }?.asJsonObject
                ?: invalidBody("binary hidden container must be an object")
        if (container.has(fieldKey) ||
            (containerKey == "root_input" && container.has("hidden_ref")) ||
            (containerKey == "output" && container.has("hidden_out_ref"))) {
            invalidBody("binary hidden must have exactly one payload source")
        }
        return container
    }

    private fun invalidBody(message: String): Nothing = throw TaskWorkerProtocolException(
        message, "invalid_frame", "data.payload",
    )

    /** 该字段能否解出非空 raw；不能则返回 null（交给既有协议校验报错）。 */
    private fun decodeField(container: JsonObject, fieldKey: String): ByteArray? {
        val encoded = container.get(fieldKey)?.takeIf { it.isJsonPrimitive && it.asJsonPrimitive.isString }?.asString
            ?: return null
        if (encoded.length % 4 != 0) return null
        return try {
            Base64.getDecoder().decode(encoded).takeIf {
                it.isNotEmpty() && Base64.getEncoder().encodeToString(it) == encoded
            }
        } catch (_: IllegalArgumentException) {
            null
        }
    }
}
