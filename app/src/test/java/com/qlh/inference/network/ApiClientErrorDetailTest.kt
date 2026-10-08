package com.qlh.inference.network

import org.junit.Assert.assertEquals
import org.junit.Test

/**
 * `errorDetailOf` 的回归 —— 让后端错误体变成**用户能读的原因**。
 *
 * 背景：此前 `ApiClientHttpException` 的 message 直接拼原始 JSON 串，用户看到的是一坨
 * 转义 JSON；而这条路正是"分布式没生效"、"来源不受信"这类原因的**唯一出口**，读不出来
 * 就等于静默。两条实测形状都钉在这里。
 */
class ApiClientErrorDetailTest {

    @Test
    fun `detail 是字符串时直接取出`() {
        // 实测形状：分布式专用模式下禁止整模回退
        assertEquals(
            "推理失败: 当前模型以分布式专用模式准备，禁止自动整模回退；请等待流水线节点就绪或显式执行普通模型加载",
            errorDetailOf(
                """{"detail":"推理失败: 当前模型以分布式专用模式准备，禁止自动整模回退；""" +
                    """请等待流水线节点就绪或显式执行普通模型加载","request_id":"f93d0e55"}"""
            ),
        )
    }

    @Test
    fun `detail 是对象时取其中的 message`() {
        // 实测形状：信任边界拒绝（src/model_api_access.py）
        assertEquals(
            "model control APIs require loopback or an explicitly trusted CIDR",
            errorDetailOf(
                """{"detail":{"code":"MODEL_API_SOURCE_UNTRUSTED",""" +
                    """"message":"model control APIs require loopback or an explicitly trusted CIDR"}}"""
            ),
        )
    }

    @Test
    fun `没有 detail 时退回顶层 message`() {
        assertEquals("boom", errorDetailOf("""{"message":"boom"}"""))
    }

    @Test
    fun `非法 JSON 时原样返回而不是吞掉`() {
        assertEquals("not-json", errorDetailOf("not-json"))
    }

    @Test
    fun `detail 为空串时退回原始 body`() {
        // 宁可难看也不静默：绝不返回空消息让用户什么都看不到
        val body = """{"detail":""}"""
        assertEquals(body, errorDetailOf(body))
    }
}
