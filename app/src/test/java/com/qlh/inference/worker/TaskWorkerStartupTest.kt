package com.qlh.inference.worker

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * ★ 2026-10-07（DIST-NEXT-5）：task-worker 的启动配置与**重建决策**。
 *
 * 审计 P1-3：`START_STICKY` 重建（intent 为空）时 worker 不恢复、配置也不在 ——
 * 这里把「配置是否够用」「重建时该做什么」抽成纯逻辑并锁住。
 */
class TaskWorkerStartupTest {
    private fun config(
        host: String = "100.90.76.108",
        port: Int = 8888,
        nodeId: String = "android_worker_01",
        secret: String = "s3cret",
    ) = TaskWorkerStartupConfig(
        coordinatorHost = host,
        coordinatorPort = port,
        nodeId = nodeId,
        clusterSecret = secret,
        hostname = "TB321FU",
        networkType = "wifi",
        deviceInfo = mapOf("runtime_profile" to "llama_cpp_only"),
        modelId = "qwen_1_8b",
        modelSha256 = "a".repeat(64),
        resourceAdmitted = true,
        resourceReason = "",
    )

    @Test
    fun `usable requires host port node and secret`() {
        assertTrue(config().usable)
        assertFalse(config(host = "  ").usable)
        assertFalse(config(port = 0).usable)
        assertFalse(config(port = 70_000).usable)
        assertFalse(config(nodeId = "").usable)
        assertFalse(config(secret = "").usable)
    }

    @Test
    fun `json round trip preserves the fields needed to rebuild`() {
        val original = config()
        val restored = TaskWorkerStartupConfig.fromJson(original.toJson())

        assertNotNull(restored)
        assertEquals(original, restored)
        // 重建路径真的用得上的字段逐项核对（不只依赖 data class equals）
        assertEquals("100.90.76.108", restored!!.coordinatorHost)
        assertEquals(8888, restored.coordinatorPort)
        assertEquals("android_worker_01", restored.nodeId)
        assertEquals("s3cret", restored.clusterSecret)
        assertEquals("llama_cpp_only", restored.deviceInfo["runtime_profile"])
        assertTrue(restored.usable)
    }

    @Test
    fun `missing or corrupt persisted config is treated as no config`() {
        assertNull(TaskWorkerStartupConfig.fromJson(null))
        assertNull(TaskWorkerStartupConfig.fromJson(""))
        assertNull(TaskWorkerStartupConfig.fromJson("   "))
        assertNull(TaskWorkerStartupConfig.fromJson("{not json"))
        // 结构合法但配置不可用 ⇒ 仍可解析，由 `usable` 判定
        val incomplete = TaskWorkerStartupConfig.fromJson(
            TaskWorkerStartupConfig(coordinatorHost = "", coordinatorPort = 0,
                nodeId = "", clusterSecret = "").toJson(),
        )
        assertNotNull(incomplete)
        assertFalse(incomplete!!.usable)
    }

    @Test
    fun `system recreation without an intent resumes from persisted config`() {
        // ★ 核心判据：intent == null（START_STICKY 重建）且有持久化配置 ⇒ 恢复，而不是停。
        assertEquals(
            TaskWorkerStartupDecision.Resume,
            decideTaskWorkerStartup(
                action = null, hasPersistedConfig = true, hasActiveClient = false,
            ),
        )
        // 未知 action 同样按重建处理（系统可能不带上我们的 action）
        assertEquals(
            TaskWorkerStartupDecision.Resume,
            decideTaskWorkerStartup(
                action = "com.android.something.else",
                hasPersistedConfig = true,
                hasActiveClient = false,
            ),
        )
    }

    @Test
    fun `recreation without config stops instead of looping`() {
        // 无配置且没有在跑的 client ⇒ 明确停止（此前是「什么都不做」）
        assertEquals(
            TaskWorkerStartupDecision.Stop,
            decideTaskWorkerStartup(
                action = null, hasPersistedConfig = false, hasActiveClient = false,
            ),
        )
        // 已经在跑且没有新配置 ⇒ 不动
        assertEquals(
            TaskWorkerStartupDecision.Ignore,
            decideTaskWorkerStartup(
                action = null, hasPersistedConfig = false, hasActiveClient = true,
            ),
        )
    }

    @Test
    fun `explicit actions keep their meaning`() {
        assertEquals(
            TaskWorkerStartupDecision.Start,
            decideTaskWorkerStartup(
                action = TaskWorkerService.ACTION_START,
                hasPersistedConfig = false,
                hasActiveClient = false,
            ),
        )
        // 用户主动停止：即使有持久化配置也必须停（否则重建会把它拉回来）
        assertEquals(
            TaskWorkerStartupDecision.Stop,
            decideTaskWorkerStartup(
                action = TaskWorkerService.ACTION_STOP,
                hasPersistedConfig = true,
                hasActiveClient = true,
            ),
        )
        assertEquals(
            TaskWorkerStartupDecision.Cancel,
            decideTaskWorkerStartup(
                action = TaskWorkerService.ACTION_CANCEL,
                hasPersistedConfig = true,
                hasActiveClient = true,
            ),
        )
    }
}
