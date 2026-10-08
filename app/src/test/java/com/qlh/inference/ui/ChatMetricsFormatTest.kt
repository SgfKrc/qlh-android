package com.qlh.inference.ui

import org.junit.Assert.assertEquals
import org.junit.Test

/**
 * 状态行判定逻辑的回归（与 TUI `tui_shared.format_metrics` **同口径**）。
 *
 * ⚠️ 这里**刻意只测纯函数** `distributedStatusLine` / `formatLayerEvidence`：
 * Android 的 JVM unit test 里 `org.json` 是**未实现的 stub**（调用即抛
 * `RuntimeException: not mocked`）⇒ 任何经由 `formatMetrics(String)` 的断言都会落进
 * `catch` 并返回原始 JSON，得到"看起来像失败、其实是环境"的假红。把判定抽成纯函数后，
 * 这些结论才能被真正钉住。`formatMetrics` 自身的 JSON 解析由真机（instrumented）覆盖。
 */
class ChatMetricsFormatTest {

    // ---------------------------------------------------------------- 状态片段四分支

    @Test
    fun `真分布式时显示生效并附承层证据`() {
        val line = distributedStatusLine(
            hasDistributedField = true,
            distributedUsed = true,
            requestedDistributed = true,
            evidence = "（2 段·层 0-24）",
        )
        assertEquals("分布式 ✓（2 段·层 0-24）", line)
    }

    @Test
    fun `请求侧要求过分布式却未生效必须提示`() {
        // 核心场景：用户用 /route required，但全局开关关着 ⇒ distributed_requested=false，
        // 只看它就会完全静默（"本地要求 vs 分布式要求"矛盾不可见）。
        val line = distributedStatusLine(
            hasDistributedField = true,
            distributedUsed = false,
            requestedDistributed = true,
            evidence = "",
        )
        assertEquals("⚠️ 已请求分布式，实际本地", line)
    }

    @Test
    fun `明确未用分布式时说本地执行`() {
        val line = distributedStatusLine(
            hasDistributedField = true,
            distributedUsed = false,
            requestedDistributed = false,
            evidence = "",
        )
        assertEquals("⚠️ 本地执行（未用分布式）", line)
    }

    @Test
    fun `没有 distributed_used 字段时不擅自宣称本地`() {
        assertEquals(
            "",
            distributedStatusLine(
                hasDistributedField = false,
                distributedUsed = false,
                requestedDistributed = false,
                evidence = "",
            ),
        )
    }

    @Test
    fun `分布式生效优先于请求侧意图`() {
        val line = distributedStatusLine(
            hasDistributedField = true,
            distributedUsed = true,
            requestedDistributed = true,
            evidence = "（层 20-24）",
        )
        assertEquals("分布式 ✓（层 20-24）", line)
    }

    // ---------------------------------------------------------------- 承层证据优先级

    @Test
    fun `layer_segments 优先并给出并集区间`() {
        assertEquals(
            "（2 段·层 0-24）",
            formatLayerEvidence(listOf(listOf(0, 20), listOf(20, 24)), claimed = listOf(4, 19), workers = 3),
        )
    }

    @Test
    fun `无 segments 时回退到 claimed_layers`() {
        assertEquals("（层 4-19）", formatLayerEvidence(emptyList(), claimed = listOf(4, 19), workers = 3))
    }

    @Test
    fun `无 claimed_layers 时回退到 workers_used`() {
        assertEquals("（1 个 worker）", formatLayerEvidence(emptyList(), claimed = null, workers = 1))
    }

    @Test
    fun `都没有时回空串而不是编造证据`() {
        assertEquals("", formatLayerEvidence(emptyList(), claimed = null, workers = 0))
        assertEquals("", formatLayerEvidence(emptyList(), claimed = listOf(1), workers = 0))
    }

    @Test
    fun `segments 元素异常时退化为段数`() {
        assertEquals(
            "（1 段）",
            formatLayerEvidence(listOf(listOf(0)), claimed = null, workers = 0),
        )
    }
}
