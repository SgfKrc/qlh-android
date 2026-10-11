package com.qlh.inference.ui

import com.qlh.inference.ModelFleetSnapshot
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * 票12（`UI-CONTRACT-01`）的**展示契约**回归 —— 模型舰队运行时标签。
 *
 * 为什么用 JVM 单测而不是 instrumented UI 测试：本仓 `SettingsScreenTest`（androidTest）在
 * Y700（TB321FU / Android 15）上**整类失败** —— SettingsScreen 内容超出 Compose 测试宿主视口，
 * 连首屏主题按钮都被判为不可见（`Assert failed: The component is not displayed!`，13/13 失败，
 * 与本次改动无关）。而本票关心的是**文案是否正确、是否互斥**，与节点是否落在视口内无关，
 * 所以把映射抽成纯函数 [modelFleetRuntimeLabel] 在这里固定。
 */
class ModelFleetRuntimeLabelTest {

    private fun snapshot(
        currentLoaded: Boolean,
        pipelinePrepared: Boolean,
        name: String = "Seg Demo",
        id: String? = "seg-demo",
    ) = ModelFleetSnapshot(
        currentModelId = id,
        currentModelName = name,
        currentLoaded = currentLoaded,
        pipelinePrepared = pipelinePrepared,
    )

    @Test
    fun wholeModelLoadedSaysLoaded() {
        assertEquals("已加载：Seg Demo", modelFleetRuntimeLabel(snapshot(currentLoaded = true, pipelinePrepared = false)))
    }

    @Test
    fun pipelineOnlySaysPipelinePreparedNotLoaded() {
        // 关键契约：流水线就绪 ≠ 整模已加载。
        val label = modelFleetRuntimeLabel(snapshot(currentLoaded = false, pipelinePrepared = true))
        assertEquals("已准备流水线：Seg Demo", label)
        assertNotEquals("已加载：Seg Demo", label)
        assertNotEquals("主节点当前未加载模型", label)
    }

    @Test
    fun nothingReadySaysNotLoaded() {
        val label = modelFleetRuntimeLabel(snapshot(currentLoaded = false, pipelinePrepared = false))
        assertEquals("主节点当前未加载模型", label)
        assertTrue(!label.startsWith("已加载") && !label.startsWith("已准备流水线"))
    }

    @Test
    fun syntheticDoubleTrueResolvesToLoaded() {
        // 合成双真 {true,true} 在现行合同下送不到该 UI（三种可达形状见函数 KDoc）。
        // 这里只固定"若真发生"时的语法优先级，不代表存在可达的互斥展示故障。
        assertEquals("已加载：Seg Demo", modelFleetRuntimeLabel(snapshot(currentLoaded = true, pipelinePrepared = true)))
    }

    @Test
    fun theThreeReachableLabelsAreMutuallyExclusive() {
        val labels = listOf(
            modelFleetRuntimeLabel(snapshot(currentLoaded = true, pipelinePrepared = false)),
            modelFleetRuntimeLabel(snapshot(currentLoaded = false, pipelinePrepared = true)),
            modelFleetRuntimeLabel(snapshot(currentLoaded = false, pipelinePrepared = false)),
        )
        assertEquals("三种可达态必须给出三条不同文案", 3, labels.toSet().size)
    }

    @Test
    fun fallsBackToModelIdThenToPlaceholder() {
        assertEquals("已加载：seg-demo", modelFleetRuntimeLabel(snapshot(currentLoaded = true, pipelinePrepared = false, name = "")))
        assertEquals(
            "主节点当前未加载模型",
            modelFleetRuntimeLabel(snapshot(currentLoaded = false, pipelinePrepared = false, name = "", id = null)),
        )
    }
}
