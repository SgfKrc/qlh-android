package com.qlh.inference

import com.qlh.inference.network.ClusterNode
import com.qlh.inference.network.ClusterStatus
import com.qlh.inference.network.PipelineCapacitySummary
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * 「在线但不干活」在移动端的投影契约（2026-10-08 第 4 条静默路径）。
 *
 * 刻意只测纯函数与 DTO→UI 投影：不触碰 `org.json`（JVM 单测里是 stub）与网络。
 */
class ClusterOverviewUiStateTest {

    @Test
    fun `容量三态各自成文，未知不得伪装成已准入`() {
        assertEquals(
            "分层容量：已准入 · 参与 2 节点",
            clusterCapacitySummaryText(capacity(admitted = true, participating = 2)),
        )
        assertEquals(
            "分层容量：未准入 · 可用流水线 worker 不足",
            clusterCapacitySummaryText(
                capacity(admitted = false, reasonCode = "pipeline_distributed_workers_unavailable"),
            ),
        )
        assertEquals(
            "分层容量：准入未知 · 容量计划尚未计算",
            clusterCapacitySummaryText(
                capacity(admitted = null, reasonCode = "pipeline_capacity_not_computed"),
            ),
        )
    }

    @Test
    fun `未知原因码原样呈现而不是被吞掉`() {
        assertEquals(
            "分层容量：未准入 · brand_new_reason",
            clusterCapacitySummaryText(capacity(admitted = false, reasonCode = "brand_new_reason")),
        )
    }

    @Test
    fun `在线但未参与的节点必须带出原因`() {
        assertEquals(
            "未参与层段（仅控制面，不参与层段）",
            clusterNodeParticipationText(
                node(participating = false, reason = "capacity_plan_control_only"),
            ),
        )
        assertEquals("参与层段", clusterNodeParticipationText(node(participating = true)))
        assertEquals(
            "未参与层段（未分配到层段）",
            clusterNodeParticipationText(node(participating = false, reason = "capacity_plan_unassigned")),
        )
    }

    @Test
    fun `status 映射保留节点参与与容量投影`() {
        val status = ClusterStatus(
            runMode = "distributed",
            nodesReady = true,
            nodes = mapOf(
                "master" to ClusterNode(
                    nodeId = "master", role = "master", state = "online",
                    isAvailable = true, pipelineParticipating = true,
                ),
                "android-1" to ClusterNode(
                    nodeId = "android-1", role = "client", nodeType = "android",
                    state = "online", isAvailable = true,
                    pipelineParticipating = false,
                    pipelineExclusionReason = "capacity_plan_control_only",
                ),
            ),
            pipelineCapacity = PipelineCapacitySummary(
                status = "rejected",
                admitted = false,
                reasonCode = "pipeline_distributed_workers_unavailable",
                controlOnlyNodes = listOf("android-1"),
                participatingNodeCount = 0,
            ),
        )

        val snapshot = toClusterOverviewSnapshot(status)

        val master = snapshot.nodes.first { it.nodeId == "master" }
        val android = snapshot.nodes.first { it.nodeId == "android-1" }
        assertTrue(master.pipelineParticipating)
        assertFalse(android.pipelineParticipating)
        assertEquals("capacity_plan_control_only", android.pipelineExclusionReason)
        assertEquals(false, snapshot.capacity?.admitted)
        assertEquals("可用流水线 worker 不足", snapshot.capacity?.reasonLabel)
        assertEquals(listOf("android-1"), snapshot.capacity?.controlOnlyNodes)
    }

    @Test
    fun `旧后端没有容量字段时不得编造准入结论`() {
        val snapshot = toClusterOverviewSnapshot(
            ClusterStatus(
                nodes = mapOf(
                    "master" to ClusterNode(nodeId = "master", role = "master", state = "online"),
                ),
            ),
        )

        assertNull(snapshot.capacity)
        assertFalse(snapshot.nodes.first().pipelineParticipating)
        assertEquals("", snapshot.nodes.first().pipelineExclusionReason)
    }

    private fun capacity(
        admitted: Boolean?,
        reasonCode: String = "",
        participating: Int = 0,
    ) = ClusterOverviewCapacity(
        admitted = admitted,
        status = "",
        reasonCode = reasonCode,
        reasonLabel = pipelineCapacityReasonLabel(reasonCode),
        participatingNodeCount = participating,
        workerCount = 0,
        controlOnlyNodes = emptyList(),
        requireDistributed = false,
    )

    private fun node(participating: Boolean, reason: String = "") = ClusterOverviewNode(
        nodeId = "android-1",
        role = "client",
        nodeType = "android",
        state = "online",
        hostname = "y700",
        networkType = "tailscale",
        taskCount = 0,
        errorCount = 0,
        reachable = true,
        pipelineParticipating = participating,
        pipelineExclusionReason = reason,
    )
}
