package com.qlh.inference

import com.qlh.inference.network.ClusterNode
import com.qlh.inference.network.ClusterStatus
import com.qlh.inference.network.PipelineCapacitySummary

/** Read-only mobile projection of the cluster control plane. It intentionally has no mutating actions. */
data class ClusterOverviewUiState(
    val snapshot: ClusterOverviewSnapshot? = null,
    val loading: Boolean = false,
    val error: String? = null,
)

data class ClusterOverviewSnapshot(
    val running: Boolean = false,
    val runMode: String = "",
    val nodesReady: Boolean = false,
    val reachableNodes: Int = 0,
    val totalNodes: Int = 0,
    val currentTaskId: String? = null,
    val currentTaskState: String? = null,
    val currentTaskElapsedSeconds: Long? = null,
    val nodes: List<ClusterOverviewNode> = emptyList(),
    val capacity: ClusterOverviewCapacity? = null,
)

/**
 * 只读容量投影。存在的理由：「节点 online」只说明心跳在发，不说明它在流水线里干活
 * （安卓重装清空 DataStore 后即如此：`state=online` 但容量里 `worker_count=0`）。
 * 这里把「是否准入 + 未准入原因 + 谁在参与」一并带到移动端，免得必须交叉
 * `/cluster/nodes` 与 `/cluster/pipeline-capacity` 才能看出真相。
 */
data class ClusterOverviewCapacity(
    /** 三态：true=已准入、false=已拒绝、null=后端没有权威决策。 */
    val admitted: Boolean?,
    val status: String,
    val reasonCode: String,
    val reasonLabel: String,
    val participatingNodeCount: Int,
    val workerCount: Int,
    val controlOnlyNodes: List<String>,
    val requireDistributed: Boolean,
)

data class ClusterOverviewNode(
    val nodeId: String,
    val role: String,
    val nodeType: String,
    val state: String,
    val hostname: String,
    val networkType: String,
    val taskCount: Int,
    val errorCount: Int,
    val reachable: Boolean,
    val pipelineParticipating: Boolean = false,
    val pipelineExclusionReason: String = "",
)

/**
 * Normalize the server's node map into a bounded-UI-friendly, deterministic snapshot.
 * A busy worker remains reachable even though scheduler `is_available` only means idle.
 */
fun toClusterOverviewSnapshot(status: ClusterStatus): ClusterOverviewSnapshot {
    val nodes = status.nodes
        .map { (key, value) -> value.toOverviewNode(key) }
        .sortedWith(
            compareBy<ClusterOverviewNode> { if (it.role.equals("master", ignoreCase = true)) 0 else 1 }
                .thenBy { it.nodeId },
        )
    val reachable = nodes.count { it.reachable }
    val fallbackTotal = if (status.totalCount > 0) status.totalCount else 0
    val fallbackReachable = if (status.onlineCount > 0) status.onlineCount else 0
    val task = status.currentTask?.takeIf { it.taskId.isNotBlank() }

    return ClusterOverviewSnapshot(
        running = status.running,
        runMode = status.runMode,
        nodesReady = status.nodesReady,
        reachableNodes = if (nodes.isEmpty()) fallbackReachable else reachable,
        totalNodes = if (nodes.isEmpty()) fallbackTotal else nodes.size,
        currentTaskId = task?.taskId,
        currentTaskState = task?.state?.takeIf { it.isNotBlank() },
        currentTaskElapsedSeconds = task?.elapsed?.takeIf { it >= 0 }?.toLong(),
        nodes = nodes,
        capacity = status.pipelineCapacity?.toOverviewCapacity(),
    )
}

private fun PipelineCapacitySummary.toOverviewCapacity(): ClusterOverviewCapacity {
    val normalizedReason = reasonCode.trim()
    return ClusterOverviewCapacity(
        admitted = admitted,
        status = status.trim(),
        reasonCode = normalizedReason,
        reasonLabel = pipelineCapacityReasonLabel(normalizedReason),
        participatingNodeCount = participatingNodeCount.coerceAtLeast(0),
        workerCount = workerCount.coerceAtLeast(0),
        controlOnlyNodes = controlOnlyNodes
            .map { it.trim() }
            .filter { it.isNotEmpty() },
        requireDistributed = requireDistributed,
    )
}

private fun ClusterNode.toOverviewNode(fallbackId: String): ClusterOverviewNode {
    val normalizedState = state.trim().lowercase()
    return ClusterOverviewNode(
        nodeId = nodeId.ifBlank { fallbackId },
        role = role.ifBlank { "client" },
        nodeType = nodeType.ifBlank { "unknown" },
        state = normalizedState.ifBlank { "unknown" },
        hostname = hostname.trim(),
        networkType = networkType.trim().ifBlank { "unknown" },
        taskCount = taskCount.coerceAtLeast(0),
        errorCount = errorCount.coerceAtLeast(0),
        reachable = isAvailable || normalizedState == "online" || normalizedState == "busy",
        pipelineParticipating = pipelineParticipating,
        pipelineExclusionReason = pipelineExclusionReason.trim(),
    )
}

/**
 * 后端容量原因码 → 用户可读文案。未知码**原样返回** —— 宁可难看，也不静默。
 */
fun pipelineCapacityReasonLabel(reasonCode: String): String = when (val code = reasonCode.trim()) {
    "" -> ""
    "pipeline_capacity_not_computed" -> "容量计划尚未计算"
    "pipeline_descriptor_unavailable" -> "尚无可用模型描述"
    "pipeline_distributed_workers_unavailable" -> "可用流水线 worker 不足"
    "pipeline_capacity_workers_unavailable" -> "可用流水线 worker 不足"
    "pipeline_layer_range_coverage_insufficient" -> "节点声明的层区间覆盖不足"
    "pipeline_segment_contract_unsatisfied" -> "层段契约未满足"
    "pipeline_node_contract_invalid" -> "节点布局契约校验失败"
    "pipeline_recovery_pending" -> "流水线恢复待权威重发"
    "distributed_forced" -> "分布式强制准入"
    else -> code
}

/** 节点级「未参与层段」原因码 → 用户可读文案。 */
fun pipelineExclusionReasonLabel(reasonCode: String): String = when (val code = reasonCode.trim()) {
    "" -> ""
    "capacity_plan_control_only" -> "仅控制面，不参与层段"
    "capacity_plan_unassigned" -> "未分配到层段"
    "capacity_plan_unknown" -> "容量计划未确定"
    "capacity_plan_rejected" -> "容量计划已拒绝"
    else -> pipelineCapacityReasonLabel(code)
}

/**
 * 容量准入的单行文案。**三态显式**：后端没有权威决策时写「准入未知」，
 * 不许把未知伪装成已准入（那正是要消灭的静默形态）。
 */
fun clusterCapacitySummaryText(capacity: ClusterOverviewCapacity): String {
    val verdict = when (capacity.admitted) {
        true -> "已准入"
        false -> "未准入"
        null -> "准入未知"
    }
    val suffix = buildString {
        when {
            capacity.admitted == true -> {
                append(" · 参与 ${capacity.participatingNodeCount} 节点")
                if (capacity.requireDistributed) append(" · 强制分布式")
            }
            capacity.reasonLabel.isNotBlank() -> append(" · ${capacity.reasonLabel}")
            capacity.reasonCode.isNotBlank() -> append(" · ${capacity.reasonCode}")
        }
    }
    return "分层容量：$verdict$suffix"
}

/** 节点参与状态一行文案：未参与时带上可读原因。 */
fun clusterNodeParticipationText(node: ClusterOverviewNode): String {
    if (node.pipelineParticipating) return "参与层段"
    val reason = pipelineExclusionReasonLabel(node.pipelineExclusionReason)
    return if (reason.isBlank()) "未参与层段" else "未参与层段（$reason）"
}
