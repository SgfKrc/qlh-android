package com.qlh.inference

import com.qlh.inference.network.ClusterNode
import com.qlh.inference.network.ClusterStatus

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
    )
}
