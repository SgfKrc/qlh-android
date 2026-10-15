package com.qlh.inference

import com.qlh.inference.network.ModelIntentCapacityDiagnostic
import com.qlh.inference.network.ModelIntentCapacityNode
import com.qlh.inference.network.ModelIntentOperation
import java.util.Locale

private val capacityReasonCodes = setOf(
    "pipeline_distributed_capacity_insufficient",
    "pipeline_cluster_capacity_insufficient",
    "pipeline_capacity_single_node_insufficient",
    "pipeline_capacity_manual_insufficient",
    "pipeline_reshard_capacity_insufficient",
    "node_capacity_unavailable",
    "SUPPLY_CAPACITY_INSUFFICIENT",
    "SUPPLY_MASTER_CAPACITY_INSUFFICIENT",
    "SUPPLY_RUNTIME_HOST_CAPACITY_INSUFFICIENT",
)

private fun ModelIntentOperation.matchesCapacityDiagnostic(): Boolean =
    !reasonCode.isNullOrBlank() && capacityDiagnostic?.reasonCode == reasonCode

private fun ModelIntentOperation.hasCapacityCause(): Boolean =
    status in setOf("waiting_nodes", "failed") && !reasonCode.isNullOrBlank() &&
        reasonCode != "SUPPLY_WAITING_NODES" &&
        (reasonCode in capacityReasonCodes || (matchesCapacityDiagnostic() &&
            capacityDiagnostic?.nodes?.any { it.deficitBytes != null && it.deficitBytes > 0L } == true))

/** A label projection only; the master's lifecycle and readyForChat remain authoritative. */
fun modelIntentStatusLabel(operation: ModelIntentOperation): String =
    if (operation.status == "waiting_nodes" && operation.hasCapacityCause()) "等待容量"
    else modelIntentStatusLabel(operation.status)

private fun capacityBytes(value: Long?): String? {
    if (value == null || value < 0) return null
    val unit = when {
        value >= 1024L * 1024 * 1024 -> "GiB" to 1024L * 1024 * 1024
        value >= 1024L * 1024 -> "MiB" to 1024L * 1024
        value >= 1024L -> "KiB" to 1024L
        else -> return "$value bytes"
    }
    // Floating point is used only for the readable units; the exact integer is kept verbatim.
    return String.format(Locale.ROOT, "%.2f %s (%d bytes)", value.toDouble() / unit.second, unit.first, value)
}

private fun ModelIntentCapacityNode.line(): String = buildList {
    add(nodeId?.takeIf(String::isNotBlank) ?: "未标识节点")
    engine?.takeIf(String::isNotBlank)?.let { add(when (it) {
        "llama_cpp" -> "llama.cpp"
        "pytorch" -> "PyTorch"
        else -> it
    }) }
    executionDevice?.takeIf(String::isNotBlank)?.let { add(when (it) {
        "cpu" -> "CPU"
        "cuda" -> "CUDA"
        else -> it
    }) }
    resource?.takeIf(String::isNotBlank)?.let { add(when (it) {
        "ram" -> "RAM"
        "vram" -> "显存（VRAM）"
        "host_ram" -> "主机暂存 RAM"
        else -> it
    }) }
    layerRange?.takeIf { it.size == 2 && it[0] >= 0 && it[0] < it[1] }?.let {
        add("层段 [${it[0]}, ${it[1]})")
    }
    val reserveLabel = when (resource) {
        "ram", "vram" -> "含边界的预留"
        "host_ram" -> "固定预留"
        else -> "预留"
    }
    for ((label, value) in listOf(
        "可用" to availableBytes, "所需" to requiredBytes,
        "缺口" to deficitBytes, reserveLabel to reserveBytes,
    )) capacityBytes(value)?.let { add("$label $it") }
}.joinToString(" · ")

/** No deficit arithmetic, local solver, or cached state: display only the matching owner's facts. */
internal fun formatModelIntentCapacity(operation: ModelIntentOperation): List<String> {
    if (!operation.hasCapacityCause() || !operation.matchesCapacityDiagnostic()) return emptyList()
    val diagnostic = operation.capacityDiagnostic ?: return emptyList()
    return diagnostic.lines(displayedReason = operation.reason)
}

private fun ModelIntentCapacityDiagnostic.lines(displayedReason: String?): List<String> = buildList {
    val scopeLabel = when (scope) {
        "candidate_plan" -> "候选方案"
        "minimum_segment" -> "最小层段"
        else -> "范围未提供"
    }
    val caveat = if (nodes.any { it.deficitBytes != null && it.deficitBytes > 0L }) "；释放所示缺口不保证可执行" else ""
    add("容量不足（$scopeLabel$caveat）")
    reason?.takeIf { it.isNotBlank() && it != displayedReason }?.let { add(it) }
    nodes.forEach { add(it.line()) }
    pendingNodeIds.filter(String::isNotBlank).takeIf { it.isNotEmpty() }?.let {
        add("仍在加入的节点：${it.joinToString(", ")}")
    }
    suggestedAction?.takeIf(String::isNotBlank)?.let { add("建议：$it") }
}
