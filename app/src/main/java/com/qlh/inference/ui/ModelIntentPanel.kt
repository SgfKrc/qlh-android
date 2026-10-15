package com.qlh.inference.ui

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Cloud
import androidx.compose.material3.Button
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.unit.dp
import com.qlh.inference.ModelIntentUiState
import com.qlh.inference.formatModelIntentCapacity
import com.qlh.inference.modelIntentStatusLabel
import com.qlh.inference.network.ServerModelSummary
import com.qlh.inference.ui.components.SettingsGroup

internal fun filterModelIntentCatalog(models: List<ServerModelSummary>, query: String): List<ServerModelSummary> {
    val term = query.trim()
    return if (term.isEmpty()) models else models.filter {
        it.name.contains(term, ignoreCase = true) || it.modelId.contains(term, ignoreCase = true)
    }
}

internal data class ModelIntentSelectionAvailability(
    val enabled: Boolean,
    val message: String,
    val reasonCode: String = "",
)

/** Only a typed reason for the selected engine blocks acquisition. */
internal fun modelIntentSelectionAvailability(
    model: ServerModelSummary,
    engine: String,
    canChooseModel: Boolean = true,
): ModelIntentSelectionAvailability {
    val selectedEngine = if (engine == "auto") model.pipelineAutoEngine?.trim()?.takeIf { it.isNotEmpty() } else engine
    val evidence = selectedEngine?.let { model.pipelineEngineAvailability?.get(it) }
    val reasonCode = evidence?.reasonCode?.trim().orEmpty()
    val engineLabel = when (selectedEngine) {
        "pytorch" -> "PyTorch"
        "llama_cpp" -> "llama.cpp"
        null -> "自动"
        else -> selectedEngine
    }.let { if (engine == "auto" && selectedEngine != null) "自动（$it）" else it }
    val message = when {
        reasonCode.isNotEmpty() -> "$engineLabel 不可选择：${evidence?.reason?.trim()?.takeIf { it.isNotEmpty() } ?: "主节点已确认当前来源不可用"} · $reasonCode"
        model.modelId.isBlank() -> "模型标识缺失，请刷新主节点模型目录。"
        evidence?.status == "missing" -> "$engineLabel 来源尚未就绪；选择后由主节点尝试获取并验证。"
        evidence?.status != "supported" -> "$engineLabel 来源可用性尚未确认；选择后由主节点获取并验证。"
        evidence.source == "provenance" -> "$engineLabel 转换来源已登记；完整模型内容仍由主节点在供给时验证。"
        else -> ""
    }
    return ModelIntentSelectionAvailability(
        enabled = canChooseModel && model.modelId.isNotBlank() && reasonCode.isEmpty(),
        message = message,
        reasonCode = reasonCode,
    )
}

/** The only master-model selector. The same operation status also appears above chat input. */
@Composable
fun ModelIntentPanel(
    state: ModelIntentUiState,
    discovering: Boolean,
    onDiscoverMaster: () -> Unit,
    onRefresh: () -> Unit,
    onSelectModel: (String, String) -> Unit,
    onRetry: () -> Unit,
    onCancel: () -> Unit,
    onChat: () -> Unit,
) {
    var engine by remember { mutableStateOf("auto") }
    var query by rememberSaveable { mutableStateOf("") }
    SettingsGroup(title = "主节点模型", icon = Icons.Default.Cloud) {
        if (discovering || state.loading || state.busy) {
            LinearProgressIndicator(modifier = Modifier.fillMaxWidth())
            Text(if (discovering) "正在发现并验证主节点" else "正在读取模型意图", style = MaterialTheme.typography.bodySmall)
        }
        // Keep the operation and its actions reachable before a potentially long catalog.
        ModelIntentProgress(state, onRetry, onCancel)
        OutlinedButton(onClick = onRefresh, enabled = !discovering && !state.busy && !state.suspended,
            modifier = Modifier.fillMaxWidth().testTag("intent_refresh")) { Text("刷新模型状态") }
        if (state.readyForChat) Button(onClick = onChat,
            modifier = Modifier.fillMaxWidth().testTag("intent_open_chat")) { Text("准备完成，开始聊天") }
        if (!state.connected) {
            Text("连接主节点后，在这里选择一次模型；主节点将自动供给并准备各节点。")
            Button(
                onClick = onDiscoverMaster, enabled = !discovering && !state.busy,
                modifier = Modifier.fillMaxWidth().testTag("intent_discover_master"),
            ) { Text("发现主节点 / 连接") }
        } else {
            Text("已连接主节点 · ${state.catalog?.models?.size ?: 0} 个模型", style = MaterialTheme.typography.bodyMedium)
            Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                listOf("auto" to "自动", "pytorch" to "PyTorch", "llama_cpp" to "llama.cpp").forEach { (id, label) ->
                    OutlinedButton(onClick = { engine = id }, enabled = state.canChooseModel,
                        modifier = Modifier.testTag("intent_engine_$id")) {
                        Text(if (engine == id) "✓ $label" else label)
                    }
                }
            }
            OutlinedTextField(
                value = query,
                onValueChange = { query = it },
                label = { Text("搜索模型名称 / ID") },
                singleLine = true,
                modifier = Modifier.fillMaxWidth().testTag("intent_model_search"),
            )
            val models = state.catalog?.models.orEmpty()
            val filteredModels = filterModelIntentCatalog(models, query)
            if (query.isNotBlank()) Text("筛选结果 ${filteredModels.size} / ${models.size}",
                style = MaterialTheme.typography.bodySmall)
            filteredModels.forEach { model ->
                Column(modifier = Modifier.fillMaxWidth().padding(vertical = 4.dp)) {
                    val selected = state.requestedIntent?.modelId == model.modelId
                    val availability = modelIntentSelectionAvailability(model, engine, state.canChooseModel)
                    OutlinedButton(
                        onClick = { onSelectModel(model.modelId, engine) },
                        enabled = availability.enabled,
                        modifier = Modifier.fillMaxWidth().testTag("intent_model_${model.modelId}"),
                    ) {
                        Text((if (selected) "已选择 · " else "选择 · ") + model.name.ifBlank { model.modelId })
                    }
                    if (availability.message.isNotBlank()) Text(
                        availability.message,
                        modifier = Modifier.testTag("intent_model_reason_${model.modelId}"),
                        style = MaterialTheme.typography.bodySmall,
                        color = if (availability.reasonCode.isNotEmpty()) MaterialTheme.colorScheme.error
                            else MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }
            }
            if (models.isEmpty()) Text("主节点没有可选择的模型")
            else if (filteredModels.isEmpty()) Text("没有匹配的模型，请修改搜索词")
        }
    }
}

@Composable
fun ModelIntentProgress(
    state: ModelIntentUiState,
    onRetry: () -> Unit,
    onCancel: () -> Unit,
    showNodes: Boolean = true,
) {
    Column(verticalArrangement = Arrangement.spacedBy(6.dp), modifier = Modifier.testTag("intent_progress")) {
        state.operation?.let { operation ->
            Text("${operation.intent.modelId} · ${modelIntentStatusLabel(operation)}",
                style = MaterialTheme.typography.titleSmall)
            if (operation.status == "ready" && !operation.loadedReady) Text("等待主节点确认推理就绪")
            if (state.readyForChat) Text("模型已准备完成，可开始聊天", color = MaterialTheme.colorScheme.primary)
            Text("操作 ${operation.operationId} · 代际 ${operation.generation}", style = MaterialTheme.typography.labelSmall)
            operation.configId?.takeIf { it.isNotBlank() }?.let {
                Text("模型配置 $it", style = MaterialTheme.typography.labelSmall)
            }
            if (!operation.reasonCode.isNullOrBlank() || !operation.reason.isNullOrBlank()) {
                Text(listOfNotNull(operation.reasonCode, operation.reason).joinToString(" · "),
                    color = MaterialTheme.colorScheme.error, style = MaterialTheme.typography.bodySmall)
            }
            val capacityLines = formatModelIntentCapacity(operation)
            if (capacityLines.isNotEmpty()) Column(modifier = Modifier.testTag("intent_capacity_diagnostic")) {
                capacityLines.forEach { line -> Text(line, style = MaterialTheme.typography.bodySmall) }
            }
            if (operation.plan != null && !operation.plan.isJsonNull) Text("供给计划已生成", style = MaterialTheme.typography.bodySmall)
            if (showNodes) operation.nodes.forEach { (id, node) ->
                Column(modifier = Modifier.fillMaxWidth().testTag("intent_node_$id")) {
                    Text("$id · ${modelIntentStatusLabel(node.status)}", style = MaterialTheme.typography.bodySmall)
                    val progress = node.progress
                    if (progress != null) LinearProgressIndicator(progress = { progress }, modifier = Modifier.fillMaxWidth())
                    if (node.totalBytes > 0 || node.bytesReceived > 0) {
                        Text("${node.bytesReceived.coerceAtLeast(0)} / ${node.totalBytes.coerceAtLeast(0)} bytes",
                            style = MaterialTheme.typography.labelSmall)
                    }
                    node.layerRange?.takeIf { it.isJsonArray && it.asJsonArray.size() == 2 &&
                        it.asJsonArray.all { value -> value.isJsonPrimitive && value.asJsonPrimitive.isNumber } }?.let {
                        Text("层区间 ${it.asJsonArray[0]}–${it.asJsonArray[1]}", style = MaterialTheme.typography.labelSmall)
                    }
                    if (!node.reasonCode.isNullOrBlank() || !node.reason.isNullOrBlank()) Text(
                        listOfNotNull(node.reasonCode, node.reason).joinToString(" · "),
                        color = MaterialTheme.colorScheme.error, style = MaterialTheme.typography.bodySmall,
                    )
                }
            }
        }
        state.error?.let { Text("${it.code} · ${it.reason}", color = MaterialTheme.colorScheme.error,
            style = MaterialTheme.typography.bodySmall, modifier = Modifier.testTag("intent_error")) }
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            if (state.canRetry) OutlinedButton(onClick = onRetry, modifier = Modifier.testTag("intent_retry")) { Text("重试模型准备") }
            if (state.canCancel) OutlinedButton(onClick = onCancel, modifier = Modifier.testTag("intent_cancel")) { Text("取消模型准备") }
        }
    }
}
