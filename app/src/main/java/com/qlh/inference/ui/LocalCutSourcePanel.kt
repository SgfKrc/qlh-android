package com.qlh.inference.ui

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.material3.Button
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.unit.dp
import com.qlh.inference.service.ModelManager
import com.qlh.inference.service.ModelSupplyException
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

/** Registers a read-only disk source. Layer selection stays in the existing master model intent. */
@Composable
internal fun LocalCutSourcePanel(reference: String) {
    val context = LocalContext.current.applicationContext
    val registry = remember(context) { ModelManager(context).localCutSources }
    val scope = rememberCoroutineScope()
    var modelId by remember(reference) { mutableStateOf("") }
    var busy by remember { mutableStateOf(false) }
    var message by remember(reference) { mutableStateOf("") }
    LaunchedEffect(reference) {
        message = withContext(Dispatchers.IO) { runCatching { registry.summary(reference) }.getOrDefault("") }
    }
    Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
        Text("GGUF 本地裁切源", style = MaterialTheme.typography.titleSmall)
        Text("选择完整 GGUF 后，填写主节点模型目录中的准确模型 ID 并登记。自动供给时由主节点分层，本机只生成和加载分配的层段。需要与主节点相同的完整源；旧裁切文件不能登记为整源。",
            style = MaterialTheme.typography.bodySmall)
        OutlinedTextField(value = modelId, onValueChange = { modelId = it }, singleLine = true,
            enabled = !busy && reference.isNotBlank(), label = { Text("主节点模型 ID") },
            modifier = Modifier.fillMaxWidth().testTag("local_cut_source_model_id"))
        Button(enabled = !busy && reference.isNotBlank() && modelId.isNotBlank(),
            modifier = Modifier.fillMaxWidth().testTag("local_cut_register_source"), onClick = {
                busy = true
                message = "正在流式校验完整源 SHA、tokenizer 与 GGUF 结构，请保持此页打开。"
                val selectedReference = reference
                val selectedId = modelId.trim()
                scope.launch {
                    try {
                        val registered = withContext(Dispatchers.IO) { registry.register(selectedReference, selectedId) }
                        message = "已登记 ${registered.modelId} · SHA ${registered.sha256}。供给时会重新验证源身份；请在主节点模型面板选择模型并使用自动供给。"
                    } catch (cancelled: CancellationException) { throw cancelled }
                    catch (error: Exception) {
                        val code = (error as? ModelSupplyException)?.reasonCode ?: "SUPPLY_LOCAL_SOURCE_UNAVAILABLE"
                        message = "登记失败 · $code。请选择已授权目录内的完整 GGUF，检查模型 ID 和源文件；带层段 manifest 的文件不能用作整源。"
                    } finally { busy = false }
                }
            }) { Text(if (busy) "校验源文件中" else "登记为裁切源") }
        if (busy) CircularProgressIndicator()
        if (message.isNotBlank()) Text(message, style = MaterialTheme.typography.bodySmall,
            modifier = Modifier.testTag("local_cut_source_status"))
        Text("只读 SAF 授权即可裁切；产物写入应用私有目录。裁切权重不走下载，原生 manifest 仍从主节点校验接收。未登记或源不可读时，自动供给可使用主节点分发；源摘要或 tokenizer 身份不一致时会拒绝供给，请重新核对源文件。",
            style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
    }
}

/** Independent from the selected inference model: picking a disk source must never request a full load. */
@Composable
internal fun LocalCutSourcePicker(
    models: List<ModelManager.ModelDocument>,
    scanning: Boolean,
    refresh: () -> Unit,
    chooseDirectory: () -> Unit,
) {
    var reference by androidx.compose.runtime.saveable.rememberSaveable { mutableStateOf("") }
    var expanded by remember { mutableStateOf(false) }
    androidx.compose.material3.OutlinedButton(onClick = chooseDirectory, enabled = !scanning,
        modifier = Modifier.fillMaxWidth().testTag("local_cut_choose_directory")) { Text("选择源目录（沿用 SAF 授权）") }
    androidx.compose.material3.OutlinedButton(onClick = refresh, enabled = !scanning,
        modifier = Modifier.fillMaxWidth().testTag("local_cut_scan_sources")) { Text(if (scanning) "扫描中" else "扫描源文件") }
    Text("源文件放入已授权 Download/QLH/models 后点击扫描。这里只登记磁盘来源；推理使用主节点自动分配的层段。",
        style = MaterialTheme.typography.bodySmall)
    androidx.compose.material3.OutlinedButton(onClick = { expanded = !expanded },
        modifier = Modifier.fillMaxWidth().testTag("local_cut_source_list_toggle")) {
        Text(models.firstOrNull { it.uri.toString() == reference }?.name ?: "选择完整 GGUF 源文件（${models.size} 项）")
    }
    if (expanded) models.forEach { model ->
        androidx.compose.material3.OutlinedButton(onClick = { reference = model.uri.toString(); expanded = false },
            modifier = Modifier.fillMaxWidth().testTag("local_cut_source_${model.name}")) { Text(model.name) }
    }
    LocalCutSourcePanel(reference)
}
