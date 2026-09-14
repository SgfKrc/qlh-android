package com.qlh.inference.ui

import android.content.ClipData
import android.content.ClipboardManager
import android.content.Context
import android.content.Intent
import com.qlh.inference.BuildConfig
import androidx.compose.foundation.clickable
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.border
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.text.selection.SelectionContainer
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.Logout
import androidx.compose.material.icons.filled.Cloud
import androidx.compose.material.icons.filled.CloudDownload
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material.icons.filled.Description
import androidx.compose.material.icons.filled.FolderOpen
import androidx.compose.material.icons.filled.Info
import androidx.compose.material.icons.filled.Memory
import androidx.compose.material.icons.filled.NetworkCheck
import androidx.compose.material.icons.filled.Palette
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material.icons.filled.Tune
import androidx.compose.material.icons.filled.AccountCircle
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.RadioButton
import androidx.compose.material3.Slider
import androidx.compose.material3.SliderDefaults
import androidx.compose.material3.Switch
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.text.input.PasswordVisualTransformation
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import com.qlh.inference.logging.QlhLogger
import com.qlh.inference.AppUpdateUiState
import com.qlh.inference.AuthControlUiState
import com.qlh.inference.AuditAttempt
import com.qlh.inference.AuditReviewTicket
import com.qlh.inference.AuditUiState
import com.qlh.inference.AuditWorkflow
import com.qlh.inference.ClusterOverviewUiState
import com.qlh.inference.DiagnosticsUiState
import com.qlh.inference.ModelFleetEntry
import com.qlh.inference.ModelFleetStatus
import com.qlh.inference.ModelFleetUiState
import com.qlh.inference.ManagementUiState
import com.qlh.inference.ManagedBindingSnapshot
import com.qlh.inference.ManagedUserSnapshot
import com.qlh.inference.MAX_MANAGEMENT_AUDIT_EVENTS
import com.qlh.inference.network.ConnectionHealthState
import com.qlh.inference.network.ApiClient
import com.qlh.inference.network.GgufModelInfo
import com.qlh.inference.network.httpBaseUrl
import com.qlh.inference.service.ModelManager
import com.qlh.inference.status.AndroidRuntimeStatus
import com.qlh.inference.network.AndroidPresenceSnapshot
import com.qlh.inference.network.AndroidPresenceState
import com.qlh.inference.ui.components.QlhTopBar
import com.qlh.inference.ui.components.CollapsibleSettingsGroup
import com.qlh.inference.ui.components.SettingRow
import com.qlh.inference.ui.components.SettingsGroup
import com.qlh.inference.ui.components.StatusChip
import com.qlh.inference.ui.theme.qlhBrandGold
import com.qlh.inference.ui.theme.qlhBrandGoldContainer
import com.qlh.inference.ui.theme.qlhOnBrandGoldContainer
import com.qlh.inference.ui.theme.qlhNeonGreen
import com.qlh.inference.ui.theme.qlhOnNeonGreen
import kotlinx.coroutines.launch

private data class QlhStatusTone(
    val containerColor: Color,
    val contentColor: Color,
    val showDot: Boolean,
)

@Composable
private fun qlhStatusTone(status: String): QlhStatusTone {
    return when (status.trim().lowercase()) {
        "active", "approved", "authenticated", "available", "busy", "completed",
        "connected", "enabled", "online", "pass", "ready", "result_ready", "running",
        "success" -> QlhStatusTone(qlhNeonGreen(), qlhOnNeonGreen(), showDot = true)
        "created", "pending", "registering", "skipped", "waiting", "unknown" ->
            QlhStatusTone(qlhBrandGoldContainer(), qlhOnBrandGoldContainer(), showDot = false)
        "denied", "disabled", "error", "fail", "failed", "missing", "offline", "revoked",
        "stopped" -> QlhStatusTone(
            MaterialTheme.colorScheme.errorContainer,
            MaterialTheme.colorScheme.onErrorContainer,
            showDot = true,
        )
        else -> QlhStatusTone(
            MaterialTheme.colorScheme.tertiaryContainer,
            MaterialTheme.colorScheme.onTertiaryContainer,
            showDot = false,
        )
    }
}

@Composable
private fun QlhSemanticStatusChip(
    text: String,
    status: String = text,
    showDot: Boolean? = null,
    modifier: Modifier = Modifier,
) {
    val tone = qlhStatusTone(status)
    StatusChip(
        text = text,
        modifier = modifier,
        containerColor = tone.containerColor,
        contentColor = tone.contentColor,
        showDot = showDot ?: tone.showDot,
    )
}

// ================================================================
// 设置界面 — 按分组卡片组织：外观 / 主节点连接 / 推理模式 / 模型管理 /
// 推理参数 / 设备状态 / 日志管理 / 关于
// ================================================================

@Composable
fun SettingsScreen(
    serverHost: String,
    serverPort: Int,
    inferenceMode: String,
    maxTokens: Int,
    temperature: Float,
    topP: Float,
    contextSize: Int,
    showThinking: Boolean,
    themeMode: String,
    modelTreeUri: String,
    selectedModelUri: String,
    modelStorageMode: String,
    availableModels: List<ModelManager.ModelDocument>,
    selectedModelName: String,
    selectedModelSizeBytes: Long,
    isScanningModels: Boolean,
    modelMessage: String?,
    onServerHostChange: (String) -> Unit,
    onServerPortChange: (Int) -> Unit,
    onInferenceModeChange: (String) -> Unit,
    onMaxTokensChange: (Int) -> Unit,
    onTemperatureChange: (Float) -> Unit,
    onTopPChange: (Float) -> Unit,
    onContextSizeChange: (Int) -> Unit,
    onShowThinkingChange: (Boolean) -> Unit,
    onThemeModeChange: (String) -> Unit,
    onChooseModelDirectory: () -> Unit,
    onRefreshModels: () -> Unit,
    onModelSelected: (ModelManager.ModelDocument) -> Unit,
    onDeleteSelectedModel: () -> Unit,
    runtimeStatus: AndroidRuntimeStatus?,
    runtimeStatusLoading: Boolean,
    runtimeStatusError: String?,
    onRefreshRuntimeStatus: () -> Unit,
    presence: AndroidPresenceSnapshot = AndroidPresenceSnapshot(),
    onConnectionTestSuccess: () -> Unit = {},
    remoteModels: List<GgufModelInfo> = emptyList(),
    remoteModelsLoading: Boolean = false,
    remoteDownloadModelName: String? = null,
    remoteDownloadProgress: ModelManager.DownloadProgress? = null,
    remoteModelMessage: String? = null,
    onRefreshRemoteModels: () -> Unit = {},
    onDownloadRemoteModel: (GgufModelInfo) -> Unit = {},
    diagnostics: DiagnosticsUiState = DiagnosticsUiState(),
    onRefreshDiagnostics: () -> Unit = {},
    onUploadDiagnostics: () -> Unit = {},
    clusterOverview: ClusterOverviewUiState = ClusterOverviewUiState(),
    onRefreshClusterOverview: () -> Unit = {},
    modelFleet: ModelFleetUiState = ModelFleetUiState(),
    onRefreshModelFleet: () -> Unit = {},
    audit: AuditUiState = AuditUiState(),
    onRefreshAudit: () -> Unit = {},
    authControl: AuthControlUiState = AuthControlUiState(),
    onRefreshAuthControl: () -> Unit = {},
    onLogin: (String, String?, String?) -> Unit = { _, _, _ -> },
    onLogout: () -> Unit = {},
    management: ManagementUiState = ManagementUiState(),
    onRefreshManagement: () -> Unit = {},
    onRevokeManagedUser: (ManagedUserSnapshot) -> Unit = {},
    onRevokeManagedBinding: (ManagedBindingSnapshot) -> Unit = {},
    appUpdate: AppUpdateUiState = AppUpdateUiState(),
    onCheckForAppUpdate: () -> Unit = {},
    onDownloadAppUpdate: () -> Unit = {},
    onOpenInstallPermissionSettings: () -> Unit = {},
    onInstallDownloadedUpdate: () -> Unit = {},
    modifier: Modifier = Modifier
) {
    val isLite = BuildConfig.IS_LITE

    LaunchedEffect(Unit) {
        onRefreshRuntimeStatus()
        onRefreshDiagnostics()
        onRefreshClusterOverview()
        onRefreshModelFleet()
        onRefreshAudit()
        onRefreshAuthControl()
    }

    Column(modifier = modifier.fillMaxSize()) {
        // ---- 顶栏 ----
        QlhTopBar(
            title = "设置",
            subtitle = if (isLite) "QLH 极简版 · 仅远程推理" else "QLH 完整版"
        )
        HorizontalDivider(color = MaterialTheme.colorScheme.outlineVariant.copy(alpha = 0.5f))

        // ---- 分组内容（保持 Column + verticalScroll） ----
        Column(
            modifier = Modifier
                .testTag("settings_screen")
                .fillMaxSize()
                .verticalScroll(rememberScrollState())
                .padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(20.dp)
        ) {
            DeviceStatusGroups(
                status = runtimeStatus,
                loading = runtimeStatusLoading,
                error = runtimeStatusError,
                onRefresh = onRefreshRuntimeStatus
            )
            ConnectionGroup(
                serverHost = serverHost,
                serverPort = serverPort,
                onServerHostChange = onServerHostChange,
                onServerPortChange = onServerPortChange,
                onConnectionTestSuccess = onConnectionTestSuccess
            )

            InferenceModeGroup(
                inferenceMode = inferenceMode,
                isLite = isLite,
                onInferenceModeChange = onInferenceModeChange
            )

            PresenceStatusGroup(presence)

            ClusterOverviewGroup(
                state = clusterOverview,
                onRefresh = onRefreshClusterOverview,
            )

            ModelFleetOverviewGroup(
                state = modelFleet,
                onRefresh = onRefreshModelFleet,
            )

            AuditOverviewGroup(
                state = audit,
                onRefresh = onRefreshAudit,
            )

            AuthControlGroup(
                state = authControl,
                onRefresh = onRefreshAuthControl,
                onLogin = onLogin,
                onLogout = onLogout,
            )

            if (authControl.account?.role?.lowercase() in setOf("owner", "admin")) {
                ManagementControlGroup(
                    state = management,
                    onRefresh = onRefreshManagement,
                    onRevokeUser = onRevokeManagedUser,
                    onRevokeBinding = onRevokeManagedBinding,
                )
            }

            DiagnosticsGroup(
                state = diagnostics,
                onRefresh = onRefreshDiagnostics,
                onUpload = onUploadDiagnostics,
            )

            AppUpdateGroup(
                state = appUpdate,
                onCheck = onCheckForAppUpdate,
                onDownload = onDownloadAppUpdate,
                onOpenInstallPermissionSettings = onOpenInstallPermissionSettings,
                onInstall = onInstallDownloadedUpdate,
            )

            AppearanceGroup(
                themeMode = themeMode,
                onThemeModeChange = onThemeModeChange
            )

            if (!isLite && inferenceMode == "full") {
                SettingsGroup(title = "模型管理", icon = Icons.Default.FolderOpen) {
                    ModelManagementPanel(
                        modelTreeUri = modelTreeUri,
                        selectedModelUri = selectedModelUri,
                        modelStorageMode = modelStorageMode,
                        availableModels = availableModels,
                        selectedModelName = selectedModelName,
                        selectedModelSizeBytes = selectedModelSizeBytes,
                        isScanningModels = isScanningModels,
                        modelMessage = modelMessage,
                        onChooseModelDirectory = onChooseModelDirectory,
                        onRefreshModels = onRefreshModels,
                        onModelSelected = onModelSelected,
                        onDeleteSelectedModel = onDeleteSelectedModel
                    )
                }

                SettingsGroup(title = "主节点模型", icon = Icons.Default.CloudDownload) {
                    RemoteModelPanel(
                        modelTreeUri = modelTreeUri,
                        models = remoteModels,
                        loading = remoteModelsLoading,
                        activeModelName = remoteDownloadModelName,
                        progress = remoteDownloadProgress,
                        message = remoteModelMessage,
                        onRefresh = onRefreshRemoteModels,
                        onDownload = onDownloadRemoteModel,
                    )
                }
            }

            InferenceParamsGroup(
                maxTokens = maxTokens,
                temperature = temperature,
                topP = topP,
                contextSize = contextSize,
                showThinking = showThinking,
                onMaxTokensChange = onMaxTokensChange,
                onTemperatureChange = onTemperatureChange,
                onTopPChange = onTopPChange,
                onContextSizeChange = onContextSizeChange,
                onShowThinkingChange = onShowThinkingChange
            )

            LogManagementGroup(isLite = isLite)

            AboutGroup()

            Spacer(modifier = Modifier.height(24.dp))
        }
    }
}

// ================================================================
// 外观
// ================================================================

@Composable
private fun AppearanceGroup(
    themeMode: String,
    onThemeModeChange: (String) -> Unit
) {
    SettingsGroup(title = "外观", icon = Icons.Default.Palette) {
        Text(
            text = "主题模式",
            style = MaterialTheme.typography.bodyLarge,
            color = MaterialTheme.colorScheme.onSurface
        )
        Spacer(modifier = Modifier.height(10.dp))
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            listOf(
                "system" to "跟随系统",
                "light" to "浅色",
                "dark" to "深色"
            ).forEach { (mode, label) ->
                val selected = themeMode == mode
                OutlinedButton(
                    onClick = { onThemeModeChange(mode) },
                    modifier = Modifier
                        .weight(1f)
                        .testTag("settings_theme_$mode"),
                    colors = ButtonDefaults.outlinedButtonColors(
                        containerColor = if (selected) {
                            MaterialTheme.colorScheme.primary
                        } else {
                            MaterialTheme.colorScheme.surfaceContainerLowest
                        },
                        contentColor = if (selected) {
                            MaterialTheme.colorScheme.onPrimary
                        } else {
                            MaterialTheme.colorScheme.onSurface
                        }
                    )
                ) {
                    Text(label, maxLines = 1, overflow = TextOverflow.Ellipsis)
                }
            }
        }
    }
}

// ================================================================
// 主节点连接
// ================================================================

@Composable
private fun PresenceStatusGroup(snapshot: AndroidPresenceSnapshot) {
    val label = when (snapshot.state) {
        AndroidPresenceState.ONLINE -> "online"
        AndroidPresenceState.REGISTERING -> "registering"
        AndroidPresenceState.BACKING_OFF -> "offline / retrying"
        AndroidPresenceState.OFFLINE -> "offline"
        AndroidPresenceState.STOPPED -> "stopped"
    }
    SettingsGroup(title = "Cluster presence", icon = Icons.Default.Cloud) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Text("Status", style = MaterialTheme.typography.bodyMedium)
            QlhSemanticStatusChip(
                text = label,
                status = when (snapshot.state) {
                    AndroidPresenceState.ONLINE -> "online"
                    AndroidPresenceState.REGISTERING -> "registering"
                    AndroidPresenceState.BACKING_OFF -> "offline"
                    AndroidPresenceState.OFFLINE -> "offline"
                    AndroidPresenceState.STOPPED -> "stopped"
                },
            )
        }
        if (!snapshot.lastErrorCode.isNullOrBlank()) {
            Text(
                text = "Code: ${snapshot.lastErrorCode}",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.error,
            )
        }
    }
}

@Composable
private fun ClusterOverviewGroup(
    state: ClusterOverviewUiState,
    onRefresh: () -> Unit,
) {
    val snapshot = state.snapshot
    val summary = when {
        state.loading -> "正在读取主节点状态"
        snapshot != null -> "${snapshot.reachableNodes}/${snapshot.totalNodes} 个节点可达"
        state.error != null -> "暂时无法读取"
        else -> "尚未读取"
    }

    CollapsibleSettingsGroup(
        title = "集群概览",
        summary = summary,
        icon = Icons.Default.Cloud,
        testTag = "cluster_overview_details",
    ) {
        if (state.loading) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                CircularProgressIndicator(modifier = Modifier.size(16.dp), strokeWidth = 2.dp)
                Spacer(modifier = Modifier.width(8.dp))
                Text("正在刷新", style = MaterialTheme.typography.bodySmall)
            }
        }
        state.error?.let { error ->
            Text(
                text = "集群状态不可用：$error",
                color = MaterialTheme.colorScheme.error,
                style = MaterialTheme.typography.bodySmall,
                maxLines = 2,
                overflow = TextOverflow.Ellipsis,
            )
        }
        snapshot?.let { value ->
            Text(
                text = "运行：${if (value.running) "已启动" else "未启动"} · ${if (value.nodesReady) "节点已就绪" else "节点待就绪"}",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
            if (value.runMode.isNotBlank()) {
                Text(
                    text = "模式：${value.runMode}",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
            value.currentTaskId?.let { taskId ->
                val taskState = value.currentTaskState?.ifBlank { "unknown" } ?: "unknown"
                val elapsed = value.currentTaskElapsedSeconds?.let { " · ${it}s" }.orEmpty()
                Text(
                    text = "当前任务：$taskState$elapsed",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    maxLines = 1,
                    overflow = TextOverflow.Ellipsis,
                    modifier = Modifier.testTag("cluster_overview_task_$taskId"),
                )
            }
            if (value.nodes.isEmpty()) {
                Text(
                    text = "主节点未返回节点明细",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            } else {
                value.nodes.take(8).forEach { node ->
                    SettingRow(
                        title = node.hostname.ifBlank { node.nodeId },
                        subtitle = "${node.role} · ${node.nodeType} · ${node.networkType} · 任务 ${node.taskCount} · 错误 ${node.errorCount}",
                        modifier = Modifier.testTag("cluster_overview_node_${node.nodeId}"),
                        trailing = {
                            QlhSemanticStatusChip(
                                text = node.state,
                                status = if (node.reachable) "online" else node.state,
                                showDot = node.reachable,
                            )
                        },
                    )
                }
                if (value.nodes.size > 8) {
                    Text(
                        text = "其余 ${value.nodes.size - 8} 个节点未在移动端展开；请使用 PC 控制台处理详细运维。",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }
            }
        }
        OutlinedButton(
            onClick = onRefresh,
            enabled = !state.loading,
            modifier = Modifier
                .fillMaxWidth()
                .testTag("cluster_overview_refresh"),
        ) {
            Icon(Icons.Default.Refresh, contentDescription = null, modifier = Modifier.size(16.dp))
            Spacer(Modifier.width(6.dp))
            Text("刷新集群状态")
        }
    }
}

@Composable
private fun ModelFleetOverviewGroup(
    state: ModelFleetUiState,
    onRefresh: () -> Unit,
) {
    val snapshot = state.snapshot
    val summary = when {
        state.loading -> "正在读取主节点模型"
        snapshot != null -> "注册 ${snapshot.registryCount} · 资产 ${snapshot.localAssetCount} · 已验签 GGUF ${snapshot.verifiedGgufCount}"
        state.error != null -> "暂时无法读取"
        else -> "尚未读取"
    }

    CollapsibleSettingsGroup(
        title = "模型舰队",
        summary = summary,
        icon = Icons.Default.Memory,
        testTag = "model_fleet_details",
    ) {
        if (state.loading) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                CircularProgressIndicator(modifier = Modifier.size(16.dp), strokeWidth = 2.dp)
                Spacer(modifier = Modifier.width(8.dp))
                Text("正在刷新", style = MaterialTheme.typography.bodySmall)
            }
        }
        state.error?.let { error ->
            Text(
                text = "模型舰队不可用：$error",
                color = MaterialTheme.colorScheme.error,
                style = MaterialTheme.typography.bodySmall,
                maxLines = 2,
                overflow = TextOverflow.Ellipsis,
            )
        }
        snapshot?.let { value ->
            val runtimeLabel = when {
                value.currentLoaded -> "已加载：${value.currentModelName.ifBlank { value.currentModelId ?: "当前模型" }}"
                value.pipelinePrepared -> "已准备流水线：${value.currentModelName.ifBlank { value.currentModelId ?: "当前模型" }}"
                else -> "主节点当前未加载模型"
            }
            Text(
                text = runtimeLabel,
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
            if (value.currentEngine.isNotBlank() || !value.currentQuantType.isNullOrBlank()) {
                Text(
                    text = listOf(value.currentEngine, value.currentQuantType.orEmpty())
                        .filter { it.isNotBlank() }
                        .joinToString(" · "),
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
            if (value.androidSelectedModelName.isNotBlank()) {
                val size = value.androidSelectedModelSizeBytes.takeIf { it > 0L }
                    ?.let { " · ${formatBytes(it)}" }
                    .orEmpty()
                Text(
                    text = "本机已选：${value.androidSelectedModelName}$size",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    maxLines = 1,
                    overflow = TextOverflow.Ellipsis,
                )
            } else {
                Text(
                    text = "本机尚未选择 GGUF 模型",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
            if (value.entries.isEmpty()) {
                Text(
                    text = "主节点未返回可展示的模型条目",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            } else {
                value.entries.take(8).forEach { entry ->
                    ModelFleetEntryRow(entry)
                }
                if (value.entries.size > 8) {
                    Text(
                        text = "其余 ${value.entries.size - 8} 项未在移动端展开；下载治理、注册表编辑与部署操作请使用 PC 控制台。",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }
            }
        }
        OutlinedButton(
            onClick = onRefresh,
            enabled = !state.loading,
            modifier = Modifier
                .fillMaxWidth()
                .testTag("model_fleet_refresh"),
        ) {
            Icon(Icons.Default.Refresh, contentDescription = null, modifier = Modifier.size(16.dp))
            Spacer(Modifier.width(6.dp))
            Text("刷新模型舰队")
        }
    }
}

@Composable
private fun ModelFleetEntryRow(entry: ModelFleetEntry) {
    val details = buildList {
        if (entry.modelType.isNotBlank()) add(entry.modelType)
        if (entry.formats.isNotEmpty()) add(entry.formats.joinToString("/"))
        if (entry.totalBytes > 0L) add(formatBytes(entry.totalBytes))
        if (entry.sources.isNotEmpty()) add(entry.sources.joinToString(" · "))
    }.joinToString(" · ")
    val statusColors = when (entry.status) {
        ModelFleetStatus.ACTIVE -> qlhNeonGreen() to qlhOnNeonGreen()
        ModelFleetStatus.AVAILABLE -> qlhBrandGoldContainer() to qlhOnBrandGoldContainer()
        ModelFleetStatus.UNVERIFIED -> MaterialTheme.colorScheme.tertiaryContainer to MaterialTheme.colorScheme.onTertiaryContainer
        ModelFleetStatus.MISSING -> MaterialTheme.colorScheme.errorContainer to MaterialTheme.colorScheme.onErrorContainer
    }
    SettingRow(
        title = entry.name,
        subtitle = details.ifBlank { entry.modelId },
        modifier = Modifier.testTag("model_fleet_entry_${entry.modelId}"),
        trailing = {
            StatusChip(
                text = when (entry.status) {
                    ModelFleetStatus.ACTIVE -> "运行中"
                    ModelFleetStatus.AVAILABLE -> "可用"
                    ModelFleetStatus.UNVERIFIED -> "待验证"
                    ModelFleetStatus.MISSING -> "缺失"
                },
                containerColor = statusColors.first,
                contentColor = statusColors.second,
                showDot = entry.status == ModelFleetStatus.ACTIVE || entry.status == ModelFleetStatus.AVAILABLE,
            )
        },
    )
}

@Composable
private fun AuditOverviewGroup(
    state: AuditUiState,
    onRefresh: () -> Unit,
) {
    val snapshot = state.snapshot
    val summary = when {
        state.loading -> "姝ｅ湪璇诲彇"
        snapshot != null -> "娲诲姩 ${snapshot.workflows.size} · 澶嶆牳 ${snapshot.reviews.size}"
        state.error != null -> "鏆傛椂鏃犳硶璇诲彇"
        else -> "灏氭湭璇诲彇"
    }
    CollapsibleSettingsGroup(
        title = "瀹¤涓庢椿鍔?",
        summary = summary,
        icon = Icons.Default.Description,
        testTag = "audit_overview_details",
    ) {
        if (state.loading) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                CircularProgressIndicator(modifier = Modifier.size(16.dp), strokeWidth = 2.dp)
                Spacer(modifier = Modifier.width(8.dp))
                Text("姝ｅ湪鍒锋柊", style = MaterialTheme.typography.bodySmall)
            }
        }
        state.error?.let { error ->
            Text(
                text = "瀹¤璧勬枡涓嶅彲鐢細$error",
                color = MaterialTheme.colorScheme.error,
                style = MaterialTheme.typography.bodySmall,
                maxLines = 2,
                overflow = TextOverflow.Ellipsis,
            )
        }
        snapshot?.let { value ->
            Text(
                text = "工作流 ${value.workflows.size} 条 · 复核票 ${value.reviews.size} 条",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
            value.workflows.take(8).forEach { workflow ->
                AuditWorkflowRow(workflow)
            }
            value.reviews.take(8).forEach { ticket ->
                AuditReviewRow(ticket)
            }
            if (value.workflows.isEmpty() && value.reviews.isEmpty()) {
                Text(
                    text = "暂无可显示的活动或复核摘要",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
        }
        OutlinedButton(
            onClick = onRefresh,
            enabled = !state.loading,
            modifier = Modifier
                .fillMaxWidth()
                .testTag("audit_overview_refresh"),
        ) {
            Icon(Icons.Default.Refresh, contentDescription = null, modifier = Modifier.size(16.dp))
            Spacer(Modifier.width(6.dp))
            Text("刷新审计摘要")
        }
    }
}

@Composable
private fun AuditWorkflowRow(workflow: AuditWorkflow) {
    val stageSummary = "阶段 ${workflow.completedStageCount}/${workflow.stageCount} · 尝试 ${workflow.attemptCount} · 重试 ${workflow.retryCount}"
    SettingRow(
        title = workflow.template,
        subtitle = "$stageSummary · ${workflow.workflowId}",
        modifier = Modifier.testTag("audit_workflow_${workflow.workflowId}"),
        trailing = {
            QlhSemanticStatusChip(
                text = workflow.state,
            )
        },
    )
    workflow.stages.take(4).forEach { stage ->
        val attemptKinds = stage.attempts.map { it.providerKind }.distinct().filter { it != "unknown" }
        val errorCode = stage.errorCode.ifBlank { stage.attempts.firstOrNull()?.errorCode.orEmpty() }
        SettingRow(
            title = stage.stageType,
            subtitle = "${stage.state} · 尝试 ${stage.attempts.size} · 重试 ${stage.retryCount}" +
                (if (attemptKinds.isEmpty()) "" else " · ${attemptKinds.joinToString(", ")}") +
                (if (errorCode.isBlank()) "" else " · 错误 $errorCode"),
            modifier = Modifier.testTag("audit_stage_${workflow.workflowId}_${stage.stageId}"),
            trailing = {
                QlhSemanticStatusChip(text = stage.state, showDot = stage.state == "running")
            },
        )
    }
}

@Composable
private fun AuditReviewRow(ticket: AuditReviewTicket) {
    SettingRow(
        title = "复核 ${ticket.targetNodeId}",
        subtitle = "${ticket.ticketId} · 票数 ${ticket.voteCount} · 分数 ${ticket.score}",
        modifier = Modifier.testTag("audit_review_${ticket.ticketId}"),
        trailing = {
            QlhSemanticStatusChip(
                text = ticket.status,
            )
        },
    )
}

@Composable
private fun AuthControlGroup(
    state: AuthControlUiState,
    onRefresh: () -> Unit,
    onLogin: (String, String?, String?) -> Unit,
    onLogout: () -> Unit,
) {
    var username by remember { mutableStateOf("") }
    var code by remember { mutableStateOf("") }
    var recoveryCode by remember { mutableStateOf("") }
    var useRecovery by remember { mutableStateOf(false) }

    val summary = when {
        state.loading -> "正在读取认证能力"
        state.account != null -> "已登录 · ${state.account.role}"
        state.capability == null -> "尚未读取"
        !state.capability.canAuthenticate -> "认证控制面未启用"
        state.error != null -> "会话待确认"
        else -> "需要 Auth App 验证"
    }

    CollapsibleSettingsGroup(
        title = "账户与 Auth App",
        summary = summary,
        icon = Icons.Default.AccountCircle,
        testTag = "auth_control_details",
    ) {
        if (state.loading) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                CircularProgressIndicator(modifier = Modifier.size(16.dp), strokeWidth = 2.dp)
                Spacer(modifier = Modifier.width(8.dp))
                Text("正在确认认证控制面", style = MaterialTheme.typography.bodySmall)
            }
        }
        state.error?.let { error ->
            Text(
                text = error,
                color = MaterialTheme.colorScheme.error,
                style = MaterialTheme.typography.bodySmall,
                maxLines = 2,
                overflow = TextOverflow.Ellipsis,
            )
        }
        state.capability?.let { capability ->
            val mode = capability.mode.ifBlank { "unknown" }
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Text(
                    text = if (capability.canAuthenticate) "认证模式：$mode" else "主节点未启用用户认证",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    modifier = Modifier.weight(1f),
                )
                QlhSemanticStatusChip(
                    text = if (capability.canAuthenticate) "ready" else "disabled",
                    status = if (capability.canAuthenticate) "ready" else "disabled",
                    showDot = false,
                )
            }
            if (capability.reasonCode.isNotBlank() && !capability.canAuthenticate) {
                Text(
                    text = "认证状态：${capability.reasonCode}",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    maxLines = 2,
                    overflow = TextOverflow.Ellipsis,
                )
            }
        }

        state.account?.let { account ->
            SettingRow(
                title = account.displayName,
                subtitle = "${account.username} · 角色 ${account.role} · 到期 ${account.expiresAt}",
                modifier = Modifier.testTag("auth_account_summary"),
                trailing = {
                    QlhSemanticStatusChip(
                        text = account.role,
                        status = if (account.role.equals("owner", true) || account.role.equals("admin", true)) {
                            "active"
                        } else {
                            "unknown"
                        },
                        showDot = false,
                    )
                },
            )
            OutlinedButton(
                onClick = onLogout,
                enabled = !state.busy,
                modifier = Modifier
                    .fillMaxWidth()
                    .testTag("auth_logout"),
            ) {
                if (state.busy) {
                    CircularProgressIndicator(modifier = Modifier.size(16.dp), strokeWidth = 2.dp)
                } else {
                    Icon(Icons.AutoMirrored.Filled.Logout, contentDescription = null, modifier = Modifier.size(16.dp))
                }
                Spacer(Modifier.width(6.dp))
                Text("退出当前会话")
            }
        } ?: state.capability?.takeIf { it.canAuthenticate }?.let {
            Text(
                text = "使用 Authenticator 应用生成 6 位验证码；恢复码仅用于无法访问 Auth App 时的一次性登录。",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
            OutlinedTextField(
                value = username,
                onValueChange = { username = it },
                label = { Text("用户名") },
                singleLine = true,
                modifier = Modifier
                    .fillMaxWidth()
                    .testTag("auth_username"),
            )
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                if (!useRecovery) {
                    Button(
                        onClick = { useRecovery = false },
                        modifier = Modifier.weight(1f),
                    ) { Text("Auth App") }
                    OutlinedButton(
                        onClick = { useRecovery = true },
                        modifier = Modifier.weight(1f),
                    ) { Text("恢复码") }
                } else {
                    OutlinedButton(
                        onClick = { useRecovery = false },
                        modifier = Modifier.weight(1f),
                    ) { Text("Auth App") }
                    Button(
                        onClick = { useRecovery = true },
                        modifier = Modifier.weight(1f),
                    ) { Text("恢复码") }
                }
            }
            OutlinedTextField(
                value = if (useRecovery) recoveryCode else code,
                onValueChange = { value -> if (useRecovery) recoveryCode = value else code = value },
                label = { Text(if (useRecovery) "恢复码" else "Auth App 验证码") },
                singleLine = true,
                visualTransformation = PasswordVisualTransformation(),
                keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number),
                modifier = Modifier
                    .fillMaxWidth()
                    .testTag(if (useRecovery) "auth_recovery" else "auth_code"),
            )
            Button(
                onClick = {
                    val submittedCode = code.takeIf { !useRecovery && it.isNotBlank() }
                    val submittedRecovery = recoveryCode.takeIf { useRecovery && it.isNotBlank() }
                    onLogin(username.trim(), submittedCode, submittedRecovery)
                    code = ""
                    recoveryCode = ""
                },
                enabled = !state.busy && username.isNotBlank() &&
                    ((!useRecovery && code.isNotBlank()) || (useRecovery && recoveryCode.isNotBlank())),
                modifier = Modifier
                    .fillMaxWidth()
                    .testTag("auth_login"),
            ) {
                if (state.busy) CircularProgressIndicator(modifier = Modifier.size(16.dp), strokeWidth = 2.dp)
                Spacer(Modifier.width(6.dp))
                Text("登录")
            }
        }

        OutlinedButton(
            onClick = onRefresh,
            enabled = !state.loading && !state.busy,
            modifier = Modifier
                .fillMaxWidth()
                .testTag("auth_refresh"),
        ) {
            Icon(Icons.Default.Refresh, contentDescription = null, modifier = Modifier.size(16.dp))
            Spacer(Modifier.width(6.dp))
            Text("刷新认证状态")
        }
    }
}

@Composable
private fun ManagementControlGroup(
    state: ManagementUiState,
    onRefresh: () -> Unit,
    onRevokeUser: (ManagedUserSnapshot) -> Unit,
    onRevokeBinding: (ManagedBindingSnapshot) -> Unit,
) {
    var pendingUser by remember { mutableStateOf<ManagedUserSnapshot?>(null) }
    var pendingBinding by remember { mutableStateOf<ManagedBindingSnapshot?>(null) }
    var showAllUsers by rememberSaveable { mutableStateOf(false) }
    var showAllAudit by rememberSaveable { mutableStateOf(false) }
    val managerActions = state.summary?.actions.orEmpty()
    val reviewPending = state.summary?.reviewAdminAuthPending == true
    val visibleUsers = if (showAllUsers) state.users else state.users.take(8)
    val visibleAudit = if (showAllAudit) state.audit else state.audit.take(8)
    val summary = when {
        state.loading -> "正在读取管理控制面"
        state.summary == null && state.error != null -> "管理控制面不可用"
        reviewPending -> "成员 ${state.users.size} · 入群审批待授权"
        else -> "成员 ${state.users.size} · 审计 ${state.audit.size}"
    }

    CollapsibleSettingsGroup(
        title = "主节点管理",
        summary = summary,
        icon = Icons.Default.AccountCircle,
        testTag = "management_control_details",
    ) {
        state.error?.let { error ->
            Text(
                text = error,
                color = MaterialTheme.colorScheme.error,
                style = MaterialTheme.typography.bodySmall,
                maxLines = 2,
                overflow = TextOverflow.Ellipsis,
            )
        }
        state.summary?.let { matrix ->
            Text(
                text = "当前角色：${matrix.role} · 策略 ${matrix.policyVersion}",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
            Text(
                text = "成员 ${matrix.counts["users"] ?: 0} · Tailnet 绑定 ${matrix.counts["bindings"] ?: 0} · 审计 ${if (matrix.auditAvailable) "可用" else "不可用"}",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
            if (reviewPending) {
                Text(
                    text = "入群审批暂不可用：review_admin 授权契约尚未迁移。此页面不会伪造审批按钮。",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
            managerActions.entries.forEach { (action, rule) ->
                SettingRow(
                    title = rule.description.ifBlank { action },
                    subtitle = "允许：${if (rule.allowed) "是" else "否"} · 二次确认：${if (rule.confirmRequired) "是" else "否"} · 审计：${if (rule.audited) "是" else "否"}",
                    trailing = {
                        QlhSemanticStatusChip(
                            text = if (rule.allowed) "allowed" else "denied",
                            status = if (rule.allowed) "enabled" else "denied",
                            showDot = false,
                        )
                    },
                )
            }
        }

        Text("成员清单", style = MaterialTheme.typography.titleSmall, color = MaterialTheme.colorScheme.onSurface)
        if (state.users.isEmpty() && state.error == null) {
            Text("没有可显示的成员", style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
        }
        visibleUsers.forEach { user ->
            SettingRow(
                title = user.displayName,
                subtitle = "${user.username} · ${user.role} · ${user.status} · 版本 ${user.aggregateVersion}",
                modifier = Modifier.testTag("managed_user_${user.userId}"),
                trailing = {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        QlhSemanticStatusChip(text = user.status, showDot = false)
                        if (user.status != "revoked" && user.aggregateVersion > 0) {
                            Spacer(Modifier.width(6.dp))
                            IconButton(
                                onClick = { pendingUser = user },
                                enabled = state.busyAction == null,
                                modifier = Modifier.testTag("managed_user_revoke_${user.userId}"),
                            ) {
                                Icon(Icons.Default.Delete, contentDescription = "撤销成员")
                            }
                        }
                    }
                },
            )
            user.bindings.forEach { binding ->
                SettingRow(
                    title = "Tailnet ${binding.tailnetId.ifBlank { "unknown" }}",
                    subtitle = "${binding.tailscaleUserId.ifBlank { "unknown user" }} · 节点 ${binding.nodeId.ifBlank { "unknown" }} · ${binding.state}",
                    modifier = Modifier
                        .padding(start = 16.dp)
                        .testTag("managed_binding_${binding.bindingId}"),
                    trailing = {
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            QlhSemanticStatusChip(text = binding.state, showDot = false)
                            if (binding.state != "revoked") {
                                Spacer(Modifier.width(6.dp))
                                IconButton(
                                    onClick = { pendingBinding = binding },
                                    enabled = state.busyAction == null,
                                    modifier = Modifier.testTag("managed_binding_revoke_${binding.bindingId}"),
                                ) {
                                    Icon(Icons.Default.Delete, contentDescription = "撤销绑定")
                                }
                            }
                        }
                    },
                )
            }
        }
        if (state.users.size > visibleUsers.size) {
            TextButton(onClick = { showAllUsers = true }, modifier = Modifier.fillMaxWidth()) {
                Text("查看全部成员 (${state.users.size})")
            }
        } else if (showAllUsers && state.users.size > 8) {
            TextButton(onClick = { showAllUsers = false }, modifier = Modifier.fillMaxWidth()) {
                Text("收起成员列表")
            }
        }

        Text("最近管理审计", style = MaterialTheme.typography.titleSmall, color = MaterialTheme.colorScheme.onSurface)
        visibleAudit.take(MAX_MANAGEMENT_AUDIT_EVENTS).forEach { event ->
            SettingRow(
                title = event.eventType.ifBlank { "管理事件" },
                subtitle = "${event.outcome.ifBlank { "unknown" }} · ${event.createdAt}",
                modifier = Modifier.testTag("management_audit_${event.eventId}"),
                trailing = {
                    QlhSemanticStatusChip(text = event.outcome.ifBlank { "unknown" }, showDot = false)
                },
            )
        }
        if (state.audit.size > visibleAudit.size) {
            TextButton(onClick = { showAllAudit = true }, modifier = Modifier.fillMaxWidth()) {
                Text("查看全部审计 (${state.audit.size})")
            }
        } else if (showAllAudit && state.audit.size > 8) {
            TextButton(onClick = { showAllAudit = false }, modifier = Modifier.fillMaxWidth()) {
                Text("收起审计列表")
            }
        }
        OutlinedButton(
            onClick = onRefresh,
            enabled = !state.loading && state.busyAction == null,
            modifier = Modifier
                .fillMaxWidth()
                .testTag("management_refresh"),
        ) {
            if (state.loading) CircularProgressIndicator(modifier = Modifier.size(16.dp), strokeWidth = 2.dp)
            else Icon(Icons.Default.Refresh, contentDescription = null, modifier = Modifier.size(16.dp))
            Spacer(Modifier.width(6.dp))
            Text("刷新管理摘要")
        }
    }

    pendingUser?.let { user ->
        AlertDialog(
            onDismissRequest = { if (state.busyAction == null) pendingUser = null },
            title = { Text("确认撤销成员") },
            text = { Text("将撤销 ${user.displayName} 的账户。服务端会要求一次性二次确认并写入审计，操作不可在此页面回滚。") },
            confirmButton = {
                TextButton(
                    onClick = {
                        pendingUser = null
                        onRevokeUser(user)
                    },
                    enabled = state.busyAction == null,
                ) { Text("确认撤销") }
            },
            dismissButton = {
                TextButton(onClick = { pendingUser = null }, enabled = state.busyAction == null) { Text("取消") }
            },
        )
    }
    pendingBinding?.let { binding ->
        AlertDialog(
            onDismissRequest = { if (state.busyAction == null) pendingBinding = null },
            title = { Text("确认撤销 Tailnet 绑定") },
            text = { Text("将撤销该用户的 Tailnet 绑定。服务端会要求一次性二次确认并写入审计。") },
            confirmButton = {
                TextButton(
                    onClick = {
                        pendingBinding = null
                        onRevokeBinding(binding)
                    },
                    enabled = state.busyAction == null,
                ) { Text("确认撤销") }
            },
            dismissButton = {
                TextButton(onClick = { pendingBinding = null }, enabled = state.busyAction == null) { Text("取消") }
            },
        )
    }
}

@Composable
private fun DiagnosticsGroup(
    state: DiagnosticsUiState,
    onRefresh: () -> Unit,
    onUpload: () -> Unit,
) {
    SettingsGroup(title = "连接诊断", icon = Icons.Default.Info) {
        state.health?.let { report ->
            Text(
                text = "网络：${report.localNetworkType}",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
            report.checks.forEach { check ->
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    Text(check.label, color = MaterialTheme.colorScheme.onSurface)
                    QlhSemanticStatusChip(
                        text = when (check.state) {
                            ConnectionHealthState.PASS -> "通过"
                            ConnectionHealthState.FAIL -> "失败"
                            ConnectionHealthState.SKIPPED -> "跳过"
                        } + (check.latencyMillis?.let { " · ${it} ms" } ?: ""),
                        status = when (check.state) {
                            ConnectionHealthState.PASS -> "pass"
                            ConnectionHealthState.FAIL -> "fail"
                            ConnectionHealthState.SKIPPED -> "skipped"
                        },
                        showDot = false,
                    )
                }
                if (check.detail.isNotBlank()) {
                    Text(
                        text = check.detail,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                        style = MaterialTheme.typography.bodySmall,
                        maxLines = 2,
                        overflow = TextOverflow.Ellipsis,
                    )
                }
            }
        }
        state.healthError?.let { error ->
            Text(error, color = MaterialTheme.colorScheme.error, style = MaterialTheme.typography.bodySmall)
        }
        state.uploadMessage?.let { message ->
            Text(message, color = qlhNeonGreen(), style = MaterialTheme.typography.bodySmall)
        }
        state.uploadError?.let { error ->
            Text(error, color = MaterialTheme.colorScheme.error, style = MaterialTheme.typography.bodySmall)
        }
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            OutlinedButton(
                onClick = onRefresh,
                enabled = !state.healthLoading && !state.uploadInProgress,
                modifier = Modifier.weight(1f),
            ) {
                if (state.healthLoading) {
                    CircularProgressIndicator(modifier = Modifier.size(16.dp), strokeWidth = 2.dp)
                } else {
                    Icon(Icons.Default.Refresh, contentDescription = null, modifier = Modifier.size(16.dp))
                }
                Spacer(Modifier.width(6.dp))
                Text("刷新")
            }
            OutlinedButton(
                onClick = onUpload,
                enabled = !state.uploadInProgress,
                modifier = Modifier.weight(1f),
            ) {
                if (state.uploadInProgress) {
                    CircularProgressIndicator(modifier = Modifier.size(16.dp), strokeWidth = 2.dp)
                }
                Spacer(Modifier.width(6.dp))
                Text("上报脱敏日志")
            }
        }
    }
}

@Composable
private fun AppUpdateGroup(
    state: AppUpdateUiState,
    onCheck: () -> Unit,
    onDownload: () -> Unit,
    onOpenInstallPermissionSettings: () -> Unit,
    onInstall: () -> Unit,
) {
    SettingsGroup(title = "应用更新", icon = Icons.Default.CloudDownload) {
        Text(
            text = "当前版本 ${BuildConfig.VERSION_NAME}",
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurface,
        )
        state.candidate?.let { candidate ->
            Text(
                text = "可用版本 ${candidate.asset.version} · ${formatBytes(candidate.asset.sizeBytes)}",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
        state.progress?.let { progress ->
            LinearProgressIndicator(
                progress = { progress.percent / 100f },
                modifier = Modifier.fillMaxWidth(),
            )
            Text(
                text = "${formatBytes(progress.downloadedBytes)} / ${formatBytes(progress.totalBytes)} (${progress.percent}%)",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
        state.message?.let { Text(it, color = MaterialTheme.colorScheme.primary, style = MaterialTheme.typography.bodySmall) }
        state.error?.let { Text(it, color = MaterialTheme.colorScheme.error, style = MaterialTheme.typography.bodySmall) }
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            OutlinedButton(
                onClick = onCheck,
                enabled = !state.checking && !state.downloading,
                modifier = Modifier.weight(1f),
            ) {
                if (state.checking) CircularProgressIndicator(modifier = Modifier.size(16.dp), strokeWidth = 2.dp)
                else Icon(Icons.Default.Refresh, contentDescription = null, modifier = Modifier.size(16.dp))
                Spacer(Modifier.width(6.dp))
                Text("检查更新")
            }
            if (state.candidate != null && !state.downloadedReady) {
                Button(
                    onClick = onDownload,
                    enabled = !state.downloading && !state.checking,
                    modifier = Modifier.weight(1f),
                ) {
                    Text(if (state.downloading) "下载中" else "下载并校验")
                }
            } else if (state.downloadedReady) {
                Button(
                    onClick = if (state.installPermissionGranted) onInstall else onOpenInstallPermissionSettings,
                    modifier = Modifier.weight(1f),
                ) {
                    Text(if (state.installPermissionGranted) "安装更新" else "允许安装")
                }
            }
        }
    }
}

@Composable
private fun ConnectionGroup(
    serverHost: String,
    serverPort: Int,
    onServerHostChange: (String) -> Unit,
    onServerPortChange: (Int) -> Unit,
    onConnectionTestSuccess: () -> Unit
) {
    val scope = rememberCoroutineScope()
    var isTesting by remember { mutableStateOf(false) }
    var connectionResult by remember { mutableStateOf<Boolean?>(null) }

    SettingsGroup(title = "主节点连接", icon = Icons.Default.NetworkCheck) {
        OutlinedTextField(
            value = serverHost,
            onValueChange = onServerHostChange,
            label = { Text("主节点地址") },
            placeholder = { Text("例如: 192.168.1.100") },
            modifier = Modifier.fillMaxWidth(),
            singleLine = true,
            shape = MaterialTheme.shapes.small,
            enabled = !isTesting
        )
        Spacer(modifier = Modifier.height(12.dp))
        OutlinedTextField(
            value = serverPort.toString(),
            onValueChange = { it.toIntOrNull()?.let(onServerPortChange) },
            label = { Text("端口") },
            modifier = Modifier.fillMaxWidth(),
            singleLine = true,
            shape = MaterialTheme.shapes.small,
            keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number),
            enabled = !isTesting
        )
        Spacer(modifier = Modifier.height(16.dp))

        Row(
            modifier = Modifier.fillMaxWidth(),
            verticalAlignment = Alignment.CenterVertically
        ) {
            // 测试连接按钮
            Button(
                onClick = {
                    isTesting = true
                    connectionResult = null
                    scope.launch {
                        val url = httpBaseUrl(serverHost, serverPort)
                        QlhLogger.i("Settings", "测试连接: $url")
                        val client = ApiClient(url)
                        val raw = client.testConnection()
                        connectionResult = raw.getOrDefault(false)
                        isTesting = false
                        raw.onSuccess {
                            QlhLogger.i("Settings", "连接测试成功: $url")
                            onConnectionTestSuccess()
                        }.onFailure { e ->
                            QlhLogger.w("Settings", "连接测试失败: $url — ${e.message ?: e.javaClass.simpleName}")
                        }
                    }
                },
                modifier = Modifier.weight(1f),
                enabled = !isTesting && serverHost.isNotBlank()
            ) {
                if (isTesting) {
                    CircularProgressIndicator(
                        modifier = Modifier.size(18.dp),
                        strokeWidth = 2.dp,
                        color = MaterialTheme.colorScheme.onPrimary
                    )
                    Spacer(modifier = Modifier.width(8.dp))
                    Text("测试中…")
                } else {
                    Text("测试连接")
                }
            }

            // 连接状态角标
            connectionResult?.let { success ->
                Spacer(modifier = Modifier.width(12.dp))
                if (success) {
                    QlhSemanticStatusChip(
                        text = "已连接",
                        status = "connected",
                        showDot = false,
                    )
                } else {
                    QlhSemanticStatusChip(
                        text = "无法连接",
                        status = "offline",
                        showDot = false,
                    )
                }
            }
        }
    }
}

// ================================================================
// 推理模式
// ================================================================

@Composable
private fun InferenceModeGroup(
    inferenceMode: String,
    isLite: Boolean,
    onInferenceModeChange: (String) -> Unit
) {
    var showModeDialog by remember { mutableStateOf(false) }

    SettingsGroup(title = "推理模式", icon = Icons.Default.Cloud) {
        SettingRow(
            title = if (inferenceMode == "thin") "全无 (远程推理)" else "全有 (本地推理)",
            subtitle = if (inferenceMode == "thin") {
                "请求发送给 PC 主节点，本机不计算"
            } else {
                "本机加载 GGUF 模型，离线完整推理"
            },
            trailing = {
                Button(
                    onClick = { showModeDialog = true },
                    enabled = !isLite,
                    colors = ButtonDefaults.buttonColors(
                        containerColor = MaterialTheme.colorScheme.secondaryContainer,
                        contentColor = MaterialTheme.colorScheme.onSecondaryContainer
                    )
                ) {
                    Text(if (isLite) "极简版" else "切换")
                }
            }
        )
    }

    // ---- 模式切换对话框 ----
    if (showModeDialog) {
        AlertDialog(
            onDismissRequest = { showModeDialog = false },
            title = { Text("选择推理模式") },
            text = {
                Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
                    ModeOptionRow(
                        selected = inferenceMode == "thin",
                        title = "全无模式 (远程推理)",
                        desc = "请求发送给 PC 主节点，本机不计算",
                        onClick = {
                            onInferenceModeChange("thin")
                            showModeDialog = false
                        }
                    )
                    if (!isLite) {
                        ModeOptionRow(
                            selected = inferenceMode == "full",
                            title = "全有模式 (本地推理)",
                            desc = "本机加载 GGUF 模型，离线完整推理",
                            onClick = {
                                onInferenceModeChange("full")
                                showModeDialog = false
                            }
                        )
                    }
                }
            },
            confirmButton = {
                TextButton(onClick = { showModeDialog = false }) {
                    Text("取消")
                }
            }
        )
    }
}

@Composable
private fun ModeOptionRow(
    selected: Boolean,
    title: String,
    desc: String,
    onClick: () -> Unit
) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .clip(MaterialTheme.shapes.small)
            .clickable(onClick = onClick)
            .padding(vertical = 8.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        RadioButton(selected = selected, onClick = onClick)
        Spacer(modifier = Modifier.width(4.dp))
        Column {
            Text(
                text = title,
                style = MaterialTheme.typography.titleSmall,
                color = if (selected) {
                    MaterialTheme.colorScheme.primary
                } else {
                    MaterialTheme.colorScheme.onSurface
                }
            )
            Text(
                text = desc,
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
        }
    }
}

// ================================================================
// 推理参数
// ================================================================

@Composable
private fun InferenceParamsGroup(
    maxTokens: Int,
    temperature: Float,
    topP: Float,
    contextSize: Int,
    showThinking: Boolean,
    onMaxTokensChange: (Int) -> Unit,
    onTemperatureChange: (Float) -> Unit,
    onTopPChange: (Float) -> Unit,
    onContextSizeChange: (Int) -> Unit,
    onShowThinkingChange: (Boolean) -> Unit
) {
    SettingsGroup(title = "推理参数", icon = Icons.Default.Tune) {
        SettingRow(
            title = "显示思考过程",
            subtitle = "仅影响对话内容的展示，不改变模型执行模式",
            trailing = {
                Switch(
                    checked = showThinking,
                    onCheckedChange = onShowThinkingChange,
                    modifier = Modifier.testTag("settings_show_thinking")
                )
            }
        )

        HorizontalDivider(color = MaterialTheme.colorScheme.outlineVariant.copy(alpha = 0.6f))

        // Max Tokens
        var maxTokensText by remember(maxTokens) {
            mutableStateOf(maxTokens.toString())
        }
        OutlinedTextField(
            value = maxTokensText,
            onValueChange = {
                maxTokensText = it
                it.toIntOrNull()?.let { v -> if (v in 1..8192) onMaxTokensChange(v) }
            },
            label = { Text("最大 Token 数") },
            modifier = Modifier.fillMaxWidth(),
            singleLine = true,
            shape = MaterialTheme.shapes.small,
            keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number)
        )

        Spacer(modifier = Modifier.height(16.dp))

        // Temperature
        ParamLabel(label = "温度", value = "%.1f".format(temperature))
        Slider(
            value = temperature,
            onValueChange = onTemperatureChange,
            valueRange = 0f..2f,
            steps = 19,
            colors = SliderDefaults.colors(
                thumbColor = MaterialTheme.colorScheme.primary,
                activeTrackColor = MaterialTheme.colorScheme.primary
            )
        )

        Spacer(modifier = Modifier.height(8.dp))

        // Top-P
        ParamLabel(label = "Top-P", value = "%.2f".format(topP))
        Slider(
            value = topP,
            onValueChange = onTopPChange,
            valueRange = 0f..1f,
            steps = 9,
            colors = SliderDefaults.colors(
                thumbColor = MaterialTheme.colorScheme.primary,
                activeTrackColor = MaterialTheme.colorScheme.primary
            )
        )

        Spacer(modifier = Modifier.height(8.dp))

        // 上下文长度
        ParamLabel(label = "上下文长度", value = "$contextSize")
        Slider(
            value = contextSize.toFloat(),
            onValueChange = { onContextSizeChange(it.toInt()) },
            valueRange = 512f..4096f,
            steps = 6,  // 512, 1024, 1536, 2048, 2560, 3072, 3584, 4096
            colors = SliderDefaults.colors(
                thumbColor = MaterialTheme.colorScheme.primary,
                activeTrackColor = MaterialTheme.colorScheme.primary
            )
        )
        Text(
            text = "更大的上下文可以处理更长的对话历史，但会增加内存占用和首次加载时间",
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant
        )
    }
}

@Composable
private fun ParamLabel(label: String, value: String) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.CenterVertically
    ) {
        Text(
            text = label,
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurface
        )
        Text(
            text = value,
            style = MaterialTheme.typography.labelLarge,
            color = MaterialTheme.colorScheme.primary
        )
    }
}

// ================================================================
// 模型管理
// ================================================================

@Composable
private fun RemoteModelPanel(
    modelTreeUri: String,
    models: List<GgufModelInfo>,
    loading: Boolean,
    activeModelName: String?,
    progress: ModelManager.DownloadProgress?,
    message: String?,
    onRefresh: () -> Unit,
    onDownload: (GgufModelInfo) -> Unit,
) {
    val busy = activeModelName != null
    Column(
        modifier = Modifier.testTag("remote_model_panel"),
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Text(
                text = "从当前主节点下载 GGUF 到已授权目录",
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
            OutlinedButton(
                onClick = onRefresh,
                enabled = !loading && !busy,
                modifier = Modifier.testTag("remote_model_refresh"),
            ) {
                if (loading) {
                    CircularProgressIndicator(modifier = Modifier.size(16.dp), strokeWidth = 2.dp)
                } else {
                    Icon(Icons.Default.Refresh, contentDescription = null, modifier = Modifier.size(16.dp))
                }
                Spacer(modifier = Modifier.width(6.dp))
                Text("刷新")
            }
        }

        if (modelTreeUri.isBlank()) {
            Text(
                text = "请先在模型管理中选择存储目录。",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.error,
            )
        }

        activeModelName?.let { filename ->
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .testTag("remote_model_progress"),
                verticalArrangement = Arrangement.spacedBy(6.dp),
            ) {
                Text(
                    text = filename,
                    style = MaterialTheme.typography.bodyMedium,
                    fontWeight = FontWeight.SemiBold,
                    maxLines = 1,
                    overflow = TextOverflow.Ellipsis,
                )
                val current = progress
                if (current != null) {
                    LinearProgressIndicator(
                        progress = { current.percent.coerceIn(0, 100) / 100f },
                        modifier = Modifier.fillMaxWidth(),
                        color = if (current.phase == "completed") qlhNeonGreen() else qlhBrandGold(),
                    )
                    Text(
                        text = when (current.phase) {
                            "verifying" -> "正在校验 SHA-256"
                            "completed" -> "校验完成"
                            else -> "${formatBytes(current.downloadedBytes)} / ${formatBytes(current.totalBytes)} (${current.percent}%)"
                        },
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                } else {
                    LinearProgressIndicator(modifier = Modifier.fillMaxWidth())
                    Text("准备下载", style = MaterialTheme.typography.bodySmall)
                }
            }
        }

        if (models.isNotEmpty()) {
            HorizontalDivider(color = MaterialTheme.colorScheme.outlineVariant.copy(alpha = 0.6f))
            Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
                models.forEach { model ->
                    Surface(
                        modifier = Modifier
                            .fillMaxWidth()
                            .testTag("remote_model_${model.filename}"),
                        color = MaterialTheme.colorScheme.surfaceContainerLow,
                        shape = MaterialTheme.shapes.small,
                        border = BorderStroke(1.dp, MaterialTheme.colorScheme.outlineVariant.copy(alpha = 0.64f)),
                    ) {
                        Row(
                            modifier = Modifier.padding(horizontal = 10.dp, vertical = 8.dp),
                            horizontalArrangement = Arrangement.spacedBy(8.dp),
                            verticalAlignment = Alignment.CenterVertically,
                        ) {
                            Column(modifier = Modifier.weight(1f)) {
                                Text(
                                    text = model.filename,
                                    style = MaterialTheme.typography.bodyMedium,
                                    maxLines = 2,
                                    overflow = TextOverflow.Ellipsis,
                                )
                                Text(
                                    text = "${formatBytes(model.sizeBytes)} · SHA-256 ${model.sha256.take(12)}…",
                                    style = MaterialTheme.typography.bodySmall,
                                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                                )
                            }
                            OutlinedButton(
                                onClick = { onDownload(model) },
                                enabled = modelTreeUri.isNotBlank() && !busy && !loading,
                                modifier = Modifier.testTag("remote_model_download_${model.filename}"),
                                colors = androidx.compose.material3.ButtonDefaults.outlinedButtonColors(
                                    contentColor = qlhBrandGold(),
                                ),
                            ) {
                                Icon(Icons.Default.CloudDownload, contentDescription = null, modifier = Modifier.size(18.dp))
                                Spacer(modifier = Modifier.width(6.dp))
                                Text("下载")
                            }
                        }
                    }
                }
            }
        }

        message?.let {
            Text(
                text = it,
                style = MaterialTheme.typography.bodySmall,
                color = if (it.contains("失败") || it.contains("无效")) {
                    MaterialTheme.colorScheme.error
                } else {
                    qlhNeonGreen()
                },
                modifier = Modifier.testTag("remote_model_message"),
            )
        }
    }
}

@Composable
private fun ModelManagementPanel(
    modelTreeUri: String,
    selectedModelUri: String,
    modelStorageMode: String,
    availableModels: List<ModelManager.ModelDocument>,
    selectedModelName: String,
    selectedModelSizeBytes: Long,
    isScanningModels: Boolean,
    modelMessage: String?,
    onChooseModelDirectory: () -> Unit,
    onRefreshModels: () -> Unit,
    onModelSelected: (ModelManager.ModelDocument) -> Unit,
    onDeleteSelectedModel: () -> Unit
) {
    var showDeleteDialog by remember { mutableStateOf(false) }
    var showModelList by rememberSaveable { mutableStateOf(false) }

    Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
        Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
            StatusRow(
                label = "存储位置",
                value = if (modelTreeUri.isBlank()) "未选择" else "SAF 外部目录"
            )
            StatusRow(
                label = "当前模型",
                value = selectedModelName.ifBlank { "未选择" }
            )
            if (selectedModelSizeBytes > 0L) {
                StatusRow(
                    label = "模型大小",
                    value = formatBytes(selectedModelSizeBytes)
                )
            }
            StatusRow(
                label = "加载策略",
                value = storageModeLabel(modelStorageMode)
            )
        }

        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            Button(
                onClick = onChooseModelDirectory,
                modifier = Modifier.weight(1f),
                enabled = !isScanningModels
            ) {
                Icon(Icons.Default.FolderOpen, contentDescription = null, modifier = Modifier.size(18.dp))
                Spacer(modifier = Modifier.width(6.dp))
                Text("选择目录")
            }
            OutlinedButton(
                onClick = onRefreshModels,
                modifier = Modifier.weight(1f),
                enabled = !isScanningModels
            ) {
                if (isScanningModels) {
                    CircularProgressIndicator(modifier = Modifier.size(18.dp), strokeWidth = 2.dp)
                } else {
                    Icon(Icons.Default.Refresh, contentDescription = null, modifier = Modifier.size(18.dp))
                }
                Spacer(modifier = Modifier.width(6.dp))
                Text("扫描")
            }
        }

        if (availableModels.isNotEmpty()) {
            SettingRow(
                title = if (showModelList) "收起模型列表" else "查看模型列表",
                subtitle = "${availableModels.size} 个可用模型 · 当前：${selectedModelName.ifBlank { "未选择" }}",
                modifier = Modifier.testTag("settings_model_list_toggle"),
                onClick = { showModelList = !showModelList },
                trailing = {
                    Text(
                        text = if (showModelList) "收起" else "展开",
                        style = MaterialTheme.typography.labelMedium,
                        color = MaterialTheme.colorScheme.primary
                    )
                }
            )
            if (showModelList) {
                HorizontalDivider(color = MaterialTheme.colorScheme.outlineVariant.copy(alpha = 0.6f))
                Column(verticalArrangement = Arrangement.spacedBy(2.dp)) {
                    availableModels.forEach { model ->
                        ModelRow(
                            model = model,
                            selected = selectedModelUri == model.uri.toString(),
                            onClick = { onModelSelected(model) }
                        )
                    }
                }
            }
        }

        modelMessage?.let { message ->
            Text(
                text = message,
                style = MaterialTheme.typography.bodySmall,
                color = if (message.contains("失败")) {
                    MaterialTheme.colorScheme.error
                } else {
                    MaterialTheme.colorScheme.primary
                }
            )
        }

        Text(
            text = "建议选择直接包含 .gguf 的目录；也支持向下扫描 2 层子目录。模型保存在用户授权的外部目录中，卸载 APK 默认保留。",
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant
        )

        OutlinedButton(
            onClick = { showDeleteDialog = true },
            enabled = selectedModelUri.isNotBlank(),
            colors = ButtonDefaults.outlinedButtonColors(
                contentColor = MaterialTheme.colorScheme.error
            ),
            modifier = Modifier.fillMaxWidth()
        ) {
            Icon(Icons.Default.Delete, contentDescription = null, modifier = Modifier.size(18.dp))
            Spacer(modifier = Modifier.width(6.dp))
            Text(if (modelStorageMode == ModelManager.STORAGE_MODE_INTERNAL_TEST) "删除内部测试模型" else "删除外部模型文件")
        }
    }

    if (showDeleteDialog) {
        AlertDialog(
            onDismissRequest = { showDeleteDialog = false },
            title = { Text(if (modelStorageMode == ModelManager.STORAGE_MODE_INTERNAL_TEST) "删除内部测试模型" else "删除外部模型文件") },
            text = {
                Text(
                    text = if (modelStorageMode == ModelManager.STORAGE_MODE_INTERNAL_TEST) {
                        "将删除应用内部测试目录中的 GGUF 文件：${selectedModelName.ifBlank { "未选择" }}\n\n此操作不可撤销。"
                    } else {
                        "将删除你授权的外部模型目录中的原始 GGUF 文件：${selectedModelName.ifBlank { "未选择" }}\n\n这不是只删除缓存，而是删除外部目录中的模型原件。删除后无法从应用内恢复，重新使用需要重新下载或导入。是否继续？"
                    }
                )
            },
            confirmButton = {
                TextButton(
                    onClick = {
                        showDeleteDialog = false
                        onDeleteSelectedModel()
                    }
                ) {
                    Text("删除", color = MaterialTheme.colorScheme.error)
                }
            },
            dismissButton = {
                TextButton(onClick = { showDeleteDialog = false }) {
                    Text("取消")
                }
            }
        )
    }
}

@Composable
private fun ModelRow(
    model: ModelManager.ModelDocument,
    selected: Boolean,
    onClick: () -> Unit
) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .border(
                BorderStroke(
                    1.dp,
                    if (selected) qlhBrandGold().copy(alpha = 0.72f)
                    else MaterialTheme.colorScheme.outlineVariant.copy(alpha = 0.42f),
                ),
                MaterialTheme.shapes.small,
            )
            .clip(MaterialTheme.shapes.small)
            .clickable(onClick = onClick)
            .padding(horizontal = 6.dp, vertical = 6.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        RadioButton(selected = selected, onClick = onClick)
        Spacer(modifier = Modifier.width(6.dp))
        Column(modifier = Modifier.weight(1f)) {
            Text(
                text = model.name,
                style = MaterialTheme.typography.bodyMedium,
                fontWeight = if (selected) FontWeight.SemiBold else FontWeight.Normal,
                color = if (selected) {
                    qlhBrandGold()
                } else {
                    MaterialTheme.colorScheme.onSurface
                },
                maxLines = 1,
                overflow = TextOverflow.Ellipsis
            )
            Text(
                text = "${formatBytes(model.sizeBytes)} · ${if (model.source == ModelManager.ModelSource.SAF) "SAF" else "内部测试"}",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
        }
    }
}

private fun storageModeLabel(mode: String): String = when (mode) {
    ModelManager.STORAGE_MODE_SAF_FD -> "SAF fd"
    ModelManager.STORAGE_MODE_SAF_CACHE -> "SAF 缓存副本"
    ModelManager.STORAGE_MODE_INTERNAL_TEST -> "内部测试目录"
    else -> mode.ifBlank { "未设置" }
}

private fun formatBytes(bytes: Long): String {
    if (bytes < 0L) return "未知"
    val units = arrayOf("B", "KB", "MB", "GB", "TB")
    var value = bytes.toDouble()
    var unitIndex = 0
    while (value >= 1024.0 && unitIndex < units.lastIndex) {
        value /= 1024.0
        unitIndex++
    }
    return if (unitIndex == 0) {
        "${bytes} ${units[unitIndex]}"
    } else {
        "%.2f %s".format(value, units[unitIndex])
    }
}

// ================================================================
// 设备状态（运行时快照 + GPU / 后端 / 模型 / 上下文 分组）
// ================================================================

@Composable
private fun DeviceStatusGroups(
    status: AndroidRuntimeStatus?,
    loading: Boolean,
    error: String?,
    onRefresh: () -> Unit
) {
    var showDeviceDetails by rememberSaveable { mutableStateOf(false) }

    Column(verticalArrangement = Arrangement.spacedBy(20.dp)) {
        SettingsGroup(title = "设备状态", icon = Icons.Default.Memory) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(
                    text = "运行时与设备信息",
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
                OutlinedButton(
                    onClick = onRefresh,
                    enabled = !loading
                ) {
                    if (loading) {
                        CircularProgressIndicator(
                            modifier = Modifier.size(16.dp),
                            strokeWidth = 2.dp
                        )
                        Spacer(modifier = Modifier.width(6.dp))
                        Text("刷新中…")
                    } else {
                        Icon(Icons.Default.Refresh, contentDescription = null, modifier = Modifier.size(16.dp))
                        Spacer(modifier = Modifier.width(6.dp))
                        Text("刷新")
                    }
                }
            }

            error?.let { err ->
                Spacer(modifier = Modifier.height(4.dp))
                Text(
                    text = "状态获取失败: $err",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.error
                )
            }

            if (status == null) {
                Spacer(modifier = Modifier.height(4.dp))
                Text(
                    text = "状态尚未加载，请点击「刷新」",
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
            } else {
                Spacer(modifier = Modifier.height(8.dp))
                Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
                    // ---- 运行时 ----
                    StatusRow(
                        label = "llama.cpp 本地运行时",
                        value = if (status.nativeRuntimeAvailable) "可用" else "不可用",
                        valueColor = if (status.nativeRuntimeAvailable) {
                            MaterialTheme.colorScheme.primary
                        } else {
                            MaterialTheme.colorScheme.error
                        }
                    )
                    status.nativeRuntimeError?.let { err ->
                        Text(
                            text = "错误: $err",
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.error
                        )
                    }

                    if (status.isLite) {
                        StatusRow("模式", "极简版 (仅远程推理)")
                    } else {
                        StatusRow("模式", if (status.inferenceMode == "thin") "全无 (远程推理)" else "全有 (本地推理)")
                    }
                    if (status.serviceRunning) {
                        StatusRow("推理服务", "运行中")
                    }

                    SettingRow(
                        title = if (showDeviceDetails) "收起设备详情" else "查看设备详情",
                        subtitle = "系统、内存、存储与热状态",
                        modifier = Modifier.testTag("settings_device_details"),
                        onClick = { showDeviceDetails = !showDeviceDetails },
                        trailing = {
                            Text(
                                text = if (showDeviceDetails) "收起" else "展开",
                                style = MaterialTheme.typography.labelMedium,
                                color = MaterialTheme.colorScheme.primary
                            )
                        }
                    )

                    if (showDeviceDetails) {
                        HorizontalDivider(color = MaterialTheme.colorScheme.outlineVariant.copy(alpha = 0.6f))

                        StatusRow("设备", "${status.system.manufacturer} ${status.system.brand} ${status.system.model}".trim())
                        StatusRow("ABI", status.system.abis.joinToString(", ").ifBlank { "未知" })
                        StatusRow("CPU 核心", "${status.system.cpuCores}")
                        StatusRow("Android", "${status.system.androidRelease} (SDK ${status.system.sdkInt})")
                        if (status.system.socModel.isNotBlank()) {
                            StatusRow("SoC", "${status.system.socManufacturer} ${status.system.socModel}".trim())
                        }
                        StatusRow("省电模式", if (status.system.powerSaveMode) "开启" else "关闭")
                        StatusRow("热状态", status.system.thermalStatus)

                        HorizontalDivider(color = MaterialTheme.colorScheme.outlineVariant.copy(alpha = 0.6f))

                        StatusRow("系统内存", "${formatBytes(status.memory.availableBytes)} / ${formatBytes(status.memory.totalBytes)}")
                        if (status.memory.lowMemory) {
                            Text(
                                text = "系统内存不足",
                                style = MaterialTheme.typography.bodySmall,
                                color = MaterialTheme.colorScheme.error
                            )
                        }
                        StatusRow("JVM Heap", "${formatBytes(status.memory.heapFreeBytes)} / ${formatBytes(status.memory.heapTotalBytes)} (max ${formatBytes(status.memory.heapMaxBytes)})")
                        if (status.memory.lowRamDevice) {
                            StatusRow("低 RAM 设备", "是")
                        }
                        StatusRow("存储 (文件)", "${formatBytes(status.storage.filesAvailableBytes)} / ${formatBytes(status.storage.filesTotalBytes)}")
                        StatusRow("存储 (缓存)", "${formatBytes(status.storage.cacheAvailableBytes)} / ${formatBytes(status.storage.cacheTotalBytes)}")
                    }
                }
            }
        }

        if (status != null) {
            GpuGroup(status = status)
            BackendGroup(status = status)
            ModelStatusGroup(status = status)
            ContextGroup(status = status)
        }
    }
}

@Composable
private fun GpuGroup(status: AndroidRuntimeStatus) {
    CollapsibleSettingsGroup(
        title = "GPU",
        summary = "渲染器、厂商和 GPU offload 能力",
        icon = Icons.Default.Memory,
        testTag = "settings_gpu_details"
    ) {
        Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
            val gpu = status.gpu
            if (gpu.probeError != null) {
                StatusRow("GPU 探测", "失败: ${gpu.probeError}")
            } else {
                StatusRow("Renderer", gpu.renderer.ifBlank { "未知" })
                StatusRow("Vendor", gpu.vendor.ifBlank { "未知" })
                StatusRow("GL Version", gpu.version.ifBlank { "未知" })
            }
            StatusRow("GPU Offload 支持", if (gpu.supportsGpuOffload) "是" else "否")
            StatusRow("当前推理后端", if (gpu.supportsGpuOffload) "GPU offload 可用" else "CPU llama.cpp")
            StatusRow("Android GPU 版", "计划单独构建")
            if (gpu.backendDevices.isNotBlank()) {
                StatusRow("GGML 后端设备", gpu.backendDevices)
            }
            Text(
                text = gpu.note.ifBlank { "GPU 仅用于设备画像展示；当前 Android Full/Lite 版本不启用 GPU 推理，GPU 版将作为单独版本规划。" },
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.outline
            )
        }
    }
}

@Composable
private fun BackendGroup(status: AndroidRuntimeStatus) {
    val be = status.backend
    if (be.systemInfo.isNotBlank()) {
        CollapsibleSettingsGroup(
            title = "llama.cpp 后端",
            summary = "引擎、mmap、mlock、RPC 和编译信息",
            icon = Icons.Default.Info,
            testTag = "settings_backend_details"
        ) {
            Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                StatusRow("引擎", be.engine)
                StatusRow("mmap", if (be.supportsMmap) "支持" else "不支持")
                StatusRow("mlock", if (be.supportsMlock) "支持" else "不支持")
                StatusRow("GPU Offload", if (be.supportsGpuOffload) "支持" else "不支持")
                StatusRow("RPC", if (be.supportsRpc) "支持" else "不支持")
                if (be.systemInfo.isNotBlank()) {
                    Text(
                        text = be.systemInfo,
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.outline,
                        maxLines = 6,
                        overflow = TextOverflow.Ellipsis,
                        fontFamily = FontFamily.Monospace
                    )
                }
            }
        }
    }
}

@Composable
private fun ModelStatusGroup(status: AndroidRuntimeStatus) {
    val mdl = status.model
    CollapsibleSettingsGroup(
        title = "模型状态",
        summary = if (mdl.loaded) "已加载：${mdl.name.ifBlank { "当前模型" }}" else "尚未加载模型",
        icon = Icons.Default.Info,
        testTag = "settings_model_details"
    ) {
        Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
            StatusRow("已加载", if (mdl.loaded) "是" else "否")
            StatusRow("已选择模型", mdl.selectedName.ifBlank { "无" })
            if (mdl.selectedSizeBytes > 0L) {
                StatusRow("选择文件大小", formatBytes(mdl.selectedSizeBytes))
            }
            StatusRow("来源", mdl.selectedSource.ifBlank { "-" })
            if (mdl.loaded) {
                StatusRow("模型名", mdl.name.ifBlank { "-" })
                StatusRow("后端", mdl.backend.ifBlank { "-" })
                StatusRow("参数量", mdl.params.ifBlank { "-" })
                StatusRow("层数", mdl.layers.ifBlank { "-" })
                StatusRow("嵌入维度", mdl.embedding.ifBlank { "-" })
                StatusRow("Heads", mdl.heads.ifBlank { "-" })
                StatusRow("词汇表", mdl.vocabTokens.ifBlank { "-" })
                if (mdl.sizeBytes > 0L) {
                    StatusRow("模型大小", formatBytes(mdl.sizeBytes))
                }
            }
        }
    }
}

@Composable
private fun ContextGroup(status: AndroidRuntimeStatus) {
    val mdl = status.model
    if (!mdl.loaded) return
    val ctx = status.context
    CollapsibleSettingsGroup(
        title = "上下文 / KV (估算)",
        summary = "上下文、批大小和 KV 缓存指标",
        icon = Icons.Default.Memory,
        testTag = "settings_context_details"
    ) {
        Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
            StatusRow("配置上下文", "${ctx.configuredContextSize}")
            StatusRow("模型上下文 (n_ctx)", ctx.modelContextSize.ifBlank { "-" })
            StatusRow("训练上下文", ctx.trainContextSize.ifBlank { "-" })
            StatusRow("Batch", ctx.batchSize.ifBlank { "-" })
            StatusRow("Micro-Batch", ctx.microBatchSize.ifBlank { "-" })
            if (ctx.lastTotalTokens > 0) {
                StatusRow("Last Prompt Tokens", "${ctx.lastPromptTokens}")
                StatusRow("Last Generated Tokens", "${ctx.lastGeneratedTokens}")
                StatusRow("Last Total Tokens", "${ctx.lastTotalTokens}")
                if (ctx.lastElapsedSeconds > 0) {
                    StatusRow("Last 耗时", "%.2f s".format(ctx.lastElapsedSeconds))
                }
                if (ctx.lastTokensPerSecond > 0) {
                    StatusRow("Last tok/s", "%.2f".format(ctx.lastTokensPerSecond))
                }
                StatusRow("停止原因", ctx.stopReason.ifBlank { "-" })
            }
            if (ctx.estimatedKvMemoryMb > 0) {
                StatusRow("估算 KV 内存", "%.1f MB".format(ctx.estimatedKvMemoryMb))
            }
            StatusRow("持久 KV 复用", if (ctx.persistentKvReuseEnabled) "开启" else "关闭")
            if (ctx.note.isNotBlank()) {
                Text(
                    text = ctx.note,
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.outline
                )
            }
        }
    }
}

@Composable
private fun StatusRow(
    label: String,
    value: String,
    valueColor: Color = MaterialTheme.colorScheme.onSurface
) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.SpaceBetween
    ) {
        Text(
            text = label,
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            modifier = Modifier.weight(0.45f)
        )
        Text(
            text = value,
            style = MaterialTheme.typography.bodyMedium,
            color = valueColor,
            modifier = Modifier.weight(0.55f),
            textAlign = TextAlign.End,
            maxLines = 2,
            overflow = TextOverflow.Ellipsis
        )
    }
}

// ================================================================
// 日志管理
// ================================================================

@Composable
private fun LogManagementGroup(isLite: Boolean) {
    val ctx = LocalContext.current
    var logRefreshKey by remember { mutableStateOf(0) }
    val logFiles = remember(logRefreshKey) { QlhLogger.getLogFiles() }
    var showLogViewer by remember { mutableStateOf(false) }
    var showClearConfirm by remember { mutableStateOf(false) }
    // L4: 日志查看器搜索状态
    var logSearchQuery by remember { mutableStateOf("") }

    SettingsGroup(title = "日志管理", icon = Icons.Default.Description) {
        Text(
            text = "日志文件: ${logFiles.size} 个",
            style = MaterialTheme.typography.bodyLarge,
            color = MaterialTheme.colorScheme.onSurface
        )
        if (logFiles.isNotEmpty()) {
            val totalSize = logFiles.sumOf { it.size }
            Text(
                text = "总大小: ${formatBytes(totalSize)} · 最新: ${logFiles.first().name} (${formatBytes(logFiles.first().size)})",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
        }
        Spacer(modifier = Modifier.height(12.dp))

        // L4: 极简版仅保留复制和分享，完整版提供查看+清理
        if (isLite) {
            // ---- 极简版日志操作 ----
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                OutlinedButton(
                    onClick = {
                        val allLogText = QlhLogger.readRedactedLogBundle()
                        val clipboard = ctx.getSystemService(Context.CLIPBOARD_SERVICE) as ClipboardManager
                        clipboard.setPrimaryClip(ClipData.newPlainText("QLH Logs", allLogText))
                    },
                    modifier = Modifier.weight(1f),
                    enabled = logFiles.isNotEmpty()
                ) {
                    Text("复制日志")
                }
                OutlinedButton(
                    onClick = {
                        val file = QlhLogger.getLogFiles().firstOrNull() ?: return@OutlinedButton
                        val content = QlhLogger.readRedactedLogBundle(maxBytes = QlhLogger.READ_MAX_BYTES)
                        val sendIntent = Intent().apply {
                            action = Intent.ACTION_SEND
                            putExtra(Intent.EXTRA_TEXT, content)
                            putExtra(Intent.EXTRA_SUBJECT, "QLH 日志: ${file.name} (${formatBytes(file.size)})")
                            type = "text/plain"
                        }
                        ctx.startActivity(Intent.createChooser(sendIntent, "分享日志 — ${file.name} (${formatBytes(file.size)})"))
                    },
                    modifier = Modifier.weight(1f),
                    enabled = logFiles.isNotEmpty()
                ) {
                    Text("分享日志")
                }
            }
        } else {
            // ---- 完整版日志操作 ----
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                Button(
                    onClick = { showLogViewer = true },
                    modifier = Modifier.weight(1f),
                    enabled = logFiles.isNotEmpty()
                ) {
                    Text("查看日志")
                }
                OutlinedButton(
                    onClick = {
                        val allLogText = QlhLogger.readRedactedLogBundle()
                        val clipboard = ctx.getSystemService(Context.CLIPBOARD_SERVICE) as ClipboardManager
                        clipboard.setPrimaryClip(ClipData.newPlainText("QLH Logs", allLogText))
                    },
                    modifier = Modifier.weight(1f),
                    enabled = logFiles.isNotEmpty()
                ) {
                    Text("复制日志")
                }
            }

            Spacer(modifier = Modifier.height(8.dp))

            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                OutlinedButton(
                    onClick = {
                        val file = QlhLogger.getLogFiles().firstOrNull() ?: return@OutlinedButton
                        val content = QlhLogger.readRedactedLogBundle(maxBytes = QlhLogger.READ_MAX_BYTES)
                        val prefix = ""
                        val sendIntent = Intent().apply {
                            action = Intent.ACTION_SEND
                            putExtra(Intent.EXTRA_TEXT, prefix + content)
                            putExtra(Intent.EXTRA_SUBJECT, "QLH 日志: ${file.name} (${formatBytes(file.size)})")
                            type = "text/plain"
                        }
                        ctx.startActivity(Intent.createChooser(sendIntent, "分享日志 — ${file.name} (${formatBytes(file.size)})"))
                    },
                    modifier = Modifier.weight(1f),
                    enabled = logFiles.isNotEmpty()
                ) {
                    Text("分享日志")
                }
                OutlinedButton(
                    onClick = { showClearConfirm = true },
                    modifier = Modifier.weight(1f),
                    colors = ButtonDefaults.outlinedButtonColors(
                        contentColor = MaterialTheme.colorScheme.error
                    ),
                    enabled = logFiles.isNotEmpty()
                ) {
                    Text("清理日志", color = MaterialTheme.colorScheme.error)
                }
            }
        }
    }

    // ---- L4 增强日志查看对话框（仅完整版） ----
    if (showLogViewer && !isLite) {
        // 构建每个文件的内容（含截断信息）
        val fileContents = remember(showLogViewer, logRefreshKey) {
            QlhLogger.getLogFiles().map { fi ->
                val result = QlhLogger.readLogFileWithInfo(fi.name)
                Triple(fi, result.content, result.truncated)
            }
        }

        // 根据搜索词过滤
        val filteredContents = remember(fileContents, logSearchQuery) {
            if (logSearchQuery.isBlank()) {
                fileContents
            } else {
                fileContents.mapNotNull { (fi, content, truncated) ->
                    if (content != null && content.contains(logSearchQuery, ignoreCase = true)) {
                        Triple(fi, content, truncated)
                    } else if (fi.name.contains(logSearchQuery, ignoreCase = true)) {
                        // 文件名匹配也保留
                        Triple(fi, content, truncated)
                    } else {
                        null
                    }
                }
            }
        }

        val displayText = remember(filteredContents, logSearchQuery) {
            if (filteredContents.isEmpty()) {
                if (logSearchQuery.isNotBlank()) {
                    "没有匹配「${logSearchQuery}」的日志内容。\n"
                } else {
                    "(无日志文件)\n"
                }
            } else {
                buildString {
                    filteredContents.forEach { (fi, content, truncated) ->
                        appendLine("=".repeat(60))
                        appendLine("  ${fi.name}")
                        appendLine("  大小: ${formatBytes(fi.size)}")
                        if (truncated) {
                            appendLine("  [⚠ 日志文件过大 (${formatBytes(fi.size)})，仅显示末尾 ${formatBytes(QlhLogger.READ_MAX_BYTES)} 内容]")
                        }
                        appendLine("=".repeat(60))
                        appendLine(QlhLogger.redactDiagnosticText(content ?: "(读取失败)"))
                        appendLine()
                    }
                }
            }
        }

        AlertDialog(
            onDismissRequest = {
                showLogViewer = false
                logSearchQuery = ""
            },
            title = {
                Column {
                    Text("日志文件 (${logFiles.size} 个)")
                    // L4: 关键词搜索
                    Spacer(modifier = Modifier.height(8.dp))
                    OutlinedTextField(
                        value = logSearchQuery,
                        onValueChange = { logSearchQuery = it },
                        label = { Text("搜索日志内容…") },
                        modifier = Modifier.fillMaxWidth(),
                        singleLine = true,
                        shape = MaterialTheme.shapes.small,
                        textStyle = MaterialTheme.typography.bodySmall
                    )
                    if (logSearchQuery.isNotBlank()) {
                        Text(
                            text = "匹配 ${filteredContents.size}/${fileContents.size} 个文件",
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant
                        )
                    }
                }
            },
            text = {
                SelectionContainer {
                    Text(
                        text = displayText,
                        fontFamily = FontFamily.Monospace,
                        fontSize = MaterialTheme.typography.bodySmall.fontSize,
                        modifier = Modifier
                            .horizontalScroll(rememberScrollState())
                            .heightIn(max = 380.dp)
                    )
                }
            },
            confirmButton = {
                TextButton(onClick = {
                    showLogViewer = false
                    logSearchQuery = ""
                }) {
                    Text("关闭")
                }
            }
        )
    }

    // ---- 极简版日志查看对话框（保持简洁，无搜索） ----
    if (showLogViewer && isLite) {
        val logContent = remember(showLogViewer, logRefreshKey) {
            buildString {
                QlhLogger.getLogFiles().forEach { fi ->
                    val result = QlhLogger.readLogFileWithInfo(fi.name)
                    appendLine("===== ${fi.name} (${formatBytes(fi.size)}) =====")
                    if (result.truncated) {
                        appendLine("[⚠ 文件过大，仅显示末尾 ${formatBytes(QlhLogger.READ_MAX_BYTES)}]")
                    }
                    appendLine(QlhLogger.redactDiagnosticText(result.content ?: "(读取失败)"))
                    appendLine()
                }
            }
        }

        AlertDialog(
            onDismissRequest = { showLogViewer = false },
            title = { Text("日志文件") },
            text = {
                SelectionContainer {
                    Text(
                        text = logContent,
                        fontFamily = FontFamily.Monospace,
                        fontSize = MaterialTheme.typography.bodySmall.fontSize,
                        modifier = Modifier
                            .horizontalScroll(rememberScrollState())
                            .heightIn(max = 400.dp)
                    )
                }
            },
            confirmButton = {
                TextButton(onClick = { showLogViewer = false }) {
                    Text("关闭")
                }
            }
        )
    }

    // ---- 清理确认对话框 ----
    if (showClearConfirm) {
        AlertDialog(
            onDismissRequest = { showClearConfirm = false },
            title = { Text("清理日志") },
            text = { Text("确定删除所有日志文件？此操作不可撤销。\n删除后当前日志会自动重新生成。") },
            confirmButton = {
                TextButton(
                    onClick = {
                        QlhLogger.clearLogs()
                        logRefreshKey++
                        showClearConfirm = false
                    }
                ) {
                    Text("删除", color = MaterialTheme.colorScheme.error)
                }
            },
            dismissButton = {
                TextButton(onClick = { showClearConfirm = false }) {
                    Text("取消")
                }
            }
        )
    }
}

// ================================================================
// 关于
// ================================================================

@Composable
private fun AboutGroup() {
    SettingsGroup(title = "关于", icon = Icons.Default.Info) {
        Text(
            text = "轻量化大模型分布式边缘推理系统",
            style = MaterialTheme.typography.titleMedium,
            color = MaterialTheme.colorScheme.onSurface
        )
        Spacer(modifier = Modifier.height(4.dp))
        Text(
            text = "版本 ${BuildConfig.VERSION_NAME} (Android)",
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant
        )
        Spacer(modifier = Modifier.height(4.dp))
        Text(
            text = "北京交通大学 · 大学生创新创业训练计划",
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.outline
        )
        Spacer(modifier = Modifier.height(8.dp))
        Text(
            text = "© 2026 北京交通大学 · 项目团队",
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.outline
        )
    }
}
