package com.qlh.inference

import android.content.Intent
import android.os.Bundle
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.viewModels
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.Chat
import androidx.compose.material.icons.filled.Forum
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.NavigationBar
import androidx.compose.material3.NavigationBarItem
import androidx.compose.material3.NavigationBarItemDefaults
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat
import androidx.core.splashscreen.SplashScreen.Companion.installSplashScreen
import com.qlh.inference.logging.QlhLogger
import com.qlh.inference.service.InferenceService
import com.qlh.inference.ui.ChatScreen
import com.qlh.inference.ui.SessionListScreen
import com.qlh.inference.ui.SettingsScreen
import com.qlh.inference.ui.theme.QlhTheme

class MainActivity : ComponentActivity() {

    private val viewModel: MainViewModel by viewModels()
    private val modelDirectoryLauncher = registerForActivityResult(
        ActivityResultContracts.OpenDocumentTree()
    ) { uri ->
        uri?.let { viewModel.selectModelDirectory(it) }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        installSplashScreen()
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()

        setContent {
            val uiState by viewModel.uiState.collectAsState()
            val systemDark = isSystemInDarkTheme()
            val darkTheme = when (uiState.themeMode) {
                "light" -> false
                "dark" -> true
                else -> systemDark
            }

            QlhTheme(darkTheme = darkTheme) {
                MainApp(
                    viewModel = viewModel,
                    onChooseModelDirectory = { modelDirectoryLauncher.launch(null) },
                    onInferenceModeChanged = { mode ->
                        if (mode != com.qlh.inference.data.SettingsDataStore.MODE_DISTRIBUTED && !BuildConfig.IS_LITE) {
                            runCatching {
                                ContextCompat.startForegroundService(
                                    this,
                                    Intent(this, InferenceService::class.java),
                                )
                            }.onFailure { error ->
                                QlhLogger.w(
                                    "MainActivity",
                                    "Inference service unavailable: ${error.message ?: error.javaClass.simpleName}",
                                )
                            }
                        } else if (!BuildConfig.IS_LITE) {
                            stopService(Intent(this, InferenceService::class.java))
                        }
                    }
                )
            }
        }
    }

    override fun onResume() {
        super.onResume()
        viewModel.refreshAppInstallPermission()
    }
}

// ================================================================
// 主导航结构
// ================================================================

data class BottomNavItem(
    val label: String,
    val icon: ImageVector,
    val route: String
)

private val bottomNavItems = listOf(
    BottomNavItem("对话", Icons.AutoMirrored.Filled.Chat, "chat"),
    BottomNavItem("会话", Icons.Default.Forum, "sessions"),
    BottomNavItem("设置", Icons.Default.Settings, "settings")
)

@Composable
fun MainApp(
    viewModel: MainViewModel,
    onChooseModelDirectory: () -> Unit = {},
    onInferenceModeChanged: (String) -> Unit = {}
) {
    val uiState by viewModel.uiState.collectAsState()

    LaunchedEffect(uiState.inferenceMode) {
        onInferenceModeChanged(uiState.inferenceMode)
    }

    Scaffold(
        modifier = Modifier.fillMaxSize(),
        containerColor = MaterialTheme.colorScheme.background,
        contentColor = MaterialTheme.colorScheme.onBackground,
        bottomBar = {
            NavigationBar(
                containerColor = MaterialTheme.colorScheme.surfaceContainerLow,
                tonalElevation = 0.dp,
            ) {
                bottomNavItems.forEach { item ->
                    NavigationBarItem(
                        selected = uiState.currentTab == item.route,
                        onClick = { viewModel.selectTab(item.route) },
                        icon = { Icon(item.icon, contentDescription = item.label) },
                        label = { Text(item.label, style = MaterialTheme.typography.labelMedium) },
                        alwaysShowLabel = true,
                        colors = NavigationBarItemDefaults.colors(
                            selectedIconColor = MaterialTheme.colorScheme.onTertiaryContainer,
                            selectedTextColor = MaterialTheme.colorScheme.onSurface,
                            indicatorColor = MaterialTheme.colorScheme.tertiaryContainer,
                            unselectedIconColor = MaterialTheme.colorScheme.onSurfaceVariant,
                            unselectedTextColor = MaterialTheme.colorScheme.onSurfaceVariant
                        )
                    )
                }
            }
        }
    ) { innerPadding ->
        when (uiState.currentTab) {
            "chat" -> ChatScreen(
                sessionId = uiState.currentSessionId,
                sessionTitle = uiState.currentSessionTitle,
                messages = uiState.messages,
                isLoading = uiState.isLoading,
                error = uiState.error,
                onSendMessage = { viewModel.sendMessage(it) },
                onSendMessageWithImages = { message, imageDataUrls ->
                    viewModel.sendMessage(message, imageDataUrls)
                },
                onRetry = { viewModel.retryLastMessage() },
                onClearError = { viewModel.clearError() },
                modifier = Modifier.padding(innerPadding),
                inferenceMode = uiState.inferenceMode
            )
            "sessions" -> SessionListScreen(
                sessions = uiState.sessions,
                currentSessionId = uiState.currentSessionId,
                onSessionClick = { viewModel.selectSession(it) },
                onCreateSession = { viewModel.createSession() },
                onDeleteSession = { viewModel.deleteSession(it) },
                modifier = Modifier.padding(innerPadding)
            )
            "settings" -> SettingsScreen(
                serverHost = uiState.serverHost,
                serverPort = uiState.serverPort,
                inferenceMode = uiState.inferenceMode,
                maxTokens = uiState.maxTokens,
                temperature = uiState.temperature,
                topP = uiState.topP,
                contextSize = uiState.contextSize,
                showThinking = uiState.showThinking,
                themeMode = uiState.themeMode,
                onServerHostChange = { viewModel.setServerHost(it) },
                onServerPortChange = { viewModel.setServerPort(it) },
                onInferenceModeChange = { viewModel.setInferenceMode(it) },
                onMaxTokensChange = { viewModel.setMaxTokens(it) },
                onTemperatureChange = { viewModel.setTemperature(it) },
                onTopPChange = { viewModel.setTopP(it) },
                onContextSizeChange = { viewModel.setContextSize(it) },
                onShowThinkingChange = { /* no-op for now */ },
                onThemeModeChange = { viewModel.setThemeMode(it) },
                modelTreeUri = uiState.modelTreeUri,
                selectedModelUri = uiState.selectedModelUri,
                modelStorageMode = uiState.modelStorageMode,
                availableModels = uiState.availableModels,
                selectedModelName = uiState.selectedModelName,
                selectedModelSizeBytes = uiState.selectedModelSizeBytes,
                isScanningModels = uiState.isScanningModels,
                modelMessage = uiState.modelMessage,
                onChooseModelDirectory = onChooseModelDirectory,
                onRefreshModels = { viewModel.refreshModels() },
                onModelSelected = { viewModel.selectModel(it) },
                onDeleteSelectedModel = { viewModel.deleteSelectedModel() },
                runtimeStatus = uiState.runtimeStatus,
                runtimeStatusLoading = uiState.runtimeStatusLoading,
                runtimeStatusError = uiState.runtimeStatusError,
                presence = uiState.presence,
                onRefreshRuntimeStatus = { viewModel.refreshRuntimeStatus() },
                onConnectionTestSuccess = { viewModel.onConnectionTestSuccess() },
                remoteModels = uiState.remoteModels,
                remoteModelsLoading = uiState.remoteModelsLoading,
                remoteDownloadModelName = uiState.remoteDownloadModelName,
                remoteDownloadProgress = uiState.remoteDownloadProgress,
                remoteModelMessage = uiState.remoteModelMessage,
                onRefreshRemoteModels = { viewModel.refreshRemoteModels() },
                onDownloadRemoteModel = { viewModel.downloadRemoteModel(it) },
                diagnostics = uiState.diagnostics,
                onRefreshDiagnostics = { viewModel.refreshDiagnostics() },
                onUploadDiagnostics = { viewModel.uploadDiagnostics() },
                clusterOverview = uiState.clusterOverview,
                onRefreshClusterOverview = { viewModel.refreshClusterOverview() },
                modelFleet = uiState.modelFleet,
                onRefreshModelFleet = { viewModel.refreshModelFleet() },
                audit = uiState.audit,
                onRefreshAudit = { viewModel.refreshAudit() },
                authControl = uiState.authControl,
                onRefreshAuthControl = { viewModel.refreshAuthControl() },
                onLogin = { username, code, recoveryCode -> viewModel.login(username, code, recoveryCode) },
                onLogout = { viewModel.logout() },
                management = uiState.management,
                onRefreshManagement = { viewModel.refreshManagement() },
                onRevokeManagedUser = { viewModel.revokeManagedUser(it) },
                onRevokeManagedBinding = { viewModel.revokeManagedBinding(it) },
                appUpdate = uiState.appUpdate,
                onCheckForAppUpdate = { viewModel.checkForAppUpdate() },
                onDownloadAppUpdate = { viewModel.downloadAppUpdate() },
                onOpenInstallPermissionSettings = { viewModel.openInstallPermissionSettings() },
                onInstallDownloadedUpdate = { viewModel.installDownloadedUpdate() },
                modifier = Modifier.padding(innerPadding)
            )
        }
    }
}
