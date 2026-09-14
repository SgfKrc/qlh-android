package com.qlh.inference.ui

import androidx.compose.ui.test.assertIsDisplayed
import androidx.compose.ui.test.assertIsEnabled
import androidx.compose.ui.test.junit4.createComposeRule
import androidx.compose.ui.test.onNodeWithTag
import androidx.compose.ui.test.performClick
import androidx.compose.ui.test.performTextInput
import com.qlh.inference.ClusterOverviewNode
import com.qlh.inference.ClusterOverviewSnapshot
import com.qlh.inference.ClusterOverviewUiState
import com.qlh.inference.AuditReviewTicket
import com.qlh.inference.AuditSnapshot
import com.qlh.inference.AuditUiState
import com.qlh.inference.AuditWorkflow
import com.qlh.inference.AuthAccountSnapshot
import com.qlh.inference.AuthCapabilitySnapshot
import com.qlh.inference.AuthControlUiState
import com.qlh.inference.ModelFleetEntry
import com.qlh.inference.ModelFleetSnapshot
import com.qlh.inference.ModelFleetStatus
import com.qlh.inference.ModelFleetUiState
import com.qlh.inference.network.GgufModelInfo
import com.qlh.inference.service.ModelManager
import com.qlh.inference.status.AndroidRuntimeStatus
import com.qlh.inference.ui.theme.QlhTheme
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test

class SettingsScreenTest {
    @get:Rule
    val composeRule = createComposeRule()

    @Test
    fun themeModeButtonsAreVisibleAndEmitExpectedModes() {
        val selectedModes = mutableListOf<String>()
        var refreshCalled = false

        setSettingsContent(
            themeMode = "system",
            onThemeModeChange = { selectedModes += it },
            onRefreshRuntimeStatus = { refreshCalled = true },
        )

        composeRule.onNodeWithTag("settings_screen").assertIsDisplayed()
        composeRule.onNodeWithTag("settings_theme_system").assertIsDisplayed().assertIsEnabled()
        composeRule.onNodeWithTag("settings_theme_light").assertIsDisplayed().assertIsEnabled()
        composeRule.onNodeWithTag("settings_theme_dark").assertIsDisplayed().assertIsEnabled()

        composeRule.onNodeWithTag("settings_theme_light").performClick()
        composeRule.onNodeWithTag("settings_theme_dark").performClick()
        composeRule.onNodeWithTag("settings_theme_system").performClick()

        composeRule.runOnIdle {
            assertEquals(listOf("light", "dark", "system"), selectedModes)
            assertTrue(refreshCalled)
        }
    }

    @Test
    fun settingsScreenRendersThemeControlsInDarkTheme() {
        setSettingsContent(themeMode = "dark", darkTheme = true)

        composeRule.onNodeWithTag("settings_screen").assertIsDisplayed()
        composeRule.onNodeWithTag("settings_theme_dark").assertIsDisplayed().assertIsEnabled()
        composeRule.onNodeWithTag("settings_theme_light").assertIsDisplayed().assertIsEnabled()
        composeRule.onNodeWithTag("settings_theme_system").assertIsDisplayed().assertIsEnabled()
    }

    @Test
    fun remoteModelCatalogShowsVerifiedModelAndEmitsDownload() {
        var selected: GgufModelInfo? = null
        val model = GgufModelInfo(
            filename = "tiny.gguf",
            sizeBytes = 1_048_576L,
            sizeMb = 1.0,
            sha256 = "a".repeat(64),
            downloadUrl = "/api/models/download/tiny.gguf",
        )

        setSettingsContent(
            themeMode = "system",
            inferenceMode = "full",
            modelTreeUri = "content://example/tree/models",
            remoteModels = listOf(model),
            onDownloadRemoteModel = { selected = it },
        )

        composeRule.onNodeWithTag("remote_model_panel").assertIsDisplayed()
        composeRule.onNodeWithTag("remote_model_download_tiny.gguf").assertIsEnabled().performClick()
        composeRule.runOnIdle { assertEquals(model, selected) }
    }

    @Test
    fun highFrequencyStatusAndThinkingControlStayAvailable() {
        var showThinking = false
        setSettingsContent(
            themeMode = "system",
            onShowThinkingChange = { showThinking = it }
        )

        composeRule.onNodeWithTag("settings_device_details").assertIsDisplayed()
        composeRule.onNodeWithTag("settings_show_thinking").assertIsDisplayed().performClick()
        composeRule.runOnIdle { assertTrue(showThinking) }
    }

    @Test
    fun clusterOverviewIsBoundedAndRefreshable() {
        var refreshCalls = 0
        setSettingsContent(
            themeMode = "system",
            clusterOverview = ClusterOverviewUiState(
                snapshot = ClusterOverviewSnapshot(
                    running = true,
                    nodesReady = true,
                    reachableNodes = 2,
                    totalNodes = 2,
                    nodes = listOf(
                        ClusterOverviewNode(
                            nodeId = "master",
                            role = "master",
                            nodeType = "pc",
                            state = "online",
                            hostname = "main",
                            networkType = "tailscale",
                            taskCount = 0,
                            errorCount = 0,
                            reachable = true,
                        ),
                    ),
                ),
            ),
            onRefreshClusterOverview = { refreshCalls += 1 },
        )

        composeRule.onNodeWithTag("cluster_overview_details").assertIsDisplayed().performClick()
        composeRule.onNodeWithTag("cluster_overview_node_master").assertIsDisplayed()
        composeRule.onNodeWithTag("cluster_overview_refresh").assertIsEnabled().performClick()
        composeRule.runOnIdle { assertTrue(refreshCalls >= 2) }
    }

    @Test
    fun modelFleetIsReadOnlyBoundedAndRefreshable() {
        var refreshCalls = 0
        setSettingsContent(
            themeMode = "system",
            modelFleet = ModelFleetUiState(
                snapshot = ModelFleetSnapshot(
                    currentModelId = "qwen-1_8b",
                    currentModelName = "Qwen 1.8B",
                    currentLoaded = true,
                    registryCount = 2,
                    localAssetCount = 1,
                    verifiedGgufCount = 1,
                    entries = listOf(
                        ModelFleetEntry(
                            modelId = "qwen-1_8b",
                            name = "Qwen 1.8B",
                            modelType = "both",
                            status = ModelFleetStatus.ACTIVE,
                            sources = listOf("主节点注册表"),
                            formats = listOf("gguf", "safetensors"),
                        ),
                    ),
                ),
            ),
            onRefreshModelFleet = { refreshCalls += 1 },
        )

        composeRule.onNodeWithTag("model_fleet_details").assertIsDisplayed().performClick()
        composeRule.onNodeWithTag("model_fleet_entry_qwen-1_8b").assertIsDisplayed()
        composeRule.onNodeWithTag("model_fleet_refresh").assertIsEnabled().performClick()
        composeRule.runOnIdle { assertTrue(refreshCalls >= 2) }
    }

    @Test
    fun auditOverviewIsReadOnlyBoundedAndRefreshable() {
        var refreshCalls = 0
        setSettingsContent(
            themeMode = "system",
            audit = AuditUiState(
                snapshot = AuditSnapshot(
                    enabled = true,
                    available = true,
                    workflows = listOf(
                        AuditWorkflow(
                            workflowId = "wf-1",
                            template = "chat",
                            state = "completed",
                            createdAt = 1.0,
                            finishedAt = 2.0,
                            stageCount = 1,
                            completedStageCount = 1,
                            failedStageCount = 0,
                            attemptCount = 1,
                            retryCount = 0,
                            resultRejectionCount = 0,
                            recoveredAfterRestart = false,
                        ),
                    ),
                    reviews = listOf(
                        AuditReviewTicket(
                            ticketId = "review-1",
                            status = "pending",
                            createdAt = 1.0,
                            targetNodeId = "node-2",
                            score = 0,
                            voteCount = 0,
                        ),
                    ),
                ),
            ),
            onRefreshAudit = { refreshCalls += 1 },
        )

        composeRule.onNodeWithTag("audit_overview_details").assertIsDisplayed().performClick()
        composeRule.onNodeWithTag("audit_workflow_wf-1").assertIsDisplayed()
        composeRule.onNodeWithTag("audit_review_review-1").assertIsDisplayed()
        composeRule.onNodeWithTag("audit_overview_refresh").assertIsEnabled().performClick()
        composeRule.runOnIdle { assertTrue(refreshCalls >= 2) }
    }

    @Test
    fun accountSessionShowsSafeProjectionAndLogout() {
        var logoutCalls = 0
        var refreshCalls = 0
        setSettingsContent(
            themeMode = "system",
            authControl = AuthControlUiState(
                capability = AuthCapabilitySnapshot(
                    required = true,
                    available = true,
                    mode = "local_totp",
                ),
                account = AuthAccountSnapshot(
                    username = "alice",
                    displayName = "Alice",
                    role = "member",
                    expiresAt = "2030-01-01T00:00:00Z",
                ),
            ),
            onLogout = { logoutCalls += 1 },
            onRefreshAuthControl = { refreshCalls += 1 },
        )

        composeRule.onNodeWithTag("auth_control_details").assertIsDisplayed().performClick()
        composeRule.onNodeWithTag("auth_account_summary").assertIsDisplayed()
        composeRule.onNodeWithTag("auth_logout").assertIsEnabled().performClick()
        composeRule.onNodeWithTag("auth_refresh").assertIsEnabled().performClick()
        composeRule.runOnIdle {
            assertEquals(1, logoutCalls)
            assertTrue(refreshCalls >= 2)
        }
    }

    @Test
    fun accountLoginDoesNotPersistSecretFieldsInUiState() {
        var login: Triple<String, String?, String?>? = null
        setSettingsContent(
            themeMode = "system",
            authControl = AuthControlUiState(
                capability = AuthCapabilitySnapshot(required = true, available = true, mode = "local_totp"),
            ),
            onLogin = { username, code, recovery -> login = Triple(username, code, recovery) },
        )
        composeRule.onNodeWithTag("auth_control_details").performClick()
        composeRule.onNodeWithTag("auth_username").assertIsDisplayed().performTextInput("alice")
        composeRule.onNodeWithTag("auth_code").assertIsDisplayed().performTextInput("123456")
        composeRule.onNodeWithTag("auth_login").assertIsEnabled().performClick()
        composeRule.runOnIdle { assertEquals(Triple("alice", "123456", null), login) }
    }

    private fun setSettingsContent(
        themeMode: String,
        darkTheme: Boolean = false,
        inferenceMode: String = "thin",
        modelTreeUri: String = "",
        remoteModels: List<GgufModelInfo> = emptyList(),
        clusterOverview: ClusterOverviewUiState = ClusterOverviewUiState(),
        modelFleet: ModelFleetUiState = ModelFleetUiState(),
        audit: AuditUiState = AuditUiState(),
        authControl: AuthControlUiState = AuthControlUiState(),
        onThemeModeChange: (String) -> Unit = {},
        onRefreshRuntimeStatus: () -> Unit = {},
        onDownloadRemoteModel: (GgufModelInfo) -> Unit = {},
        onShowThinkingChange: (Boolean) -> Unit = {},
        onRefreshClusterOverview: () -> Unit = {},
        onRefreshModelFleet: () -> Unit = {},
        onRefreshAudit: () -> Unit = {},
        onRefreshAuthControl: () -> Unit = {},
        onLogin: (String, String?, String?) -> Unit = { _, _, _ -> },
        onLogout: () -> Unit = {},
    ) {
        composeRule.setContent {
            QlhTheme(darkTheme = darkTheme) {
                SettingsScreen(
                    serverHost = "100.64.0.1",
                    serverPort = 8000,
                    inferenceMode = inferenceMode,
                    maxTokens = 512,
                    temperature = 0.7f,
                    topP = 0.9f,
                    contextSize = 2048,
                    showThinking = false,
                    themeMode = themeMode,
                    modelTreeUri = modelTreeUri,
                    selectedModelUri = "",
                    modelStorageMode = "saf",
                    availableModels = emptyList<ModelManager.ModelDocument>(),
                    selectedModelName = "",
                    selectedModelSizeBytes = 0L,
                    isScanningModels = false,
                    modelMessage = null,
                    onServerHostChange = {},
                    onServerPortChange = {},
                    onInferenceModeChange = {},
                    onMaxTokensChange = {},
                    onTemperatureChange = {},
                    onTopPChange = {},
                    onContextSizeChange = {},
                    onShowThinkingChange = onShowThinkingChange,
                    onThemeModeChange = onThemeModeChange,
                    onChooseModelDirectory = {},
                    onRefreshModels = {},
                    onModelSelected = {},
                    onDeleteSelectedModel = {},
                    runtimeStatus = AndroidRuntimeStatus(),
                    runtimeStatusLoading = false,
                    runtimeStatusError = null,
                    onRefreshRuntimeStatus = onRefreshRuntimeStatus,
                    remoteModels = remoteModels,
                    onDownloadRemoteModel = onDownloadRemoteModel,
                    clusterOverview = clusterOverview,
                    onRefreshClusterOverview = onRefreshClusterOverview,
                    modelFleet = modelFleet,
                    onRefreshModelFleet = onRefreshModelFleet,
                    audit = audit,
                    onRefreshAudit = onRefreshAudit,
                    authControl = authControl,
                    onRefreshAuthControl = onRefreshAuthControl,
                    onLogin = onLogin,
                    onLogout = onLogout,
                )
            }
        }
    }
}
