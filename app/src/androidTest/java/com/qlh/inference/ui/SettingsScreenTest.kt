package com.qlh.inference.ui

import androidx.compose.ui.test.assertIsDisplayed
import androidx.compose.ui.test.assertIsEnabled
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.ui.test.junit4.createEmptyComposeRule
import androidx.compose.ui.test.onNodeWithTag
import androidx.compose.ui.test.performClick
import androidx.compose.ui.test.performScrollTo
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
import androidx.test.core.app.ActivityScenario
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Rule
import org.junit.Test

class SettingsScreenTest {
    /**
     * ★ 2026-10-11：**不用** `createComposeRule()` —— 它内部经 `ActivityScenarioRule`，其
     * `after()` 会调 `ActivityScenario.close()`，而 `close()` 会经
     * `InstrumentationActivityInvoker` 启动 `androidx.test.core` 的 `EmptyActivity`；
     * 本设备上这一步恒抛 `ActivityNotFoundException`（见 `docs/彻查后修复与重构排票计划` 票12
     * 的设备门小节），并且**该异常会替换掉测试体抛出的断言**，使 instrumented 面变成「哑」的。
     *
     * 改为：`createEmptyComposeRule()`（只提供 Compose 测试环境与空闲同步，不启动 activity）
     * + 自建 `ActivityScenario.launch(ComponentActivity)`
     * + `@After` 用 `onActivity { finish() }` 清理（**不**调 `close()`）。
     */
    @get:Rule
    val composeRule = createEmptyComposeRule()

    private lateinit var scenario: ActivityScenario<ComponentActivity>

    @Before
    fun launchHost() {
        scenario = ActivityScenario.launch(ComponentActivity::class.java)
    }

    @After
    fun finishHost() {
        // 故意不调 scenario.close()：那正是触发 invoker activity 的那一步。
        // 直接 finish() 达到同样的清理效果，且不依赖测试 APK 之外的任何 activity。
        if (::scenario.isInitialized) {
            runCatching { scenario.onActivity { it.finish() } }
        }
    }

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
        composeRule.onNodeWithTag("settings_theme_system").performScrollTo().assertIsDisplayed().assertIsEnabled()
        composeRule.onNodeWithTag("settings_theme_light").performScrollTo().assertIsDisplayed().assertIsEnabled()
        composeRule.onNodeWithTag("settings_theme_dark").performScrollTo().assertIsDisplayed().assertIsEnabled()

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
        composeRule.onNodeWithTag("settings_theme_dark").performScrollTo().assertIsDisplayed().assertIsEnabled()
        composeRule.onNodeWithTag("settings_theme_light").performScrollTo().assertIsDisplayed().assertIsEnabled()
        composeRule.onNodeWithTag("settings_theme_system").performScrollTo().assertIsDisplayed().assertIsEnabled()
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

        composeRule.onNodeWithTag("remote_model_panel").performScrollTo().assertIsDisplayed()
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
        composeRule.onNodeWithTag("settings_show_thinking").performScrollTo().assertIsDisplayed().performClick()
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

        composeRule.onNodeWithTag("cluster_overview_details").performScrollTo().assertIsDisplayed().performClick()
        composeRule.onNodeWithTag("cluster_overview_node_master").performScrollTo().assertIsDisplayed()
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

        composeRule.onNodeWithTag("model_fleet_details").performScrollTo().assertIsDisplayed().performClick()
        composeRule.onNodeWithTag("model_fleet_entry_qwen-1_8b").performScrollTo().assertIsDisplayed()
        composeRule.onNodeWithTag("model_fleet_refresh").assertIsEnabled().performClick()
        composeRule.runOnIdle { assertTrue(refreshCalls >= 2) }
    }

    // ------------------------------------------------------------------
    // 票12（UI-CONTRACT-01）的模型舰队**文案契约**回归不在本文件 —— 见
    // `src/test/java/com/qlh/inference/ui/ModelFleetRuntimeLabelTest.kt`。
    //
    // 原因：本类在 Y700（TB321FU / Android 15）上**整类失败** —— SettingsScreen 内容超出
    // Compose 测试宿主视口，连首屏主题按钮都被判为不可见
    // （`Assert failed: The component is not displayed!`，13/13，与本次改动无关）。
    // 本票关心的是文案是否正确/互斥，与节点是否落在视口内无关 ⇒ 改用纯函数 + JVM 单测固定，
    // 不在此处新增依赖UI可见性的用例，避免为该设备的环境问题再添红灯。
    // ------------------------------------------------------------------

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

        composeRule.onNodeWithTag("audit_overview_details").performScrollTo().assertIsDisplayed().performClick()
        composeRule.onNodeWithTag("audit_workflow_wf-1").performScrollTo().assertIsDisplayed()
        composeRule.onNodeWithTag("audit_review_review-1").performScrollTo().assertIsDisplayed()
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

        composeRule.onNodeWithTag("auth_control_details").performScrollTo().assertIsDisplayed().performClick()
        composeRule.onNodeWithTag("auth_account_summary").performScrollTo().assertIsDisplayed()
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
        composeRule.onNodeWithTag("auth_username").performScrollTo().assertIsDisplayed().performTextInput("alice")
        composeRule.onNodeWithTag("auth_code").performScrollTo().assertIsDisplayed().performTextInput("123456")
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
        scenario.onActivity { activity ->
            activity.setContent {
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
}
