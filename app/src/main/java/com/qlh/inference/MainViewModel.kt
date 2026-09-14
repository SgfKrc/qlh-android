package com.qlh.inference

import com.google.gson.Gson
import android.app.Application
import android.content.Context
import android.net.ConnectivityManager
import android.net.NetworkCapabilities
import android.net.Uri
import java.io.File
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import androidx.core.content.ContextCompat
import com.qlh.inference.BuildConfig
import com.qlh.inference.data.MessageEntity
import com.qlh.inference.data.SessionEntity
import com.qlh.inference.data.SettingsDataStore
import com.qlh.inference.logging.QlhLogger
import com.qlh.inference.network.ApiClient
import com.qlh.inference.network.AndroidPresenceSnapshot
import com.qlh.inference.network.AndroidPresenceState
import com.qlh.inference.network.GgufModelInfo
import com.qlh.inference.network.httpBaseUrl
import com.qlh.inference.network.ClientErrorReport
import com.qlh.inference.network.BootstrapRequest
import com.qlh.inference.network.ChatRepository
import com.qlh.inference.service.InferenceService
import com.qlh.inference.service.AndroidPresenceService
import com.qlh.inference.service.ModelManager
import com.qlh.inference.status.AndroidRuntimeStatus
import com.qlh.inference.system.AndroidDeviceInfoProvider
import com.qlh.inference.security.AuthTokenStore
import com.qlh.inference.security.StoredAuthSession
import com.qlh.inference.update.AndroidAppUpdateManager
import com.qlh.inference.update.AndroidUpdateCandidate
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Job
import kotlinx.coroutines.async
import kotlinx.coroutines.awaitAll
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.distinctUntilChanged
import kotlinx.coroutines.flow.filter
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.flatMapLatest
import kotlinx.coroutines.flow.map
import kotlinx.coroutines.launch

// ================================================================
// UI 状态
// ================================================================

data class MainUiState(
    // 导航
    val currentTab: String = "chat",

    // 当前会话
    val currentSessionId: Long = 0,
    val currentSessionTitle: String = "新对话",

    // 消息
    val messages: List<MessageEntity> = emptyList(),
    val isLoading: Boolean = false,
    val error: String? = null,

    // 会话列表
    val sessions: List<SessionEntity> = emptyList(),

    // 设置
    val serverHost: String = SettingsDataStore.DEFAULT_HOST,
    val serverPort: Int = SettingsDataStore.DEFAULT_PORT,
    val inferenceMode: String = SettingsDataStore.DEFAULT_MODE,
    val maxTokens: Int = SettingsDataStore.DEFAULT_MAX_TOKENS,
    val temperature: Float = SettingsDataStore.DEFAULT_TEMPERATURE,
    val topP: Float = SettingsDataStore.DEFAULT_TOP_P,
    val contextSize: Int = SettingsDataStore.DEFAULT_CONTEXT_SIZE,
    val showThinking: Boolean = false,

    // 模型管理
    val modelTreeUri: String = "",
    val selectedModelUri: String = "",
    val modelStorageMode: String = SettingsDataStore.DEFAULT_MODEL_STORAGE_MODE,
    val availableModels: List<ModelManager.ModelDocument> = emptyList(),
    val selectedModelName: String = "",
    val selectedModelSizeBytes: Long = 0L,
    val isScanningModels: Boolean = false,
    val modelMessage: String? = null,
    val remoteModels: List<GgufModelInfo> = emptyList(),
    val remoteModelsLoading: Boolean = false,
    val remoteDownloadModelName: String? = null,
    val remoteDownloadProgress: ModelManager.DownloadProgress? = null,
    val remoteModelMessage: String? = null,

    // 本地运行时状态
    val runtimeStatus: AndroidRuntimeStatus? = null,
    val runtimeStatusLoading: Boolean = false,
    val runtimeStatusError: String? = null,
    val themeMode: String = SettingsDataStore.DEFAULT_THEME_MODE,
    val diagnostics: DiagnosticsUiState = DiagnosticsUiState(),
    val clusterOverview: ClusterOverviewUiState = ClusterOverviewUiState(),
    val modelFleet: ModelFleetUiState = ModelFleetUiState(),
    val audit: AuditUiState = AuditUiState(),
    val authControl: AuthControlUiState = AuthControlUiState(),
    val management: ManagementUiState = ManagementUiState(),
    val appUpdate: AppUpdateUiState = AppUpdateUiState(),

    // 上次发送的消息（用于重试）
    val lastSentMessage: String? = null,
    val lastSentImageDataUrls: List<String> = emptyList(),

    // Local control-plane authentication
    val authSession: StoredAuthSession? = null,
    val authBusy: Boolean = false,
    val authError: String? = null,
    val presence: AndroidPresenceSnapshot = AndroidPresenceSnapshot(),
)

// ================================================================
// ViewModel
// ================================================================

@OptIn(kotlinx.coroutines.ExperimentalCoroutinesApi::class)
class MainViewModel(application: Application) : AndroidViewModel(application) {

    private val database = QlhApplication.instance.database
    private val settings = SettingsDataStore(application)
    private val modelManager = ModelManager(application)
    private val authStore = AuthTokenStore(application)
    private val appUpdateManager = AndroidAppUpdateManager(application)
    private var downloadedUpdateFile: File? = null

    private val _uiState = MutableStateFlow(MainUiState())
    val uiState: StateFlow<MainUiState> = _uiState.asStateFlow()

    private var lastAutoRegisterKey: String = ""

    private fun apiClient(state: MainUiState = _uiState.value): ApiClient =
        ApiClient(httpBaseUrl(state.serverHost, state.serverPort), authStore = authStore)

    // ---- 仓库（根据模式动态创建 ApiClient 或使用本地引擎） ----
    private val repository = ChatRepository(
        sessionDao = database.sessionDao(),
        messageDao = database.messageDao(),
        apiClient = {
            val state = _uiState.value
            if (state.inferenceMode == "thin") {
                apiClient(state)
            } else {
                null // 全有模式 — 使用本地推理引擎
            }
        },
        inferenceService = {
            // 全有模式：返回 InferenceService 实例（由 Application 管理）
            QlhApplication.instance.inferenceService
        }
    ).also { repo ->
        repo.setThinClientMetadataProvider { settings.getOrCreateAndroidNodeId() }
        repo.setThinPresenceHook { autoRegisterAndroidNode(force = true) }
    }

    init {
        // 加载设置
        viewModelScope.launch {
            try {
                val host = settings.getServerHost()
                val port = settings.getServerPort()
                val mode = if (BuildConfig.IS_LITE) "thin" else settings.getInferenceMode()
                val maxTokens = settings.getMaxTokens()
                val temp = settings.getTemperature()
                val topP = settings.getTopP()
                val contextSize = settings.getContextSize()
                val modelTreeUri = settings.getModelTreeUri()
                val selectedModelUri = settings.getSelectedModelUri()
                val storageMode = settings.getModelStorageMode()
                val themeMode = settings.getThemeMode()
                val selectedModel = modelManager.getSelectedModel()
                val sessions = database.sessionDao().getAllSessions().first()
                val initialSessionId = sessions.firstOrNull()?.id ?: repository.createSession("新对话")
                val initialSession = database.sessionDao().getById(initialSessionId)

                _uiState.value = _uiState.value.copy(
                    currentSessionId = initialSessionId,
                    currentSessionTitle = initialSession?.title ?: "新对话",
                    serverHost = host,
                    serverPort = port,
                    inferenceMode = mode,
                    maxTokens = maxTokens,
                    temperature = temp,
                    topP = topP,
                    contextSize = contextSize,
                    modelTreeUri = modelTreeUri,
                    selectedModelUri = selectedModelUri,
                    modelStorageMode = storageMode,
                    themeMode = themeMode,
                    selectedModelName = selectedModel?.name.orEmpty(),
                    selectedModelSizeBytes = selectedModel?.sizeBytes ?: 0L,
                    authSession = authStore.read(),
                    authControl = AuthControlUiState(
                        localSessionPresent = authStore.read() != null,
                    ),
                )
                QlhApplication.instance.inferenceService?.modelContextSize = contextSize
                ensureAndroidBootstrap()
                autoRegisterAndroidNode()
                refreshModels(showMessage = false)
                refreshRuntimeStatus()
                refreshManagement()
            } catch (e: CancellationException) {
                throw e
            } catch (e: Exception) {
                QlhLogger.e(
                    "MainViewModel",
                    "Android startup initialization failed: ${e.message ?: e.javaClass.simpleName}",
                    e,
                )
                _uiState.value = _uiState.value.copy(
                    error = "本地启动初始化失败，请进入设置重试：${e.message ?: e.javaClass.simpleName}",
                )
            }
        }

        // 监听会话列表
        viewModelScope.launch {
            database.sessionDao().getAllSessions().collect { sessions ->
                _uiState.value = _uiState.value.copy(sessions = sessions)

                // 如果没有会话，自动创建默认会话
                if (sessions.isEmpty()) {
                    createSessionInternal("新对话")
                } else if (_uiState.value.currentSessionId == 0L) {
                    // 选择最近更新的会话
                    selectSession(sessions.first().id)
                }
            }
        }

        // Android Full 薄客户端 presence 心跳：只在 thin 模式生效，Lite 仍跳过。
        viewModelScope.launch {
            AndroidPresenceService.snapshot.collect { snapshot ->
                _uiState.value = _uiState.value.copy(presence = snapshot)
            }
        }

        // 监听当前会话消息 — flatMapLatest 自动取消旧 collector，避免泄漏
        viewModelScope.launch {
            _uiState
                .map { it.currentSessionId }
                .distinctUntilChanged()
                .filter { it > 0L }
                .flatMapLatest { sessionId ->
                    database.messageDao().getMessagesBySession(sessionId)
                }
                .collect { messages ->
                    _uiState.value = _uiState.value.copy(messages = messages)
                }
        }

        // 监听设置变化
        viewModelScope.launch {
            settings.serverHost.collect { host ->
                _uiState.value = _uiState.value.copy(serverHost = host)
            }
        }
        viewModelScope.launch {
            settings.serverPort.collect { port ->
                _uiState.value = _uiState.value.copy(serverPort = port)
            }
        }
        viewModelScope.launch {
            settings.inferenceMode.collect { mode ->
                _uiState.value = _uiState.value.copy(
                    inferenceMode = if (BuildConfig.IS_LITE) "thin" else mode
                )
            }
        }
        viewModelScope.launch {
            settings.modelTreeUri.collect { uri ->
                _uiState.value = _uiState.value.copy(modelTreeUri = uri)
            }
        }
        viewModelScope.launch {
            settings.selectedModelUri.collect { uri ->
                val selected = modelManager.getSelectedModel()
                _uiState.value = _uiState.value.copy(
                    selectedModelUri = uri,
                    selectedModelName = selected?.name.orEmpty(),
                    selectedModelSizeBytes = selected?.sizeBytes ?: 0L
                )
            }
        }
        viewModelScope.launch {
            settings.modelStorageMode.collect { mode ->
                _uiState.value = _uiState.value.copy(modelStorageMode = mode)
            }
        }
        viewModelScope.launch {
            settings.contextSize.collect { size ->
                _uiState.value = _uiState.value.copy(contextSize = size)
            }
        }
        viewModelScope.launch {
            settings.themeMode.collect { mode ->
                _uiState.value = _uiState.value.copy(themeMode = mode)
            }
        }
    }

    // ==================== 导航 ====================

    fun selectTab(tab: String) {
        _uiState.value = selectUiTab(_uiState.value, tab)
    }

    // ==================== 会话管理 ====================

    fun createSession() {
        viewModelScope.launch {
            createSessionInternal("新对话")
            _uiState.value = _uiState.value.copy(currentTab = "chat")
        }
    }

    private suspend fun createSessionInternal(title: String) {
        val id = repository.createSession(title)
        selectSessionInternal(id)
    }

    fun selectSession(sessionId: Long) {
        viewModelScope.launch {
            selectSessionInternal(sessionId)
        }
    }

    /** 同步版本 — 供内部 suspend 函数直接调用，避免 race condition */
    private suspend fun selectSessionInternal(sessionId: Long) {
        val session = repository.getSession(sessionId)
        _uiState.value = selectUiSession(
            _uiState.value,
            sessionId,
            session?.title ?: "新对话",
        )
    }

    fun deleteSession(sessionId: Long) {
        viewModelScope.launch {
            repository.deleteSession(sessionId)
            // 如果删除的是当前会话，切换到其他会话
            if (_uiState.value.currentSessionId == sessionId) {
                val sessions = database.sessionDao().getAllSessions().first()
                if (sessions.isNotEmpty()) {
                    selectSessionInternal(sessions.first().id)
                } else {
                    createSessionInternal("新对话")
                }
            }
        }
    }

    /** 确保当前会话 ID 一定存在，避免首次启动竞态导致外键崩溃。 */
    private suspend fun ensureActiveSession(): MainUiState {
        val current = _uiState.value
        if (current.currentSessionId > 0L && repository.getSession(current.currentSessionId) != null) {
            return current
        }
        val sessions = database.sessionDao().getAllSessions().first()
        val sessionId = sessions.firstOrNull()?.id ?: repository.createSession("新对话")
        val session = repository.getSession(sessionId)
        return current.copy(
            currentSessionId = sessionId,
            currentSessionTitle = session?.title ?: "新对话",
            currentTab = "chat"
        ).also { _uiState.value = it }
    }

    // ==================== 消息 ====================

    fun sendMessage(message: String, imageDataUrls: List<String> = emptyList()) {
        val normalizedImages = normalizeChatImageDataUrls(imageDataUrls)
        QlhLogger.i(
            "MainViewModel",
            "sendMessage start: ${message.length} chars, images=${normalizedImages.size}"
        )
        viewModelScope.launch {
            try {
                val state = ensureActiveSession()
                val validationError = validateChatImageSubmission(state.inferenceMode, imageDataUrls)
                if (validationError != null) {
                    _uiState.value = failMessageSubmission(state, validationError)
                    return@launch
                }
                _uiState.value = startMessageSubmission(state, message, normalizedImages)

                val result = repository.sendMessage(
                    sessionId = state.currentSessionId,
                    message = message,
                    maxTokens = state.maxTokens,
                    temperature = state.temperature,
                    topP = state.topP,
                    showThinking = state.showThinking,
                    imageDataUrls = normalizedImages,
                )

                result.onSuccess {
                    _uiState.value = completeMessageSubmission(_uiState.value)
                    refreshRuntimeStatus()
                }.onFailure { e ->
                    QlhLogger.e("MainViewModel", "sendMessage failed", e)
                    _uiState.value = failMessageSubmission(
                        _uiState.value,
                        formatMessageSendError(e),
                    )
                }
            } catch (e: Exception) {
                QlhLogger.e("MainViewModel", "sendMessage crashed", e)
                _uiState.value = failMessageSubmission(
                    _uiState.value,
                    formatMessageSendError(e),
                )
            }
        }
    }

    fun retryLastMessage() {
        val retryState = _uiState.value
        val lastMsg = retryState.lastSentMessage ?: return
        val retryImages = retryState.lastSentImageDataUrls
        viewModelScope.launch {
            try {
                val state = ensureActiveSession()
                val validationError = validateChatImageSubmission(state.inferenceMode, retryImages)
                if (validationError != null) {
                    _uiState.value = failMessageSubmission(state, validationError)
                    return@launch
                }
                _uiState.value = startMessageRetry(state)

                // 跳过用户消息保存（上次失败的尝试已保存），只重新调用 API
                val result = repository.sendMessage(
                    sessionId = state.currentSessionId,
                    message = lastMsg,
                    maxTokens = state.maxTokens,
                    temperature = state.temperature,
                    topP = state.topP,
                    showThinking = state.showThinking,
                    skipUserSave = true,  // ★ 避免重复用户消息
                    imageDataUrls = retryImages,
                )

                result.onSuccess {
                    _uiState.value = completeMessageSubmission(_uiState.value)
                    refreshRuntimeStatus()
                }.onFailure { e ->
                    QlhLogger.e("MainViewModel", "retryLastMessage failed", e)
                    _uiState.value = failMessageSubmission(
                        _uiState.value,
                        formatMessageSendError(e),
                    )
                }
            } catch (e: Exception) {
                QlhLogger.e("MainViewModel", "retryLastMessage crashed", e)
                _uiState.value = failMessageSubmission(
                    _uiState.value,
                    formatMessageSendError(e),
                )
            }
        }
    }

    fun clearError() {
        _uiState.value = clearMessageError(_uiState.value)
    }

    /** Authenticate the Android client; the token is persisted only in Android Keystore-backed storage. */
    fun login(username: String, code: String? = null, recoveryCode: String? = null) {
        viewModelScope.launch {
            _uiState.value = _uiState.value.copy(
                authBusy = true,
                authError = null,
                authControl = _uiState.value.authControl.copy(busy = true, error = null),
            )
            val result = apiClient().login(username, code, recoveryCode)
            val session = result.getOrNull()
            val error = result.exceptionOrNull()?.let(::formatAuthError)
            _uiState.value = _uiState.value.copy(
                authBusy = false,
                authSession = session ?: authStore.read(),
                authError = error,
                authControl = _uiState.value.authControl.copy(
                    busy = false,
                    account = session?.toAccountSnapshot(),
                    localSessionPresent = session != null || authStore.read() != null,
                    error = error,
                ),
            )
            if (session != null) refreshManagement()
            else _uiState.value = _uiState.value.copy(management = ManagementUiState())
        }
    }

    /** Refresh capability first, then validate a stored session; failure never shows a stale account. */
    fun refreshAuthControl() {
        viewModelScope.launch {
            val current = _uiState.value
            _uiState.value = current.copy(
                authControl = current.authControl.copy(
                    loading = true,
                    error = null,
                    account = null,
                    localSessionPresent = authStore.read() != null,
                ),
            )
            val capabilityResult = apiClient().getAuthCapability()
            if (capabilityResult.isFailure) {
                val error = formatAuthError(capabilityResult.exceptionOrNull()!!)
                _uiState.value = _uiState.value.copy(
                    authBusy = false,
                    authError = error,
                    management = ManagementUiState(),
                    authControl = _uiState.value.authControl.copy(
                        capability = null,
                        loading = false,
                        account = null,
                        error = error,
                    ),
                )
                return@launch
            }

            val capability = capabilityResult.getOrThrow().toSnapshot()
            val localSession = authStore.read()
            if (!capability.canAuthenticate || localSession == null) {
                _uiState.value = _uiState.value.copy(
                    authBusy = false,
                    authError = null,
                    management = ManagementUiState(),
                    authControl = _uiState.value.authControl.copy(
                        capability = capability,
                        loading = false,
                        account = null,
                        localSessionPresent = localSession != null,
                        error = null,
                    ),
                )
                return@launch
            }

            val sessionResult = apiClient().getAuthSession()
            val error = sessionResult.exceptionOrNull()?.let(::formatAuthError)
            _uiState.value = _uiState.value.copy(
                authBusy = false,
                authError = error,
                authSession = if (sessionResult.isSuccess) localSession else authStore.read(),
                authControl = _uiState.value.authControl.copy(
                    capability = capability,
                    loading = false,
                    account = sessionResult.getOrNull()?.toAccountSnapshot(),
                    localSessionPresent = authStore.read() != null,
                    error = error,
                ),
            )
            if (sessionResult.isSuccess && managementRoleAllowed(sessionResult.getOrNull()?.user?.role)) {
                refreshManagement()
            } else {
                _uiState.value = _uiState.value.copy(management = ManagementUiState())
            }
        }
    }

    /** Revalidate the local session against the control plane without exposing credentials to UI code. */
    fun refreshAuthSession() {
        refreshAuthControl()
    }

    /** Clear local credentials even when the control plane is offline. */
    fun logout() {
        viewModelScope.launch {
            _uiState.value = _uiState.value.copy(
                authBusy = true,
                authError = null,
                authControl = _uiState.value.authControl.copy(busy = true, account = null, error = null),
            )
            val result = apiClient().logout()
            val error = result.exceptionOrNull()?.let(::formatAuthError)
            _uiState.value = _uiState.value.copy(
                authBusy = false,
                authSession = null,
                authError = error,
                authControl = _uiState.value.authControl.copy(
                    busy = false,
                    account = null,
                    localSessionPresent = false,
                    error = error,
                ),
            )
            _uiState.value = _uiState.value.copy(management = ManagementUiState())
        }
    }

    fun refreshRuntimeStatus() {
        viewModelScope.launch {
            _uiState.value = _uiState.value.copy(
                runtimeStatusLoading = true,
                runtimeStatusError = null
            )
            try {
                val state = _uiState.value
                val service = QlhApplication.instance.inferenceService
                val status = service?.getRuntimeStatus(state.inferenceMode, BuildConfig.IS_LITE)
                    ?: createPassiveRuntimeStatus(state.inferenceMode)
                _uiState.value = _uiState.value.copy(
                    runtimeStatus = status,
                    runtimeStatusLoading = false,
                    runtimeStatusError = null
                )
            } catch (e: Exception) {
                QlhLogger.e("MainViewModel", "refreshRuntimeStatus failed", e)
                _uiState.value = _uiState.value.copy(
                    runtimeStatusLoading = false,
                    runtimeStatusError = e.message ?: e.javaClass.simpleName
                )
            }
        }
    }

    fun refreshDiagnostics() {
        viewModelScope.launch {
            val state = _uiState.value
            _uiState.value = state.copy(
                diagnostics = state.diagnostics.copy(
                    healthLoading = true,
                    healthError = null,
                )
            )
            apiClient(state).probeConnectionHealth(detectNetworkType())
                .onSuccess { report ->
                    _uiState.value = _uiState.value.copy(
                        diagnostics = _uiState.value.diagnostics.copy(
                            health = report,
                            healthLoading = false,
                            healthError = null,
                        )
                    )
                }
                .onFailure { error ->
                    _uiState.value = _uiState.value.copy(
                        diagnostics = _uiState.value.diagnostics.copy(
                            healthLoading = false,
                            healthError = error.message ?: error.javaClass.simpleName,
                        )
                    )
                }
        }
    }

    /** Refresh only the safe, read-only cluster projection used by Settings. */
    fun refreshClusterOverview() {
        if (_uiState.value.clusterOverview.loading) return
        viewModelScope.launch {
            val state = _uiState.value
            _uiState.value = state.copy(
                clusterOverview = state.clusterOverview.copy(loading = true, error = null),
            )
            apiClient(state).getClusterStatus()
                .onSuccess { status ->
                    _uiState.value = _uiState.value.copy(
                        clusterOverview = ClusterOverviewUiState(
                            snapshot = toClusterOverviewSnapshot(status),
                        ),
                    )
                }
                .onFailure { error ->
                    _uiState.value = _uiState.value.copy(
                        clusterOverview = _uiState.value.clusterOverview.copy(
                            loading = false,
                            error = formatClusterOverviewError(error),
                        ),
                    )
                }
        }
    }

    /** Refresh the mobile-safe model-fleet projection; it has no deployment controls. */
    fun refreshModelFleet() {
        if (_uiState.value.modelFleet.loading) return
        viewModelScope.launch {
            val state = _uiState.value
            _uiState.value = state.copy(
                modelFleet = state.modelFleet.copy(loading = true, error = null),
            )
            apiClient(state).getModelFleetData()
                .onSuccess { data ->
                    _uiState.value = _uiState.value.copy(
                        modelFleet = ModelFleetUiState(
                            snapshot = toModelFleetSnapshot(
                                data = data,
                                androidSelectedModelName = state.selectedModelName,
                                androidSelectedModelSizeBytes = state.selectedModelSizeBytes,
                            ),
                        ),
                    )
                }
                .onFailure { error ->
                    _uiState.value = _uiState.value.copy(
                        modelFleet = _uiState.value.modelFleet.copy(
                            loading = false,
                            error = formatModelFleetError(error),
                        ),
                    )
                }
        }
    }

    /** Refresh the bounded, content-free audit projection used by Android Settings. */
    fun refreshAudit() {
        if (_uiState.value.audit.loading) return
        viewModelScope.launch {
            val state = _uiState.value
            _uiState.value = state.copy(
                audit = state.audit.copy(loading = true, error = null),
            )
            apiClient(state).getAuditData()
                .onSuccess { data ->
                    _uiState.value = _uiState.value.copy(
                        audit = AuditUiState(snapshot = toAuditSnapshot(data)),
                    )
                }
                .onFailure { error ->
                    _uiState.value = _uiState.value.copy(
                        audit = _uiState.value.audit.copy(
                            loading = false,
                            error = formatAuditError(error),
                        ),
                    )
                }
        }
    }

    /**
     * Refresh the owner/admin-only mobile management projection. The server remains
     * authoritative; a stale local role never grants access because every request
     * still carries the bearer and is checked by the control/gateway policy.
     */
    fun refreshManagement() {
        if (_uiState.value.management.loading) return
        val localRole = authStore.read()?.role
        if (!managementRoleAllowed(localRole)) {
            _uiState.value = _uiState.value.copy(management = ManagementUiState())
            return
        }
        viewModelScope.launch {
            val state = _uiState.value
            _uiState.value = state.copy(
                management = state.management.copy(loading = true, error = null),
            )
            val client = apiClient(state)
            try {
                val summary = client.fetchManageSummary().getOrThrow()
                val usersResponse = client.fetchManagedUsers().getOrThrow()
                val users = usersResponse.users
                    .asSequence()
                    .filter { it.userId.isNotBlank() && it.username.isNotBlank() }
                    .take(MAX_MANAGED_USERS)
                    .toList()

                // Bindings are fetched per user because the manager endpoint deliberately
                // avoids a global identity dump. Keep fan-out bounded by the user cap.
                val usersWithBindings = coroutineScope {
                    users.map { user ->
                        async {
                            val bindings = client.fetchUserTailscaleBindings(user.userId)
                                .getOrThrow()
                                .bindings
                                .asSequence()
                                .filter { it.bindingId.isNotBlank() }
                                .take(MAX_MANAGED_BINDINGS)
                                .map { it.toManagedBindingSnapshot(user.userId) }
                                .toList()
                            user.toManagedUserSnapshot(bindings)
                        }
                    }.awaitAll()
                }

                val audit = if (summary.auditAvailable) {
                    client.fetchManageAudit(MAX_MANAGEMENT_AUDIT_EVENTS)
                        .getOrThrow()
                        .events
                        .take(MAX_MANAGEMENT_AUDIT_EVENTS)
                } else {
                    emptyList()
                }
                _uiState.value = _uiState.value.copy(
                    management = ManagementUiState(
                        summary = summary,
                        users = usersWithBindings,
                        audit = audit,
                    ),
                )
            } catch (e: Exception) {
                QlhLogger.e("MainViewModel", "refreshManagement failed", e)
                _uiState.value = _uiState.value.copy(
                    management = _uiState.value.management.copy(
                        loading = false,
                        error = formatManagementError(e),
                        summary = null,
                        users = emptyList(),
                        audit = emptyList(),
                    ),
                )
            }
        }
    }

    /** Request and consume a one-shot confirmation token entirely inside the ViewModel. */
    fun revokeManagedUser(user: ManagedUserSnapshot) {
        if (user.userId.isBlank() || user.aggregateVersion < 1) {
            _uiState.value = _uiState.value.copy(
                management = _uiState.value.management.copy(error = "成员版本不可用，已拒绝撤销"),
            )
            return
        }
        runManagedMutation("revoke-user:${user.userId}") { client ->
            val confirmation = client.requestManageConfirm("user_manage", user.userId).getOrThrow()
            client.revokeUser(user.userId, user.aggregateVersion, confirmation.confirmToken).getOrThrow()
        }
    }

    /** Request and consume a one-shot confirmation token for a cross-user binding revoke. */
    fun revokeManagedBinding(binding: ManagedBindingSnapshot) {
        if (binding.bindingId.isBlank()) {
            _uiState.value = _uiState.value.copy(
                management = _uiState.value.management.copy(error = "绑定标识不可用，已拒绝撤销"),
            )
            return
        }
        runManagedMutation("revoke-binding:${binding.bindingId}") { client ->
            val confirmation = client.requestManageConfirm("tailnet_bind", binding.bindingId).getOrThrow()
            client.revokeTailscaleBinding(binding.bindingId, confirmation.confirmToken).getOrThrow()
        }
    }

    private fun runManagedMutation(action: String, operation: suspend (ApiClient) -> Unit) {
        if (_uiState.value.management.busyAction != null) return
        viewModelScope.launch {
            _uiState.value = _uiState.value.copy(
                management = _uiState.value.management.copy(busyAction = action, error = null),
            )
            try {
                operation(apiClient())
                _uiState.value = _uiState.value.copy(
                    management = _uiState.value.management.copy(busyAction = null),
                )
                refreshManagement()
            } catch (e: Exception) {
                QlhLogger.e("MainViewModel", "management mutation failed", e)
                _uiState.value = _uiState.value.copy(
                    management = _uiState.value.management.copy(
                        busyAction = null,
                        error = formatManagementError(e),
                    ),
                )
            }
        }
    }

    fun uploadDiagnostics() {
        viewModelScope.launch {
            val state = _uiState.value
            val diagnostics = state.diagnostics.copy(
                uploadInProgress = true,
                uploadMessage = null,
                uploadError = null,
            )
            _uiState.value = state.copy(diagnostics = diagnostics)
            val logText = QlhLogger.readRedactedLogBundle(maxBytes = 20_000L)
            val report = ClientErrorReport(
                message = "manual Android diagnostic upload",
                source = "manual",
                stack = logText,
                userAgent = "QLH Android ${BuildConfig.VERSION_NAME}",
                extra = mapOf(
                    "variant" to if (BuildConfig.IS_LITE) "lite" else "full",
                    "network_type" to (state.diagnostics.health?.localNetworkType ?: detectNetworkType()),
                ),
            )
            apiClient(state).reportClientError(report)
                .onSuccess {
                    _uiState.value = _uiState.value.copy(
                        diagnostics = _uiState.value.diagnostics.copy(
                            uploadInProgress = false,
                            uploadMessage = "已上报脱敏诊断摘要",
                        )
                    )
                }
                .onFailure { error ->
                    _uiState.value = _uiState.value.copy(
                        diagnostics = _uiState.value.diagnostics.copy(
                            uploadInProgress = false,
                            uploadError = error.message ?: error.javaClass.simpleName,
                        )
                    )
                }
        }
    }

    fun checkForAppUpdate() {
        if (_uiState.value.appUpdate.checking || _uiState.value.appUpdate.downloading) return
        viewModelScope.launch {
            _uiState.value = _uiState.value.copy(
                appUpdate = _uiState.value.appUpdate.copy(
                    checking = true,
                    candidate = null,
                    downloadedReady = false,
                    message = null,
                    error = null,
                )
            )
            downloadedUpdateFile = null
            appUpdateManager.checkForUpdate(
                host = _uiState.value.serverHost,
                currentVersion = BuildConfig.VERSION_NAME,
            ).onSuccess { candidate ->
                _uiState.value = _uiState.value.copy(
                    appUpdate = _uiState.value.appUpdate.copy(
                        checking = false,
                        candidate = candidate,
                        installPermissionGranted = appUpdateManager.canRequestPackageInstalls(),
                        message = if (candidate == null) "当前已是最新版本" else "发现 ${candidate.asset.version}",
                    )
                )
            }.onFailure { error ->
                _uiState.value = _uiState.value.copy(
                    appUpdate = _uiState.value.appUpdate.copy(
                        checking = false,
                        error = error.message ?: error.javaClass.simpleName,
                    )
                )
            }
        }
    }

    fun downloadAppUpdate() {
        val candidate = _uiState.value.appUpdate.candidate ?: return
        if (_uiState.value.appUpdate.downloading) return
        viewModelScope.launch {
            _uiState.value = _uiState.value.copy(
                appUpdate = _uiState.value.appUpdate.copy(
                    downloading = true,
                    progress = null,
                    downloadedReady = false,
                    message = null,
                    error = null,
                )
            )
            appUpdateManager.downloadAndVerify(candidate) { progress ->
                _uiState.value = _uiState.value.copy(
                    appUpdate = _uiState.value.appUpdate.copy(progress = progress)
                )
            }.onSuccess { file ->
                downloadedUpdateFile = file
                _uiState.value = _uiState.value.copy(
                    appUpdate = _uiState.value.appUpdate.copy(
                        downloading = false,
                        downloadedReady = true,
                        installPermissionGranted = appUpdateManager.canRequestPackageInstalls(),
                        message = "下载并校验完成",
                    )
                )
            }.onFailure { error ->
                _uiState.value = _uiState.value.copy(
                    appUpdate = _uiState.value.appUpdate.copy(
                        downloading = false,
                        error = error.message ?: error.javaClass.simpleName,
                    )
                )
            }
        }
    }

    fun openInstallPermissionSettings() {
        appUpdateManager.openInstallPermissionSettings()
            .onSuccess {
                _uiState.value = _uiState.value.copy(
                    appUpdate = _uiState.value.appUpdate.copy(
                        installPermissionGranted = appUpdateManager.canRequestPackageInstalls(),
                    )
                )
            }
            .onFailure { error ->
                _uiState.value = _uiState.value.copy(
                    appUpdate = _uiState.value.appUpdate.copy(
                        error = error.message ?: error.javaClass.simpleName,
                    )
                )
        }
    }

    fun refreshAppInstallPermission() {
        _uiState.value = _uiState.value.copy(
            appUpdate = _uiState.value.appUpdate.copy(
                installPermissionGranted = appUpdateManager.canRequestPackageInstalls(),
            )
        )
    }

    fun installDownloadedUpdate() {
        val file = downloadedUpdateFile
        if (file == null || !file.exists()) {
            _uiState.value = _uiState.value.copy(
                appUpdate = _uiState.value.appUpdate.copy(error = "请先下载并校验更新包")
            )
            return
        }
        appUpdateManager.launchInstaller(file)
            .onFailure { error ->
                _uiState.value = _uiState.value.copy(
                    appUpdate = _uiState.value.appUpdate.copy(
                        installPermissionGranted = appUpdateManager.canRequestPackageInstalls(),
                        error = error.message ?: error.javaClass.simpleName,
                    )
                )
            }
    }

    private fun createPassiveRuntimeStatus(inferenceMode: String): AndroidRuntimeStatus {
        val provider = AndroidDeviceInfoProvider(getApplication())
        val nativeResult = runCatching { System.loadLibrary("qlh_llama_jni") }
        return AndroidRuntimeStatus(
            nativeRuntimeAvailable = nativeResult.isSuccess,
            nativeRuntimeError = nativeResult.exceptionOrNull()?.message,
            serviceRunning = false,
            inferenceMode = inferenceMode,
            isLite = BuildConfig.IS_LITE,
            system = provider.getSystemStatus(),
            memory = provider.getMemoryStatus(),
            storage = provider.getStorageStatus(),
            gpu = provider.getGpuStatus(),
        )
    }

    private suspend fun ensureAndroidBootstrap(force: Boolean = false) {
        if (BuildConfig.IS_LITE) return
        if (!force && settings.isBootstrapped()) return

        val state = _uiState.value
        val nodeId = settings.getOrCreateAndroidNodeId()
        val hostname = listOf(
            android.os.Build.MANUFACTURER,
            android.os.Build.MODEL,
        ).filter { it.isNotBlank() }
            .joinToString(" ")
            .ifBlank { nodeId }
        val client = apiClient(state)
        val result = client.firstConnectBootstrap(
            BootstrapRequest(
                nodeId = nodeId,
                hostname = hostname,
                appVariant = if (BuildConfig.IS_LITE) "lite" else "full",
                appVersion = BuildConfig.VERSION_NAME,
                capabilities = buildAndroidPresenceDeviceInfo(),
            )
        )

        result.onSuccess { response ->
            val cluster = response.cluster
            val android = response.android
            val newHost = cluster.masterApiHost.ifBlank { state.serverHost }
            val newPort = cluster.masterApiPort.takeIf { it in 1..65535 } ?: state.serverPort
            settings.saveBootstrapConfig(
                serverHost = newHost,
                serverPort = newPort,
                masterTcpPort = cluster.masterTcpPort,
                clusterId = cluster.clusterId,
                nodeId = response.node.nodeId.ifBlank { nodeId },
                modelManifestUrl = android.modelManifestUrl,
            )
            _uiState.value = _uiState.value.copy(
                serverHost = newHost,
                serverPort = newPort,
            )
            QlhLogger.i(
                "MainViewModel",
                "Android bootstrap completed: nodeId=${response.node.nodeId.ifBlank { nodeId }} host=$newHost:$newPort"
            )
        }.onFailure { e ->
            QlhLogger.w(
                "MainViewModel",
                "Android bootstrap failed: ${e.message ?: e.javaClass.simpleName}"
            )
        }
    }

    private suspend fun autoRegisterAndroidNode(force: Boolean = false) {
        // Lite 变体不参与分布式计算，跳过注册
        if (BuildConfig.IS_LITE) {
            QlhLogger.i("MainViewModel", "autoRegisterAndroidNode: lite variant, skip registration")
            return
        }
        val state = _uiState.value
        if (state.inferenceMode != "thin") {
            getApplication<Application>().stopService(AndroidPresenceService.stopIntent(getApplication()))
            return
        }
        val key = "${state.serverHost}:${state.serverPort}:${state.inferenceMode}"
        if (key == lastAutoRegisterKey && state.presence.state in setOf(
                AndroidPresenceState.ONLINE,
                AndroidPresenceState.REGISTERING,
                AndroidPresenceState.BACKING_OFF,
            )
        ) return
        lastAutoRegisterKey = key

        val nodeId = settings.getOrCreateAndroidNodeId()
        val hostname = listOf(
            android.os.Build.MANUFACTURER,
            android.os.Build.MODEL,
        ).filter { it.isNotBlank() }
            .joinToString(" ")
            .ifBlank { nodeId }

        val networkType = detectNetworkType()
        val deviceInfoJson = Gson().toJson(buildAndroidPresenceDeviceInfo())
        runCatching {
            ContextCompat.startForegroundService(
                getApplication(),
                AndroidPresenceService.startIntent(
                    getApplication(),
                    state.serverHost,
                    state.serverPort,
                    nodeId,
                    hostname,
                    networkType,
                    deviceInfoJson,
                )
            )
        }.onFailure { error ->
            // Presence is an optional control-plane lease. A platform FGS policy
            // failure must not take down the chat UI during cold start.
            QlhLogger.w(
                "MainViewModel",
                "Android presence service unavailable: ${error.message ?: error.javaClass.simpleName}",
            )
        }.onSuccess {
            QlhLogger.i("MainViewModel", "Android presence service started: nodeId=$nodeId host=${state.serverHost}:${state.serverPort}")
        }
    }

    override fun onCleared() {
        getApplication<Application>().stopService(AndroidPresenceService.stopIntent(getApplication()))
        super.onCleared()
    }

    private fun buildAndroidPresenceDeviceInfo(): Map<String, Any?> {
        val provider = AndroidDeviceInfoProvider(getApplication())
        val system = provider.getSystemStatus()
        val memory = provider.getMemoryStatus()
        val gpu = provider.getGpuStatus()
        val runtime = _uiState.value.runtimeStatus
        return buildAndroidPresencePayload(
            inferenceMode = _uiState.value.inferenceMode,
            appVariant = if (BuildConfig.IS_LITE) "lite" else "full",
            appVersion = BuildConfig.VERSION_NAME,
            system = system,
            memory = memory,
            gpu = gpu,
            runtime = runtime,
        )
    }

    /** 通过 ConnectivityManager 检测当前网络类型。 */
    private fun detectNetworkType(): String {
        return try {
            val cm = getApplication<Application>()
                .getSystemService(Context.CONNECTIVITY_SERVICE) as? ConnectivityManager
            val network = cm?.activeNetwork ?: return "unknown"
            val caps = cm.getNetworkCapabilities(network) ?: return "unknown"
            networkTypeFromTransports(
                hasWifi = caps.hasTransport(NetworkCapabilities.TRANSPORT_WIFI),
                hasCellular = caps.hasTransport(NetworkCapabilities.TRANSPORT_CELLULAR),
                hasEthernet = caps.hasTransport(NetworkCapabilities.TRANSPORT_ETHERNET),
                hasVpn = caps.hasTransport(NetworkCapabilities.TRANSPORT_VPN),
            )
        } catch (e: Exception) {
            QlhLogger.w("MainViewModel", "detectNetworkType failed: ${e.message}")
            "unknown"
        }
    }

    // ==================== 连接测试回调 ====================

    /** 测试连接成功后调用，补一刀自动注册（防止之前因服务端未就绪而失败）。 */
    fun onConnectionTestSuccess() {
        viewModelScope.launch {
            ensureAndroidBootstrap(force = true)
            autoRegisterAndroidNode(force = true)
        }
    }

    // ==================== 设置 ====================

    fun setServerHost(host: String) {
        viewModelScope.launch {
            settings.setServerHost(host)
            settings.clearBootstrapConfig()
            _uiState.value = _uiState.value.copy(serverHost = host)
            ensureAndroidBootstrap(force = true)
            autoRegisterAndroidNode(force = true)
        }
    }

    fun setServerPort(port: Int) {
        viewModelScope.launch {
            settings.setServerPort(port)
            settings.clearBootstrapConfig()
            _uiState.value = _uiState.value.copy(serverPort = port)
            ensureAndroidBootstrap(force = true)
            autoRegisterAndroidNode(force = true)
        }
    }

    fun setInferenceMode(mode: String) {
        if (BuildConfig.IS_LITE && mode != "thin") return
        viewModelScope.launch {
            settings.setInferenceMode(mode)
            _uiState.value = _uiState.value.copy(inferenceMode = mode)
            autoRegisterAndroidNode(force = true)
            refreshRuntimeStatus()
        }
    }

    fun setMaxTokens(tokens: Int) {
        viewModelScope.launch {
            settings.setMaxTokens(tokens)
            _uiState.value = _uiState.value.copy(maxTokens = tokens)
        }
    }

    fun setTemperature(temp: Float) {
        viewModelScope.launch {
            settings.setTemperature(temp)
            _uiState.value = _uiState.value.copy(temperature = temp)
        }
    }

    fun setTopP(topP: Float) {
        viewModelScope.launch {
            settings.setTopP(topP)
            _uiState.value = _uiState.value.copy(topP = topP)
        }
    }

    fun setContextSize(size: Int) {
        viewModelScope.launch {
            settings.setContextSize(size)
            _uiState.value = _uiState.value.copy(contextSize = size)
            QlhApplication.instance.inferenceService?.modelContextSize = size
            refreshRuntimeStatus()
        }
    }

    fun setThemeMode(mode: String) {
        viewModelScope.launch {
            settings.setThemeMode(mode)
            _uiState.value = _uiState.value.copy(themeMode = mode)
        }
    }

    // ==================== 模型管理 ====================

    fun selectModelDirectory(treeUri: Uri) {
        viewModelScope.launch {
            _uiState.value = _uiState.value.copy(isScanningModels = true, modelMessage = null)
            unloadRunningModel()
            val result = modelManager.selectModelDirectory(treeUri)
            result.onSuccess { models ->
                val selected = modelManager.getSelectedModel()
                _uiState.value = _uiState.value.copy(
                    availableModels = models,
                    selectedModelUri = selected?.uri?.toString().orEmpty(),
                    selectedModelName = selected?.name.orEmpty(),
                    selectedModelSizeBytes = selected?.sizeBytes ?: 0L,
                    isScanningModels = false,
                    modelMessage = if (models.isEmpty()) {
                        "目录已授权，但未发现 .gguf 模型文件"
                    } else {
                        "已发现 ${models.size} 个 GGUF 模型"
                    }
                )
                refreshRuntimeStatus()
            }.onFailure { e ->
                _uiState.value = _uiState.value.copy(
                    isScanningModels = false,
                    modelMessage = "目录授权失败: ${e.message}"
                )
            }
        }
    }

    fun refreshModels(showMessage: Boolean = true) {
        viewModelScope.launch {
            _uiState.value = _uiState.value.copy(isScanningModels = true, modelMessage = null)
            val result = modelManager.listModels()
            result.onSuccess { models ->
                val selected = modelManager.getSelectedModel()
                _uiState.value = _uiState.value.copy(
                    availableModels = models,
                    selectedModelName = selected?.name.orEmpty(),
                    selectedModelSizeBytes = selected?.sizeBytes ?: 0L,
                    selectedModelUri = selected?.uri?.toString().orEmpty(),
                    isScanningModels = false,
                    modelMessage = if (showMessage) {
                        if (models.isEmpty()) "未发现 .gguf 模型文件" else "已扫描到 ${models.size} 个模型"
                    } else {
                        null
                    }
                )
            }.onFailure { e ->
                _uiState.value = _uiState.value.copy(
                    availableModels = emptyList(),
                    isScanningModels = false,
                    modelMessage = "扫描失败: ${e.message}"
                )
            }
        }
    }

    fun selectModel(model: ModelManager.ModelDocument) {
        viewModelScope.launch {
            unloadRunningModel()
            val result = modelManager.selectModel(model.uri)
            result.onSuccess {
                _uiState.value = _uiState.value.copy(
                    selectedModelUri = model.uri.toString(),
                    selectedModelName = model.name,
                    selectedModelSizeBytes = model.sizeBytes,
                    modelMessage = "已选择模型: ${model.name}"
                )
                refreshRuntimeStatus()
            }.onFailure { e ->
                _uiState.value = _uiState.value.copy(modelMessage = "选择模型失败: ${e.message}")
            }
        }
    }

    fun deleteSelectedModel() {
        viewModelScope.launch {
            val name = _uiState.value.selectedModelName
            unloadRunningModel()
            val result = modelManager.deleteSelectedModel()
            result.onSuccess {
                _uiState.value = _uiState.value.copy(
                    selectedModelUri = "",
                    selectedModelName = "",
                    selectedModelSizeBytes = 0L,
                    modelMessage = if (name.isBlank()) "没有已选择的模型" else "已删除模型: $name"
                )
                refreshModels(showMessage = false)
                refreshRuntimeStatus()
            }.onFailure { e ->
                _uiState.value = _uiState.value.copy(modelMessage = "删除模型失败: ${e.message}")
            }
        }
    }

    fun refreshRemoteModels() {
        if (_uiState.value.remoteModelsLoading || _uiState.value.remoteDownloadModelName != null) return
        viewModelScope.launch {
            val state = _uiState.value
            _uiState.value = state.copy(remoteModelsLoading = true, remoteModelMessage = null)
            apiClient(state).getGgufModels()
                .onSuccess { models ->
                    _uiState.value = _uiState.value.copy(
                        remoteModels = models,
                        remoteModelsLoading = false,
                        remoteModelMessage = if (models.isEmpty()) {
                            "主节点没有可下载的 GGUF 模型"
                        } else {
                            "主节点提供 ${models.size} 个已校验模型"
                        },
                    )
                }
                .onFailure { error ->
                    _uiState.value = _uiState.value.copy(
                        remoteModelsLoading = false,
                        remoteModelMessage = "获取主节点模型失败: ${error.message}",
                    )
                }
        }
    }

    fun downloadRemoteModel(model: GgufModelInfo) {
        if (_uiState.value.remoteDownloadModelName != null) return
        viewModelScope.launch {
            val state = _uiState.value
            _uiState.value = state.copy(
                remoteDownloadModelName = model.filename,
                remoteDownloadProgress = null,
                remoteModelMessage = null,
            )
            try {
                unloadRunningModel()
                modelManager.downloadRemoteModel(
                    model = model,
                    baseUrl = httpBaseUrl(state.serverHost, state.serverPort),
                ).collect { progress ->
                    _uiState.value = _uiState.value.copy(
                        remoteDownloadProgress = progress,
                    )
                }
                val selected = modelManager.getSelectedModel()
                _uiState.value = _uiState.value.copy(
                    selectedModelUri = selected?.uri?.toString().orEmpty(),
                    selectedModelName = selected?.name.orEmpty(),
                    selectedModelSizeBytes = selected?.sizeBytes ?: 0L,
                    remoteModelMessage = "模型已下载并通过 SHA-256 校验",
                )
                refreshModels(showMessage = false)
                refreshRuntimeStatus()
            } catch (error: CancellationException) {
                throw error
            } catch (error: Exception) {
                QlhLogger.e("MainViewModel", "remote model download failed", error)
                _uiState.value = _uiState.value.copy(
                    remoteModelMessage = "模型下载失败: ${error.message ?: error.javaClass.simpleName}",
                )
            } finally {
                _uiState.value = _uiState.value.copy(
                    remoteDownloadModelName = null,
                )
            }
        }
    }

    private suspend fun unloadRunningModel() {
        QlhApplication.instance.inferenceService?.unloadModel()
    }
}
