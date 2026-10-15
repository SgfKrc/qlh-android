package com.qlh.inference

import com.google.gson.Gson
import android.app.Application
import android.content.Context
import android.content.Intent
import android.net.ConnectivityManager
import android.net.NetworkCapabilities
import android.net.Uri
import android.os.PowerManager
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
import com.qlh.inference.network.MasterDiscovery
import com.qlh.inference.network.ModelIntentRequest
import com.qlh.inference.network.canonicalHost
import com.qlh.inference.network.validDiscoveryHost
import com.qlh.inference.network.AndroidPresenceSnapshot
import com.qlh.inference.network.AndroidPresenceState
import com.qlh.inference.network.GgufModelInfo
import com.qlh.inference.network.CurrentModelStatus
import com.qlh.inference.network.httpBaseUrl
import com.qlh.inference.network.ClientErrorReport
import com.qlh.inference.network.BootstrapRequest
import com.qlh.inference.network.ChatRepository
import com.qlh.inference.service.InferenceService
import com.qlh.inference.service.inferenceModes
import com.qlh.inference.service.AndroidPresenceService
import com.qlh.inference.service.ModelManager
import com.qlh.inference.service.ModelDirectoryGrantState
import com.qlh.inference.service.LayerArtifactCandidate
import com.qlh.inference.service.toCandidate
import com.qlh.inference.service.unambiguousLayerArtifactCandidates
import com.qlh.inference.status.AndroidRuntimeStatus
import com.qlh.inference.status.androidRuntimeStatusInvalidation
import com.qlh.inference.system.WorkerBatteryPolicy
import com.qlh.inference.system.AndroidDeviceInfoProvider
import com.qlh.inference.security.AuthTokenStore
import com.qlh.inference.security.StoredAuthSession
import com.qlh.inference.update.AndroidAppUpdateManager
import com.qlh.inference.update.AndroidUpdateCandidate
import com.qlh.inference.network.ControlPlaneProbeResult
import com.qlh.inference.network.probeControlPlaneConnection
import com.qlh.inference.worker.TaskWorkerSnapshot
import com.qlh.inference.worker.TaskWorkerService
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
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock

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
    val modelDirectoryGrant: ModelDirectoryGrantState = ModelDirectoryGrantState(),
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
    val modelIntent: ModelIntentUiState = ModelIntentUiState(),
    val masterDiscoveryBusy: Boolean = false,
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
    val taskWorker: TaskWorkerSnapshot = TaskWorkerSnapshot(),
)

// ================================================================
// ViewModel
// ================================================================

internal data class StandaloneLayerIdentity(val modelId: String, val sourceSha256: String)

internal fun reconcileRemoteModelMessage(
    message: String?,
    grant: ModelDirectoryGrantState,
): String? = if (
    grant.canDownload &&
    (message?.contains("授权已失效") == true || message?.contains("授权失效") == true)
) null else message

internal fun coordinatorSourceModelId(status: CurrentModelStatus?): String =
    status?.descriptor?.sourceModelId?.takeIf { it.isNotBlank() }
        ?: status?.descriptor?.modelId?.takeIf { it.isNotBlank() }
        ?: status?.modelId.orEmpty()

private fun canUseStandaloneLayerIdentity(coordinatorModel: CurrentModelStatus?): Boolean =
    coordinatorModel?.loaded != true && coordinatorModel?.pipelinePrepared != true &&
        coordinatorModel?.descriptor == null && coordinatorModel?.modelId.isNullOrBlank()

/** Only a uniquely selected, digest-verified crop can supply identity before a master descriptor exists. */
internal fun standaloneLayerIdentity(
    coordinatorModel: CurrentModelStatus?,
    selectedModelUri: String,
    selectedModelName: String,
    selectedModelSizeBytes: Long,
    selectedSha256: String,
    verifiedArtifacts: List<LayerArtifactCandidate>,
): StandaloneLayerIdentity? {
    if (!canUseStandaloneLayerIdentity(coordinatorModel)) return null
    val artifact = verifiedArtifacts.filter { it.reference == selectedModelUri }
        .singleOrNull() ?: return null
    val descriptor = artifact.descriptor
    if (verifiedArtifacts.count {
        it.descriptor.startLayer == descriptor.startLayer &&
            it.descriptor.endLayerExclusive == descriptor.endLayerExclusive
    } != 1) return null
    val sha256 = Regex("[0-9a-fA-F]{64}")
    val sourceModelId = descriptor.sourceModelId.orEmpty()
    if (selectedModelUri.isBlank() || selectedModelUri != artifact.reference ||
        !selectedModelName.endsWith(".gguf", ignoreCase = true) ||
        selectedModelName != descriptor.artifactName || selectedModelSizeBytes <= 0L ||
        !sha256.matches(selectedSha256) ||
        !descriptor.artifactSha256.equals(selectedSha256, ignoreCase = true) ||
        !sha256.matches(descriptor.sourceModelSha256) ||
        !Regex("[A-Za-z0-9_.:-]{1,128}").matches(sourceModelId) ||
        descriptor.hiddenSize == null || descriptor.hiddenSize <= 0 ||
        !sha256.matches(descriptor.tokenizerSha256.orEmpty())
    ) return null
    return StandaloneLayerIdentity(sourceModelId, descriptor.sourceModelSha256.lowercase())
}

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
    private var runtimeStatusRefreshJob: Job? = null
    private val modeSwitchMutex = Mutex()
    private val endpointChangeMutex = Mutex()
    private val masterBindingPublicationLock = Any()
    @Volatile private var modeSelectionSerial = 0L
    @Volatile private var masterSelectionSerial = 0L
    @Volatile private var modeSelectionPending = false
    private var endpointSelectionPending = false
    private var intentInitialized = false
    private var intentEndpoint: IntentMasterEndpoint? = null
    private var discoveryJob: Job? = null
    private val modelIntentController = ModelIntentController(viewModelScope, { endpoint ->
        ApiClient(httpBaseUrl(endpoint.host, endpoint.port), authStore = authStore)
    })

    private fun syncModelIntentContext() {
        if (!intentInitialized) return
        modelIntentController.updateContext(
            _uiState.value.inferenceMode, intentEndpoint,
            paused = modeSelectionPending || endpointSelectionPending || _uiState.value.masterDiscoveryBusy,
        )
    }

    private fun retireMasterSelection() {
        endpointSelectionPending = false
        synchronized(masterBindingPublicationLock) { masterSelectionSerial++ }
        discoveryJob?.cancel()
        modelIntentController.suspendForSelection()
        _uiState.value = _uiState.value.copy(masterDiscoveryBusy = false)
    }

    private fun apiClient(state: MainUiState = _uiState.value): ApiClient =
        ApiClient(httpBaseUrl(state.serverHost, state.serverPort), authStore = authStore)

    // ---- 仓库（根据模式动态创建 ApiClient 或使用本地引擎） ----
    private val repository = ChatRepository(
        sessionDao = database.sessionDao(),
        messageDao = database.messageDao(),
        apiClient = {
            val state = _uiState.value
            if (state.inferenceMode != SettingsDataStore.MODE_LOCAL) {
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
        repo.setLocalFallbackPolicy {
            _uiState.value.inferenceMode == SettingsDataStore.MODE_FALLBACK
        }
    }

    init {
        // 加载设置
        viewModelScope.launch {
            try {
                val host = settings.getServerHost()
                val port = settings.getServerPort()
                val mode = if (BuildConfig.IS_LITE) SettingsDataStore.MODE_FALLBACK else settings.getInferenceMode()
                val showThinking = settings.getShowThinking()
                val maxTokens = settings.getMaxTokens()
                val temp = settings.getTemperature()
                val topP = settings.getTopP()
                val contextSize = settings.getContextSize()
                val modelTreeUri = settings.getModelTreeUri()
                val modelDirectoryGrant = modelManager.modelDirectoryGrantState(modelTreeUri)
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
                    serverHost = if (masterSelectionSerial == 0L) host else _uiState.value.serverHost,
                    serverPort = if (masterSelectionSerial == 0L) port else _uiState.value.serverPort,
                    inferenceMode = if (modeSelectionSerial == 0L) mode else _uiState.value.inferenceMode,
                    showThinking = showThinking,
                    maxTokens = maxTokens,
                    temperature = temp,
                    topP = topP,
                    contextSize = contextSize,
                    modelTreeUri = modelTreeUri,
                    modelDirectoryGrant = modelDirectoryGrant,
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
                intentEndpoint = settings.getIntentMasterEndpoint()
                intentInitialized = true
                ensureAndroidBootstrap(allowDiscovery = true)
                syncModelIntentContext()
                autoRegisterAndroidNode()
                ensureAndroidTaskWorker()
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

        // HTTP presence is used only by fallback mode; distributed mode exposes
        // its lease through TaskWorkerService.
        viewModelScope.launch {
            AndroidPresenceService.snapshot.collect { snapshot ->
                _uiState.value = _uiState.value.copy(presence = snapshot)
            }
        }

        viewModelScope.launch {
            TaskWorkerService.workerSnapshot.collect { snapshot ->
                _uiState.value = _uiState.value.copy(taskWorker = snapshot)
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

        // Native layer preparation/retirement can happen without this device changing mode.
        viewModelScope.launch {
            androidRuntimeStatusInvalidation.revisions.collect { refreshRuntimeStatus() }
        }

        // 监听设置变化
        viewModelScope.launch {
            modelIntentController.state.collect { state ->
                _uiState.value = _uiState.value.copy(modelIntent = state)
            }
        }
        viewModelScope.launch {
            settings.intentMasterEndpoint.collect { endpoint ->
                intentEndpoint = endpoint
                if (endpoint != null && !endpointSelectionPending) {
                    _uiState.value = _uiState.value.copy(serverHost = endpoint.host, serverPort = endpoint.port)
                }
                syncModelIntentContext()
            }
        }
        viewModelScope.launch {
            settings.inferenceMode.collect { mode ->
                if (!modeSelectionPending) {
                    _uiState.value = _uiState.value.copy(
                        inferenceMode = if (BuildConfig.IS_LITE) SettingsDataStore.MODE_FALLBACK
                            else SettingsDataStore.normalizeInferenceMode(mode),
                    )
                    syncModelIntentContext()
                }
            }
        }
        viewModelScope.launch {
            settings.modelTreeUri.collect { uri ->
                val grant = modelManager.modelDirectoryGrantState(uri)
                _uiState.value = _uiState.value.copy(
                    modelTreeUri = uri,
                    modelDirectoryGrant = grant,
                    remoteModelMessage = reconcileRemoteModelMessage(
                        _uiState.value.remoteModelMessage,
                        grant,
                    ),
                )
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
            settings.showThinking.collect { enabled ->
                _uiState.value = _uiState.value.copy(showThinking = enabled)
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
                if (!modelIntentAllowsChat(_uiState.value.inferenceMode, modelIntentController.state.value)) {
                    _uiState.value = failMessageSubmission(_uiState.value, "请先在设置选择主节点模型，等待准备完成后聊天")
                    return@launch
                }
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
                if (!modelIntentAllowsChat(_uiState.value.inferenceMode, modelIntentController.state.value)) {
                    _uiState.value = failMessageSubmission(_uiState.value, "请先在设置选择主节点模型，等待准备完成后聊天")
                    return@launch
                }
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
        runtimeStatusRefreshJob?.cancel()
        runtimeStatusRefreshJob = viewModelScope.launch {
            _uiState.value = _uiState.value.copy(
                runtimeStatusLoading = true,
                runtimeStatusError = null
            )
            try {
                val state = _uiState.value
                val service = QlhApplication.instance.inferenceService
                val status = androidRuntimeStatusInvalidation.readSnapshot {
                    service?.getRuntimeStatus(state.inferenceMode, BuildConfig.IS_LITE)
                        ?: createPassiveRuntimeStatus(state.inferenceMode)
                } ?: return@launch
                _uiState.value = _uiState.value.copy(
                    runtimeStatus = status,
                    runtimeStatusLoading = false,
                    runtimeStatusError = null
                )
            } catch (cancelled: CancellationException) {
                throw cancelled
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

    /**
     * ★ 2026-10-08（goal ①-B）：刷新「电池优化是否已豁免」，供设置页显示。
     *
     * 与 [buildBatteryExemptionRequest] 分工：那个是**一次性引导**（弹系统对话框、落库"已问过"）；
     * 这里只读系统状态、不做引导，因此可以随 `onResume` 反复调用。取不到系统状态时保持 `null`
     * （显示"未知"），**不谎报**为"未豁免"。
     */
    fun refreshBatteryExemption() {
        val context = getApplication<Application>()
        val exempt = runCatching {
            (context.getSystemService(Context.POWER_SERVICE) as? PowerManager)
                ?.isIgnoringBatteryOptimizations(context.packageName)
        }.getOrNull()
        _uiState.value = _uiState.value.copy(
            diagnostics = _uiState.value.diagnostics.copy(
                batteryOptimizationExempt = exempt,
            ),
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

    /** Saved endpoint first, then bounded UDP discovery. A hint never changes settings by itself. */
    private suspend fun ensureAndroidBootstrap(force: Boolean = false, allowDiscovery: Boolean = false) {
        if (_uiState.value.inferenceMode == SettingsDataStore.MODE_LOCAL || modeSelectionPending ||
            _uiState.value.masterDiscoveryBusy) return
        val selection = masterSelectionSerial
        val modeSelection = modeSelectionSerial
        fun stillCurrent(): Boolean = selection == masterSelectionSerial && modeSelection == modeSelectionSerial &&
            !modeSelectionPending && _uiState.value.inferenceMode != SettingsDataStore.MODE_LOCAL
        _uiState.value = _uiState.value.copy(masterDiscoveryBusy = true)
        syncModelIntentContext()
        var connected = false
        try {
            val saved = settings.getIntentMasterEndpoint()
            val placeholder = saved?.host == SettingsDataStore.DEFAULT_HOST && !settings.isBootstrapped()
            if (saved != null && !placeholder && stillCurrent()) {
                connected = connectMasterCandidate(saved, force, ::stillCurrent)
            }
            if (!connected && allowDiscovery && stillCurrent()) {
                for (hint in MasterDiscovery().discover()) {
                    if (!stillCurrent()) return
                    connected = connectMasterCandidate(
                        IntentMasterEndpoint(hint.masterApiHost, hint.masterApiPort), true, ::stillCurrent,
                    )
                    if (connected) break
                }
            }
        } catch (cancelled: CancellationException) {
            throw cancelled
        } catch (error: Exception) {
            QlhLogger.w("MainViewModel", "master discovery/connect failed: ${error.javaClass.simpleName}")
        } finally {
            if (stillCurrent()) {
                val latestEndpoint = settings.getIntentMasterEndpoint()
                if (stillCurrent()) {
                    intentEndpoint = latestEndpoint
                    _uiState.value = _uiState.value.copy(masterDiscoveryBusy = false)
                    syncModelIntentContext()
                    if (!connected) modelIntentController.discoveryFailed()
                }
            }
        }
    }

    private suspend fun connectMasterCandidate(
        candidate: IntentMasterEndpoint,
        force: Boolean,
        stillCurrent: () -> Boolean,
    ): Boolean {
        if (!validDiscoveryHost(canonicalHost(candidate.host)) || candidate.port !in 1..65535 || !stillCurrent()) return false
        // Discovery candidates receive no stored bearer token. Confirmation precedes encrypted bootstrap.
        val client = ApiClient(httpBaseUrl(candidate.host, candidate.port), callTimeoutSeconds = 4)
        if (!force && candidate == settings.getIntentMasterEndpoint() && settings.isBootstrapped() &&
            settings.getClusterSecret().isNotBlank() && settings.getMasterTcpHost().isNotBlank()
        ) return client.getBootstrapInfo().getOrNull()?.isMaster == true && stillCurrent()
        val nodeId = settings.getOrCreateAndroidNodeId()
        val hostname = listOf(android.os.Build.MANUFACTURER, android.os.Build.MODEL)
            .filter { it.isNotBlank() }.joinToString(" ").ifBlank { nodeId }
        val bootstrap = client.firstConnectVerifiedMaster(BootstrapRequest(
            nodeId = nodeId, hostname = hostname,
            appVariant = if (BuildConfig.IS_LITE) "lite" else "full",
            appVersion = BuildConfig.VERSION_NAME, capabilities = buildAndroidPresenceDeviceInfo(),
        ), stillCurrent).getOrNull() ?: return false
        if (!stillCurrent()) return false
        val response = bootstrap.response
        val cluster = response.cluster
        val newHost = cluster.masterApiHost.ifBlank { candidate.host }
        val newPort = cluster.masterApiPort.takeIf { it in 1..65535 } ?: candidate.port
        if (!validDiscoveryHost(canonicalHost(newHost)) ||
            !validDiscoveryHost(canonicalHost(cluster.masterTcpHost)) || cluster.masterTcpPort !in 1..65535
        ) return false
        val saved = settings.saveBootstrapConfig(
            serverHost = newHost, serverPort = newPort,
            masterTcpHost = cluster.masterTcpHost, masterTcpPort = cluster.masterTcpPort,
            clusterId = cluster.clusterId, clusterSecret = bootstrap.clusterSecret,
            clusterSecretEpoch = cluster.clusterSecretEpoch,
            nodeId = response.node.nodeId.ifBlank { nodeId },
            modelManifestUrl = response.android.modelManifestUrl,
            stillCurrent = stillCurrent,
            publicationLock = masterBindingPublicationLock,
        )
        if (!saved || !stillCurrent()) return false
        intentEndpoint = settings.getIntentMasterEndpoint()
        if (!stillCurrent()) return false
        _uiState.value = _uiState.value.copy(serverHost = newHost, serverPort = newPort)
        return true
    }

    fun onDiscoverMaster() {
        if (!intentInitialized || modeSelectionPending || _uiState.value.masterDiscoveryBusy ||
            _uiState.value.inferenceMode == SettingsDataStore.MODE_LOCAL) return
        retireMasterSelection()
        discoveryJob = viewModelScope.launch {
            ensureAndroidBootstrap(force = true, allowDiscovery = true)
            autoRegisterAndroidNode(force = true)
            ensureAndroidTaskWorker()
        }
    }

    fun refreshModelIntent() {
        if (!intentInitialized || modeSelectionPending || endpointSelectionPending || _uiState.value.masterDiscoveryBusy) return
        syncModelIntentContext()
        modelIntentController.refresh()
    }

    fun selectMasterModel(modelId: String, engine: String = "auto") {
        modelIntentController.submit(ModelIntentRequest(modelId, engine))
    }

    fun retryModelIntent() = modelIntentController.retry()
    fun cancelModelIntent() = modelIntentController.cancel()

    fun openPreparedChat() {
        if (modelIntentAllowsChat(_uiState.value.inferenceMode, modelIntentController.state.value)) selectTab("chat")
    }

    private suspend fun autoRegisterAndroidNode(force: Boolean = false) {
        // Lite 变体不参与分布式计算，跳过注册
        if (BuildConfig.IS_LITE) {
            QlhLogger.i("MainViewModel", "autoRegisterAndroidNode: lite variant, skip registration")
            return
        }
        val state = _uiState.value
        if (state.inferenceMode != SettingsDataStore.MODE_FALLBACK) {
            // Distributed mode owns the node's TCP task-worker lease. The HTTP
            // presence client is reserved for fallback mode and must not race
            // the TCP registration with a weaker device profile.
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

    /** Start the authenticated TCP worker only for distributed execution. */
    /**
     * ★ 2026-10-07（DIST-NEXT-5b）：需要时返回「申请忽略电池优化」的系统 Intent，否则返回 null。
     *
     * 只问一次（标记持久化到设置），且**不**静默改设置：用户拒绝后 worker 照常跑，只是可能被
     * 厂商省电策略回收。判定本身在 `WorkerBatteryPolicy.shouldRequestExemption()`（已单测）；
     * 这里只负责读系统状态、组装 Intent —— 拉起对话框由 Activity 负责，**标记在拉起成功后**
     * 由 `markBatteryExemptionAsked()` 落库（拉起失败不算"问过"，下次仍会引导）。
     */
    suspend fun buildBatteryExemptionRequest(): Intent? {
        val context = getApplication<Application>()
        val alreadyAsked = runCatching { settings.hasAskedBatteryExemption() }
            .getOrDefault(false)
        val ignoring = runCatching {
            (context.getSystemService(Context.POWER_SERVICE) as? PowerManager)
                ?.isIgnoringBatteryOptimizations(context.packageName) == true
        }.getOrDefault(false)
        if (!WorkerBatteryPolicy.shouldRequestExemption(ignoring, alreadyAsked)) {
            return null
        }
        return Intent(
            WorkerBatteryPolicy.ACTION_REQUEST_IGNORE_BATTERY_OPTIMIZATIONS,
            Uri.parse("package:${context.packageName}"),
        )
    }

    /** ★ 2026-10-07（DIST-NEXT-5b）：引导对话框成功拉起后落库，避免重复打扰。 */
    suspend fun markBatteryExemptionAsked() {
        runCatching { settings.markBatteryExemptionAsked() }
    }

    private suspend fun ensureAndroidTaskWorker() {
        if (BuildConfig.IS_LITE || _uiState.value.inferenceMode != SettingsDataStore.MODE_DISTRIBUTED) {
            stopAndroidTaskWorker("worker_disabled")
            return
        }
        val secret = settings.getClusterSecret()
        val secretEpoch = settings.getClusterSecretEpoch()
        val host = settings.getMasterTcpHost().ifBlank { _uiState.value.serverHost }
        val port = settings.getMasterTcpPort()
        val nodeId = settings.getOrCreateAndroidNodeId()
        if (secret.isBlank() || host.isBlank() || port !in 1..65535) {
            stopAndroidTaskWorker("worker_credentials_or_endpoint_unavailable")
            return
        }

        val selected = modelManager.getSelectedModel()
        val selectedSha256 = modelManager.getSelectedModelSha256()
        val selectedModelId = selected?.name
            ?.substringBeforeLast('.', selected.name)
            ?.takeIf { it.isNotBlank() }
            .orEmpty()
        val coordinatorModel = apiClient().getCurrentModel().getOrNull()
        // ★ 2026-10-11（三机实测 / #92）：身份以 `descriptor.model_id`（模型名，与工件的
        //   `source_model_id` 同口径）为准；顶层 `model_id` 是**注册表 id**，两者不同名时
        //   用它过滤会把工件整片丢掉（真机表现为扫描 ok 但 ranges=[]）。仅在 descriptor
        //   缺失该字段时回退顶层 id（兼容旧服务端）。
        val coordinatorModelId = coordinatorSourceModelId(coordinatorModel)
        // ★ 2026-10-11（Y700 反复重选无效排查）：跨段校验必须用主节点报告的**源模型**摘要。
        //   主节点现在同时给出 `model_sha256`（**该段工件**的 sha，每段不同）与
        //   `source_model_sha256`（源模型 sha，三段共享）。此前只用前者 ⇒ 非主节点段
        //   （如 tail）两边都不等 ⇒ `source_digest_mismatch` ⇒ ranges 不上报。
        //   这里优先取源摘要，缺失时回退工件摘要（兼容旧服务端）。
        val coordinatorSha256 = coordinatorModel?.descriptor?.sourceModelSha256
            ?.trim()?.lowercase()?.takeIf { it.isNotBlank() }
            ?: coordinatorModel?.descriptor?.modelSha256
                ?.trim()?.lowercase().orEmpty()
        val hasCoordinatorIdentity = coordinatorModelId.isNotBlank() &&
            coordinatorSha256.matches(Regex("[a-f0-9]{64}"))
        val noCoordinatorDescriptor = canUseStandaloneLayerIdentity(coordinatorModel)
        val verifiedStandaloneArtifacts = if (noCoordinatorDescriptor && selected != null &&
            selected.name.endsWith(".gguf", ignoreCase = true) && selected.sizeBytes > 0L &&
            selectedSha256.matches(Regex("[a-fA-F0-9]{64}"))
        ) {
            modelManager.listLayerArtifacts(verifyArtifactDigest = true).getOrNull().orEmpty()
        } else emptyList()
        val layerArtifacts = when {
            hasCoordinatorIdentity -> modelManager.listLayerArtifacts(
                expectedModelSha256 = coordinatorSha256,
                verifyArtifactDigest = false,
            ).getOrNull().orEmpty().filter {
                it.sourceModelId == null || it.sourceModelId == coordinatorModelId
            }
            noCoordinatorDescriptor -> verifiedStandaloneArtifacts.filter {
                it.document.uri.toString() == selected?.uri?.toString() &&
                    it.artifactSha256.equals(selectedSha256, ignoreCase = true)
            }
            else -> emptyList()
        }
        val selectedIsCrop = selectedSha256.isNotBlank() && layerArtifacts.any {
            it.artifactSha256.equals(selectedSha256, ignoreCase = true)
        }
        val hasFullModelIdentity = hasCoordinatorIdentity && selectedModelId.isNotBlank() &&
            selectedSha256.matches(Regex("[a-fA-F0-9]{64}")) && !selectedIsCrop &&
            coordinatorModel?.pipelinePrepared != true &&
            selectedSha256.equals(coordinatorSha256, ignoreCase = true)
        val publishableLayers = unambiguousLayerArtifactCandidates(
            layerArtifacts.map { it.toCandidate() },
        )
        val hasAmbiguousLayers = publishableLayers.size != layerArtifacts.size
        val standaloneIdentity = if (noCoordinatorDescriptor) standaloneLayerIdentity(
            coordinatorModel = coordinatorModel,
            selectedModelUri = selected?.uri?.toString().orEmpty(),
            selectedModelName = selected?.name.orEmpty(),
            selectedModelSizeBytes = selected?.sizeBytes ?: 0L,
            selectedSha256 = selectedSha256,
            verifiedArtifacts = verifiedStandaloneArtifacts.map { it.toCandidate() },
        ) else null
        val layerOnlyIdentity = !hasFullModelIdentity &&
            (hasCoordinatorIdentity && publishableLayers.isNotEmpty() || standaloneIdentity != null)
        val modelId = when {
            hasFullModelIdentity -> selectedModelId
            layerOnlyIdentity -> standaloneIdentity?.modelId ?: coordinatorModelId
            else -> ""
        }
        val modelSha256 = when {
            hasFullModelIdentity -> selectedSha256
            layerOnlyIdentity -> standaloneIdentity?.sourceSha256 ?: coordinatorSha256
            else -> ""
        }
        val resourceAdmitted = hasFullModelIdentity || layerOnlyIdentity
        // An admitted resource must carry an empty reason_code: the scheduler
        // treats any non-empty reason as a closed resource gate. The log keeps
        // the distinction between full-model and layer-only admission.
        val resourceReason = when {
            resourceAdmitted -> ""
            layerArtifacts.isNotEmpty() && (hasCoordinatorIdentity || hasAmbiguousLayers) ->
                "layer_artifact_ambiguous"
            hasCoordinatorIdentity -> "master_model_artifact_not_found"
            else -> "model_identity_not_verified"
        }
        if (hasAmbiguousLayers || !resourceAdmitted) {
            _uiState.value = _uiState.value.copy(
                modelMessage = if (hasAmbiguousLayers) {
                    "同一层区间存在多个不同工件；请先移走冲突工件或仅授权目标工件目录，再重新扫描"
                } else {
                    "当前主节点模型工件未匹配：请确认已授权 Download/QLH/models 子目录，并确认所选工件的源模型摘要（source_model_sha256）与主节点报告的源摘要一致、层区间与 manifest 匹配，然后重试"
                },
            )
        }
        QlhLogger.i(
            "MainViewModel",
            "task worker identity: modelId=$modelId sha=${modelSha256.take(12)} " +
                "full=$hasFullModelIdentity layerOnly=$layerOnlyIdentity ranges=" +
                layerArtifacts.map { "[${it.startLayer},${it.endLayerExclusive})" },
        )
        // ★ 2026-09-30（审计「能力撒谎」）：`pipeline_worker` / `backend_id` / `capabilities`
        //   必须**同源**。此前这里无条件硬写前两者，而 `capabilities` 仍来自
        //   `buildAndroidPresenceDeviceInfo()`（按 `nativeRuntimeAvailable` 计算）——
        //   真机实测因此出现 `pipeline_worker:true` + `backend_id:"llama_cpp"` +
        //   `capabilities:[]` 的**自相矛盾**上报；而调度侧 `_client_supports_forward_layers` /
        //   `_node_supports_forward_layers` 优先读 `capabilities`、读不到才用 `backend_id` 兜底
        //   ⇒ 一旦设备 native runtime 真没就绪，字段会一起说谎。
        //   组装逻辑抽到 `buildTaskWorkerDeviceInfo`（纯函数）以便被单测锁死。
        val deviceInfo = buildTaskWorkerDeviceInfo(buildAndroidPresenceDeviceInfo())
        if ((deviceInfo["pipeline_worker"] as? Boolean) != true) {
            QlhLogger.w(
                "MainViewModel",
                "native 运行时未就绪：不上报 pipeline_worker/backend_id" +
                    "（上报值=${deviceInfo["pipeline_worker"]}, " +
                    "能力=${deviceInfo["capabilities"]}）",
            )
        }
        if (settings.getInferenceMode() != SettingsDataStore.MODE_DISTRIBUTED) return
        runCatching {
            ContextCompat.startForegroundService(
                getApplication(),
                TaskWorkerService.startIntent(
                    context = getApplication(),
                    host = host,
                    port = port,
                    nodeId = nodeId,
                    clusterSecret = secret,
                    clusterSecretEpoch = secretEpoch,
                    hostname = listOf(android.os.Build.MANUFACTURER, android.os.Build.MODEL)
                        .filter { it.isNotBlank() }.joinToString(" ").ifBlank { nodeId },
                    networkType = detectNetworkType(),
                    deviceInfo = deviceInfo,
                    modelId = modelId,
                    modelSha256 = modelSha256,
                    modelRevision = modelSha256.takeIf { it.length == 64 }
                        ?.let { "local-${it.take(12)}" } ?: "local",
                    resourceAdmitted = resourceAdmitted,
                    resourceReason = resourceReason,
                    fullInferenceAvailable = hasFullModelIdentity,
                ),
            )
        }.onFailure { error ->
            QlhLogger.w(
                "MainViewModel",
                "Android task worker unavailable: ${error.message ?: error.javaClass.simpleName}",
            )
        }
    }

    private suspend fun stopAndroidTaskWorker(reason: String) {
        val application = getApplication<Application>()
        try {
            TaskWorkerService.requestStop(application)
        } catch (error: RuntimeException) {
            application.stopService(Intent(application, TaskWorkerService::class.java))
            settings.clearTaskWorkerStartupConfig()
            QlhLogger.w(
                "MainViewModel",
                "task worker stop action unavailable; forced stop and cleared startup config: " +
                    "reason=$reason error=${error.javaClass.simpleName}",
            )
        }
    }

    override fun onCleared() {
        val application = getApplication<Application>()
        discoveryJob?.cancel()
        modelIntentController.suspendForSelection()
        application.stopService(AndroidPresenceService.stopIntent(application))
        runCatching { TaskWorkerService.requestStop(application) }
            .onFailure { error ->
                application.stopService(Intent(application, TaskWorkerService::class.java))
                QlhLogger.w(
                    "MainViewModel",
                    "task worker stop action unavailable during ViewModel cleanup: " +
                        error.javaClass.simpleName,
                )
            }
        super.onCleared()
    }

    private fun buildAndroidPresenceDeviceInfo(): Map<String, Any?> {
        val provider = AndroidDeviceInfoProvider(getApplication())
        val system = provider.getSystemStatus()
        val memory = provider.getMemoryStatus()
        val gpu = provider.getGpuStatus()
        // The startup path launches the task worker before the asynchronous
        // runtime-status refresh completes.  A null UI status must not be
        // interpreted as "native unavailable": doing so permanently publishes
        // pipeline_worker=false for the first worker lease even when the JNI
        // library is present and loadable.  Probe synchronously as a fallback;
        // the normal refreshed status still wins once available.
        val runtime = _uiState.value.runtimeStatus
            ?: createPassiveRuntimeStatus(_uiState.value.inferenceMode)
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

    suspend fun testControlPlaneConnection(host: String, apiPort: Int): ControlPlaneProbeResult {
        // A saved TCP endpoint belongs only to its saved API binding. Draft endpoints use the default TCP port.
        val sameBinding = canonicalHost(host) == canonicalHost(settings.getServerHost()) &&
            apiPort == settings.getServerPort()
        val tcpHost = if (sameBinding) settings.getMasterTcpHost().ifBlank { host } else host
        val tcpPort = if (sameBinding) settings.getMasterTcpPort() else 8888
        return probeControlPlaneConnection(tcpHost, tcpPort, bootstrapInfo = {
            ApiClient(httpBaseUrl(host, apiPort), callTimeoutSeconds = 4).getBootstrapInfo()
        })
    }

    /** 测试连接成功后调用，补一刀自动注册（防止之前因服务端未就绪而失败）。 */
    fun onConnectionTestSuccess() {
        viewModelScope.launch {
            ensureAndroidBootstrap(force = true)
            autoRegisterAndroidNode(force = true)
            ensureAndroidTaskWorker()
        }
    }

    // ==================== 设置 ====================

    fun setServerHost(host: String) {
        retireMasterSelection()
        endpointSelectionPending = true
        val selection = masterSelectionSerial
        _uiState.value = _uiState.value.copy(serverHost = host)
        viewModelScope.launch {
            endpointChangeMutex.withLock {
                if (selection != masterSelectionSerial) return@withLock
                settings.setServerHost(host)
                if (selection != masterSelectionSerial) return@withLock
                settings.clearBootstrapConfig()
            }
            if (selection != masterSelectionSerial) return@launch
            val latestEndpoint = settings.getIntentMasterEndpoint()
            if (selection != masterSelectionSerial) return@launch
            intentEndpoint = latestEndpoint
            endpointSelectionPending = false
            syncModelIntentContext()
            ensureAndroidBootstrap(force = true)
            autoRegisterAndroidNode(force = true)
            ensureAndroidTaskWorker()
        }
    }

    fun setServerPort(port: Int) {
        // ★ 2026-10-09 修 #79：兜住非法端口。UI 已做校验，但 datastore 里可能已存在历史脏值
        //   （实测 `802500`），且任何调用方都不该把非法值写进去 —— 非法端口会让
        //   `httpBaseUrl()` 产出非法 URL（`Invalid URL port`）⇒ `ensureAndroidBootstrap` 直接失败
        //   ⇒ 表现为「App 启动即坏、还不说原因」。
        if (port !in 1..65535) {
            QlhLogger.w("MainViewModel", "忽略非法端口: $port（应为 1..65535），保持 ${_uiState.value.serverPort}")
            return
        }
        retireMasterSelection()
        endpointSelectionPending = true
        val selection = masterSelectionSerial
        _uiState.value = _uiState.value.copy(serverPort = port)
        viewModelScope.launch {
            endpointChangeMutex.withLock {
                if (selection != masterSelectionSerial) return@withLock
                settings.setServerPort(port)
                if (selection != masterSelectionSerial) return@withLock
                settings.clearBootstrapConfig()
            }
            if (selection != masterSelectionSerial) return@launch
            val latestEndpoint = settings.getIntentMasterEndpoint()
            if (selection != masterSelectionSerial) return@launch
            intentEndpoint = latestEndpoint
            endpointSelectionPending = false
            syncModelIntentContext()
            ensureAndroidBootstrap(force = true)
            autoRegisterAndroidNode(force = true)
            ensureAndroidTaskWorker()
        }
    }

    fun setInferenceMode(mode: String) {
        if (BuildConfig.IS_LITE && mode != SettingsDataStore.MODE_FALLBACK) return
        val normalized = SettingsDataStore.normalizeInferenceMode(mode)
        val selection = synchronized(masterBindingPublicationLock) { ++modeSelectionSerial }
        retireMasterSelection()
        modeSelectionPending = true
        viewModelScope.launch {
            modeSwitchMutex.withLock {
                if (selection != modeSelectionSerial) return@withLock
                try {
                    inferenceModes.switchMode(
                        mode = normalized,
                        persistMode = { settings.setInferenceMode(it) },
                        releaseModel = {
                            QlhApplication.instance.inferenceService?.unloadModelLocked()
                                ?: Result.success(Unit)
                        },
                    )
                    if (selection != modeSelectionSerial) return@withLock
                    modeSelectionPending = false
                    _uiState.value = _uiState.value.copy(inferenceMode = normalized, modelMessage = null)
                    syncModelIntentContext()
                    autoRegisterAndroidNode(force = true)
                    ensureAndroidTaskWorker()
                    refreshRuntimeStatus()
                } catch (cancelled: CancellationException) {
                    throw cancelled
                } catch (error: Exception) {
                    if (selection == modeSelectionSerial) {
                        modeSelectionPending = false
                        syncModelIntentContext()
                        _uiState.value = _uiState.value.copy(modelMessage = "模式切换失败: ${error.message}")
                    }
                }
            }
        }
    }

    fun setShowThinking(enabled: Boolean) {
        viewModelScope.launch {
            settings.setShowThinking(enabled)
            _uiState.value = _uiState.value.copy(showThinking = enabled)
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

    fun refreshModelDirectoryAuthorization() {
        val current = _uiState.value
        val grant = modelManager.modelDirectoryGrantState(current.modelTreeUri)
        _uiState.value = current.copy(
            modelDirectoryGrant = grant,
            remoteModelMessage = reconcileRemoteModelMessage(
                current.remoteModelMessage,
                grant,
            ),
        )
    }

    fun selectModelDirectory(treeUri: Uri) {
        viewModelScope.launch {
            _uiState.value = _uiState.value.copy(
                isScanningModels = true,
                modelMessage = null,
                remoteModelMessage = null,
            )
            unloadRunningModel()
            val result = modelManager.selectModelDirectory(treeUri)
            result.onSuccess { models ->
                val selected = modelManager.getSelectedModel()
                val grant = modelManager.modelDirectoryGrantState(treeUri.toString())
                _uiState.value = _uiState.value.copy(
                    availableModels = models,
                    modelDirectoryGrant = grant,
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
                ensureAndroidTaskWorker()
                refreshRuntimeStatus()
            }.onFailure { e ->
                val grant = modelManager.modelDirectoryGrantState(_uiState.value.modelTreeUri)
                _uiState.value = _uiState.value.copy(
                    modelDirectoryGrant = grant,
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
                val grant = modelManager.modelDirectoryGrantState(_uiState.value.modelTreeUri)
                _uiState.value = _uiState.value.copy(
                    availableModels = models,
                    modelDirectoryGrant = grant,
                    selectedModelName = selected?.name.orEmpty(),
                    selectedModelSizeBytes = selected?.sizeBytes ?: 0L,
                    selectedModelUri = selected?.uri?.toString().orEmpty(),
                    isScanningModels = false,
                    modelMessage = if (showMessage) {
                        if (models.isEmpty()) {
                            "未发现 .gguf 模型；请授权 Download/QLH/models 子目录后重试"
                        } else "已扫描到 ${models.size} 个模型"
                    } else {
                        null
                    },
                    remoteModelMessage = reconcileRemoteModelMessage(
                        _uiState.value.remoteModelMessage,
                        grant,
                    ),
                )
            }.onFailure { e ->
                val grant = modelManager.modelDirectoryGrantState(_uiState.value.modelTreeUri)
                _uiState.value = _uiState.value.copy(
                    availableModels = emptyList(),
                    modelDirectoryGrant = grant,
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
                ensureAndroidTaskWorker()
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
                ensureAndroidTaskWorker()
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
                ensureAndroidTaskWorker()
                refreshModels(showMessage = false)
                refreshRuntimeStatus()
            } catch (error: CancellationException) {
                throw error
            } catch (error: Exception) {
                QlhLogger.e("MainViewModel", "remote model download failed", error)
                val grant = modelManager.modelDirectoryGrantState(_uiState.value.modelTreeUri)
                _uiState.value = _uiState.value.copy(
                    modelDirectoryGrant = grant,
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
