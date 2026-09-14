package com.qlh.inference.network

import com.google.gson.Gson
import com.google.gson.JsonParser
import com.google.gson.annotations.SerializedName
import com.qlh.inference.BuildConfig
import com.qlh.inference.security.AuthSessionStore
import com.qlh.inference.security.StoredAuthSession
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.delay
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlinx.coroutines.withContext
import okhttp3.Call
import okhttp3.Callback
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import okhttp3.Response
import okhttp3.logging.HttpLoggingInterceptor
import java.io.IOException
import java.util.concurrent.TimeoutException
import java.util.concurrent.TimeUnit
import kotlin.coroutines.resume
import kotlin.coroutines.resumeWithException

// ================================================================
// DTO — 匹配 PC 端 api_server.py 的 ChatRequest / ChatResponse
// ================================================================

data class ChatRequest(
    val message: String,
    /** PC /api/chat 的多模态输入；空列表保持原有纯文本请求语义。 */
    @SerializedName("image_data_urls")
    val imageDataUrls: List<String> = emptyList(),
    @SerializedName("max_new_tokens")
    val maxNewTokens: Int = 1024,
    val temperature: Float = 0.7f,
    @SerializedName("top_p")
    val topP: Float = 0.9f,
    @SerializedName("show_thinking")
    val showThinking: Boolean = false,
    @SerializedName("session_id")
    val sessionId: String? = null,
    @SerializedName("streaming_mode")
    val streamingMode: String = "full",    // "full" | "fast"（对应 PC 端 streaming_mode）
    @SerializedName("client_node_id")
    val clientNodeId: String? = null,
    @SerializedName("client_node_type")
    val clientNodeType: String? = null,
    @SerializedName("client_mode")
    val clientMode: String? = null,
    @SerializedName("routing_preference")
    val routingPreference: String = "distributed_preferred",
    @SerializedName("client_app_variant")
    val clientAppVariant: String? = null,
    /** Only populated for an image request; PC requires explicit multimodal routing consent. */
    @SerializedName("allow_external")
    val allowExternal: Boolean? = null,
    @SerializedName("prefer_external")
    val preferExternal: Boolean? = null,
)

data class ChatResponse(
    val content: String = "",
    val metrics: Map<String, Any>? = null,
    val followups: List<String>? = null,
    val error: String? = null
)

data class SessionInfo(
    val id: String = "",
    val title: String = "新对话",
    @SerializedName("message_count")
    val messageCount: Int = 0,
    @SerializedName("created_at")
    val createdAt: String? = null,
    @SerializedName("updated_at")
    val updatedAt: String? = null
)

data class ClusterStatus(
    val running: Boolean = false,
    @SerializedName("run_mode")
    val runMode: String = "",
    @SerializedName("nodes_ready")
    val nodesReady: Boolean = false,
    /** `api_server.py` returns this as a node-id keyed object, not an array. */
    val nodes: Map<String, ClusterNode> = emptyMap(),
    @SerializedName("current_task")
    val currentTask: ClusterTask? = null,
    @SerializedName("online_count")
    val onlineCount: Int = 0,
    @SerializedName("total_count")
    val totalCount: Int = 0,
    @SerializedName("distributed_enabled")
    val distributedEnabled: Boolean = false
)

data class ClusterTask(
    @SerializedName("task_id")
    val taskId: String = "",
    val state: String = "",
    val elapsed: Double = 0.0,
)

data class ClusterNode(
    @SerializedName("node_id")
    val nodeId: String = "",
    val role: String = "",
    @SerializedName("node_type")
    val nodeType: String = "pc",
    val state: String = "",
    val hostname: String = "",
    val address: String = "",
    @SerializedName("network_type")
    val networkType: String = "",
    @SerializedName("task_count")
    val taskCount: Int = 0,
    @SerializedName("avg_rtt_ms")
    val avgRttMs: Float = 0f,
    @SerializedName("error_count")
    val errorCount: Int = 0,
    @SerializedName("is_available")
    val isAvailable: Boolean = false,
)

data class RegisterNodeRequest(
    @SerializedName("node_id")
    val nodeId: String,
    val hostname: String,
    val address: String = "",
    @SerializedName("network_type")
    val networkType: String = "unknown",
    @SerializedName("node_type")
    val nodeType: String = "android",
    @SerializedName("device_info")
    val deviceInfo: Map<String, Any?> = emptyMap(),
    @SerializedName("client_mode")
    val clientMode: String = "thin",
    @SerializedName("app_variant")
    val appVariant: String = if (BuildConfig.IS_LITE) "lite" else "full",
    @SerializedName("app_version")
    val appVersion: String = BuildConfig.VERSION_NAME,
    @SerializedName("presence_generation")
    val presenceGeneration: Long = 0L,
    @SerializedName("presence_lease_id")
    val presenceLeaseId: String = ""
)

data class RegisterNodeResponse(
    val status: String = "",
    @SerializedName("node_id")
    val nodeId: String? = null,
    val message: String? = null,
    val state: String? = null,
    @SerializedName("server_time_ms")
    val serverTimeMs: Long = 0L,
    @SerializedName("presence_generation")
    val presenceGeneration: Long = 0L,
    @SerializedName("presence_lease_id")
    val presenceLeaseId: String = "",
    @SerializedName("lease_expires_at_ms")
    val leaseExpiresAtMs: Long = 0L,
    @SerializedName("heartbeat_interval_seconds")
    val heartbeatIntervalSeconds: Int = 45
)

data class AndroidPresenceHeartbeatRequest(
    @SerializedName("node_id") val nodeId: String,
    @SerializedName("presence_generation") val presenceGeneration: Long,
    @SerializedName("presence_lease_id") val presenceLeaseId: String,
)

class ApiClientHttpException(
    val statusCode: Int,
    val errorCode: String,
    val responseBody: String,
) : IOException("HTTP $statusCode${if (errorCode.isBlank()) "" else " [$errorCode]"}: $responseBody")

data class BootstrapRequest(
    @SerializedName("node_id")
    val nodeId: String,
    @SerializedName("node_type")
    val nodeType: String = "android",
    val hostname: String = "",
    val platform: String = "android",
    @SerializedName("app_variant")
    val appVariant: String = if (BuildConfig.IS_LITE) "lite" else "full",
    @SerializedName("app_version")
    val appVersion: String = BuildConfig.VERSION_NAME,
    val capabilities: Map<String, Any?> = emptyMap()
)

data class BootstrapCluster(
    @SerializedName("cluster_id")
    val clusterId: String = "",
    @SerializedName("master_api_host")
    val masterApiHost: String = "",
    @SerializedName("master_api_port")
    val masterApiPort: Int = 8000,
    @SerializedName("master_tcp_host")
    val masterTcpHost: String = "",
    @SerializedName("master_tcp_port")
    val masterTcpPort: Int = 8888,
    @SerializedName("cluster_secret")
    val clusterSecret: String = ""
)

data class BootstrapNode(
    @SerializedName("node_id")
    val nodeId: String = "",
    val role: String = "client",
    @SerializedName("node_type")
    val nodeType: String = "android",
    @SerializedName("pipeline_worker")
    val pipelineWorker: Boolean = false
)

data class BootstrapAndroid(
    @SerializedName("presence_interval_seconds")
    val presenceIntervalSeconds: Int = 45,
    @SerializedName("pipeline_worker")
    val pipelineWorker: Boolean = false,
    @SerializedName("model_manifest_url")
    val modelManifestUrl: String = ""
)

data class BootstrapResponse(
    val status: String = "",
    val cluster: BootstrapCluster = BootstrapCluster(),
    val node: BootstrapNode = BootstrapNode(),
    val android: BootstrapAndroid = BootstrapAndroid()
)

// ================================================================
data class GgufModelInfo(
    val filename: String = "",
    @SerializedName("size_bytes") val sizeBytes: Long = 0L,
    @SerializedName("size_mb") val sizeMb: Double = 0.0,
    val sha256: String = "",
    @SerializedName("download_url") val downloadUrl: String = "",
)

data class GgufModelsResponse(
    val models: List<GgufModelInfo> = emptyList(),
    val exists: Boolean = false,
    val count: Int = 0,
)

/** Path-free model configuration returned by the main-node registry. */
data class ServerModelSummary(
    @SerializedName("model_id") val modelId: String = "",
    val name: String = "",
    @SerializedName("model_type") val modelType: String = "",
    @SerializedName("is_builtin") val isBuiltin: Boolean = false,
    @SerializedName("is_experimental") val isExperimental: Boolean = false,
    @SerializedName("is_available") val isAvailable: Boolean = false,
    @SerializedName("unavailable_reason") val unavailableReason: String = "",
    @SerializedName("available_formats") val availableFormats: List<String> = emptyList(),
    @SerializedName("has_safetensors") val hasSafetensors: Boolean = false,
    @SerializedName("has_gguf") val hasGguf: Boolean = false,
    @SerializedName("preferred_engine") val preferredEngine: String = "",
    @SerializedName("recommended_vram_gb") val recommendedVramGb: Double? = null,
)

data class ServerModelsResponse(
    val models: List<ServerModelSummary> = emptyList(),
    @SerializedName("active_model_id") val activeModelId: String? = null,
)

/** Runtime status deliberately excludes server filesystem paths. */
data class CurrentModelStatus(
    val loaded: Boolean = false,
    @SerializedName("pipeline_prepared") val pipelinePrepared: Boolean = false,
    @SerializedName("model_id") val modelId: String? = null,
    @SerializedName("model_name") val modelName: String = "",
    val engine: String = "",
    @SerializedName("quant_type") val quantType: String? = null,
)

/** Read-only local-asset projection. Paths and runtime hints stay on the main node. */
data class LocalModelAssetSummary(
    @SerializedName("model_id") val modelId: String = "",
    val name: String = "",
    @SerializedName("model_type") val modelType: String = "",
    @SerializedName("available_formats") val availableFormats: List<String> = emptyList(),
    @SerializedName("total_bytes") val totalBytes: Long = 0L,
    val integrity: String = "",
    @SerializedName("runtime_status") val runtimeStatus: String = "",
)

data class LocalModelAssetsResponse(
    val assets: List<LocalModelAssetSummary> = emptyList(),
    val summary: LocalModelAssetsSummary = LocalModelAssetsSummary(),
)

data class LocalModelAssetsSummary(
    val total: Int = 0,
    @SerializedName("total_bytes") val totalBytes: Long = 0L,
)

/** Inputs for the Android read-only model-fleet projection. */
data class MasterModelFleetData(
    val current: CurrentModelStatus = CurrentModelStatus(),
    val registry: ServerModelsResponse = ServerModelsResponse(),
    val localAssets: LocalModelAssetsResponse = LocalModelAssetsResponse(),
    val verifiedGguf: List<GgufModelInfo> = emptyList(),
)

/** Content-free audit summaries returned by /api/workflows?summary=1. */
data class AuditAttemptSummary(
    @SerializedName("attempt_id") val attemptId: String = "",
    @SerializedName("provider_kind") val providerKind: String = "",
    @SerializedName("provider_node_id") val providerNodeId: String = "",
    val state: String = "",
    @SerializedName("error_code") val errorCode: String = "",
    @SerializedName("started_at") val startedAt: Double = 0.0,
    @SerializedName("finished_at") val finishedAt: Double = 0.0,
    @SerializedName("duration_seconds") val durationSeconds: Double = 0.0,
)

data class AuditStageSummary(
    @SerializedName("stage_id") val stageId: String = "",
    @SerializedName("stage_type") val stageType: String = "",
    val state: String = "",
    @SerializedName("started_at") val startedAt: Double = 0.0,
    @SerializedName("finished_at") val finishedAt: Double = 0.0,
    @SerializedName("duration_seconds") val durationSeconds: Double = 0.0,
    @SerializedName("retry_count") val retryCount: Int = 0,
    @SerializedName("result_rejection_count") val resultRejectionCount: Int = 0,
    @SerializedName("error_code") val errorCode: String = "",
    @SerializedName("attempt_count") val attemptCount: Int = 0,
    val attempts: List<AuditAttemptSummary> = emptyList(),
)

data class AuditObservabilitySummary(
    val state: String = "",
    @SerializedName("result_ready") val resultReady: Boolean = false,
    val terminal: Boolean = false,
    @SerializedName("partial_result") val partialResult: Boolean = false,
    @SerializedName("recovered_after_restart") val recoveredAfterRestart: Boolean = false,
    @SerializedName("retry_count") val retryCount: Int = 0,
    @SerializedName("same_provider_retry_count") val sameProviderRetryCount: Int = 0,
    @SerializedName("reassignment_count") val reassignmentCount: Int = 0,
    val retrying: Boolean = false,
    @SerializedName("result_rejection_count") val resultRejectionCount: Int = 0,
    @SerializedName("winner_count") val winnerCount: Int = 0,
    @SerializedName("actual_providers") val actualProviders: List<String> = emptyList(),
    @SerializedName("actual_nodes") val actualNodes: List<String> = emptyList(),
)

data class AuditWorkflowSummary(
    @SerializedName("workflow_id") val workflowId: String = "",
    val template: String = "",
    val state: String = "",
    @SerializedName("created_at") val createdAt: Double = 0.0,
    @SerializedName("started_at") val startedAt: Double = 0.0,
    @SerializedName("result_ready_at") val resultReadyAt: Double = 0.0,
    @SerializedName("finished_at") val finishedAt: Double = 0.0,
    @SerializedName("duration_seconds") val durationSeconds: Double = 0.0,
    @SerializedName("stage_count") val stageCount: Int = 0,
    @SerializedName("completed_stage_count") val completedStageCount: Int = 0,
    @SerializedName("failed_stage_count") val failedStageCount: Int = 0,
    @SerializedName("attempt_count") val attemptCount: Int = 0,
    @SerializedName("retry_count") val retryCount: Int = 0,
    @SerializedName("same_provider_retry_count") val sameProviderRetryCount: Int = 0,
    @SerializedName("result_rejection_count") val resultRejectionCount: Int = 0,
    @SerializedName("cancel_requested") val cancelRequested: Boolean = false,
    val observability: AuditObservabilitySummary = AuditObservabilitySummary(),
    val stages: List<AuditStageSummary> = emptyList(),
)

data class AuditWorkflowsResponse(
    val enabled: Boolean = false,
    val available: Boolean = false,
    val role: String = "",
    val workflows: List<AuditWorkflowSummary> = emptyList(),
)

/** Review ticket projection excludes transfer reasons, comments and voter identities. */
data class ReviewTicketSummary(
    @SerializedName("ticket_id") val ticketId: String = "",
    val status: String = "",
    @SerializedName("created_at") val createdAt: Double = 0.0,
    @SerializedName("target_node_id") val targetNodeId: String = "",
    val score: Int = 0,
    @SerializedName("expires_at") val expiresAt: Double = 0.0,
    @SerializedName("resolved_at") val resolvedAt: Double = 0.0,
    @SerializedName("vote_count") val voteCount: Int = 0,
)

data class AuditReviewTicketsResponse(
    val tickets: List<ReviewTicketSummary> = emptyList(),
    val count: Int = 0,
)

data class AuditData(
    val workflows: AuditWorkflowsResponse = AuditWorkflowsResponse(),
    val reviewTickets: AuditReviewTicketsResponse = AuditReviewTicketsResponse(),
)

data class AuthLoginRequest(
    val username: String,
    val code: String? = null,
    @SerializedName("recovery_code") val recoveryCode: String? = null,
)

data class AuthCapabilityResponse(
    val required: Boolean = false,
    val enforced: Boolean = false,
    val available: Boolean = false,
    val mode: String = "",
    @SerializedName("bootstrap_available") val bootstrapAvailable: Boolean = false,
    @SerializedName("reason_code") val reasonCode: String = "",
    @SerializedName("policy_version") val policyVersion: String = "",
    val service: String = "",
)

data class AuthUser(
    @SerializedName("user_id") val userId: String = "",
    val username: String = "",
    @SerializedName("display_name") val displayName: String? = null,
    val role: String = "",
    val status: String? = null,
    @SerializedName("totp_state") val totpState: String? = null,
    @SerializedName("active_session_count") val activeSessionCount: Int? = null,
    @SerializedName("aggregate_version") val aggregateVersion: Int? = null,
)

data class AuthLoginResponse(
    @SerializedName("access_token") val accessToken: String = "",
    @SerializedName("token_type") val tokenType: String = "Bearer",
    @SerializedName("session_id") val sessionId: String = "",
    @SerializedName("expires_at") val expiresAt: String = "",
    val user: AuthUser = AuthUser(),
) {
    fun toStoredSession(): StoredAuthSession {
        require(tokenType.equals("Bearer", ignoreCase = true)) { "unsupported auth token type" }
        return StoredAuthSession(
        accessToken = accessToken,
        sessionId = sessionId,
        expiresAt = expiresAt,
        userId = user.userId,
        username = user.username,
        displayName = user.displayName,
        role = user.role,
        )
    }
}

data class AuthSessionResponse(
    @SerializedName("session_id") val sessionId: String = "",
    @SerializedName("expires_at") val expiresAt: String = "",
    val user: AuthUser = AuthUser(),
)

// ---- AND-CTRL-05 前置契约：管理授权矩阵摘要 / 审计 / 二次确认 ----

data class ManageActionRule(
    val allowed: Boolean = false,
    @SerializedName("confirm_required") val confirmRequired: Boolean = false,
    val audited: Boolean = false,
    val description: String = "",
)

data class ManageSummaryResponse(
    val role: String = "",
    @SerializedName("policy_version") val policyVersion: String = "",
    @SerializedName("audit_available") val auditAvailable: Boolean = false,
    @SerializedName("confirm_ttl_seconds") val confirmTtlSeconds: Int = 120,
    val actions: Map<String, ManageActionRule> = emptyMap(),
    val counts: Map<String, Int> = emptyMap(),
    @SerializedName("review_admin_auth_pending") val reviewAdminAuthPending: Boolean = true,
)

data class ManageAuditEvent(
    @SerializedName("event_id") val eventId: String = "",
    @SerializedName("event_type") val eventType: String = "",
    val outcome: String = "",
    @SerializedName("reason_code") val reasonCode: String? = null,
    @SerializedName("actor_user_id") val actorUserId: String? = null,
    @SerializedName("user_id") val userId: String? = null,
    @SerializedName("subject_id") val subjectId: String? = null,
    @SerializedName("created_at") val createdAt: String = "",
)

data class ManageAuditResponse(
    val events: List<ManageAuditEvent> = emptyList(),
)

data class ManageConfirmResponse(
    @SerializedName("confirm_token") val confirmToken: String = "",
    @SerializedName("expires_at") val expiresAt: String = "",
    val action: String = "",
    @SerializedName("target_id") val targetId: String = "",
) {
    fun isValid(): Boolean = confirmToken.isNotBlank() && action.isNotBlank()
}

data class ManagedUsersResponse(
    val users: List<AuthUser> = emptyList(),
)

data class TailscaleBindingsResponse(
    val bindings: List<TailscaleBinding> = emptyList(),
)

data class TailscaleBinding(
    @SerializedName("binding_id") val bindingId: String = "",
    @SerializedName("user_id") val userId: String? = null,
    @SerializedName("tailnet_id") val tailnetId: String? = null,
    @SerializedName("tailscale_user_id") val tailscaleUserId: String? = null,
    @SerializedName("node_id") val nodeId: String? = null,
    val state: String? = null,
    @SerializedName("authorization_method") val authorizationMethod: String? = null,
    @SerializedName("confirmed_at") val confirmedAt: String? = null,
    @SerializedName("updated_at") val updatedAt: String? = null,
)

// ================================================================
// API 客户端
// ================================================================

class ApiClient(
    private val baseUrl: String,
    private val gson: Gson = Gson(),
    private val authStore: AuthSessionStore? = null,
) {
    /** Login is anonymous; all other control-plane requests may use the local session. */
    private val anonymousPaths = setOf(
        "/api/auth/capability",
        "/api/auth/login",
        "/api/auth/bootstrap",
        "/api/auth/totp/verify",
        "/api/bootstrap/first-connect",
        "/api/cluster/android/register",
        "/api/cluster/android/heartbeat",
    )

    private val client: OkHttpClient = OkHttpClient.Builder()
        .connectTimeout(10, TimeUnit.SECONDS)
        .readTimeout(120, TimeUnit.SECONDS)   // 推理可能较慢
        .writeTimeout(30, TimeUnit.SECONDS)
        .addInterceptor { chain ->
            val original = chain.request()
            val anonymous = anonymousPaths.contains(original.url.encodedPath)
            val token = if (!anonymous) authStore?.read()?.accessToken else null
            val request = if (!token.isNullOrBlank() && original.header("Authorization") == null) {
                original.newBuilder().header("Authorization", "Bearer $token").build()
            } else {
                original
            }
            val response = chain.proceed(request)
            if (response.code == 401 && !anonymous) authStore?.clear()
            response
        }
        .addInterceptor(HttpLoggingInterceptor().apply {
            redactHeader("Authorization")
            redactHeader("X-QLH-Confirm-Token")
            // Never log request/response bodies: chat content and one-shot
            // management confirmation tokens are user secrets.
            level = if (BuildConfig.DEBUG) {
                HttpLoggingInterceptor.Level.HEADERS
            } else {
                HttpLoggingInterceptor.Level.NONE
            }
        })
        .build()

    private val jsonMediaType = "application/json; charset=utf-8".toMediaType()


    /** Read the public auth boundary without requiring a local bearer token. */
    suspend fun getAuthCapability(): Result<AuthCapabilityResponse> = withContext(Dispatchers.IO) {
        try {
            val request = Request.Builder()
                .url("$baseUrl/api/auth/capability")
                .get()
                .build()
            executeJson(request, AuthCapabilityResponse::class.java)
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    /** Authenticate against the local control plane and persist the encrypted session. */
    suspend fun login(
        username: String,
        code: String? = null,
        recoveryCode: String? = null,
    ): Result<StoredAuthSession> = withContext(Dispatchers.IO) {
        try {
            require(username.isNotBlank()) { "username is required" }
            require(!code.isNullOrBlank() || !recoveryCode.isNullOrBlank()) {
                "code or recoveryCode is required"
            }
            val payload = AuthLoginRequest(username, code, recoveryCode)
            val body = gson.toJson(payload).toRequestBody(jsonMediaType)
            val request = Request.Builder()
                .url("$baseUrl/api/auth/login")
                .post(body)
                .header("Content-Type", "application/json")
                .build()
            val response = executeAsync(request)
            val responseBody = response.body?.string() ?: "{}"
            if (!response.isSuccessful) {
                return@withContext Result.failure(IOException("HTTP ${response.code}"))
            }
            val session = gson.fromJson(responseBody, AuthLoginResponse::class.java).toStoredSession()
            require(session.isValid()) { "auth login response is invalid" }
            authStore?.save(session)
                ?: return@withContext Result.failure(IllegalStateException("auth session store unavailable"))
            Result.success(session)
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    /** Read the current authenticated session without exposing the bearer token to callers. */
    suspend fun getAuthSession(): Result<AuthSessionResponse> = withContext(Dispatchers.IO) {
        try {
            val request = Request.Builder()
                .url("$baseUrl/api/auth/session")
                .get()
                .build()
            executeJson(request, AuthSessionResponse::class.java)
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    /** Revoke the server session when reachable and always clear local credentials. */
    suspend fun logout(): Result<Unit> = withContext(Dispatchers.IO) {
        try {
            val request = Request.Builder()
                .url("$baseUrl/api/auth/logout")
                .post(ByteArray(0).toRequestBody(null))
                .build()
            val response = executeAsync(request)
            if (!response.isSuccessful) {
                return@withContext Result.failure(IOException("HTTP ${response.code}"))
            }
            Result.success(Unit)
        } catch (e: Exception) {
            Result.failure(e)
        } finally {
            authStore?.clear()
        }
    }

    // ==================== AND-CTRL-05 前置：管理摘要/审计/二次确认 ====================

    /** 管理授权矩阵摘要（仅 owner/admin 有完整内容；member 侧 allowed=false）。 */
    suspend fun fetchManageSummary(): Result<ManageSummaryResponse> = withContext(Dispatchers.IO) {
        try {
            val request = Request.Builder()
                .url("$baseUrl/api/auth/manage/summary")
                .get()
                .build()
            executeJson(request, ManageSummaryResponse::class.java)
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    /** 管理写操作审计列表（只读，服务端有界分页；Android 不信任调用方 page size）。 */
    suspend fun fetchManageAudit(limit: Int = 50): Result<ManageAuditResponse> = withContext(Dispatchers.IO) {
        try {
            val bounded = limit.coerceIn(1, 200)
            val request = Request.Builder()
                .url("$baseUrl/api/auth/manage/audit?limit=$bounded")
                .get()
                .build()
            executeJson(request, ManageAuditResponse::class.java)
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    /** 危险管理写操作的一次性确认令牌（二次确认第一步）。 */
    suspend fun requestManageConfirm(action: String, targetId: String): Result<ManageConfirmResponse> = withContext(Dispatchers.IO) {
        try {
            require(action.isNotBlank()) { "action is required" }
            require(targetId.isNotBlank()) { "target id is required" }
            // Use the JSON encoder rather than string interpolation: target IDs are
            // user-owned data and must not be able to alter the confirmation body.
            val payload = gson.toJson(mapOf("action" to action, "target_id" to targetId))
            val request = Request.Builder()
                .url("$baseUrl/api/auth/manage/confirm")
                .post(payload.toRequestBody(jsonMediaType))
                .header("Content-Type", "application/json")
                .build()
            val response = executeAsync(request)
            val body = response.body?.string() ?: "{}"
            if (!response.isSuccessful) {
                return@withContext Result.failure(IOException("HTTP ${response.code}: ${response.message}"))
            }
            val confirmation = gson.fromJson(body, ManageConfirmResponse::class.java)
            if (!confirmation.isValid()) {
                return@withContext Result.failure(IOException("confirm token response invalid"))
            }
            Result.success(confirmation)
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    /** Bounded manager-only user projection; secret/authenticator fields are discarded by DTO. */
    suspend fun fetchManagedUsers(): Result<ManagedUsersResponse> = withContext(Dispatchers.IO) {
        try {
            val request = Request.Builder()
                .url("$baseUrl/api/users")
                .get()
                .build()
            executeJson(request, ManagedUsersResponse::class.java)
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    /** Read one user's safe Tailnet binding projection for the manager surface. */
    suspend fun fetchUserTailscaleBindings(userId: String): Result<TailscaleBindingsResponse> = withContext(Dispatchers.IO) {
        try {
            require(userId.isNotBlank()) { "user id is required" }
            val encodedUserId = java.net.URLEncoder.encode(userId, Charsets.UTF_8.name())
            val request = Request.Builder()
                .url("$baseUrl/api/auth/users/$encodedUserId/tailscale")
                .get()
                .build()
            executeJson(request, TailscaleBindingsResponse::class.java)
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    /** 撤销用户（管理写操作，需先 requestManageConfirm("user_manage", userId) 取令牌）。 */
    suspend fun revokeUser(userId: String, expectedVersion: Int, confirmToken: String): Result<Unit> = withContext(Dispatchers.IO) {
        try {
            require(confirmToken.isNotBlank()) { "confirm token is required for revoke" }
            val payload = "{\"expected_version\":$expectedVersion}"
            val request = Request.Builder()
                .url("$baseUrl/api/auth/users/${userId}")
                .delete(payload.toRequestBody(jsonMediaType))
                .header("Content-Type", "application/json")
                .header("X-QLH-Confirm-Token", confirmToken)
                .build()
            val response = executeAsync(request)
            if (!response.isSuccessful) {
                return@withContext Result.failure(IOException("HTTP ${response.code}"))
            }
            Result.success(Unit)
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    /** 撤销 Tailnet 绑定（跨用户需确认令牌；成员撤销自己绑定免确认）。 */
    suspend fun revokeTailscaleBinding(bindingId: String, confirmToken: String? = null): Result<Unit> = withContext(Dispatchers.IO) {
        try {
            val builder = Request.Builder()
                .url("$baseUrl/api/auth/tailscale/bindings/$bindingId/revoke")
                .post(ByteArray(0).toRequestBody(null))
            if (!confirmToken.isNullOrBlank()) {
                builder.header("X-QLH-Confirm-Token", confirmToken)
            }
            val response = executeAsync(builder.build())
            if (!response.isSuccessful) {
                return@withContext Result.failure(IOException("HTTP ${response.code}"))
            }
            Result.success(Unit)
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    // ==================== 聊天 ====================

    /** 发送消息并等待完整回复（非流式） */
    suspend fun chat(request: ChatRequest): Result<ChatResponse> = withContext(Dispatchers.IO) {
        try {
            val body = gson.toJson(request).toRequestBody(jsonMediaType)
            val httpRequest = Request.Builder()
                .url("$baseUrl/api/chat")
                .post(body)
                .header("Content-Type", "application/json")
                .build()

            val response = executeAsync(httpRequest)
            val responseBody = response.body?.string() ?: "{}"
            if (!response.isSuccessful) {
                return@withContext Result.failure(
                    httpException(response, responseBody)
                )
            }
            val chatResponse = gson.fromJson(responseBody, ChatResponse::class.java)
            Result.success(chatResponse)
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    // ==================== 会话管理 ====================

    /** 获取服务端会话列表 */
    suspend fun getSessions(): Result<List<SessionInfo>> = withContext(Dispatchers.IO) {
        try {
            val request = Request.Builder()
                .url("$baseUrl/api/sessions")
                .get()
                .build()

            val response = executeAsync(request)
            val body = response.body?.string() ?: "[]"
            if (!response.isSuccessful) {
                return@withContext Result.failure(IOException("HTTP ${response.code}"))
            }

            // 响应可能是 { "sessions": [...] } 或直接是数组
            val sessions: List<SessionInfo> = try {
                val array = gson.fromJson(body, Array<SessionInfo>::class.java)
                array.toList()
            } catch (e: Exception) {
                // 尝试解析为包装对象
                val wrapper = gson.fromJson(body, SessionsWrapper::class.java)
                wrapper.sessions ?: emptyList()
            }
            Result.success(sessions)
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    /** 创建服务端会话 */
    suspend fun createSession(title: String = "新对话"): Result<SessionInfo> = withContext(Dispatchers.IO) {
        try {
            val body = gson.toJson(mapOf("title" to title)).toRequestBody(jsonMediaType)
            val request = Request.Builder()
                .url("$baseUrl/api/sessions")
                .post(body)
                .header("Content-Type", "application/json")
                .build()

            val response = executeAsync(request)
            val responseBody = response.body?.string() ?: "{}"
            if (!response.isSuccessful) {
                return@withContext Result.failure(IOException("HTTP ${response.code}"))
            }
            Result.success(gson.fromJson(responseBody, SessionInfo::class.java))
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    /** 删除服务端会话 */
    suspend fun deleteSession(sessionId: String): Result<Unit> = withContext(Dispatchers.IO) {
        try {
            val request = Request.Builder()
                .url("$baseUrl/api/sessions/$sessionId")
                .delete()
                .build()

            val response = executeAsync(request)
            if (!response.isSuccessful && response.code != 404) {
                return@withContext Result.failure(IOException("HTTP ${response.code}"))
            }
            Result.success(Unit)
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    /** 获取服务端某会话的历史消息 */
    suspend fun getSessionMessages(sessionId: String): Result<List<MessageDto>> = withContext(Dispatchers.IO) {
        try {
            val request = Request.Builder()
                .url("$baseUrl/api/sessions/$sessionId/messages")
                .get()
                .build()

            val response = executeAsync(request)
            val body = response.body?.string() ?: "[]"
            if (!response.isSuccessful) {
                return@withContext Result.failure(IOException("HTTP ${response.code}"))
            }
            val messages = gson.fromJson(body, Array<MessageDto>::class.java)
            Result.success(messages.toList())
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    // ==================== 集群状态 ====================

    /** 获取集群状态 */
    suspend fun getClusterStatus(): Result<ClusterStatus> = withContext(Dispatchers.IO) {
        try {
            val request = Request.Builder()
                .url("$baseUrl/api/cluster/status")
                .get()
                .build()

            val response = executeAsync(request)
            val body = response.body?.string() ?: "{}"
            if (!response.isSuccessful) {
                return@withContext Result.failure(IOException("HTTP ${response.code}"))
            }
            Result.success(gson.fromJson(body, ClusterStatus::class.java))
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    /** 测试与主节点的连接 */
    suspend fun testConnection(): Result<Boolean> = withContext(Dispatchers.IO) {
        try {
            val request = Request.Builder()
                .url("$baseUrl/api/cluster/status")
                .get()
                .build()

            val response = executeAsync(request)
            Result.success(response.isSuccessful)
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    /** Probe the control plane without exposing response bodies or credentials to the UI. */
    suspend fun probeConnectionHealth(localNetworkType: String = "unknown"): Result<ConnectionHealthReport> =
        withContext(Dispatchers.IO) {
            val checks = mutableListOf<ConnectionHealthCheck>()

            suspend fun probe(id: String, label: String, path: String) {
                val started = System.nanoTime()
                try {
                    val request = Request.Builder()
                        .url("${baseUrl.trimEnd('/')}$path")
                        .get()
                        .build()
                    executeAsync(request).use { response ->
                        checks += ConnectionHealthCheck(
                            id = id,
                            label = label,
                            state = if (response.isSuccessful) ConnectionHealthState.PASS else ConnectionHealthState.FAIL,
                            latencyMillis = (System.nanoTime() - started) / 1_000_000L,
                            detail = "HTTP ${response.code}",
                        )
                    }
                } catch (error: Exception) {
                    checks += ConnectionHealthCheck(
                        id = id,
                        label = label,
                        state = ConnectionHealthState.FAIL,
                        latencyMillis = (System.nanoTime() - started) / 1_000_000L,
                        detail = error.message ?: error.javaClass.simpleName,
                    )
                }
            }

            probe("cluster_status", "主节点状态", "/api/cluster/status")
            if (authStore?.read()?.accessToken.isNullOrBlank()) {
                checks += ConnectionHealthCheck(
                    id = "auth_session",
                    label = "认证会话",
                    state = ConnectionHealthState.SKIPPED,
                    detail = "未登录",
                )
            } else {
                probe("auth_session", "认证会话", "/api/auth/session")
            }
            Result.success(
                ConnectionHealthReport(
                    checks = checks,
                    localNetworkType = localNetworkType,
                )
            )
        }

    /** Upload a bounded, already-redacted client diagnostic report. */
    suspend fun reportClientError(report: ClientErrorReport): Result<Unit> = withContext(Dispatchers.IO) {
        try {
            val body = gson.toJson(report).toRequestBody(jsonMediaType)
            val request = Request.Builder()
                .url("${baseUrl.trimEnd('/')}/api/logs/client-error")
                .post(body)
                .header("Content-Type", "application/json")
                .build()
            executeAsync(request).use { response ->
                if (!response.isSuccessful) {
                    return@withContext Result.failure(IOException("HTTP ${response.code}"))
                }
            }
            Result.success(Unit)
        } catch (error: Exception) {
            Result.failure(error)
        }
    }

    /** Android 薄客户端通过 HTTP 向主节点登记自身存在（非 TCP worker 注册）。 */
    suspend fun registerAndroidNode(request: RegisterNodeRequest): Result<RegisterNodeResponse> = withContext(Dispatchers.IO) {
        try {
            val body = gson.toJson(request).toRequestBody(jsonMediaType)
            val httpRequest = Request.Builder()
                .url("$baseUrl/api/cluster/android/register")
                .post(body)
                .header("Content-Type", "application/json")
                .build()

            val response = executeAsync(httpRequest)
            val responseBody = response.body?.string() ?: "{}"
            if (!response.isSuccessful) {
                return@withContext Result.failure(httpException(response, responseBody))
            }
            Result.success(gson.fromJson(responseBody, RegisterNodeResponse::class.java))
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    /** Refresh an existing Android presence lease without sending device metadata again. */
    suspend fun heartbeatAndroidNode(
        request: AndroidPresenceHeartbeatRequest,
    ): Result<RegisterNodeResponse> = withContext(Dispatchers.IO) {
        try {
            val body = gson.toJson(request).toRequestBody(jsonMediaType)
            val httpRequest = Request.Builder()
                .url("$baseUrl/api/cluster/android/heartbeat")
                .post(body)
                .header("Content-Type", "application/json")
                .build()
            val response = executeAsync(httpRequest)
            val responseBody = response.body?.string() ?: "{}"
            if (!response.isSuccessful) {
                return@withContext Result.failure(httpException(response, responseBody))
            }
            Result.success(gson.fromJson(responseBody, RegisterNodeResponse::class.java))
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    private fun httpException(response: Response, body: String): ApiClientHttpException {
        val headerCode = response.header("X-QLH-Error-Code").orEmpty()
        val bodyCode = runCatching {
            JsonParser.parseString(body).asJsonObject.get("error_code")?.asString.orEmpty()
        }.getOrDefault("")
        return ApiClientHttpException(response.code, headerCode.ifBlank { bodyCode }, body)
    }

    /** 首次连接自动部署：从主节点获取 Android 节点配置。 */
    suspend fun firstConnectBootstrap(request: BootstrapRequest): Result<BootstrapResponse> = withContext(Dispatchers.IO) {
        try {
            val body = gson.toJson(request).toRequestBody(jsonMediaType)
            val httpRequest = Request.Builder()
                .url("$baseUrl/api/bootstrap/first-connect")
                .post(body)
                .header("Content-Type", "application/json")
                .build()

            val response = executeAsync(httpRequest)
            val responseBody = response.body?.string() ?: "{}"
            if (!response.isSuccessful) {
                return@withContext Result.failure(IOException("HTTP ${response.code}: $responseBody"))
            }
            Result.success(gson.fromJson(responseBody, BootstrapResponse::class.java))
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    // ==================== Remote GGUF catalog ====================

    suspend fun getModels(): Result<ServerModelsResponse> = withContext(Dispatchers.IO) {
        try {
            val request = Request.Builder()
                .url("${baseUrl.trimEnd('/')}/api/models")
                .get()
                .build()
            executeJson(request, ServerModelsResponse::class.java)
        } catch (e: CancellationException) {
            throw e
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    suspend fun getCurrentModel(): Result<CurrentModelStatus> = withContext(Dispatchers.IO) {
        try {
            val request = Request.Builder()
                .url("${baseUrl.trimEnd('/')}/api/models/current")
                .get()
                .build()
            executeJson(request, CurrentModelStatus::class.java)
        } catch (e: CancellationException) {
            throw e
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    suspend fun getLocalModelAssets(): Result<LocalModelAssetsResponse> = withContext(Dispatchers.IO) {
        try {
            val request = Request.Builder()
                .url("${baseUrl.trimEnd('/')}/api/models/local-assets")
                .get()
                .build()
            executeJson(request, LocalModelAssetsResponse::class.java)
        } catch (e: CancellationException) {
            throw e
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    /**
     * Read the four existing model inventory endpoints as one bounded snapshot.
     * This method only aggregates server-owned metadata; it never starts a model,
     * changes a registry entry, or mutates the Android SAF model directory.
     */
    suspend fun getModelFleetData(): Result<MasterModelFleetData> = withContext(Dispatchers.IO) {
        try {
            val current = getCurrentModel().getOrThrow()
            val registry = getModels().getOrThrow()
            val localAssets = getLocalModelAssets().getOrThrow()
            val verifiedGguf = getGgufModels().getOrThrow()
            Result.success(MasterModelFleetData(current, registry, localAssets, verifiedGguf))
        } catch (e: CancellationException) {
            throw e
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    /** Read only bounded audit summaries; the server omits prompts, errors, paths and comments. */
    suspend fun getAuditData(): Result<AuditData> = withContext(Dispatchers.IO) {
        try {
            val workflowRequest = Request.Builder()
                .url("${baseUrl.trimEnd('/')}/api/workflows?limit=8&summary=1")
                .get()
                .build()
            val reviewRequest = Request.Builder()
                .url("${baseUrl.trimEnd('/')}/api/cluster/review/tickets?limit=8&summary=1")
                .get()
                .build()
            val workflows = executeJson(workflowRequest, AuditWorkflowsResponse::class.java).getOrThrow()
            val reviews = executeJson(reviewRequest, AuditReviewTicketsResponse::class.java).getOrThrow()
            Result.success(AuditData(workflows, reviews))
        } catch (e: CancellationException) {
            throw e
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    suspend fun getGgufModels(): Result<List<GgufModelInfo>> = withContext(Dispatchers.IO) {
        try {
            val request = Request.Builder()
                .url("${baseUrl.trimEnd('/')}/api/models/gguf")
                .get()
                .build()
            executeJson(request, GgufModelsResponse::class.java).map { response ->
                response.models.filter { model ->
                    model.filename.endsWith(".gguf", ignoreCase = true) &&
                        model.sizeBytes > 0L &&
                        model.sha256.matches(Regex("[a-fA-F0-9]{64}"))
                }
            }
        } catch (e: CancellationException) {
            throw e
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    // ==================== 内部方法 ====================

    private suspend fun executeAsync(request: Request): Response =
        suspendCancellableCoroutine { continuation ->
            val call = client.newCall(request)
            call.enqueue(object : Callback {
                override fun onResponse(call: Call, response: Response) {
                    continuation.resume(response)
                }

                override fun onFailure(call: Call, e: IOException) {
                    if (continuation.isCancelled) return
                    continuation.resumeWithException(e)
                }
            })
            continuation.invokeOnCancellation {
                call.cancel()
            }
        }

    private suspend fun <T> executeJson(request: Request, responseType: Class<T>): Result<T> {
        val response = executeAsync(request)
        val body = response.body?.string() ?: "{}"
        if (!response.isSuccessful) {
            return Result.failure(IOException("HTTP ${response.code}: $body"))
        }
        return Result.success(gson.fromJson(body, responseType))
    }

    // ---- DTO 包装 ----

    private data class SessionsWrapper(
        val sessions: List<SessionInfo>? = null
    )
}

/** 服务端消息 DTO */
data class MessageDto(
    val id: Long = 0,
    val role: String = "",
    val content: String = "",
    val timestamp: String? = null,
    val metrics: Map<String, Any>? = null
)
