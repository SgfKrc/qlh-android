package com.qlh.inference.network

import com.google.gson.Gson
import com.google.gson.GsonBuilder
import com.google.gson.ToNumberPolicy
import com.google.gson.JsonParser
import com.google.gson.Strictness
import com.google.gson.stream.JsonReader
import com.google.gson.stream.JsonToken
import java.io.StringReader
import com.google.gson.annotations.SerializedName
import com.qlh.inference.BuildConfig
import com.qlh.inference.security.BootstrapCredentialEnvelope
import com.qlh.inference.security.BootstrapCredentialExchange
import com.qlh.inference.security.BootstrapCredentialException
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
    /** The Android thinking switch controls generation as well as display. */
    @SerializedName("enable_thinking")
    val enableThinking: Boolean = showThinking,
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
    val distributedEnabled: Boolean = false,
    @SerializedName("pipeline_capacity")
    val pipelineCapacity: PipelineCapacitySummary? = null
)

/**
 * 只读容量决策投影（`/api/cluster/status` 的 `pipeline_capacity`）。
 *
 * 存在的理由：「节点 online」只说心跳在发，不代表它在流水线里干活 ——
 * 这里把「是否准入 + 没准入的原因 + 谁在参与」一并带给移动端，
 * 免得必须交叉 `/cluster/nodes` 与 `/cluster/pipeline-capacity` 才能看出真相。
 */
data class PipelineCapacitySummary(
    val status: String = "",
    /** 三态：true=已准入、false=已拒绝、null=后端没有权威决策。 */
    val admitted: Boolean? = null,
    @SerializedName("reason_code")
    val reasonCode: String = "",
    val reason: String = "",
    @SerializedName("participating_node_count")
    val participatingNodeCount: Int = 0,
    @SerializedName("control_only_nodes")
    val controlOnlyNodes: List<String> = emptyList(),
    @SerializedName("worker_count")
    val workerCount: Int = 0,
    @SerializedName("prepared_node_count")
    val preparedNodeCount: Int = 0,
    @SerializedName("ready_node_count")
    val readyNodeCount: Int = 0,
    @SerializedName("require_distributed")
    val requireDistributed: Boolean = false,
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
    @SerializedName("pipeline_participating")
    val pipelineParticipating: Boolean = false,
    @SerializedName("pipeline_exclusion_reason")
    val pipelineExclusionReason: String = "",
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
    val clientMode: String = "distributed",
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
) : IOException("HTTP $statusCode${if (errorCode.isBlank()) "" else " [$errorCode]"}: ${errorDetailOf(responseBody)}")

/**
 * 从后端错误响应体里取出**人类可读原因**（供 UI 直接显示）。
 *
 * ★ 2026-10-08：此前异常 message 直接拼**原始 JSON 串**（`{"detail":"推理失败: ..."}`），
 * 用户看到的是转义 JSON、得自己找重点。后端两类错误体的形状都在这儿收拢：
 *   - `{"detail": "<文本>"}`                     —— 例如"禁止自动整模回退"那条；
 *   - `{"detail": {"code": "...", "message": "..."}}` —— 例如信任边界的
 *     `MODEL_API_SOURCE_UNTRUSTED`（`src/model_api_access.py`）。
 *
 * 取不到时**回退到原始 body**（绝不吞掉信息，宁可难看也不静默）。
 */
internal fun errorDetailOf(body: String): String {
    val extracted = runCatching {
        val obj = JsonParser.parseString(body).asJsonObject
        val detail = obj.get("detail")
        when {
            detail == null || detail.isJsonNull -> obj.get("message")?.asString.orEmpty()
            detail.isJsonObject -> detail.asJsonObject.get("message")?.asString.orEmpty()
            else -> detail.asString
        }
    }.getOrDefault("")
    return extracted.ifBlank { body }
}

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
    val capabilities: Map<String, Any?> = emptyMap(),
    @SerializedName("credential_public_key")
    val credentialPublicKey: String = "",
    @SerializedName("credential_request_nonce")
    val credentialRequestNonce: String = "",
    @SerializedName("credential_requested_at")
    val credentialRequestedAt: Long = 0L,
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
    @SerializedName("cluster_secret_epoch")
    val clusterSecretEpoch: Int = 1,
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
    val android: BootstrapAndroid = BootstrapAndroid(),
    @SerializedName("credential_envelope")
    val credentialEnvelope: BootstrapCredentialEnvelope? = null,
)

class BootstrapResult internal constructor(
    val response: BootstrapResponse,
    val clusterSecret: String,
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

/** Per-engine distributed preflight metadata; content proof remains with supply. */
data class PipelineEngineAvailability(
    val status: String? = null,
    @SerializedName("runtime_model_type") val runtimeModelType: String? = null,
    @SerializedName("reason_code") val reasonCode: String? = null,
    val reason: String? = null,
    val source: String? = null,
    @SerializedName("content_verification") val contentVerification: String? = null,
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
    @SerializedName("supported_engines") val supportedEngines: List<String>? = null,
    @SerializedName("pipeline_engine_availability")
    val pipelineEngineAvailability: Map<String, PipelineEngineAvailability?>? = null,
    @SerializedName("pipeline_auto_engine") val pipelineAutoEngine: String? = null,
    @SerializedName("pipeline_auto_unavailable_reason") val pipelineAutoUnavailableReason: String? = null,
)

data class ServerModelsResponse(
    val models: List<ServerModelSummary> = emptyList(),
    @SerializedName("active_model_id") val activeModelId: String? = null,
)

/** Runtime status deliberately excludes server filesystem paths. */
data class CurrentModelDescriptor(
    @SerializedName("model_sha256") val modelSha256: String = "",
    @SerializedName("source_model_id") val sourceModelId: String = "",
    // ★ 2026-10-11（Y700 反复重选无效排查）：主节点报告的**源模型**摘要（三段共享）。
    //   裁层工件的跨段校验必须用它 —— 每段的 `artifact_sha256` 天然不同（head/tail 各异），
    //   只有 `source_model_sha256` 在三段间相同；服务端此前把它过滤掉了
    //   （见 src/pipeline_model_descriptor.py 的 public_pipeline_descriptor 白名单）。
    @SerializedName("source_model_sha256") val sourceModelSha256: String = "",
    // ★ 2026-10-11（三机实测 / #92）：`descriptor.model_id` 是**模型名**，与工件的
    //   `source_model_id` 同口径；而顶层 `model_id` 是模型**注册表 id**。身份判定必须优先用
    //   前者，否则注册 id 与工件命名不一致时会把工件整片过滤掉（真机表现为 ranges=[]）。
    @SerializedName("model_id") val modelId: String = "",
)

data class CurrentModelStatus(
    val loaded: Boolean = false,
    @SerializedName("pipeline_prepared") val pipelinePrepared: Boolean = false,
    @SerializedName("model_id") val modelId: String? = null,
    @SerializedName("model_name") val modelName: String = "",
    val descriptor: CurrentModelDescriptor? = null,
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

/** Content-free request ownership and physical cleanup observations. */
data class AuditRequestSummary(
    @SerializedName("request_id") val requestId: String = "",
    @SerializedName("deployment_id") val deploymentId: String = "",
    val phase: String = "",
    @SerializedName("execution_state") val executionState: String = "",
    val terminal: String? = null,
    val outcome: String? = null,
    @SerializedName("observed_at") val observedAt: Double = 0.0,
)
data class AuditRequestsResponse(
    @SerializedName("schema_version") val schemaVersion: Int = 0,
    val revision: Long? = null,
    val identity: Map<String, String> = emptyMap(),
    @SerializedName("observed_at") val observedAt: Double = 0.0,
    val validity: String = "unknown",
    val requests: List<AuditRequestSummary> = emptyList(),
)

data class AuditData(
    val requests: AuditRequestsResponse = AuditRequestsResponse(),
)

// ================================================================
// API 客户端
// ================================================================

class ApiClient(
    private val baseUrl: String,
    // Keep native generation integers exact; fractional metrics still use Double.
    private val gson: Gson = GsonBuilder()
        .setObjectToNumberStrategy(ToNumberPolicy.LONG_OR_DOUBLE)
        .create(),
    private val callTimeoutSeconds: Long = 0,
) : ModelIntentApi {
    private val auditCursor = ClientSnapshotCursor()
    private val conversationClientId = "android:" + java.util.UUID.randomUUID().toString()

    private val client: OkHttpClient = OkHttpClient.Builder()
        .callTimeout(callTimeoutSeconds, TimeUnit.SECONDS)
        .connectTimeout(10, TimeUnit.SECONDS)
        .readTimeout(120, TimeUnit.SECONDS)   // 推理可能较慢
        .writeTimeout(30, TimeUnit.SECONDS)
        .addInterceptor { chain ->
            val request = chain.request().newBuilder()
                .header("X-QLH-Client-ID", conversationClientId)
                .build()
            chain.proceed(request)
        }
        .addInterceptor(HttpLoggingInterceptor().apply {
            redactHeader("Authorization")
            // Never log request/response bodies: chat content may contain user data.
            level = if (BuildConfig.DEBUG) {
                HttpLoggingInterceptor.Level.HEADERS
            } else {
                HttpLoggingInterceptor.Level.NONE
            }
        })
        .build()

    private val jsonMediaType = "application/json; charset=utf-8".toMediaType()


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
    suspend fun firstConnectBootstrap(request: BootstrapRequest): Result<BootstrapResult> = withContext(Dispatchers.IO) {
        try {
            val exchange = BootstrapCredentialExchange.create(request.nodeId)
            val credentialRequest = exchange.requestFields
            val protectedRequest = request.copy(
                credentialPublicKey = credentialRequest.publicKey,
                credentialRequestNonce = credentialRequest.requestNonce,
                credentialRequestedAt = credentialRequest.requestedAt,
            )
            val body = gson.toJson(protectedRequest).toRequestBody(jsonMediaType)
            val httpRequest = Request.Builder()
                .url("$baseUrl/api/bootstrap/first-connect")
                .post(body)
                .header("Content-Type", "application/json")
                .build()

            val response = executeAsync(httpRequest)
            val responseBody = response.body?.string() ?: "{}"
            if (!response.isSuccessful) {
                return@withContext Result.failure(IOException("Bootstrap failed: HTTP ${response.code}"))
            }
            val bootstrapResponse = try {
                val responseJson = JsonParser.parseString(responseBody).asJsonObject
                if (responseJson.getAsJsonObject("cluster")?.has("cluster_secret") == true) {
                    throw BootstrapCredentialException("plaintext_secret_rejected")
                }
                val aadFields = responseJson.getAsJsonObject("credential_envelope")
                    ?.getAsJsonObject("aad")
                    ?.keySet()
                if (aadFields != setOf(
                        "cluster_id",
                        "expires_at",
                        "issued_at",
                        "node_id",
                        "request_nonce",
                        "secret_epoch",
                    )
                ) {
                    throw BootstrapCredentialException("aad_invalid")
                }
                gson.fromJson(responseJson, BootstrapResponse::class.java)
                    ?: throw BootstrapCredentialException("response_invalid")
            } catch (e: BootstrapCredentialException) {
                throw e
            } catch (_: Exception) {
                throw BootstrapCredentialException("response_invalid")
            }
            if (bootstrapResponse.status !in setOf("ok", "registered", "updated")) {
                throw BootstrapCredentialException("response_status_invalid")
            }
            val envelope = bootstrapResponse.credentialEnvelope
                ?: throw BootstrapCredentialException("envelope_missing")
            val secret = exchange.decrypt(
                envelope = envelope,
                clusterId = bootstrapResponse.cluster.clusterId,
                nodeId = bootstrapResponse.node.nodeId,
                secretEpoch = bootstrapResponse.cluster.clusterSecretEpoch,
            )
            Result.success(BootstrapResult(bootstrapResponse, secret))
        } catch (e: CancellationException) {
            throw e
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    // ==================== Master model intent ====================

    suspend fun getBootstrapInfo(): Result<MasterBootstrapInfo> =
        intentJson("", "GET", MasterBootstrapInfo::class.java, endpointPath = "/api/bootstrap/info")

    /** Never bootstrap a UDP hint before confirming the HTTP master role. */
    suspend fun firstConnectVerifiedMaster(
        request: BootstrapRequest,
        stillCurrent: () -> Boolean = { true },
    ): Result<BootstrapResult> {
        val info = getBootstrapInfo()
        if (info.isFailure) return Result.failure(info.exceptionOrNull()!!)
        if (!info.getOrThrow().isMaster) return Result.failure(IOException("INTENT_MASTER_NOT_FOUND"))
        if (!stillCurrent()) return Result.failure(IOException("master_selection_changed"))
        return firstConnectBootstrap(request)
    }

    override suspend fun getModelIntentCatalog(): Result<ModelIntentCatalog> =
        intentJson("catalog", "GET", ModelIntentCatalog::class.java).mapCatching { catalog ->
            require(catalog.binding.status in setOf("connected", "not_found", "local")) {
                "intent_binding_invalid"
            }
            catalog
        }

    override suspend fun getCurrentModelIntent(): Result<ModelIntentOperation?> =
        intentJsonNullable("current", "GET").mapCatching { it?.validate() }

    override suspend fun submitModelIntent(intent: ModelIntentRequest): Result<ModelIntentOperation> =
        intentJson("", "POST", ModelIntentOperation::class.java, intent).mapCatching { it.validate() }

    override suspend fun getModelIntent(operationId: String): Result<ModelIntentOperation> =
        intentOperationRequest(operationId, "GET")

    override suspend fun cancelModelIntent(operationId: String): Result<ModelIntentOperation> =
        intentOperationRequest(operationId, "DELETE")

    private suspend fun intentOperationRequest(id: String, method: String): Result<ModelIntentOperation> {
        return try {
            validateModelIntentOperationId(id)
            intentJson(id, method, ModelIntentOperation::class.java).mapCatching {
                it.validate()
                require(it.operationId == id) { "intent_operation_mismatch" }
                it
            }
        } catch (e: CancellationException) {
            throw e
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    private suspend fun intentJsonNullable(path: String, method: String): Result<ModelIntentOperation?> =
        withContext(Dispatchers.IO) {
            try {
                val request = Request.Builder()
                    .url("${baseUrl.trimEnd('/')}/api/models/intent/$path")
                    .method(method, null).build()
                executeAsync(request).use { response ->
                    val body = response.body?.string().orEmpty()
                    if (!response.isSuccessful) return@withContext Result.failure(httpException(response, body))
                    require(body.isNotBlank()) { "intent_response_invalid" }
                    val json = parseIntentJson(body)
                    Result.success(if (json.isJsonNull) null else gson.fromJson(json, ModelIntentOperation::class.java))
                }
            } catch (e: CancellationException) {
                throw e
            } catch (e: Exception) {
                Result.failure(e)
            }
        }

    private fun parseIntentJson(body: String): com.google.gson.JsonElement {
        require(body.isNotBlank()) { "intent_response_invalid" }
        return JsonReader(StringReader(body)).use { reader ->
            reader.strictness = Strictness.STRICT
            val json = JsonParser.parseReader(reader)
            require(reader.peek() == JsonToken.END_DOCUMENT) { "intent_response_trailing_data" }
            json
        }
    }

    private suspend fun <T> intentJson(
        path: String,
        method: String,
        type: Class<T>,
        payload: ModelIntentRequest? = null,
        endpointPath: String? = null,
    ): Result<T> = withContext(Dispatchers.IO) {
        try {
            payload?.validate()
            val suffix = if (path.isEmpty()) "" else "/$path"
            val request = Request.Builder()
                .url("${baseUrl.trimEnd('/')}${endpointPath ?: "/api/models/intent$suffix"}")
                .method(method, payload?.let { gson.toJson(it).toRequestBody(jsonMediaType) })
                .build()
            executeAsync(request).use { response ->
                val body = response.body?.string().orEmpty()
                if (!response.isSuccessful) return@withContext Result.failure(httpException(response, body))
                if (type == MasterBootstrapInfo::class.java) {
                    val isMaster = parseIntentJson(body).asJsonObject.get("is_master")
                    require(isMaster != null && isMaster.isJsonPrimitive && isMaster.asJsonPrimitive.isBoolean) {
                        "bootstrap_info_invalid"
                    }
                }
                val parsed = gson.fromJson(parseIntentJson(body), type) ?: throw IOException("intent_response_invalid")
                Result.success(parsed)
            }
        } catch (e: CancellationException) {
            throw e
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

    /** Read a bounded workflow summary; the server omits prompts, errors and paths. */
    suspend fun getAuditData(): Result<AuditData> = withContext(Dispatchers.IO) {
        try {
            val workflowRequest = Request.Builder()
                .url("${baseUrl.trimEnd('/')}/api/requests?limit=8")
                .get()
                .build()
            val requests = executeJson(workflowRequest, AuditRequestsResponse::class.java).getOrThrow()
            require(auditCursor.accept(requests.schemaVersion, requests.revision, requests.identity, requests.observedAt, requests.validity)) { "request_snapshot_invalid" }
            Result.success(AuditData(requests))
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
