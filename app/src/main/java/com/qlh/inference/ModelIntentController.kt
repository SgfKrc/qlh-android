package com.qlh.inference

import com.google.gson.JsonParser
import com.qlh.inference.data.SettingsDataStore
import com.qlh.inference.network.ApiClientHttpException
import com.qlh.inference.network.ModelIntentApi
import com.qlh.inference.network.ModelIntentCatalog
import com.qlh.inference.network.ModelIntentOperation
import com.qlh.inference.network.ModelIntentRequest
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch

/** A saved bootstrap/API endpoint, including the persisted cluster binding generation. */
data class IntentMasterEndpoint(
    val host: String,
    val port: Int,
    val clusterId: String = "",
    val secretEpoch: Int = 0,
)

data class ModelIntentError(val code: String, val reason: String)

data class ModelIntentUiState(
    val catalog: ModelIntentCatalog? = null,
    val operation: ModelIntentOperation? = null,
    val requestedIntent: ModelIntentRequest? = null,
    val loading: Boolean = false,
    val busy: Boolean = false,
    val error: ModelIntentError? = null,
    val suspended: Boolean = false,
) {
    val connected: Boolean get() = catalog?.binding?.status == "connected"
    // Registry availability describes cached source files, not the master's ability to acquire them.
    val canChooseModel: Boolean get() = connected && !loading && !busy && !suspended
    val readyForChat: Boolean get() = connected && operation?.readyForChat == true &&
        !busy && !loading && !suspended && error == null
    val canCancel: Boolean get() = !busy && !suspended && operation?.finished == false
    val canRetry: Boolean get() = connected && !busy && !suspended && requestedIntent != null &&
        (error != null || operation?.status in setOf("failed", "cancelled", "superseded"))
}

fun modelIntentAllowsChat(mode: String, state: ModelIntentUiState): Boolean =
    SettingsDataStore.normalizeInferenceMode(mode) == SettingsDataStore.MODE_LOCAL || state.readyForChat

fun modelIntentStatusLabel(status: String): String = when (status) {
    "queued" -> "已排队"
    "waiting_nodes" -> "等待节点"
    "inspecting" -> "检查模型"
    "inventory" -> "清点节点资源"
    "planning" -> "制定供给计划"
    "building" -> "构建模型段"
    "supplying" -> "供给模型段"
    "downloading" -> "下载模型段"
    "verifying" -> "校验模型段"
    "preparing" -> "准备推理"
    "available" -> "模型段已供给，等待准备"
    "loading" -> "加载模型段"
    "prepared" -> "模型段已加载，等待提交"
    "ready" -> "准备完成"
    "failed" -> "失败"
    "cancelled" -> "已取消"
    "superseded" -> "已被新请求替代"
    else -> status
}

internal fun modelIntentError(error: Throwable): ModelIntentError {
    if (error is ApiClientHttpException) {
        val detail = runCatching {
            JsonParser.parseString(error.responseBody).asJsonObject
        }.getOrNull()
        val nested = detail?.get("detail")?.takeIf { it.isJsonObject }?.asJsonObject
        fun text(key: String): String? = runCatching {
            (nested?.get(key) ?: detail?.get(key))?.takeIf { it.isJsonPrimitive }?.asString
        }.getOrNull()?.takeIf { it.isNotBlank() }
        val code = text("reason_code") ?: text("code") ?: error.errorCode.takeIf { it.isNotBlank() }
            ?: when (error.statusCode) {
                401, 403 -> "master_auth_required"
                404 -> "model_intent_api_unavailable"
                else -> "model_intent_http_${error.statusCode}"
            }
        val reason = text("reason") ?: text("message") ?: when (error.statusCode) {
            401, 403 -> "请先在账户设置完成主节点登录"
            404 -> "主节点尚未提供模型 intent API"
            else -> "主节点拒绝了模型请求，请刷新状态后重试"
        }
        return ModelIntentError(code, reason)
    }
    return when (error) {
        is java.net.ConnectException -> ModelIntentError("master_unreachable", "无法连接已保存的主节点")
        is java.net.SocketTimeoutException -> ModelIntentError("master_timeout", "读取主节点模型状态超时")
        is IllegalArgumentException -> ModelIntentError("model_intent_contract_invalid", "主节点模型契约无效")
        else -> ModelIntentError("model_intent_request_failed", "模型请求失败，请刷新状态后重试")
    }
}

/**
 * All calls share one request lane. The epoch changes even for A -> B -> A and on a user mode
 * selection before unload finishes. Cancellation saves work; epoch/ticket checks protect against
 * transports that complete after cancellation. This controller never writes inference mode.
 */
internal class ModelIntentController(
    private val scope: CoroutineScope,
    private val client: (IntentMasterEndpoint) -> ModelIntentApi,
    private val polling: Boolean = true,
    private val pollIntervalMillis: Long = 1500,
) {
    private data class Context(val mode: String, val endpoint: IntentMasterEndpoint?, val paused: Boolean)
    private var context = Context(SettingsDataStore.MODE_LOCAL, null, false)
    private var epoch = 0L
    private var ticket = 0L
    private var work: Job? = null
    private val mutableState = MutableStateFlow(ModelIntentUiState())
    val state: StateFlow<ModelIntentUiState> = mutableState.asStateFlow()

    fun updateContext(mode: String, endpoint: IntentMasterEndpoint?, paused: Boolean = false) {
        val next = Context(SettingsDataStore.normalizeInferenceMode(mode), endpoint, paused)
        if (next == context) return
        context = next
        epoch++
        ticket++
        work?.cancel()
        mutableState.value = ModelIntentUiState(suspended = paused)
        if (enabled()) refresh()
    }

    /** Retire pending reads immediately on an explicit user mode/endpoint choice. */
    fun suspendForSelection() {
        epoch++
        ticket++
        work?.cancel()
        context = context.copy(paused = true)
        mutableState.value = ModelIntentUiState(suspended = true)
    }

    private fun enabled(): Boolean = !context.paused && context.mode != SettingsDataStore.MODE_LOCAL
    private fun current(generation: Long, request: Long): Boolean =
        enabled() && generation == epoch && request == ticket

    private fun launchRequest(busy: Boolean, action: suspend (ModelIntentApi, Long, Long) -> Unit) {
        if (!enabled()) return
        val endpoint = context.endpoint
        if (endpoint == null) {
            mutableState.value = ModelIntentUiState(error = ModelIntentError(
                "INTENT_MASTER_NOT_FOUND", "尚未绑定主节点，请使用发现主节点入口重试",
            ))
            return
        }
        work?.cancel()
        val generation = epoch
        val request = ++ticket
        mutableState.value = mutableState.value.copy(loading = !busy, busy = busy, error = null)
        work = scope.launch {
            var api: ModelIntentApi? = null
            try {
                api = client(endpoint)
                action(api, generation, request)
            } catch (cancelled: CancellationException) {
                throw cancelled
            } catch (error: Exception) {
                if (current(generation, request)) mutableState.value = mutableState.value.copy(
                    loading = false, busy = false, error = modelIntentError(error),
                )
            }
            // A bound endpoint may become admissible only after the worker registers. Keep the
            // read lane alive even if the first catalog fails or reports an unconnected binding.
            // New local writes and mode/endpoint selections retire this lane before publication.
            val boundApi = api ?: return@launch
            if (current(generation, request)) poll(boundApi, generation, request)
        }
    }

    fun discoveryFailed(reason: String = "未发现可验证的主节点，请确认主节点在线后重试") {
        if (!enabled()) return
        ticket++
        work?.cancel()
        mutableState.value = ModelIntentUiState(error = ModelIntentError("INTENT_MASTER_NOT_FOUND", reason))
    }

    fun refresh() {
        if (mutableState.value.busy) return
        launchRequest(busy = false) { api, generation, request ->
            if (!readCatalog(api, generation, request)) return@launchRequest
            val operation = api.getCurrentModelIntent().getOrThrow()
            if (!current(generation, request)) return@launchRequest
            publish(operation)
        }
    }

    fun submit(intent: ModelIntentRequest) {
        if (mutableState.value.busy || !mutableState.value.connected || !enabled()) return
        mutableState.value = mutableState.value.copy(requestedIntent = intent, operation = null)
        launchRequest(busy = true) { api, generation, request ->
            val operation = api.submitModelIntent(intent).getOrThrow()
            if (!current(generation, request)) return@launchRequest
            require(operation.intent == intent) { "intent_response_mismatch" }
            publish(operation)
        }
    }

    fun retry() {
        if (!mutableState.value.canRetry) return
        val intent = mutableState.value.requestedIntent ?: return
        // A transport error can leave the binding known; an unbound catalog must be refreshed first.
        submit(intent)
    }

    fun cancel() {
        if (!mutableState.value.canCancel || !enabled()) return
        val id = mutableState.value.operation?.operationId ?: return
        launchRequest(busy = true) { api, generation, request ->
            val operation = api.cancelModelIntent(id).getOrThrow()
            if (!current(generation, request)) return@launchRequest
            require(operation.operationId == id) { "intent_operation_mismatch" }
            publish(operation)
        }
    }

    private fun publish(operation: ModelIntentOperation?) {
        val previous = mutableState.value.operation
        if (operation != null && previous?.operationId == operation.operationId &&
            operation.generation < previous.generation) {
            mutableState.value = mutableState.value.copy(loading = false, busy = false)
            return
        }
        mutableState.value = mutableState.value.copy(
            operation = operation, requestedIntent = operation?.intent,
            loading = false, busy = false, error = null,
        )
    }

    private suspend fun readCatalog(api: ModelIntentApi, generation: Long, request: Long): Boolean {
        val catalog = api.getModelIntentCatalog().getOrThrow()
        if (!current(generation, request) || mutableState.value.busy) return false
        if (catalog.binding.status != "connected") {
            mutableState.value = mutableState.value.copy(
                catalog = catalog, operation = null, requestedIntent = null, loading = false,
                error = ModelIntentError(catalog.binding.reasonCode?.takeIf { it.isNotBlank() }
                    ?: "INTENT_MASTER_NOT_FOUND", "尚未连接主节点，请使用发现主节点入口"),
            )
            return false
        }
        mutableState.value = mutableState.value.copy(catalog = catalog)
        return true
    }

    private suspend fun poll(api: ModelIntentApi, generation: Long, request: Long) {
        while (polling && current(generation, request)) {
            delay(pollIntervalMillis)
            if (!current(generation, request)) return
            if (mutableState.value.busy) continue
            try {
                // Retry catalog until the same bound endpoint is connected; never issue current
                // for an unconnected catalog or let a retired catalog overwrite a newer choice.
                if (!mutableState.value.connected && !readCatalog(api, generation, request)) continue
                // current is authoritative across devices, including after ready/failed/cancelled.
                // Polling an old operation ID would pin Android to another device's retired choice.
                val operation = api.getCurrentModelIntent().getOrThrow()
                if (!current(generation, request)) return
                if (!mutableState.value.busy) publish(operation)
            } catch (cancelled: CancellationException) {
                throw cancelled
            } catch (error: Exception) {
                if (!current(generation, request)) return
                mutableState.value = mutableState.value.copy(error = modelIntentError(error))
            }
        }
    }
}

internal fun modelIntentMetricsLabel(operationId: String, modelConfigId: String, generation: Long?): String {
    if (!Regex("supply_[A-Za-z0-9_-]+").matches(operationId)) return ""
    return listOfNotNull(
        "模型意图 $operationId",
        modelConfigId.takeIf { it.isNotBlank() }?.let { "模型配置 $it" },
        generation?.takeIf { it >= 0 }?.let { "模型代际 $it" },
    ).joinToString(" · ")
}
