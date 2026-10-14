package com.qlh.inference.network

import com.google.gson.JsonElement
import com.google.gson.annotations.SerializedName

/** MODEL-INTENT-UX-01: Android sends a model choice; the master owns supply and prepare. */
data class ModelIntentRequest(
    @SerializedName("model_id") val modelId: String,
    val engine: String = "auto",
    @SerializedName("quant_type") val quantType: String = "",
    val strategy: String = "auto",
    @SerializedName("timeout_seconds") val timeoutSeconds: Int = 3600,
) {
    fun validate() {
        require(modelId.isNotBlank()) { "model_id_required" }
        require(engine in setOf("auto", "pytorch", "llama_cpp")) { "intent_engine_invalid" }
        require(strategy == "auto" && timeoutSeconds > 0) { "intent_options_invalid" }
    }
}

data class ModelIntentBinding(
    val status: String = "not_found",
    @SerializedName("is_master") val isMaster: Boolean = false,
    @SerializedName("master_host") val masterHost: String? = null,
    @SerializedName("master_api_port") val masterApiPort: Int? = null,
    @SerializedName("reason_code") val reasonCode: String? = null,
)

data class ModelIntentCatalog(
    val models: List<ServerModelSummary> = emptyList(),
    @SerializedName("active_model_id") val activeModelId: String? = null,
    val binding: ModelIntentBinding = ModelIntentBinding(),
)

data class ModelIntentNode(
    val status: String = "queued",
    @SerializedName("bytes_received") val bytesReceived: Long = 0,
    @SerializedName("total_bytes") val totalBytes: Long = 0,
    @SerializedName("reason_code") val reasonCode: String? = null,
    val reason: String? = null,
    // The contract does not prescribe the shape of layer_range or plan.
    @SerializedName("layer_range") val layerRange: JsonElement? = null,
) {
    val progress: Float? get() = if (totalBytes > 0) {
        (bytesReceived.toDouble() / totalBytes).coerceIn(0.0, 1.0).toFloat()
    } else null
}

data class ModelIntentOperation(
    @SerializedName("operation_id") val operationId: String = "",
    val intent: ModelIntentRequest = ModelIntentRequest(""),
    val status: String = "queued",
    @SerializedName("reason_code") val reasonCode: String? = null,
    val reason: String? = null,
    @SerializedName("config_id") val configId: String? = null,
    val generation: Long = 0,
    @SerializedName("loaded_ready") val loadedReady: Boolean = false,
    val nodes: Map<String, ModelIntentNode> = emptyMap(),
    val plan: JsonElement? = null,
    @SerializedName("updated_at") val updatedAt: JsonElement? = null,
) {
    val readyForChat: Boolean get() = status == "ready" && loadedReady
    val finished: Boolean get() = readyForChat || status in setOf("failed", "cancelled", "superseded")
}

interface ModelIntentApi {
    suspend fun getModelIntentCatalog(): Result<ModelIntentCatalog>
    suspend fun getCurrentModelIntent(): Result<ModelIntentOperation?>
    suspend fun submitModelIntent(intent: ModelIntentRequest): Result<ModelIntentOperation>
    suspend fun getModelIntent(operationId: String): Result<ModelIntentOperation>
    suspend fun cancelModelIntent(operationId: String): Result<ModelIntentOperation>
}

internal fun validateModelIntentOperationId(id: String) {
    require(Regex("supply_[A-Za-z0-9_-]+").matches(id)) { "intent_operation_id_invalid" }
}

internal val MODEL_INTENT_STATUSES = setOf(
    "queued", "waiting_nodes", "inspecting", "inventory", "planning", "building", "supplying",
    "verifying", "preparing", "ready", "failed", "cancelled", "superseded",
)

internal fun ModelIntentOperation.validate(): ModelIntentOperation = apply {
    validateModelIntentOperationId(operationId)
    intent.validate()
    require(status in MODEL_INTENT_STATUSES && generation >= 0) { "intent_operation_invalid" }
}

/** Discovery verification is a separate HTTP step; UDP hints alone never confer membership. */
data class MasterBootstrapInfo(
    @SerializedName("is_master") val isMaster: Boolean = false,
)
