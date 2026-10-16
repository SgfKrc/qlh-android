package com.qlh.inference.network

import com.google.gson.JsonDeserializationContext
import com.google.gson.JsonDeserializer
import com.google.gson.JsonElement
import com.google.gson.JsonObject
import com.google.gson.annotations.JsonAdapter
import com.google.gson.annotations.SerializedName
import java.lang.reflect.Type

/** MODEL-INTENT-UX-01: Android sends a model choice; the master owns supply and prepare. */
data class ModelIntentRequest(
    @SerializedName("model_id") val modelId: String,
    val engine: String = "auto",
    @SerializedName("quant_type") val quantType: String = "",
    val strategy: String = "auto",
    @SerializedName("timeout_seconds") val timeoutSeconds: Int = 3600,
) {
    /** Android submits automatic supply; explicit strategies belong to the public API. */
    fun validate() = validateStrategies(setOf("auto"))

    /** An operation may have been created by another API client using an explicit strategy. */
    internal fun validateForOperation() = validateStrategies(setOf("auto", "local_cut", "distribute"))

    private fun validateStrategies(strategies: Set<String>) {
        require(modelId.isNotBlank()) { "model_id_required" }
        require(engine in setOf("auto", "pytorch", "llama_cpp")) { "intent_engine_invalid" }
        require(strategy in strategies && timeoutSeconds > 0) { "intent_options_invalid" }
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

/** Optional owner diagnostics. Missing fields are unknown, never zero budgets. */
@JsonAdapter(ModelIntentCapacityDiagnosticDeserializer::class)
data class ModelIntentCapacityDiagnostic(
    @SerializedName("reason_code") val reasonCode: String? = null,
    val reason: String? = null,
    val scope: String? = null,
    @SerializedName("suggested_action") val suggestedAction: String? = null,
    val nodes: List<ModelIntentCapacityNode> = emptyList(),
    @SerializedName("pending_node_ids") val pendingNodeIds: List<String> = emptyList(),
)

data class ModelIntentCapacityNode(
    @SerializedName("node_id") val nodeId: String? = null,
    val engine: String? = null,
    @SerializedName("execution_device") val executionDevice: String? = null,
    val resource: String? = null,
    @SerializedName("available_bytes") val availableBytes: Long? = null,
    @SerializedName("required_bytes") val requiredBytes: Long? = null,
    @SerializedName("deficit_bytes") val deficitBytes: Long? = null,
    @SerializedName("reserve_bytes") val reserveBytes: Long? = null,
    @SerializedName("layer_range") val layerRange: List<Long>? = null,
)

/** Gson's default Long adapter also accepts strings and truncates fractions. Do neither here. */
internal class ModelIntentCapacityDiagnosticDeserializer : JsonDeserializer<ModelIntentCapacityDiagnostic?> {
    private val integer = Regex("0|[1-9][0-9]*")

    private fun JsonElement?.text(): String? = this
        ?.takeIf { it.isJsonPrimitive && it.asJsonPrimitive.isString }?.asString

    private fun JsonElement?.bytes(): Long? {
        if (this == null || !isJsonPrimitive || !asJsonPrimitive.isNumber) return null
        val literal = asString
        return literal.takeIf { integer.matches(it) }?.toLongOrNull()
    }

    private fun JsonElement?.range(): List<Long>? {
        if (this == null || !isJsonArray || asJsonArray.size() != 2) return null
        val start = asJsonArray[0].bytes() ?: return null
        val end = asJsonArray[1].bytes() ?: return null
        return listOf(start, end).takeIf { start < end }
    }

    private fun node(value: JsonObject) = ModelIntentCapacityNode(
        nodeId = value["node_id"].text(),
        engine = value["engine"].text(),
        executionDevice = value["execution_device"].text(),
        resource = value["resource"].text(),
        availableBytes = value["available_bytes"].bytes(),
        requiredBytes = value["required_bytes"].bytes(),
        deficitBytes = value["deficit_bytes"].bytes(),
        reserveBytes = value["reserve_bytes"].bytes(),
        layerRange = value["layer_range"].range(),
    )

    override fun deserialize(
        json: JsonElement, typeOfT: Type, context: JsonDeserializationContext,
    ): ModelIntentCapacityDiagnostic? {
        if (!json.isJsonObject) return null
        val value = json.asJsonObject
        return ModelIntentCapacityDiagnostic(
            reasonCode = value["reason_code"].text(),
            reason = value["reason"].text(),
            scope = value["scope"].text(),
            suggestedAction = value["suggested_action"].text(),
            nodes = value["nodes"]?.takeIf { it.isJsonArray }?.asJsonArray
                ?.mapNotNull { it.takeIf { item -> item.isJsonObject }?.asJsonObject?.let(::node) }.orEmpty(),
            pendingNodeIds = value["pending_node_ids"]?.takeIf { it.isJsonArray }?.asJsonArray
                ?.mapNotNull { it.text()?.takeIf(String::isNotBlank) }.orEmpty(),
        )
    }
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
    @SerializedName("capacity_diagnostic") val capacityDiagnostic: ModelIntentCapacityDiagnostic? = null,
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
    intent.validateForOperation()
    require(status in MODEL_INTENT_STATUSES && generation >= 0) { "intent_operation_invalid" }
}

/** Discovery verification is a separate HTTP step; UDP hints alone never confer membership. */
data class MasterBootstrapInfo(
    @SerializedName("is_master") val isMaster: Boolean = false,
)
