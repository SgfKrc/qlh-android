package com.qlh.inference.worker

import com.google.gson.Gson
import com.google.gson.GsonBuilder
import com.google.gson.JsonArray
import com.google.gson.JsonElement
import com.google.gson.JsonNull
import com.google.gson.JsonObject
import com.google.gson.JsonParser
import java.nio.charset.StandardCharsets
import java.nio.ByteBuffer
import java.nio.charset.CharacterCodingException
import java.nio.charset.CodingErrorAction
import java.security.MessageDigest
import java.util.LinkedHashMap

/** Stable task-worker protocol errors which are safe to expose to the coordinator. */
class TaskWorkerProtocolException(
    message: String,
    val code: String,
    val field: String = "",
) : IllegalArgumentException(message)

data class TaskWorkerEnvelope(
    val protocol: String,
    val version: Int,
    val messageType: String,
    val messageId: String,
    val sentAtMs: Long,
    val payload: Map<String, Any?>,
)

data class TaskWorkerAttemptIdentity(
    val workflowId: String,
    val stageId: String,
    val attemptId: String,
    val leaseId: String,
    val leaseEpoch: Int,
) {
    fun asPayload(): Map<String, Any?> = mapOf(
        "workflow_id" to workflowId,
        "stage_id" to stageId,
        "attempt_id" to attemptId,
        "lease_id" to leaseId,
        "lease_epoch" to leaseEpoch,
    )
}

/**
 * Android's transport-independent v2 task-worker contract.
 *
 * This codec deliberately accepts only the Android full-worker role and the
 * text `full_inference` stage. Socket framing, authentication and execution
 * remain the responsibility of AND-B-02/03.
 */
object TaskWorkerProtocol {
    const val PROTOCOL = "qlh.task_worker"
    const val VERSION = 2
    const val MAX_MESSAGE_BYTES = 8 * 1024 * 1024
    const val ANDROID_WORKER_KIND = "android_full_worker"

    const val HELLO = "hello"
    const val HELLO_ACK = "hello_ack"
    const val STAGE_OFFER = "stage_offer"
    const val STAGE_ACCEPT = "stage_accept"
    const val LEASE_RENEW = "lease_renew"
    const val STAGE_RESULT = "stage_result"
    const val STAGE_ERROR = "stage_error"
    const val STAGE_CANCEL = "stage_cancel"
    const val STAGE_CANCELLED = "stage_cancelled"

    private val gson = GsonBuilder()
        .disableHtmlEscaping()
        .serializeNulls()
        .create()
    private val envelopeFields = setOf(
        "protocol", "version", "message_type", "message_id", "sent_at_ms", "payload",
    )
    private val identityFields = setOf(
        "workflow_id", "stage_id", "attempt_id", "lease_id", "lease_epoch",
    )
    private val safeId = Regex("^[A-Za-z0-9_.:-]{1,128}$")
    private val messageId = Regex("^msg_[A-Za-z0-9_-]{8,96}$")
    private val workflowId = Regex("^wf_[A-Za-z0-9_-]{8,96}$")
    private val attemptId = Regex("^att_[A-Za-z0-9_-]{8,96}$")
    private val leaseId = Regex("^lease_[A-Za-z0-9_-]{8,96}$")
    private val sha256 = Regex("^[0-9a-f]{64}$")
    private val safeCode = Regex("^[a-z][a-z0-9_]{0,63}$")
    private val messageTypes = setOf(
        HELLO, HELLO_ACK, STAGE_OFFER, STAGE_ACCEPT, LEASE_RENEW,
        STAGE_RESULT, STAGE_ERROR, STAGE_CANCEL, STAGE_CANCELLED,
    )

    fun buildHello(
        nodeId: String,
        capabilities: Map<String, Any?>,
        messageId: String,
        sentAtMs: Long,
    ): TaskWorkerEnvelope = build(
        messageType = HELLO,
        payload = mapOf(
            "node_id" to nodeId,
            "worker_kind" to ANDROID_WORKER_KIND,
            "min_version" to VERSION,
            "max_version" to VERSION,
            "capabilities" to capabilities,
        ),
        messageId = messageId,
        sentAtMs = sentAtMs,
    )

    fun buildHelloAck(
        coordinatorNodeId: String,
        accepted: Boolean,
        selectedVersion: Int,
        reasonCode: String,
        messageId: String,
        sentAtMs: Long,
    ): TaskWorkerEnvelope = build(
        messageType = HELLO_ACK,
        payload = mapOf(
            "coordinator_node_id" to coordinatorNodeId,
            "accepted" to accepted,
            "selected_version" to selectedVersion,
            "reason_code" to reasonCode,
        ),
        messageId = messageId,
        sentAtMs = sentAtMs,
    )

    fun buildStageOffer(
        identity: TaskWorkerAttemptIdentity,
        requestId: String,
        stageType: String,
        providerId: String,
        leaseExpiresAtMs: Long,
        rootInput: Map<String, Any?>,
        dependencies: Map<String, Any?>,
        modelIdentity: Map<String, Any?>,
        messageId: String,
        sentAtMs: Long,
    ): TaskWorkerEnvelope {
        val payload = identity.asPayload() + mapOf(
            "request_id" to requestId,
            "stage_type" to stageType,
            "provider_id" to providerId,
            "lease_expires_at_ms" to leaseExpiresAtMs,
            "root_input" to rootInput,
            "dependencies" to dependencies,
            "input_sha256" to stageInputSha256(rootInput, dependencies),
            "model_identity" to modelIdentity,
        )
        return build(STAGE_OFFER, payload, messageId, sentAtMs)
    }

    fun buildStageAccept(
        identity: TaskWorkerAttemptIdentity,
        providerId: String,
        accepted: Boolean,
        reasonCode: String,
        retryable: Boolean,
        messageId: String,
        sentAtMs: Long,
    ): TaskWorkerEnvelope = build(
        STAGE_ACCEPT,
        identity.asPayload() + mapOf(
            "provider_id" to providerId,
            "accepted" to accepted,
            "reason_code" to reasonCode,
            "retryable" to retryable,
        ),
        messageId,
        sentAtMs,
    )

    fun buildLeaseRenew(
        identity: TaskWorkerAttemptIdentity,
        leaseExpiresAtMs: Long,
        messageId: String,
        sentAtMs: Long,
    ): TaskWorkerEnvelope = build(
        LEASE_RENEW,
        identity.asPayload() + mapOf("lease_expires_at_ms" to leaseExpiresAtMs),
        messageId,
        sentAtMs,
    )

    fun buildStageResult(
        identity: TaskWorkerAttemptIdentity,
        providerId: String,
        output: Map<String, Any?>,
        metadata: Map<String, Any?>,
        messageId: String,
        sentAtMs: Long,
    ): TaskWorkerEnvelope {
        val outputSha256 = stageOutputSha256(output)
        return build(
            STAGE_RESULT,
            identity.asPayload() + mapOf(
                "provider_id" to providerId,
                "output" to output,
                "output_sha256" to outputSha256,
                "metadata" to metadata,
            ),
            messageId,
            sentAtMs,
        )
    }

    fun buildStageError(
        identity: TaskWorkerAttemptIdentity,
        providerId: String,
        errorCode: String,
        retryable: Boolean,
        messageId: String,
        sentAtMs: Long,
    ): TaskWorkerEnvelope = build(
        STAGE_ERROR,
        identity.asPayload() + mapOf(
            "provider_id" to providerId,
            "error_code" to errorCode,
            "retryable" to retryable,
        ),
        messageId,
        sentAtMs,
    )

    fun buildStageCancel(
        identity: TaskWorkerAttemptIdentity,
        reasonCode: String,
        messageId: String,
        sentAtMs: Long,
    ): TaskWorkerEnvelope = build(
        STAGE_CANCEL,
        identity.asPayload() + mapOf("reason_code" to reasonCode),
        messageId,
        sentAtMs,
    )

    fun buildStageCancelled(
        identity: TaskWorkerAttemptIdentity,
        providerId: String,
        reasonCode: String,
        messageId: String,
        sentAtMs: Long,
    ): TaskWorkerEnvelope = build(
        STAGE_CANCELLED,
        identity.asPayload() + mapOf(
            "provider_id" to providerId,
            "reason_code" to reasonCode,
        ),
        messageId,
        sentAtMs,
    )

    fun build(
        messageType: String,
        payload: Map<String, Any?>,
        messageId: String,
        sentAtMs: Long,
    ): TaskWorkerEnvelope {
        val envelope = TaskWorkerEnvelope(
            protocol = PROTOCOL,
            version = VERSION,
            messageType = messageType,
            messageId = messageId,
            sentAtMs = sentAtMs,
            payload = payload,
        )
        validate(envelope)
        return envelope
    }

    fun encode(envelope: TaskWorkerEnvelope): ByteArray {
        validate(envelope)
        val encoded = canonicalJson(envelopeSnapshot(envelope)).toByteArray(StandardCharsets.UTF_8)
        if (encoded.size > MAX_MESSAGE_BYTES) {
            fail("message exceeds maximum size", "message_too_large", "message")
        }
        return encoded
    }

    fun decode(bytes: ByteArray): TaskWorkerEnvelope {
        if (bytes.size > MAX_MESSAGE_BYTES) {
            fail("message exceeds maximum size", "message_too_large", "message")
        }
        val root = try {
            JsonParser.parseString(decodeUtf8(bytes))
        } catch (error: TaskWorkerProtocolException) {
            throw error
        } catch (error: Exception) {
            fail("message is not valid JSON", "invalid_json", "message", error)
        }
        if (!root.isJsonObject) fail("message must be an object", "invalid_envelope", "message")
        val objectValue = root.asJsonObject
        requireExact(objectValue, envelopeFields, "message")
        val payloadElement = objectValue.get("payload")
        if (payloadElement == null || !payloadElement.isJsonObject) {
            fail("payload must be an object", "invalid_object", "payload")
        }
        val envelope = TaskWorkerEnvelope(
            protocol = requiredString(objectValue, "protocol"),
            version = requiredInt(objectValue, "version"),
            messageType = requiredString(objectValue, "message_type"),
            messageId = requiredString(objectValue, "message_id"),
            sentAtMs = requiredLong(objectValue, "sent_at_ms"),
            payload = jsonValue(payloadElement) as Map<String, Any?>,
        )
        validate(envelope)
        return envelope
    }

    fun canonicalJson(value: Any?): String {
        val element = if (value is JsonElement) value else gson.toJsonTree(value)
        return gson.toJson(sortJson(element))
    }

    fun canonicalBytes(envelope: TaskWorkerEnvelope): ByteArray =
        canonicalJson(envelopeSnapshot(envelope)).toByteArray(StandardCharsets.UTF_8)

    fun messageDigest(envelope: TaskWorkerEnvelope): String = sha256(canonicalBytes(envelope))

    fun stageInputSha256(
        rootInput: Map<String, Any?>,
        dependencies: Map<String, Any?>,
    ): String = sha256(
        canonicalJson(
            mapOf("dependencies" to dependencies, "root_input" to rootInput),
        ).toByteArray(StandardCharsets.UTF_8)
    )

    fun stageOutputSha256(output: Map<String, Any?>): String =
        sha256(canonicalJson(output).toByteArray(StandardCharsets.UTF_8))

    /** Attempt + lease epoch + digest is the result idempotency key. */
    fun attemptIdempotencyKey(identity: TaskWorkerAttemptIdentity, digest: String): String {
        requireSha(digest, "digest")
        return "${identity.attemptId}:${identity.leaseEpoch}:$digest"
    }

    fun validate(envelope: TaskWorkerEnvelope) {
        if (envelope.protocol != PROTOCOL) fail("unsupported protocol", "unsupported_protocol", "protocol")
        if (envelope.version != VERSION) {
            fail("Android worker requires protocol v2", "unsupported_protocol_version", "version")
        }
        if (envelope.messageType !in messageTypes) {
            fail("unsupported message type", "unsupported_message_type", "message_type")
        }
        requirePattern(envelope.messageId, messageId, "message_id")
        if (envelope.sentAtMs < 0) fail("sent_at_ms must be non-negative", "invalid_integer", "sent_at_ms")
        requireExact(envelope.payload.keys, payloadFields(envelope.messageType), "payload")

        when (envelope.messageType) {
            HELLO -> validateHello(envelope.payload)
            HELLO_ACK -> validateHelloAck(envelope.payload, envelope.version)
            else -> validateStagePayload(envelope)
        }
    }

    private fun validateHello(payload: Map<String, Any?>) {
        requirePattern(string(payload, "node_id"), safeId, "payload.node_id")
        if (payload["worker_kind"] != ANDROID_WORKER_KIND) {
            fail("worker_kind must be $ANDROID_WORKER_KIND", "unsupported_worker_kind", "payload.worker_kind")
        }
        if (integer(payload, "min_version") != VERSION || integer(payload, "max_version") != VERSION) {
            fail("Android worker requires protocol v2", "invalid_version_range", "payload.min_version")
        }
        val capabilities = objectValue(payload, "capabilities")
        val expectedCapabilityFields = setOf("stage_types", "engines", "models", "max_concurrency")
        requireExact(
            capabilities.keys,
            expectedCapabilityFields + if (capabilities.containsKey("resource_gate")) setOf("resource_gate") else emptySet(),
            "payload.capabilities",
        )
        val stageTypes = stringList(capabilities, "stage_types")
        if (stageTypes != listOf("full_inference")) {
            fail("Android workers may advertise only full_inference", "invalid_capabilities", "payload.capabilities.stage_types")
        }
        val engines = stringList(capabilities, "engines")
        if (engines != listOf("llama_cpp")) {
            fail("Android workers must advertise llama_cpp", "invalid_capabilities", "payload.capabilities.engines")
        }
        val models = list(capabilities, "models")
        models.forEachIndexed { index, item -> validateModelIdentity(item as? Map<*, *>, "payload.capabilities.models[$index]") }
        val maxConcurrency = integer(capabilities, "max_concurrency")
        if (maxConcurrency != 1) fail("Android workers support one concurrent task", "invalid_capabilities", "payload.capabilities.max_concurrency")
        if (capabilities.containsKey("resource_gate")) {
            validateResourceGate(capabilities["resource_gate"] as? Map<*, *>, "payload.capabilities.resource_gate")
        }
    }

    private fun validateHelloAck(payload: Map<String, Any?>, version: Int) {
        requirePattern(string(payload, "coordinator_node_id"), safeId, "payload.coordinator_node_id")
        val accepted = boolean(payload, "accepted")
        val selected = integer(payload, "selected_version", allowZero = true)
        val reason = code(payload, "reason_code", allowEmpty = true)
        if (accepted && (selected != version || reason.isNotEmpty())) {
            fail("accepted hello_ack must select v2 without a reason", "invalid_negotiation_result", "payload")
        }
        if (!accepted && (selected != 0 || reason.isEmpty())) {
            fail("rejected hello_ack must include a reason", "invalid_negotiation_result", "payload")
        }
    }

    private fun validateStagePayload(envelope: TaskWorkerEnvelope) {
        val payload = envelope.payload
        validateIdentity(payload)
        val type = envelope.messageType
        if (type in setOf(STAGE_OFFER, STAGE_ACCEPT, STAGE_RESULT, STAGE_ERROR, STAGE_CANCELLED)) {
            requirePattern(string(payload, "provider_id"), safeId, "payload.provider_id")
        }
        when (type) {
            STAGE_OFFER -> {
                requirePattern(string(payload, "request_id", allowEmpty = true), safeId, "payload.request_id")
                if (payload["stage_type"] != "full_inference") fail("unsupported stage type", "unsupported_stage_type", "payload.stage_type")
                val deadline = long(payload, "lease_expires_at_ms")
                if (deadline <= envelope.sentAtMs) fail("lease deadline must be later than message timestamp", "invalid_lease_deadline", "payload.lease_expires_at_ms")
                val rootInput = objectValue(payload, "root_input")
                val dependencies = objectValue(payload, "dependencies")
                requireSha(string(payload, "input_sha256"), "payload.input_sha256")
                if (payload["input_sha256"] != stageInputSha256(rootInput, dependencies)) {
                    fail("stage input digest does not match payload", "input_digest_mismatch", "payload.input_sha256")
                }
                validateModelIdentity(payload["model_identity"] as? Map<*, *>, "payload.model_identity")
            }
            STAGE_ACCEPT -> {
                val accepted = boolean(payload, "accepted")
                val reason = code(payload, "reason_code", allowEmpty = true)
                val retryable = boolean(payload, "retryable")
                if (accepted == reason.isNotEmpty()) fail("acceptance and reason_code disagree", "invalid_acceptance_result", "payload")
                if (accepted && retryable) fail("accepted stage cannot be retryable", "invalid_acceptance_result", "payload.retryable")
            }
            LEASE_RENEW -> validateDeadline(payload, envelope.sentAtMs)
            STAGE_RESULT -> {
                val output = objectValue(payload, "output")
                requireSha(string(payload, "output_sha256"), "payload.output_sha256")
                if (payload["output_sha256"] != stageOutputSha256(output)) {
                    fail("stage output digest does not match output", "output_digest_mismatch", "payload.output_sha256")
                }
                objectValue(payload, "metadata")
            }
            STAGE_ERROR -> {
                code(payload, "error_code")
                boolean(payload, "retryable")
            }
            STAGE_CANCEL, STAGE_CANCELLED -> code(payload, "reason_code")
        }
    }

    private fun validateIdentity(payload: Map<String, Any?>) {
        requirePattern(string(payload, "workflow_id"), workflowId, "payload.workflow_id")
        requirePattern(string(payload, "stage_id"), safeId, "payload.stage_id")
        requirePattern(string(payload, "attempt_id"), attemptId, "payload.attempt_id")
        requirePattern(string(payload, "lease_id"), leaseId, "payload.lease_id")
        val epoch = integer(payload, "lease_epoch")
        if (epoch < 1) fail("lease_epoch must be positive", "invalid_integer", "payload.lease_epoch")
    }

    private fun validateDeadline(payload: Map<String, Any?>, sentAtMs: Long) {
        val deadline = long(payload, "lease_expires_at_ms")
        if (deadline <= sentAtMs) fail("lease deadline must be later than message timestamp", "invalid_lease_deadline", "payload.lease_expires_at_ms")
    }

    private fun validateModelIdentity(value: Map<*, *>?, field: String) {
        if (value == null) fail("model identity must be an object", "invalid_object", field)
        requireExact(value.keys.map { it.toString() }.toSet(), setOf("model_id", "engine", "format", "revision", "sha256"), field)
        requirePattern(value["model_id"].toString(), safeId, "$field.model_id")
        if (value["engine"] != "llama_cpp") fail("Android worker requires llama_cpp", "invalid_model_identity", "$field.engine")
        requirePattern(value["format"].toString(), safeId, "$field.format")
        requirePattern(value["revision"].toString(), safeId, "$field.revision")
        requireSha(value["sha256"].toString(), "$field.sha256")
    }

    private fun validateResourceGate(value: Map<*, *>?, field: String) {
        if (value == null) fail("resource gate must be an object", "invalid_object", field)
        requireExact(value.keys.map { it.toString() }.toSet(), setOf("admitted", "reason_code"), field)
        val admitted = value["admitted"]
        if (admitted !is Boolean) fail("admitted must be a boolean", "invalid_boolean", "$field.admitted")
        val reason = value["reason_code"]
        if (reason !is String || (reason.isNotEmpty() && !safeCode.matches(reason))) {
            fail("reason_code is invalid", "invalid_code", "$field.reason_code")
        }
        if (admitted && reason.isNotEmpty()) fail("admitted resource gate cannot carry a reason", "invalid_resource_gate", "$field.reason_code")
        if (!admitted && reason.isEmpty()) fail("rejected resource gate requires a reason", "invalid_resource_gate", "$field.reason_code")
    }

    private fun envelopeSnapshot(envelope: TaskWorkerEnvelope): Map<String, Any?> = mapOf(
        "protocol" to envelope.protocol,
        "version" to envelope.version,
        "message_type" to envelope.messageType,
        "message_id" to envelope.messageId,
        "sent_at_ms" to envelope.sentAtMs,
        "payload" to envelope.payload,
    )

    private fun payloadFields(type: String): Set<String> = when (type) {
        HELLO -> setOf("node_id", "worker_kind", "min_version", "max_version", "capabilities")
        HELLO_ACK -> setOf("coordinator_node_id", "accepted", "selected_version", "reason_code")
        STAGE_OFFER -> identityFields + setOf("request_id", "stage_type", "provider_id", "lease_expires_at_ms", "root_input", "dependencies", "input_sha256", "model_identity")
        STAGE_ACCEPT -> identityFields + setOf("provider_id", "accepted", "reason_code", "retryable")
        LEASE_RENEW -> identityFields + setOf("lease_expires_at_ms")
        STAGE_RESULT -> identityFields + setOf("provider_id", "output", "output_sha256", "metadata")
        STAGE_ERROR -> identityFields + setOf("provider_id", "error_code", "retryable")
        STAGE_CANCEL -> identityFields + setOf("reason_code")
        STAGE_CANCELLED -> identityFields + setOf("provider_id", "reason_code")
        else -> emptySet()
    }

    private fun objectValue(container: Map<String, Any?>, key: String): Map<String, Any?> {
        val value = container[key]
        if (value !is Map<*, *>) fail("$key must be an object", "invalid_object", "payload.$key")
        return value.entries.associate { it.key.toString() to it.value }
    }

    private fun list(container: Map<String, Any?>, key: String): List<Any?> {
        val value = container[key]
        if (value !is List<*>) fail("$key must be a list", "invalid_array", "payload.$key")
        return value
    }

    private fun stringList(container: Map<String, Any?>, key: String): List<String> =
        list(container, key).map { if (it !is String) fail("$key must contain strings", "invalid_string", "payload.$key") else it }

    private fun string(container: Map<String, Any?>, key: String, allowEmpty: Boolean = false): String {
        val value = container[key]
        if (value !is String || (!allowEmpty && value.isEmpty())) fail("$key must be a string", "invalid_string", "payload.$key")
        return value
    }

    private fun code(container: Map<String, Any?>, key: String, allowEmpty: Boolean = false): String {
        val value = string(container, key, allowEmpty)
        if (value.isNotEmpty() && !safeCode.matches(value)) fail("$key is invalid", "invalid_code", "payload.$key")
        return value
    }

    private fun boolean(container: Map<String, Any?>, key: String): Boolean {
        val value = container[key]
        if (value !is Boolean) fail("$key must be a boolean", "invalid_boolean", "payload.$key")
        return value
    }

    private fun integer(container: Map<String, Any?>, key: String, allowZero: Boolean = false): Int {
        val value = container[key]
        val number = when (value) {
            is Number -> value.toDouble()
            else -> Double.NaN
        }
        val minimum = if (allowZero) 0 else 1
        if (!number.isFinite() || number % 1.0 != 0.0 || number < minimum || number > Int.MAX_VALUE) {
            fail("$key must be an integer", "invalid_integer", "payload.$key")
        }
        return number.toInt()
    }

    private fun long(container: Map<String, Any?>, key: String): Long {
        val value = container[key]
        val number = when (value) {
            is Number -> value.toDouble()
            else -> Double.NaN
        }
        if (!number.isFinite() || number % 1.0 != 0.0 || number < 0 || number > Long.MAX_VALUE.toDouble()) {
            fail("$key must be a non-negative integer", "invalid_integer", "payload.$key")
        }
        return number.toLong()
    }

    private fun requiredString(value: JsonObject, key: String): String {
        val item = value.get(key)
        if (item == null || !item.isJsonPrimitive || !item.asJsonPrimitive.isString) {
            fail("$key must be a string", "invalid_string", key)
        }
        return item.asString
    }

    private fun requiredInt(value: JsonObject, key: String): Int {
        val item = value.get(key)
        if (item == null || !item.isJsonPrimitive || !item.asJsonPrimitive.isNumber) {
            fail("$key must be an integer", "invalid_integer", key)
        }
        return item.asInt
    }

    private fun requiredLong(value: JsonObject, key: String): Long {
        val item = value.get(key)
        if (item == null || !item.isJsonPrimitive || !item.asJsonPrimitive.isNumber) {
            fail("$key must be an integer", "invalid_integer", key)
        }
        return item.asLong
    }

    private fun requireExact(actual: Set<String>, expected: Set<String>, field: String) {
        if (actual != expected) fail("$field fields do not match schema", "invalid_fields", field)
    }

    private fun requireExact(actual: JsonObject, expected: Set<String>, field: String) {
        requireExact(actual.keySet(), expected, field)
    }

    private fun requirePattern(value: String, pattern: Regex, field: String) {
        if (!pattern.matches(value)) fail("$field is invalid", "invalid_identifier", field)
    }

    private fun requireSha(value: String, field: String) = requirePattern(value, sha256, field)

    private fun sortJson(element: JsonElement): JsonElement = when {
        element.isJsonObject -> JsonObject().also { output ->
            element.asJsonObject.entrySet().sortedBy { it.key }.forEach { (key, child) ->
                output.add(key, sortJson(child))
            }
        }
        element.isJsonArray -> JsonArray().also { output ->
            element.asJsonArray.forEach { output.add(sortJson(it)) }
        }
        element.isJsonNull -> JsonNull.INSTANCE
        else -> element
    }

    private fun decodeUtf8(bytes: ByteArray): String = try {
        StandardCharsets.UTF_8.newDecoder()
            .onMalformedInput(CodingErrorAction.REPORT)
            .onUnmappableCharacter(CodingErrorAction.REPORT)
            .decode(ByteBuffer.wrap(bytes))
            .toString()
    } catch (error: CharacterCodingException) {
        fail("message must be valid UTF-8", "invalid_encoding", "message", error)
    }

    /** Convert parsed JSON while preserving integer tokens for digest stability. */
    private fun jsonValue(element: JsonElement): Any? = when {
        element.isJsonObject -> element.asJsonObject.entrySet().associate {
            it.key to jsonValue(it.value)
        }
        element.isJsonArray -> element.asJsonArray.map(::jsonValue)
        element.isJsonNull -> null
        element.asJsonPrimitive.isBoolean -> element.asBoolean
        element.asJsonPrimitive.isString -> element.asString
        else -> {
            val raw = element.asString
            raw.toLongOrNull() ?: raw.toDoubleOrNull()
                ?: fail("invalid JSON number", "invalid_json_value", "message.payload")
        }
    }

    private fun sha256(bytes: ByteArray): String = MessageDigest.getInstance("SHA-256")
        .digest(bytes)
        .joinToString("") { "%02x".format(it) }

    private fun fail(message: String, code: String, field: String, cause: Throwable? = null): Nothing {
        throw TaskWorkerProtocolException(message, code, field).also {
            if (cause != null) it.initCause(cause)
        }
    }
}

/** Bounded message-id replay cache. Same ID/digest replays; same ID/different content rejects. */
class TaskWorkerReplayCache(private val maxEntries: Int = 1024) {
    private data class Entry(val digest: String, val response: TaskWorkerEnvelope?)

    private val entries = LinkedHashMap<String, Entry>(maxEntries, 0.75f, true)

    @Synchronized
    fun remember(message: TaskWorkerEnvelope, response: TaskWorkerEnvelope? = null) {
        val digest = TaskWorkerProtocol.messageDigest(message)
        entries[message.messageId] = Entry(digest, response)
        while (entries.size > maxEntries) entries.remove(entries.entries.first().key)
    }

    @Synchronized
    fun replay(message: TaskWorkerEnvelope): TaskWorkerEnvelope? {
        val entry = entries[message.messageId] ?: return null
        val digest = TaskWorkerProtocol.messageDigest(message)
        if (entry.digest != digest) {
            throw TaskWorkerProtocolException(
                "message_id was reused with different content",
                "message_id_conflict",
                "message_id",
            )
        }
        return entry.response
    }

    @Synchronized
    fun size(): Int = entries.size
}
