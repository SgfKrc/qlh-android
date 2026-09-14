package com.qlh.inference.worker

import kotlin.math.roundToInt

/** Stable worker-side failures; TaskWorkerClient maps these to protocol error codes. */
class AndroidFullWorkerStageException(
    val code: String,
    message: String,
) : IllegalStateException(message)

/**
 * Text-only Full Worker executor shared by the foreground service and JVM fake
 * worker tests. It never accepts a path or a model downloaded by the coordinator.
 */
class AndroidFullWorkerStageExecutor(
    private val expectedModelIdentity: () -> Map<String, Any?>?,
    private val ensureModelLoaded: suspend (contextSize: Int) -> Result<Unit>,
    private val generate: suspend (
        prompt: String,
        maxTokens: Int,
        temperature: Float,
        topP: Float,
    ) -> Result<String>,
) : TaskWorkerStageHandler {
    override suspend fun execute(offer: TaskWorkerEnvelope): TaskWorkerStageExecution {
        if (offer.messageType != TaskWorkerProtocol.STAGE_OFFER) {
            throw AndroidFullWorkerStageException(
                "invalid_stage_message",
                "Android Full Worker accepts stage_offer only",
            )
        }
        val payload = offer.payload
        if (payload["stage_type"] != "full_inference") {
            throw AndroidFullWorkerStageException(
                "unsupported_stage_type",
                "Android Full Worker accepts full_inference only",
            )
        }
        val advertised = expectedModelIdentity()
        val requested = payload["model_identity"] as? Map<*, *>
        if (advertised == null || requested == null || !sameIdentity(advertised, requested)) {
            throw AndroidFullWorkerStageException(
                "model_identity_mismatch",
                "Stage model identity does not match the Android worker",
            )
        }
        val rootInput = payload["root_input"] as? Map<*, *>
            ?: throw AndroidFullWorkerStageException("invalid_stage_input", "root_input must be an object")
        val prompt = extractPrompt(rootInput)
        val maxTokens = boundedInt(rootInput["max_new_tokens"], 1024, 1, 4096)
        val temperature = boundedFloat(rootInput["temperature"], 0.7f, 0.0f, 2.0f)
        val topP = boundedFloat(rootInput["top_p"], 0.9f, 0.0f, 1.0f)
        val contextSize = boundedInt(rootInput["context_size"], 2048, 256, 32768)

        ensureModelLoaded(contextSize).getOrElse { error ->
            throw AndroidFullWorkerStageException(
                "model_not_ready",
                error.message ?: "Android inference model is not ready",
            )
        }
        val content = generate(prompt, maxTokens, temperature, topP).getOrElse { error ->
            throw AndroidFullWorkerStageException(
                "worker_execution_failed",
                error.message ?: "Android inference failed",
            )
        }
        return TaskWorkerStageExecution(
            output = mapOf("content" to content),
            metadata = mapOf(
                // v2 text results allow `model`; provider/node provenance is
                // persisted by the coordinator's attempt journal.
                "model" to (advertised["model_id"] as? String).orEmpty(),
            ),
        )
    }

    private fun extractPrompt(rootInput: Map<*, *>): String {
        val direct = rootInput["prompt"] as? String
        if (!direct.isNullOrBlank()) return direct.take(MAX_PROMPT_CHARS)
        val message = rootInput["message"] as? String
        if (!message.isNullOrBlank()) return message.take(MAX_PROMPT_CHARS)
        val messages = rootInput["messages"] as? List<*>
        if (messages != null) {
            val joined = messages.mapNotNull { item ->
                val entry = item as? Map<*, *> ?: return@mapNotNull null
                val content = entry["content"] as? String ?: return@mapNotNull null
                val role = (entry["role"] as? String).orEmpty().ifBlank { "user" }
                "$role: $content"
            }.joinToString("\n")
            if (joined.isNotBlank()) return joined.take(MAX_PROMPT_CHARS)
        }
        throw AndroidFullWorkerStageException(
            "invalid_stage_input",
            "root_input must contain prompt, message, or messages",
        )
    }

    private fun sameIdentity(expected: Map<String, Any?>, requested: Map<*, *>): Boolean =
        listOf("model_id", "engine", "format", "revision", "sha256").all { key ->
            expected[key] == requested[key]
        }

    private fun boundedInt(value: Any?, fallback: Int, minimum: Int, maximum: Int): Int {
        val number = (value as? Number)?.toDouble() ?: return fallback
        if (!number.isFinite()) return fallback
        return number.roundToInt().coerceIn(minimum, maximum)
    }

    private fun boundedFloat(value: Any?, fallback: Float, minimum: Float, maximum: Float): Float {
        val number = (value as? Number)?.toFloat() ?: return fallback
        return if (number.isFinite()) number.coerceIn(minimum, maximum) else fallback
    }

    companion object {
        private const val MAX_PROMPT_CHARS = 64 * 1024
    }
}
