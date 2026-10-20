package com.qlh.inference.worker

/** Frozen layer token-output contract shared by the codec and direct executor entry. */
internal object MultiPositionArgmaxContract {
    const val ALL_POSITIONS = "all_positions"
    private val resultFields = setOf("token_argmax_by_position", "positions", "n_vocab")

    // JSON 1.0, booleans, strings and overflowing numbers are not int32 tokens/positions.
    fun nonNegativeInt32(value: Any?): Int? {
        val number = when (value) {
            is Byte -> value.toLong()
            is Short -> value.toLong()
            is Int -> value.toLong()
            is Long -> value
            else -> return null
        }
        return number.takeIf { it in 0..Int.MAX_VALUE.toLong() }?.toInt()
    }

    private fun invalid(message: String, field: String): Nothing =
        throw TaskWorkerProtocolException(
            message, "invalid_token_output", field,
        )

    fun validateRequest(payload: Map<String, Any?>): Boolean {
        if (!payload.containsKey("token_output")) return false
        if (payload["stage_type"] != TaskWorkerProtocol.LAYER_FORWARD_STAGE ||
            payload["token_output"] != ALL_POSITIONS) {
            invalid("token_output must be all_positions on layer_forward", "payload.token_output")
        }
        val spec = payload["hidden_spec"] as? Map<*, *>
        val count = nonNegativeInt32(spec?.get("n_tokens"))?.takeIf { it > 0 }
            ?: invalid("n_tokens must be a positive int32", "payload.hidden_spec.n_tokens")
        val root = payload["root_input"] as? Map<*, *>
        if (root?.get("want_hidden") != false) {
            invalid("all_positions requires explicit want_hidden=false", "payload.root_input.want_hidden")
        }
        validatePositions(payload["positions"], count, "payload.positions")
        if (payload.containsKey("seq_ids")) {
            val ids = payload["seq_ids"] as? List<*>
            if (ids == null || ids.size != count ||
                ids.any { nonNegativeInt32(it) == null } || ids.map(::nonNegativeInt32).distinct().size != 1) {
                invalid("seq_ids must contain n_tokens non-negative int32 entries for one sequence",
                    "payload.seq_ids")
            }
        }
        return true
    }

    private fun validatePositions(value: Any?, count: Int, field: String) {
        val positions = value as? List<*>
        if (positions == null || positions.size != count || positions.isEmpty()) {
            invalid("positions must be an explicit list of length n_tokens", field)
        }
        var previous = -1L
        positions.forEachIndexed { index, raw ->
            val current = nonNegativeInt32(raw)?.toLong()
                ?: invalid("positions entries must be non-negative int32", "$field[$index]")
            if (index > 0 && current != previous + 1L) {
                invalid("positions must be contiguous and increasing", field)
            }
            previous = current
        }
    }

    fun validateResult(output: Map<String, Any?>) {
        if (resultFields.none(output::containsKey)) return // Legacy scalar output.
        val field = "payload.output"
        if (output.containsKey("hidden_out_f32") || output.containsKey("hidden_out_ref")) {
            invalid("multi-position token output must not contain hidden output", field)
        }
        if (!(resultFields + setOf("token_argmax", "n_tokens")).all(output::containsKey)) {
            invalid("multi-position output fields must be supplied together", field)
        }
        val count = nonNegativeInt32(output["n_tokens"])?.takeIf { it > 0 }
            ?: invalid("n_tokens must be a positive int32", "$field.n_tokens")
        val vocab = nonNegativeInt32(output["n_vocab"])?.takeIf { it > 0 }
            ?: invalid("n_vocab must be a positive int32", "$field.n_vocab")
        val tokens = output["token_argmax_by_position"] as? List<*>
        if (tokens == null || tokens.size != count || tokens.any {
                val token = nonNegativeInt32(it)
                token == null || token >= vocab
            }) {
            invalid("token_argmax_by_position must contain n_tokens int32 tokens in vocabulary",
                "$field.token_argmax_by_position")
        }
        validatePositions(output["positions"], count, "$field.positions")
        val scalar = nonNegativeInt32(output["token_argmax"])
        if (scalar == null || scalar >= vocab || scalar != nonNegativeInt32(tokens.last())) {
            invalid("token_argmax must equal the final position token", "$field.token_argmax")
        }
    }
}
