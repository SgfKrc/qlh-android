package com.qlh.inference

import com.qlh.inference.network.AuditData

const val MAX_AUDIT_REQUESTS = 8
const val MAX_AUDIT_REVIEWS = 8

data class AuditUiState(val snapshot: AuditSnapshot? = null, val loading: Boolean = false, val error: String? = null)
data class AuditSnapshot(
    val available: Boolean = false,
    val requests: List<AuditRequest> = emptyList(),
    val reviews: List<AuditReviewTicket> = emptyList(),
)
data class AuditRequest(val requestId: String, val deploymentId: String, val phase: String,
    val executionState: String, val terminal: String, val outcome: String)
data class AuditReviewTicket(val ticketId: String, val status: String, val createdAt: Double,
    val targetNodeId: String, val score: Int, val voteCount: Int)

fun toAuditSnapshot(data: AuditData): AuditSnapshot {
    val valid = data.requests.schemaVersion == 2 && data.requests.validity == "valid" && data.requests.revision != null
    val requests = if (!valid) emptyList() else data.requests.requests.asSequence()
        .filter { it.requestId.isNotBlank() }.take(MAX_AUDIT_REQUESTS)
        .map { AuditRequest(it.requestId.trim(), it.deploymentId.trim(), it.phase.trim(),
            it.executionState.trim(), it.terminal.orEmpty(), it.outcome.orEmpty()) }.toList()
    val reviews = data.reviewTickets.tickets
        .asSequence()
        .filter { it.ticketId.isNotBlank() }
        .take(MAX_AUDIT_REVIEWS)
        .map { ticket ->
            AuditReviewTicket(
                ticketId = ticket.ticketId.trim(),
                status = ticket.status.trim().ifBlank { "unknown" },
                createdAt = safeAuditTimestamp(ticket.createdAt),
                targetNodeId = ticket.targetNodeId.trim().ifBlank { "unknown" },
                score = ticket.score.coerceIn(-100, 100),
                voteCount = ticket.voteCount.coerceIn(0, 100),
            )
        }
        .sortedByDescending { it.createdAt }
        .toList()

    return AuditSnapshot(available = valid, requests = requests, reviews = reviews)
}

private fun safeAuditTimestamp(value: Double): Double =
    value.takeIf { it.isFinite() && it >= 0.0 && it <= 1.0e15 } ?: 0.0
