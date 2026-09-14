package com.qlh.inference

import com.qlh.inference.network.AuditAttemptSummary
import com.qlh.inference.network.AuditData
import com.qlh.inference.network.AuditStageSummary

const val MAX_AUDIT_WORKFLOWS = 8
const val MAX_AUDIT_STAGES = 8
const val MAX_AUDIT_ATTEMPTS = 4
const val MAX_AUDIT_REVIEWS = 8

/** Read-only mobile audit state. It contains bounded operational summaries only. */
data class AuditUiState(
    val snapshot: AuditSnapshot? = null,
    val loading: Boolean = false,
    val error: String? = null,
)

data class AuditSnapshot(
    val enabled: Boolean = false,
    val available: Boolean = false,
    val role: String = "",
    val workflows: List<AuditWorkflow> = emptyList(),
    val reviews: List<AuditReviewTicket> = emptyList(),
)

data class AuditWorkflow(
    val workflowId: String,
    val template: String,
    val state: String,
    val createdAt: Double,
    val finishedAt: Double,
    val stageCount: Int,
    val completedStageCount: Int,
    val failedStageCount: Int,
    val attemptCount: Int,
    val retryCount: Int,
    val resultRejectionCount: Int,
    val recoveredAfterRestart: Boolean,
    val stages: List<AuditStage> = emptyList(),
)

data class AuditStage(
    val stageId: String,
    val stageType: String,
    val state: String,
    val retryCount: Int,
    val resultRejectionCount: Int,
    val errorCode: String,
    val attempts: List<AuditAttempt> = emptyList(),
)

data class AuditAttempt(
    val attemptId: String,
    val providerKind: String,
    val providerNodeId: String,
    val state: String,
    val errorCode: String,
)

data class AuditReviewTicket(
    val ticketId: String,
    val status: String,
    val createdAt: Double,
    val targetNodeId: String,
    val score: Int,
    val voteCount: Int,
)

fun toAuditSnapshot(data: AuditData): AuditSnapshot {
    val workflows = data.workflows.workflows
        .asSequence()
        .filter { it.workflowId.isNotBlank() }
        .take(MAX_AUDIT_WORKFLOWS)
        .map { workflow ->
            AuditWorkflow(
                workflowId = workflow.workflowId.trim(),
                template = workflow.template.trim().ifBlank { "unknown" },
                state = workflow.state.trim().ifBlank { "unknown" },
                createdAt = safeAuditTimestamp(workflow.createdAt),
                finishedAt = safeAuditTimestamp(workflow.finishedAt),
                stageCount = workflow.stageCount.coerceAtLeast(0),
                completedStageCount = workflow.completedStageCount.coerceAtLeast(0),
                failedStageCount = workflow.failedStageCount.coerceAtLeast(0),
                attemptCount = workflow.attemptCount.coerceAtLeast(0),
                retryCount = workflow.retryCount.coerceAtLeast(0),
                resultRejectionCount = workflow.resultRejectionCount.coerceAtLeast(0),
                recoveredAfterRestart = workflow.observability.recoveredAfterRestart,
                stages = workflow.stages
                    .asSequence()
                    .take(MAX_AUDIT_STAGES)
                    .map(::toAuditStage)
                    .toList(),
            )
        }
        .sortedByDescending { it.createdAt }
        .toList()

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

    return AuditSnapshot(
        enabled = data.workflows.enabled,
        available = data.workflows.available,
        role = data.workflows.role.trim(),
        workflows = workflows,
        reviews = reviews,
    )
}

private fun toAuditStage(stage: AuditStageSummary): AuditStage = AuditStage(
    stageId = stage.stageId.trim().ifBlank { "unknown" },
    stageType = stage.stageType.trim().ifBlank { "unknown" },
    state = stage.state.trim().ifBlank { "unknown" },
    retryCount = stage.retryCount.coerceAtLeast(0),
    resultRejectionCount = stage.resultRejectionCount.coerceAtLeast(0),
    errorCode = stage.errorCode.trim().take(64),
    attempts = stage.attempts
        .asSequence()
        .take(MAX_AUDIT_ATTEMPTS)
        .map(::toAuditAttempt)
        .toList(),
)

private fun toAuditAttempt(attempt: AuditAttemptSummary): AuditAttempt = AuditAttempt(
    attemptId = attempt.attemptId.trim().ifBlank { "unknown" },
    providerKind = attempt.providerKind.trim().ifBlank { "unknown" },
    providerNodeId = attempt.providerNodeId.trim().ifBlank { "unknown" },
    state = attempt.state.trim().ifBlank { "unknown" },
    errorCode = attempt.errorCode.trim().take(64),
)

private fun safeAuditTimestamp(value: Double): Double =
    value.takeIf { it.isFinite() && it >= 0.0 && it <= 1.0e15 } ?: 0.0
