package com.qlh.inference

import com.qlh.inference.network.AuthUser
import com.qlh.inference.network.ManageAuditEvent
import com.qlh.inference.network.ManageSummaryResponse
import com.qlh.inference.network.TailscaleBinding

const val MAX_MANAGED_USERS = 32
const val MAX_MANAGED_BINDINGS = 64
const val MAX_MANAGEMENT_AUDIT_EVENTS = 50

/**
 * Owner/admin-only mobile management projection. Confirmation tokens never enter
 * this state; the ViewModel keeps them in a single request and immediately discards
 * them after the server consumes them.
 */
data class ManagementUiState(
    val summary: ManageSummaryResponse? = null,
    val users: List<ManagedUserSnapshot> = emptyList(),
    val audit: List<ManageAuditEvent> = emptyList(),
    val loading: Boolean = false,
    val busyAction: String? = null,
    val error: String? = null,
)

data class ManagedUserSnapshot(
    val userId: String,
    val username: String,
    val displayName: String,
    val role: String,
    val status: String,
    val aggregateVersion: Int,
    val bindings: List<ManagedBindingSnapshot> = emptyList(),
)

data class ManagedBindingSnapshot(
    val bindingId: String,
    val userId: String,
    val tailnetId: String,
    val tailscaleUserId: String,
    val nodeId: String,
    val state: String,
)

fun AuthUser.toManagedUserSnapshot(bindings: List<ManagedBindingSnapshot> = emptyList()): ManagedUserSnapshot =
    ManagedUserSnapshot(
        userId = userId.trim(),
        username = username.trim(),
        displayName = displayName?.trim().orEmpty().ifBlank { username.trim() },
        role = role.trim().ifBlank { "unknown" },
        status = status?.trim().orEmpty().ifBlank { "unknown" },
        aggregateVersion = (aggregateVersion ?: 0).coerceAtLeast(0),
        bindings = bindings.take(MAX_MANAGED_BINDINGS),
    )

fun TailscaleBinding.toManagedBindingSnapshot(fallbackUserId: String): ManagedBindingSnapshot =
    ManagedBindingSnapshot(
        bindingId = bindingId.trim(),
        userId = userId?.trim().takeUnless { it.isNullOrBlank() } ?: fallbackUserId,
        tailnetId = tailnetId?.trim().orEmpty(),
        tailscaleUserId = tailscaleUserId?.trim().orEmpty(),
        nodeId = nodeId?.trim().orEmpty(),
        state = state?.trim().orEmpty().ifBlank { "unknown" },
    )

fun managementRoleAllowed(role: String?): Boolean =
    role?.trim()?.lowercase() in setOf("owner", "admin")
