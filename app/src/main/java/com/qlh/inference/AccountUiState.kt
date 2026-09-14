package com.qlh.inference

import com.qlh.inference.network.AuthCapabilityResponse
import com.qlh.inference.network.AuthSessionResponse
import com.qlh.inference.security.StoredAuthSession

/** Public capability projection; it deliberately contains no token or secret material. */
data class AuthCapabilitySnapshot(
    val required: Boolean = false,
    val available: Boolean = false,
    val mode: String = "",
    val bootstrapAvailable: Boolean = false,
    val reasonCode: String = "",
) {
    val canAuthenticate: Boolean
        get() = available || required
}

/** Account projection used by Android settings. Session id and bearer are never exposed. */
data class AuthAccountSnapshot(
    val username: String,
    val displayName: String,
    val role: String,
    val expiresAt: String,
)

data class AuthControlUiState(
    val capability: AuthCapabilitySnapshot? = null,
    val account: AuthAccountSnapshot? = null,
    val loading: Boolean = false,
    val busy: Boolean = false,
    val localSessionPresent: Boolean = false,
    val error: String? = null,
)

fun AuthCapabilityResponse.toSnapshot(): AuthCapabilitySnapshot = AuthCapabilitySnapshot(
    required = required,
    // The gateway exposes `required/enforced` while the standalone API exposes
    // `available`; both describe a reachable auth control boundary.
    available = available || enforced || required,
    mode = mode.take(64),
    bootstrapAvailable = bootstrapAvailable,
    reasonCode = reasonCode.take(96),
)

fun StoredAuthSession.toAccountSnapshot(): AuthAccountSnapshot = AuthAccountSnapshot(
    username = username,
    displayName = displayName?.takeIf { it.isNotBlank() } ?: username,
    role = role,
    expiresAt = expiresAt,
)

fun AuthSessionResponse.toAccountSnapshot(): AuthAccountSnapshot = AuthAccountSnapshot(
    username = user.username,
    displayName = user.displayName?.takeIf { it.isNotBlank() } ?: user.username,
    role = user.role,
    expiresAt = expiresAt,
)
