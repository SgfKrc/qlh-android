package com.qlh.inference

import com.qlh.inference.network.ConnectionHealthReport
import com.qlh.inference.update.AndroidUpdateCandidate
import com.qlh.inference.update.UpdateDownloadProgress

/** State shown by the settings maintenance controls. No credential is kept here. */
data class DiagnosticsUiState(
    val health: ConnectionHealthReport? = null,
    val healthLoading: Boolean = false,
    val healthError: String? = null,
    val uploadInProgress: Boolean = false,
    val uploadMessage: String? = null,
    val uploadError: String? = null,
)

data class AppUpdateUiState(
    val checking: Boolean = false,
    val downloading: Boolean = false,
    val candidate: AndroidUpdateCandidate? = null,
    val progress: UpdateDownloadProgress? = null,
    val downloadedReady: Boolean = false,
    val installPermissionGranted: Boolean = false,
    val message: String? = null,
    val error: String? = null,
)
