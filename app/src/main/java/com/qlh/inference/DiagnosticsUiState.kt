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
    /**
     * ★ 2026-10-08：当前是否已豁免电池优化（`PowerManager.isIgnoringBatteryOptimizations`）。
     *
     * 三态：`true` 已豁免 / `false` 未豁免（厂商省电可能回收层段 worker）/ `null` 尚未探测。
     * 引导本身（弹一次系统对话框、落库「已问过」）在 `MainViewModel.buildBatteryExemptionRequest()`；
     * 这里只让**状态可见**，便于用户手动复查与再次申请。
     */
    val batteryOptimizationExempt: Boolean? = null,
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
