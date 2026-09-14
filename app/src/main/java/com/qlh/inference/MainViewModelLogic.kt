package com.qlh.inference

import com.qlh.inference.network.ApiClientHttpException
import com.qlh.inference.status.AndroidRuntimeStatus
import com.qlh.inference.status.GpuStatus
import com.qlh.inference.status.MemoryStatus
import com.qlh.inference.status.SystemStatus

/**
 * MainViewModel 的纯逻辑（不访问 Android API，JVM 可单测）。
 */

/** 由 ConnectivityManager transport 标志判定网络类型（顺序敏感：wifi 优先）。 */
fun networkTypeFromTransports(
    hasWifi: Boolean,
    hasCellular: Boolean,
    hasEthernet: Boolean,
    hasVpn: Boolean,
): String = when {
    hasWifi -> "wifi"
    hasCellular -> "mobile"
    hasEthernet -> "ethernet"
    hasVpn -> "vpn"
    else -> "other"
}

// ---- 聊天与会话 UI 状态机（不访问 Room、DataStore 或网络，JVM 可单测） ----

fun selectUiTab(state: MainUiState, tab: String): MainUiState =
    state.copy(currentTab = tab)

fun selectUiSession(
    state: MainUiState,
    sessionId: Long,
    sessionTitle: String,
): MainUiState = state.copy(
    currentSessionId = sessionId,
    currentSessionTitle = sessionTitle,
    currentTab = "chat",
)

const val MAX_CHAT_IMAGE_DATA_URLS = 4
const val MAX_CHAT_IMAGE_BYTES = 8 * 1024 * 1024
const val MAX_CHAT_IMAGE_TOTAL_BYTES = 16 * 1024 * 1024

private val CHAT_IMAGE_DATA_URL = Regex(
    "^data:image/(png|jpeg|webp);base64,([A-Za-z0-9+/=]+)$",
    RegexOption.IGNORE_CASE,
)

/** Normalize client-provided image payloads before they reach the HTTP contract. */
fun normalizeChatImageDataUrls(imageDataUrls: List<String>): List<String> =
    imageDataUrls
        .asSequence()
        .map(String::trim)
        .filter(String::isNotEmpty)
        .take(MAX_CHAT_IMAGE_DATA_URLS)
        .toList()

/**
 * Image payload validation is independent of the runtime capability gate. Full mode
 * may proceed to the InferenceService mmproj gate; thin mode sends the same contract
 * to the PC coordinator.
 */
fun validateChatImageSubmission(inferenceMode: String, imageDataUrls: List<String>): String? {
    val normalized = normalizeChatImageDataUrls(imageDataUrls)
    if (imageDataUrls.count { it.trim().isNotEmpty() } > MAX_CHAT_IMAGE_DATA_URLS) {
        return "一次最多附加 $MAX_CHAT_IMAGE_DATA_URLS 张图片"
    }
    val maxPayloadChars = ((MAX_CHAT_IMAGE_BYTES + 2) / 3) * 4
    var totalBytes = 0
    for (dataUrl in normalized) {
        val match = CHAT_IMAGE_DATA_URL.matchEntire(dataUrl)
            ?: return "图像仅支持 PNG/JPEG/WebP base64 data URL"
        val payload = match.groupValues[2]
        if (payload.length > maxPayloadChars) {
            return "单张图片不得超过 8 MiB"
        }
        val decoded = try {
            java.util.Base64.getDecoder().decode(payload)
        } catch (_: IllegalArgumentException) {
            return "图像 data URL 的 base64 数据无效"
        }
        if (decoded.isEmpty() || decoded.size > MAX_CHAT_IMAGE_BYTES) {
            return "单张图片必须为 1 byte 至 8 MiB"
        }
        val kind = match.groupValues[1].lowercase()
        val signatureValid = when (kind) {
            "png" -> decoded.size >= 8 && decoded.copyOfRange(0, 8).contentEquals(
                byteArrayOf(0x89.toByte(), 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a)
            )
            "jpeg" -> decoded.size >= 3 && decoded[0] == 0xff.toByte() &&
                decoded[1] == 0xd8.toByte() && decoded[2] == 0xff.toByte()
            "webp" -> decoded.size >= 12 &&
                decoded.copyOfRange(0, 4).contentEquals(byteArrayOf(0x52, 0x49, 0x46, 0x46)) &&
                decoded.copyOfRange(8, 12).contentEquals(byteArrayOf(0x57, 0x45, 0x42, 0x50))
            else -> false
        }
        if (!signatureValid) {
            return "图像 MIME 类型与文件签名不一致"
        }
        totalBytes += decoded.size
        if (totalBytes > MAX_CHAT_IMAGE_TOTAL_BYTES) {
            return "图像总大小不得超过 16 MiB"
        }
    }
    if (normalized.isNotEmpty() && inferenceMode !in setOf("thin", "full")) {
        return "当前模式不支持图像理解"
    }
    return null
}

fun startMessageSubmission(
    state: MainUiState,
    message: String,
    imageDataUrls: List<String> = emptyList(),
): MainUiState = state.copy(
    isLoading = true,
    error = null,
    lastSentMessage = message,
    lastSentImageDataUrls = normalizeChatImageDataUrls(imageDataUrls),
)

fun completeMessageSubmission(state: MainUiState): MainUiState =
    state.copy(isLoading = false, error = null)

fun failMessageSubmission(state: MainUiState, error: String): MainUiState =
    state.copy(isLoading = false, error = error)

fun startMessageRetry(state: MainUiState): MainUiState =
    if (state.lastSentMessage == null) state else state.copy(isLoading = true, error = null)

fun clearMessageError(state: MainUiState): MainUiState =
    state.copy(error = null)

fun formatMessageSendError(error: Throwable): String = when (error) {
    is ApiClientHttpException -> when (error.statusCode) {
        401, 403 -> "远程推理需要先在账户页完成 Auth App 登录"
        503 -> "主节点暂时拒绝远程推理，请检查分布式节点连接与模型准入"
        else -> "发送失败（HTTP ${error.statusCode}）"
    }
    is java.net.ConnectException -> "无法连接主节点，请检查地址和网络"
    is java.net.SocketTimeoutException -> "连接超时，请检查主节点是否运行"
    is UnsupportedOperationException -> error.message ?: "当前模式暂不支持"
    else -> "发送失败: ${error.message ?: error.javaClass.simpleName}"
}

/** Cluster status is diagnostic-only; do not echo endpoints, response bodies, or credentials into the UI. */
fun formatClusterOverviewError(error: Throwable): String = when (error) {
    is java.net.ConnectException -> "无法连接主节点"
    is java.net.SocketTimeoutException -> "读取主节点状态超时"
    else -> "读取集群状态失败"
}

/** Model inventory errors must not disclose the endpoint, response body, or credentials. */
fun formatModelFleetError(error: Throwable): String = when (error) {
    is java.net.ConnectException -> "无法连接主节点"
    is java.net.SocketTimeoutException -> "读取模型舰队超时"
    else -> "读取模型舰队失败"
}

/** Audit summaries are diagnostic-only and must not echo response bodies or credentials. */
fun formatAuditError(error: Throwable): String = when (error) {
    is java.net.ConnectException -> "无法连接主节点"
    is java.net.SocketTimeoutException -> "读取审计资料超时"
    else -> "读取审计资料失败"
}

/** Management errors are intentionally generic; never echo response bodies or IDs. */
fun formatManagementError(error: Throwable): String = when (error) {
    is java.net.ConnectException -> "无法连接主节点"
    is java.net.SocketTimeoutException -> "读取管理控制面超时"
    else -> "管理控制面不可用或权限不足"
}

/** Authentication failures are intentionally coarse; never echo response bodies or secrets. */
fun formatAuthError(error: Throwable): String = when (error) {
    is java.net.ConnectException -> "无法连接认证控制面"
    is java.net.SocketTimeoutException -> "认证控制面响应超时"
    else -> when {
        error.message?.contains("HTTP 401") == true || error.message?.contains("HTTP 403") == true ->
            "认证会话无效或已过期"
        else -> "认证服务不可用"
    }
}

/** Android 节点 presence 设备信息 payload（纯组装；不包含模型绝对路径/密钥）。 */
fun buildAndroidPresencePayload(
    inferenceMode: String,
    appVariant: String,
    appVersion: String,
    system: SystemStatus,
    memory: MemoryStatus,
    gpu: GpuStatus,
    runtime: AndroidRuntimeStatus?,
): Map<String, Any?> = mapOf(
    "connection_type" to "http_thin",
    "pipeline_worker" to false,
    "client_mode" to inferenceMode,
    "app_variant" to appVariant,
    "app_version" to appVersion,
    "android" to mapOf(
        "manufacturer" to system.manufacturer,
        "brand" to system.brand,
        "model" to system.model,
        "device" to system.device,
        "hardware" to system.hardware,
        "soc_manufacturer" to system.socManufacturer,
        "soc_model" to system.socModel,
        "sdk_int" to system.sdkInt,
        "android_release" to system.androidRelease,
        "abis" to system.abis,
        "cpu_cores" to system.cpuCores,
        "thermal_status" to system.thermalStatus,
    ),
    "memory" to mapOf(
        "available_bytes" to memory.availableBytes,
        "total_bytes" to memory.totalBytes,
        "low_memory" to memory.lowMemory,
        "low_ram_device" to memory.lowRamDevice,
    ),
    "gpu" to mapOf(
        "vendor" to gpu.vendor,
        "renderer" to gpu.renderer,
        "version" to gpu.version,
        "probe_error" to gpu.probeError,
        "supports_gpu_offload" to (runtime?.gpu?.supportsGpuOffload ?: false),
        "backend_devices" to (runtime?.gpu?.backendDevices ?: ""),
        "note" to gpu.note,
    ),
    "backend" to mapOf(
        "engine" to (runtime?.backend?.engine ?: ""),
        "supports_gpu_offload" to (runtime?.backend?.supportsGpuOffload ?: false),
    ),
    "multimodal" to mapOf(
        "vision_supported" to (runtime?.multimodal?.visionSupported ?: false),
        "assets_present" to (runtime?.multimodal?.assetsPresent ?: false),
        "asset_sizes_verified" to (runtime?.multimodal?.assetSizesVerified ?: false),
        "ready" to (runtime?.multimodal?.ready ?: false),
        "reason" to (runtime?.multimodal?.reason ?: ""),
    ),
)
