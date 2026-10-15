package com.qlh.inference.network

import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.net.InetSocketAddress
import java.net.Socket

/** TCP reachability and master identity are independent checks; neither registers a worker. */
enum class ControlPlaneProbeState { VERIFIED, IDENTITY_ERROR, NETWORK_ERROR, BOOTSTRAP_UNAVAILABLE, PROTOCOL_ERROR }

data class ControlPlaneProbeResult(
    val state: ControlPlaneProbeState,
    val code: String,
    val tcpHost: String,
    val tcpPort: Int,
) {
    val controlReachable: Boolean get() = state != ControlPlaneProbeState.NETWORK_ERROR
    val message: String get() = when (state) {
        ControlPlaneProbeState.VERIFIED -> "控制面可达，主节点身份已确认"
        ControlPlaneProbeState.IDENTITY_ERROR -> "控制面可达，但主节点身份未确认"
        ControlPlaneProbeState.NETWORK_ERROR -> "控制面网络连接失败"
        ControlPlaneProbeState.BOOTSTRAP_UNAVAILABLE -> "控制面可达，但引导 API 不可达，身份尚未验证"
        ControlPlaneProbeState.PROTOCOL_ERROR -> "控制面可达，但引导 API 响应无效"
    }
}

suspend fun probeControlPlaneConnection(
    tcpHost: String,
    tcpPort: Int,
    bootstrapInfo: suspend () -> Result<MasterBootstrapInfo>,
    connectTimeoutMs: Int = 3_000,
): ControlPlaneProbeResult = withContext(Dispatchers.IO) {
    fun result(state: ControlPlaneProbeState, code: String) = ControlPlaneProbeResult(state, code, tcpHost, tcpPort)
    try {
        require(tcpHost.isNotBlank() && tcpPort in 1..65535)
        Socket().use { it.connect(InetSocketAddress(canonicalHost(tcpHost), tcpPort), connectTimeoutMs) }
    } catch (cancelled: CancellationException) {
        throw cancelled
    } catch (_: Exception) {
        return@withContext result(ControlPlaneProbeState.NETWORK_ERROR, "control_plane_unreachable")
    }
    val info = try {
        bootstrapInfo().getOrThrow()
    } catch (cancelled: CancellationException) {
        throw cancelled
    } catch (error: Exception) {
        return@withContext when {
            error is ApiClientHttpException && error.statusCode in setOf(401, 403) ->
                result(ControlPlaneProbeState.IDENTITY_ERROR, error.errorCode.ifBlank { "bootstrap_identity_rejected" })
            error is IllegalArgumentException || error.message == "intent_response_invalid" ->
                result(ControlPlaneProbeState.PROTOCOL_ERROR, "bootstrap_info_invalid")
            else -> result(ControlPlaneProbeState.BOOTSTRAP_UNAVAILABLE, "bootstrap_api_unreachable")
        }
    }
    if (info.isMaster) result(ControlPlaneProbeState.VERIFIED, "control_plane_reachable")
    else result(ControlPlaneProbeState.IDENTITY_ERROR, "master_identity_unconfirmed")
}
