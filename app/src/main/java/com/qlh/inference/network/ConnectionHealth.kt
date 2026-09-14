package com.qlh.inference.network

import com.google.gson.annotations.SerializedName
enum class ConnectionHealthState {
    PASS,
    FAIL,
    SKIPPED,
}

data class ConnectionHealthCheck(
    val id: String,
    val label: String,
    val state: ConnectionHealthState,
    val latencyMillis: Long? = null,
    val detail: String = "",
)

data class ConnectionHealthReport(
    val checks: List<ConnectionHealthCheck>,
    val generatedAtMillis: Long = System.currentTimeMillis(),
    val localNetworkType: String = "unknown",
)

data class ClientErrorReport(
    val message: String,
    val source: String,
    val stack: String = "",
    val url: String = "",
    val line: Int = 0,
    val col: Int = 0,
    @SerializedName("user_agent") val userAgent: String = "",
    val extra: Map<String, Any?> = emptyMap(),
)
