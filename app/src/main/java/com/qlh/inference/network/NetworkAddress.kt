package com.qlh.inference.network

/** Keep hosts bare in settings and bracket IPv6 only when building URLs. */
fun canonicalHost(host: String): String {
    val value = host.trim()
    return if (value.startsWith("[") && value.endsWith("]")) {
        value.substring(1, value.length - 1)
    } else {
        value
    }
}

fun formatUrlHost(host: String): String {
    val value = canonicalHost(host)
    return if (value.contains(':')) {
        "[${value.replace("%", "%25")}]"
    } else {
        value
    }
}

fun httpBaseUrl(host: String, port: Int): String =
    "http://${formatUrlHost(host)}:$port"
