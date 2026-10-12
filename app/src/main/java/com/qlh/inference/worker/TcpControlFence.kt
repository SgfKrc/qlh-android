package com.qlh.inference.worker

import com.google.gson.JsonObject

/** Shared admission/decorating boundary for every fenced outer TCP control frame. */
internal class TcpControlFence(
    var required: Boolean = false,
    private val validate: ((JsonObject) -> Unit)? = null,
) {
    private var certificate: JsonObject? = null
    @Synchronized fun admit(data: JsonObject) {
        val supplied = data["control_certificate"]
        if (supplied == null && !required && certificate == null) return
        if (supplied == null) throw TaskWorkerProtocolException("control certificate missing", "control_certificate_missing")
        if (!supplied.isJsonObject) throw TaskWorkerProtocolException("invalid control certificate", "control_certificate_invalid")
        val validator = validate ?: throw TaskWorkerProtocolException("Android has no configured voter authority", "control_fence_unavailable")
        validator(supplied.asJsonObject)
        certificate = supplied.asJsonObject.deepCopy()
        required = true
    }
    @Synchronized fun requireCurrent() {
        if (!required && certificate == null) return
        val current = certificate ?: throw TaskWorkerProtocolException("control certificate missing", "control_certificate_missing")
        val validator = validate ?: throw TaskWorkerProtocolException("Android has no configured voter authority", "control_fence_unavailable")
        validator(current)
    }
    @Synchronized fun decorate(data: JsonObject): JsonObject {
        if (!required && certificate == null) return data
        val current = certificate ?: throw TaskWorkerProtocolException("control certificate missing", "control_certificate_missing")
        val validator = validate ?: throw TaskWorkerProtocolException("Android has no configured voter authority", "control_fence_unavailable")
        validator(current)
        return data.deepCopy().apply { add("control_certificate", current.deepCopy()) }
    }
}
