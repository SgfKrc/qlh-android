package com.qlh.inference.logging

private val bearerPattern = Regex("(?i)\\bBearer\\s+[A-Za-z0-9._~+/=-]{8,}")
private val secretFieldPattern = Regex(
    "(?i)(authorization|access[_-]?token|refresh[_-]?token|password|credential|recovery[_-]?code)([\\\"']?\\s*[=:]\\s*[\\\"']?)[^,\\s\\\"'}]+",
)

internal fun redactDiagnosticText(value: String): String = value
    .replace(bearerPattern, "Bearer [REDACTED]")
    .replace(secretFieldPattern, "\$1\$2[REDACTED]")
