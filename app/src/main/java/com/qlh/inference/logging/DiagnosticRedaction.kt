package com.qlh.inference.logging

private val bearerPattern = Regex("(?i)\\bBearer\\s+[A-Za-z0-9._~+/=-]{8,}")
private val secretFieldPattern = Regex(
    "(?i)(authorization|access[_-]?token|refresh[_-]?token|password|credential|cluster[_-]?secret|private[_-]?key|join(?:[_-]?(?:token|credential|grant))?|confirm[_-]?token|recovery[_-]?code|secret)([\\\"']?\\s*[=:]\\s*[\\\"']?)[^,\\s\\\"'}]+",
)
private val pemPrivateKeyPattern = Regex(
    "-----BEGIN(?: [A-Z0-9]+)? PRIVATE KEY-----[\\s\\S]*?-----END(?: [A-Z0-9]+)? PRIVATE KEY-----",
    RegexOption.IGNORE_CASE,
)

internal fun redactDiagnosticText(value: String): String = value
    .replace(bearerPattern, "Bearer [REDACTED]")
    .replace(secretFieldPattern, "\$1\$2[REDACTED]")
    .replace(pemPrivateKeyPattern, "[REDACTED PRIVATE KEY]")
