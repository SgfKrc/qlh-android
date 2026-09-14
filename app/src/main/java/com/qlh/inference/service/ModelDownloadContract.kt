package com.qlh.inference.service

data class DownloadWritePlan(
    val startBytes: Long,
    val append: Boolean,
    val totalBytes: Long,
)

private val CONTENT_RANGE = Regex("^bytes (\\d+)-(\\d+)/(\\d+)$", RegexOption.IGNORE_CASE)

/** Decide whether a model response may append to a persistent partial file. */
fun planModelDownloadWrite(
    existingBytes: Long,
    statusCode: Int,
    contentLength: Long,
    contentRange: String?,
    expectedTotalBytes: Long,
): DownloadWritePlan {
    require(existingBytes >= 0L) { "existingBytes must be non-negative" }
    require(expectedTotalBytes > 0L) { "expectedTotalBytes must be positive" }
    return when (statusCode) {
        200 -> {
            if (contentLength > 0L) {
                require(contentLength == expectedTotalBytes) {
                    "full response size does not match the model manifest"
                }
            }
            DownloadWritePlan(0L, append = false, totalBytes = expectedTotalBytes)
        }

        206 -> {
            val match = contentRange?.let(CONTENT_RANGE::matchEntire)
                ?: throw IllegalArgumentException("partial response has no valid Content-Range")
            val start = match.groupValues[1].toLong()
            val end = match.groupValues[2].toLong()
            val total = match.groupValues[3].toLong()
            require(start == existingBytes) { "partial response starts at the wrong offset" }
            require(end >= start && end < total && total == expectedTotalBytes) {
                "partial response does not match the model manifest"
            }
            if (contentLength > 0L) {
                require(contentLength == end - start + 1L) {
                    "partial response Content-Length is inconsistent"
                }
            }
            DownloadWritePlan(start, append = start > 0L, totalBytes = total)
        }

        else -> throw IllegalArgumentException("model download requires HTTP 200 or 206")
    }
}
