package com.qlh.inference.media

import android.content.ContentResolver
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.util.Base64
import java.io.ByteArrayOutputStream

/** Bounded Android-side image normalization for the remote multimodal contract. */
data class EncodedImageAttachment(
    val dataUrl: String,
    val previewBytes: ByteArray,
    val width: Int,
    val height: Int,
    val byteCount: Int,
)

object ImageAttachmentEncoder {
    const val MAX_SOURCE_BYTES = 32 * 1024 * 1024
    const val MAX_OUTPUT_BYTES = 8 * 1024 * 1024
    const val MAX_DIMENSION = 2048

    fun encode(resolver: ContentResolver, uri: Uri): Result<EncodedImageAttachment> =
        runCatching {
            val source = readBounded(resolver, uri)
            val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
            BitmapFactory.decodeByteArray(source, 0, source.size, bounds)
            require(bounds.outWidth > 0 && bounds.outHeight > 0) {
                "无法读取图片尺寸或文件格式不受支持"
            }

            val options = BitmapFactory.Options().apply {
                inSampleSize = calculateSampleSize(bounds.outWidth, bounds.outHeight)
                inPreferredConfig = Bitmap.Config.ARGB_8888
            }
            val bitmap = BitmapFactory.decodeByteArray(source, 0, source.size, options)
                ?: error("图片解码失败")
            try {
                val normalized = compressWithinLimit(bitmap)
                val dataUrl = "data:image/jpeg;base64," +
                    Base64.encodeToString(normalized, Base64.NO_WRAP)
                EncodedImageAttachment(
                    dataUrl = dataUrl,
                    previewBytes = normalized,
                    width = bitmap.width,
                    height = bitmap.height,
                    byteCount = normalized.size,
                )
            } finally {
                bitmap.recycle()
            }
        }

    private fun readBounded(resolver: ContentResolver, uri: Uri): ByteArray {
        val output = ByteArrayOutputStream()
        resolver.openInputStream(uri)?.use { input ->
            val buffer = ByteArray(DEFAULT_BUFFER_SIZE)
            var total = 0
            while (true) {
                val count = input.read(buffer)
                if (count < 0) break
                total += count
                require(total <= MAX_SOURCE_BYTES) {
                    "源图片不得超过 ${MAX_SOURCE_BYTES / (1024 * 1024)} MiB"
                }
                output.write(buffer, 0, count)
            }
        } ?: error("无法读取所选图片")
        require(output.size() > 0) { "所选图片为空" }
        return output.toByteArray()
    }

    private fun calculateSampleSize(width: Int, height: Int): Int {
        var sample = 1
        while (maxOf(width, height) / sample > MAX_DIMENSION) {
            sample *= 2
        }
        return sample
    }

    private fun compressWithinLimit(bitmap: Bitmap): ByteArray {
        var current = bitmap
        var quality = 88
        var ownedScaledBitmap: Bitmap? = null
        try {
            repeat(5) {
                val output = ByteArrayOutputStream()
                check(current.compress(Bitmap.CompressFormat.JPEG, quality, output)) {
                    "图片压缩失败"
                }
                val bytes = output.toByteArray()
                if (bytes.size <= MAX_OUTPUT_BYTES) return bytes

                if (quality > 56) {
                    quality -= 12
                } else {
                    val nextWidth = (current.width * 0.75f).toInt().coerceAtLeast(1)
                    val nextHeight = (current.height * 0.75f).toInt().coerceAtLeast(1)
                    val scaled = Bitmap.createScaledBitmap(current, nextWidth, nextHeight, true)
                    if (current !== bitmap) current.recycle()
                    ownedScaledBitmap = scaled
                    current = scaled
                    quality = 82
                }
            }
            error("压缩后的图片仍超过 ${MAX_OUTPUT_BYTES / (1024 * 1024)} MiB")
        } finally {
            if (ownedScaledBitmap != null && !ownedScaledBitmap.isRecycled) {
                ownedScaledBitmap.recycle()
            }
        }
    }
}
