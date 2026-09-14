package com.qlh.inference.model

/** A model file discovered from SAF or the app's internal model directory. */
data class Gemma4AssetFile(
    val name: String,
    val sizeBytes: Long,
)

/**
 * Registration status for the fixed Gemma4 native MTMD pair.
 *
 * Size checks are an early identity gate. Full SHA-256 verification belongs to the
 * download/import pipeline and is deliberately not inferred from a filename.
 */
data class Gemma4NativeAssetStatus(
    val mainPresent: Boolean = false,
    val mmprojPresent: Boolean = false,
    val mainSizeVerified: Boolean = false,
    val mmprojSizeVerified: Boolean = false,
    val pairPresent: Boolean = false,
    val sizeVerified: Boolean = false,
    val reason: String = "missing Gemma4 native assets",
)

object Gemma4NativeAssets {
    const val MAIN_FILENAME = "gemma-4-12B-it-Q4_K_M.gguf"
    const val MMPROJ_FILENAME = "mmproj-gemma-4-12B-it-bf16.gguf"

    const val MAIN_SIZE_BYTES = 7_662_533_088L
    const val MMPROJ_SIZE_BYTES = 175_115_712L

    const val MAIN_SHA256 = "3962624dcd25b947d889dc9ae1bf275b61db6cd4dbe694057f34fffef1671509"
    const val MMPROJ_SHA256 = "92de172d87a262e4873a2a1d909b1b6082a76909957648705f00cb9feaa16535"

    fun inspect(files: Collection<Gemma4AssetFile>): Gemma4NativeAssetStatus {
        val main = files.firstOrNull { it.name.equals(MAIN_FILENAME, ignoreCase = true) }
        val mmproj = files.firstOrNull { it.name.equals(MMPROJ_FILENAME, ignoreCase = true) }
        val mainPresent = main != null
        val mmprojPresent = mmproj != null
        val mainSizeVerified = main?.sizeBytes == MAIN_SIZE_BYTES
        val mmprojSizeVerified = mmproj?.sizeBytes == MMPROJ_SIZE_BYTES
        val pairPresent = mainPresent && mmprojPresent
        val sizeVerified = mainSizeVerified && mmprojSizeVerified

        val problems = buildList {
            if (!mainPresent) add("missing $MAIN_FILENAME")
            if (!mmprojPresent) add("missing $MMPROJ_FILENAME")
            if (mainPresent && main?.sizeBytes == -1L) add("main asset size unknown")
            else if (mainPresent && !mainSizeVerified) add("main asset size mismatch")
            if (mmprojPresent && mmproj?.sizeBytes == -1L) add("mmproj asset size unknown")
            else if (mmprojPresent && !mmprojSizeVerified) add("mmproj asset size mismatch")
        }
        val reason = if (problems.isEmpty()) {
            "paired assets present; SHA-256 verification deferred"
        } else {
            problems.joinToString("; ")
        }
        return Gemma4NativeAssetStatus(
            mainPresent = mainPresent,
            mmprojPresent = mmprojPresent,
            mainSizeVerified = mainSizeVerified,
            mmprojSizeVerified = mmprojSizeVerified,
            pairPresent = pairPresent,
            sizeVerified = sizeVerified,
            reason = reason,
        )
    }
}
