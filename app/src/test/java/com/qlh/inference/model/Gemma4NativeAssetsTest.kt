package com.qlh.inference.model

import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class Gemma4NativeAssetsTest {
    @Test
    fun `exact native pair passes registration gate`() {
        val status = Gemma4NativeAssets.inspect(
            listOf(
                Gemma4AssetFile(
                    Gemma4NativeAssets.MAIN_FILENAME,
                    Gemma4NativeAssets.MAIN_SIZE_BYTES,
                ),
                Gemma4AssetFile(
                    Gemma4NativeAssets.MMPROJ_FILENAME,
                    Gemma4NativeAssets.MMPROJ_SIZE_BYTES,
                ),
            )
        )

        assertTrue(status.pairPresent)
        assertTrue(status.sizeVerified)
        assertTrue(status.reason.contains("SHA-256 verification deferred"))
    }

    @Test
    fun `missing mmproj fails closed`() {
        val status = Gemma4NativeAssets.inspect(
            listOf(
                Gemma4AssetFile(
                    Gemma4NativeAssets.MAIN_FILENAME,
                    Gemma4NativeAssets.MAIN_SIZE_BYTES,
                )
            )
        )

        assertTrue(status.mainPresent)
        assertFalse(status.mmprojPresent)
        assertFalse(status.pairPresent)
        assertFalse(status.sizeVerified)
        assertTrue(status.reason.contains("missing ${Gemma4NativeAssets.MMPROJ_FILENAME}"))
    }

    @Test
    fun `unknown or mismatched sizes fail closed`() {
        val unknown = Gemma4NativeAssets.inspect(
            listOf(
                Gemma4AssetFile(Gemma4NativeAssets.MAIN_FILENAME, -1L),
                Gemma4AssetFile(
                    Gemma4NativeAssets.MMPROJ_FILENAME,
                    Gemma4NativeAssets.MMPROJ_SIZE_BYTES,
                ),
            )
        )
        val mismatch = Gemma4NativeAssets.inspect(
            listOf(
                Gemma4AssetFile(Gemma4NativeAssets.MAIN_FILENAME, 1L),
                Gemma4AssetFile(
                    Gemma4NativeAssets.MMPROJ_FILENAME,
                    Gemma4NativeAssets.MMPROJ_SIZE_BYTES,
                ),
            )
        )

        assertFalse(unknown.sizeVerified)
        assertTrue(unknown.reason.contains("size unknown"))
        assertFalse(mismatch.sizeVerified)
        assertTrue(mismatch.reason.contains("size mismatch"))
    }
}
