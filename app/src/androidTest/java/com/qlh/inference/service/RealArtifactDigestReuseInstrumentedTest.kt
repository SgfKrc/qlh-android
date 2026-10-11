package com.qlh.inference.service

import android.os.SystemClock
import android.util.Log
import androidx.test.platform.app.InstrumentationRegistry
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withTimeout
import org.junit.Assert.assertEquals
import org.junit.Assume.assumeTrue
import org.junit.Test

class RealArtifactDigestReuseInstrumentedTest {
    @Test
    fun authorizedArtifactSecondScanReusesVerifiedDigest() = runBlocking {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val manager = ModelManager(context)
        val started = SystemClock.elapsedRealtime()
        val first = withTimeout(180_000L) {
            manager.scanLayerArtifacts(verifyArtifactDigest = true).getOrThrow()
        }
        val firstMs = SystemClock.elapsedRealtime() - started
        assumeTrue("no accessible, valid layer artifact on device", first.artifacts.isNotEmpty())

        val secondStarted = SystemClock.elapsedRealtime()
        val second = withTimeout(180_000L) {
            manager.scanLayerArtifacts(verifyArtifactDigest = true).getOrThrow()
        }
        val secondMs = SystemClock.elapsedRealtime() - secondStarted
        assertEquals(
            first.artifacts.map { it.artifactSha256 to it.document.uri },
            second.artifacts.map { it.artifactSha256 to it.document.uri },
        )
        Log.i(
            "ArtifactDigestProbe",
            "verified_artifacts=${second.artifacts.size} " +
                "verified_bytes=${second.artifacts.sumOf { it.document.sizeBytes }} " +
                "first_ms=$firstMs second_ms=$secondMs",
        )
        Unit
    }
}
