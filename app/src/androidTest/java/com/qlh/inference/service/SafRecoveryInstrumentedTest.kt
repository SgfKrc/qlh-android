package com.qlh.inference.service

import android.net.Uri
import android.os.Bundle
import androidx.test.platform.app.InstrumentationRegistry
import com.google.gson.JsonObject
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File
import java.security.MessageDigest
import java.util.UUID

/** Real OS permission refusal followed by the existing authorized directory scan. */
class SafRecoveryInstrumentedTest {
    @Test
    fun deniedDirectorySelectionPreservesSavedGrantAndNextScan() = runBlocking(Dispatchers.IO) {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val context = instrumentation.targetContext
        val manager = ModelManager(context)
        fun grants() = context.contentResolver.persistedUriPermissions.map {
            "${it.uri}|${it.isReadPermission}|${it.isWritePermission}|${it.persistedTime}"
        }.sorted()
        fun digest(file: File) = MessageDigest.getInstance("SHA-256")
            .digest(file.readBytes()).joinToString("") { "%02x".format(it.toInt() and 255) }
        val settings = File(context.filesDir, "datastore/qlh_settings.preferences_pb")
        assertTrue("Existing saved user settings required", settings.isFile)
        assertTrue("Existing saved SAF read grant required", manager.modelDirectoryGrantState().canScan)
        val settingsBefore = digest(settings)
        val grantsBefore = grants()
        val denied = Uri.parse("content://com.android.externalstorage.documents/tree/primary%3Aqlh-r6-denied-${UUID.randomUUID()}")
        val deniedState = manager.modelDirectoryGrantState(denied.toString())
        assertTrue(deniedState.selected)
        assertFalse(deniedState.canScan)
        assertFalse(deniedState.canDownload)
        val rejected = manager.selectModelDirectory(denied)
        assertTrue("Unissued OS grant must be refused", rejected.isFailure)
        assertTrue("Actual OS permission failure required", rejected.exceptionOrNull() is SecurityException)
        assertEquals("Saved settings changed on permission refusal", settingsBefore, digest(settings))
        assertEquals("Persisted grants changed on refusal", grantsBefore, grants())
        assertTrue("Saved directory must remain recoverable", manager.modelDirectoryGrantState().canScan)
        val models = manager.listModels().getOrThrow()
        val source = requireNotNull(models.firstOrNull {
            it.uri.scheme == "content" && it.name.endsWith(".gguf", true) && it.sizeBytes > 1_000_000_000L
        }) { "A real authorized large GGUF source is required" }
        val header = requireNotNull(context.contentResolver.openInputStream(source.uri)).use {
            it.readNBytes(4)
        }
        assertEquals("GGUF", header.toString(Charsets.US_ASCII))
        assertEquals("Saved settings changed during recovery", settingsBefore, digest(settings))
        assertEquals("Persisted grants changed during recovery", grantsBefore, grants())
        val report = JsonObject().apply {
            addProperty("kind", "r6.saf_permission_recovery.v1")
            addProperty("status", "passed")
            addProperty("failure_type", rejected.exceptionOrNull()!!.javaClass.name)
            addProperty("saved_settings_sha256", settingsBefore)
            addProperty("existing_grants_count", grantsBefore.size)
            addProperty("recovered_model_size_bytes", source.sizeBytes)
            addProperty("recovered_gguf_header", true)
            addProperty("settings_unchanged", true)
            addProperty("persisted_grants_unchanged", true)
        }
        val output = File(context.cacheDir, "r6-saf-${UUID.randomUUID()}.json")
        output.writeText(report.toString(), Charsets.UTF_8)
        instrumentation.sendStatus(2, Bundle().apply { putString("stream", "R6_SAF_RESULT $report\n") })
        Unit
    }
}
