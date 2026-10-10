package com.qlh.inference.worker

import android.content.Intent
import androidx.test.platform.app.InstrumentationRegistry
import com.qlh.inference.data.SettingsDataStore
import kotlinx.coroutines.delay
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withTimeout
import org.junit.Assert.assertEquals
import org.junit.Test

class TaskWorkerServiceIntentInstrumentedTest {
    @Test
    fun startIntentCarriesTheNonSensitiveCredentialEpoch() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val intent = TaskWorkerService.startIntent(
            context = context,
            host = "100.90.76.108",
            port = 8888,
            nodeId = "android_test",
            clusterSecret = "transient-secret",
            clusterSecretEpoch = 7,
            hostname = "test-device",
            networkType = "wifi",
            deviceInfo = emptyMap(),
        )

        assertEquals(7, intent.getIntExtra(TaskWorkerService.EXTRA_CLUSTER_SECRET_EPOCH, 0))
        assertEquals(
            "transient-secret",
            intent.getStringExtra(TaskWorkerService.EXTRA_CLUSTER_SECRET),
        )
    }

    @Test
    fun startIntentDefaultsInvalidCredentialEpochToGenerationOne() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val intent = TaskWorkerService.startIntent(
            context = context,
            host = "100.90.76.108",
            port = 8888,
            nodeId = "android_test",
            clusterSecret = "transient-secret",
            clusterSecretEpoch = 0,
            hostname = "test-device",
            networkType = "wifi",
            deviceInfo = emptyMap(),
        )

        assertEquals(1, intent.getIntExtra(TaskWorkerService.EXTRA_CLUSTER_SECRET_EPOCH, 0))
    }

    @Test
    fun stopIntentTargetsTheWorkerStopAction() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val intent = TaskWorkerService.stopIntent(context)

        assertEquals(TaskWorkerService.ACTION_STOP, intent.action)
        assertEquals(TaskWorkerService::class.java.name, intent.component?.className)
    }

    @Test
    fun requestStopDeliversTheActionAndClearsPersistedStartupConfig() = runBlocking {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val settings = SettingsDataStore(context)
        settings.setTaskWorkerStartupConfig("{\"nodeId\":\"stale-worker\"}")

        try {
            TaskWorkerService.requestStop(context)
            withTimeout(5_000L) {
                while (settings.getTaskWorkerStartupConfig().isNotEmpty()) {
                    delay(50L)
                }
            }
            assertEquals("", settings.getTaskWorkerStartupConfig())
        } finally {
            context.stopService(Intent(context, TaskWorkerService::class.java))
            settings.clearTaskWorkerStartupConfig()
        }
    }
}
