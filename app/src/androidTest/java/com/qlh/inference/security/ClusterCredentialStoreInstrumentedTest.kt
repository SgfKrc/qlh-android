package com.qlh.inference.security

import android.content.Context
import androidx.datastore.preferences.core.PreferenceDataStoreFactory
import androidx.datastore.preferences.core.edit
import androidx.test.platform.app.InstrumentationRegistry
import com.qlh.inference.data.SettingsDataStore
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.runBlocking
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import java.io.File
import java.util.UUID

class ClusterCredentialStoreInstrumentedTest {
    private val context: Context
        get() = InstrumentationRegistry.getInstrumentation().targetContext

    private val credentialStore: ClusterCredentialStore
        get() = ClusterCredentialStore(context)

    @Before
    fun clearBefore() {
        credentialStore.clear()
    }

    @After
    fun clearAfter() {
        credentialStore.clear()
    }

    @Test
    fun storesOnlyCiphertextAndIvAndFailsClosedOnTampering() {
        val secret = "cluster-secret-${"a".repeat(40)}"
        credentialStore.save(secret)

        assertEquals(secret, credentialStore.read())
        val preferences = context.getSharedPreferences(
            ClusterCredentialStore.STORE_NAME,
            Context.MODE_PRIVATE,
        )
        assertEquals(
            setOf(ClusterCredentialStore.KEY_CIPHERTEXT, ClusterCredentialStore.KEY_IV),
            preferences.all.keys,
        )
        assertFalse(preferences.all.values.joinToString().contains(secret))

        preferences.edit()
            .putString(ClusterCredentialStore.KEY_CIPHERTEXT, "tampered")
            .commit()
        assertNull(credentialStore.read())
        assertTrue(preferences.all.isEmpty())
    }

    @Test
    fun settingsMigratesLegacyPlaintextThenRemovesIt() = withTemporarySettings { dataStore, settings ->
        val secret = "legacy-cluster-secret-${"b".repeat(32)}"
        dataStore.edit { it[SettingsDataStore.KEY_CLUSTER_SECRET] = secret }

        assertEquals(secret, settings.getClusterSecret())
        assertEquals(1, settings.getClusterSecretEpoch())
        assertFalse(dataStore.data.first().contains(SettingsDataStore.KEY_CLUSTER_SECRET))
        assertEquals(secret, credentialStore.read())
        val rawCredentialPreferences = context.getSharedPreferences(
            ClusterCredentialStore.STORE_NAME,
            Context.MODE_PRIVATE,
        ).all.values.joinToString()
        assertFalse(rawCredentialPreferences.contains(secret))
    }

    @Test
    fun bootstrapSaveAndClearUseOnlySecureCredentialStorage() =
        withTemporarySettings { dataStore, settings ->
            val secret = "new-cluster-secret-${"c".repeat(32)}"
            settings.saveBootstrapConfig(
                serverHost = "100.90.76.108",
                serverPort = 8000,
                masterTcpHost = "100.90.76.108",
                masterTcpPort = 8888,
                clusterId = "cluster_test",
                clusterSecret = secret,
                nodeId = "android_test",
                modelManifestUrl = "",
                clusterSecretEpoch = 4,
            )
            settings.setTaskWorkerStartupConfig("{\"nodeId\":\"android_test\"}")

            assertEquals(secret, settings.getClusterSecret())
            assertEquals(4, settings.getClusterSecretEpoch())
            assertFalse(dataStore.data.first().contains(SettingsDataStore.KEY_CLUSTER_SECRET))

            settings.clearBootstrapConfig()

            assertEquals("", settings.getClusterSecret())
            assertEquals(1, settings.getClusterSecretEpoch())
            assertNull(credentialStore.read())
            assertEquals("", settings.getTaskWorkerStartupConfig())
        }

    private fun withTemporarySettings(
        block: suspend (
            androidx.datastore.core.DataStore<androidx.datastore.preferences.core.Preferences>,
            SettingsDataStore,
        ) -> Unit,
    ) = runBlocking {
        val file = File(context.cacheDir, "qlh-settings-${UUID.randomUUID()}.preferences_pb")
        val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
        val dataStore = PreferenceDataStoreFactory.create(scope = scope) { file }
        val settings = SettingsDataStore(dataStore, credentialStore)
        try {
            block(dataStore, settings)
        } finally {
            scope.cancel()
            file.delete()
        }
    }
}
