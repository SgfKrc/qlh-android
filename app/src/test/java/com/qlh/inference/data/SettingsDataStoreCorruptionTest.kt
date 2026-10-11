package com.qlh.inference.data

import androidx.datastore.preferences.core.PreferenceDataStoreFactory
import androidx.datastore.preferences.core.edit
import androidx.datastore.preferences.core.stringPreferencesKey
import com.qlh.inference.security.ClusterCredentialStorage
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.async
import kotlinx.coroutines.cancel
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withContext
import kotlinx.coroutines.withTimeout
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder

class SettingsDataStoreCorruptionTest {
    @get:Rule val temporaryFolder = TemporaryFolder()

    @Test
    fun corruptPreferencesAreBackedUpThenReplacedWithWritableDefaults() = runBlocking {
        val directory = temporaryFolder.newFolder("datastore")
        val source = directory.resolve("qlh_settings.preferences_pb")
        val corruptBytes = byteArrayOf(0x0F, 0x01, 0x02)
        source.writeBytes(corruptBytes)
        val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
        val dataStore = PreferenceDataStoreFactory.create(
            corruptionHandler = settingsCorruptionHandler(source),
            scope = scope,
            produceFile = { source },
        )
        try {
            val key = stringPreferencesKey("server_host")
            assertEquals(null, withTimeout(10_000) { dataStore.data.first()[key] })
            val backups = directory.listFiles().orEmpty().filter {
                it.name.startsWith("${source.name}.corrupt-")
            }
            assertEquals(1, backups.size)
            assertArrayEquals(corruptBytes, backups.single().readBytes())
            dataStore.edit { it[key] = "100.90.76.108" }
            assertEquals("100.90.76.108", dataStore.data.first()[key])
            assertTrue(source.isFile)
            assertFalse(source.readBytes().contentEquals(corruptBytes))
        } finally {
            scope.cancel()
        }
    }

    @Test
    fun missingFileDoesNotProduceAFalseBackup() {
        val directory = temporaryFolder.newFolder("datastore")
        val source = directory.resolve("missing.preferences_pb")
        assertEquals(null, backupCorruptSettingsFile(source))
        assertTrue(directory.listFiles().orEmpty().isEmpty())
    }

    @Test
    fun credentialMigrationIsSerializedAcrossSettingsInstances() = runBlocking {
        val file = temporaryFolder.newFile("legacy.preferences_pb")
        val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
        val dataStore = PreferenceDataStoreFactory.create(
            scope = scope,
            produceFile = { file },
        )
        val readerEntered = CountDownLatch(1)
        val releaseReader = CountDownLatch(1)
        val writerStarted = CountDownLatch(1)
        val saveEntered = CountDownLatch(1)
        val storage = object : ClusterCredentialStorage {
            @Volatile var secret: String? = null

            override fun read(): String? {
                val snapshot = secret
                if (snapshot == null) {
                    readerEntered.countDown()
                    check(releaseReader.await(5, TimeUnit.SECONDS))
                }
                return snapshot
            }

            override fun save(clusterSecret: String) {
                saveEntered.countDown()
                secret = clusterSecret
            }

            override fun clear() { secret = null }
        }
        try {
            dataStore.edit { it[SettingsDataStore.KEY_CLUSTER_SECRET] = "legacy" }
            val first = SettingsDataStore(dataStore, storage)
            val second = SettingsDataStore(dataStore, storage)
            val migration = async(Dispatchers.IO) { first.getClusterSecret() }
            assertTrue(withContext(Dispatchers.IO) {
                readerEntered.await(5, TimeUnit.SECONDS)
            })
            val write = async(Dispatchers.IO) {
                writerStarted.countDown()
                second.saveClusterSecret("fresh")
            }
            assertTrue(withContext(Dispatchers.IO) {
                writerStarted.await(5, TimeUnit.SECONDS)
            })
            assertFalse(withContext(Dispatchers.IO) {
                saveEntered.await(200, TimeUnit.MILLISECONDS)
            })
            releaseReader.countDown()
            assertEquals("legacy", withTimeout(5_000) { migration.await() })
            withTimeout(5_000) { write.await() }
            assertEquals("fresh", storage.secret)
        } finally {
            releaseReader.countDown()
            scope.cancel()
        }
    }
}
