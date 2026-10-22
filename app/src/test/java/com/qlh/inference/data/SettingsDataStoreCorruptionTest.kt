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
    @Test
    fun retiredBootstrapNeverPersistsEndpointOrCredential() = runBlocking {
        val source = temporaryFolder.newFile("bootstrap.preferences_pb")
        val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
        val dataStore = PreferenceDataStoreFactory.create(scope = scope, produceFile = { source })
        val storage = object : ClusterCredentialStorage {
            var secret: String? = null
            override fun read() = secret
            override fun save(clusterSecret: String) { secret = clusterSecret }
            override fun clear() { secret = null }
        }
        try {
            val settings = SettingsDataStore(dataStore, storage)
            assertEquals(null, settings.getIntentMasterEndpoint()) // A hardcoded default is not a saved binding.
            settings.setServerHost("100.64.0.1")
            settings.setInferenceMode(SettingsDataStore.MODE_LOCAL)
            assertFalse(settings.saveBootstrapConfig(
                serverHost = "100.64.0.9", serverPort = 8001, masterTcpHost = "100.64.0.9",
                masterTcpPort = 8888, clusterId = "stale", clusterSecret = "old-secret", nodeId = "android-1",
                modelManifestUrl = "", stillCurrent = { false },
            ))
            assertEquals("100.64.0.1", settings.getServerHost())
            assertEquals(SettingsDataStore.MODE_LOCAL, settings.getInferenceMode())
            assertFalse(settings.isBootstrapped())
            assertEquals(null, storage.secret)
            assertTrue(settings.saveBootstrapConfig(
                serverHost = "100.64.0.2", serverPort = 8002, masterTcpHost = "100.64.0.2",
                masterTcpPort = 8889, clusterId = "fresh", clusterSecret = "fresh-secret", nodeId = "android-1",
                modelManifestUrl = "", clusterSecretEpoch = 2,
            ))
            assertEquals(com.qlh.inference.IntentMasterEndpoint("100.64.0.2", 8002, "fresh", 2), settings.getIntentMasterEndpoint())
            assertEquals("fresh-secret", storage.secret)
            assertEquals(SettingsDataStore.MODE_LOCAL, settings.getInferenceMode()) // Binding never changes mode.
        } finally { scope.cancel() }
    }

    @Test
    fun bootstrapRefreshPreservesVoterOnlyForSameClusterAndCredentialEpoch() = runBlocking {
        val source = temporaryFolder.newFile("voter-bootstrap.preferences_pb")
        val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
        val dataStore = PreferenceDataStoreFactory.create(scope = scope, produceFile = { source })
        val storage = object : ClusterCredentialStorage {
            var secret: String? = null
            override fun read() = secret
            override fun save(clusterSecret: String) { secret = clusterSecret }
            override fun clear() { secret = null }
        }
        try {
            val settings = SettingsDataStore(dataStore, storage)
            settings.saveBootstrapConfig(
                serverHost = "100.64.0.1", serverPort = 8001, masterTcpHost = "100.64.0.1",
                masterTcpPort = 8889, clusterId = "cluster-a", clusterSecret = "secret-a",
                nodeId = "android-1", modelManifestUrl = "", clusterSecretEpoch = 3,
            )
            val voterSet = "{\"cluster_id\":\"cluster-a\",\"voters\":{}}"
            settings.saveQuorumVoterConfig("y700-voter", voterSet, enabled = true)

            settings.saveBootstrapConfig(
                serverHost = "100.64.0.2", serverPort = 8001, masterTcpHost = "100.64.0.2",
                masterTcpPort = 8889, clusterId = "cluster-a", clusterSecret = "secret-a",
                nodeId = "android-1", modelManifestUrl = "", clusterSecretEpoch = 3,
            )
            assertTrue(settings.isQuorumVoterEnabled())
            assertEquals("y700-voter", settings.getQuorumVoterId())
            assertEquals(voterSet, settings.getQuorumVoterSet())

            settings.saveBootstrapConfig(
                serverHost = "100.64.0.2", serverPort = 8001, masterTcpHost = "100.64.0.2",
                masterTcpPort = 8889, clusterId = "cluster-a", clusterSecret = "secret-b",
                nodeId = "android-1", modelManifestUrl = "", clusterSecretEpoch = 3,
            )
            assertFalse(settings.isQuorumVoterEnabled())

            settings.saveQuorumVoterConfig("y700-voter", voterSet, enabled = true)
            settings.saveBootstrapConfig(
                serverHost = "100.64.0.2", serverPort = 8001, masterTcpHost = "100.64.0.2",
                masterTcpPort = 8889, clusterId = "cluster-a", clusterSecret = "secret-b",
                nodeId = "android-1", modelManifestUrl = "", clusterSecretEpoch = 4,
            )
            assertFalse(settings.isQuorumVoterEnabled())

            settings.saveQuorumVoterConfig("y700-voter", voterSet, enabled = true)
            settings.saveBootstrapConfig(
                serverHost = "100.64.0.3", serverPort = 8001, masterTcpHost = "100.64.0.3",
                masterTcpPort = 8889, clusterId = "cluster-b", clusterSecret = "secret-c",
                nodeId = "android-1", modelManifestUrl = "", clusterSecretEpoch = 4,
            )
            assertFalse(settings.isQuorumVoterEnabled())
        } finally { scope.cancel() }
    }

    @Test
    fun userSelectionRetiresBootstrapWaitingForPreferencePublication() = runBlocking {
        val source = temporaryFolder.newFile("delayed-bootstrap.preferences_pb")
        val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
        val dataStore = PreferenceDataStoreFactory.create(scope = scope, produceFile = { source })
        val entered = kotlinx.coroutines.CompletableDeferred<Unit>()
        val release = kotlinx.coroutines.CompletableDeferred<Unit>()
        val current = java.util.concurrent.atomic.AtomicBoolean(true)
        val publicationLock = Any()
        val storage = object : ClusterCredentialStorage {
            var saved = false
            override fun read(): String? = null
            override fun save(clusterSecret: String) { saved = true }
            override fun clear() {}
        }
        try {
            val settings = SettingsDataStore(dataStore, storage)
            settings.setServerHost("100.64.0.1")
            val blockingEdit = async {
                dataStore.edit { entered.complete(Unit); release.await() }
            }
            entered.await()
            val staleBootstrap = async(start = kotlinx.coroutines.CoroutineStart.UNDISPATCHED) {
                settings.saveBootstrapConfig(
                    serverHost = "100.64.0.9", serverPort = 8009, masterTcpHost = "100.64.0.9",
                    masterTcpPort = 8889, clusterId = "stale", clusterSecret = "secret", nodeId = "phone",
                    modelManifestUrl = "", stillCurrent = current::get, publicationLock = publicationLock,
                )
            }
            synchronized(publicationLock) { current.set(false) }
            release.complete(Unit)
            blockingEdit.await()
            assertFalse(withTimeout(2000) { staleBootstrap.await() })
            assertFalse(storage.saved)
            assertEquals("100.64.0.1", settings.getServerHost())
            assertFalse(settings.isBootstrapped())
        } finally { release.complete(Unit); scope.cancel() }
    }

    @Test
    fun noThinkingAndShortTokenLimitPersistForNextChat() = runBlocking {
        val source = temporaryFolder.newFile("chat-settings.preferences_pb")
        val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
        val dataStore = PreferenceDataStoreFactory.create(scope = scope, produceFile = { source })
        val storage = object : ClusterCredentialStorage {
            override fun read(): String? = null
            override fun save(clusterSecret: String) {}
            override fun clear() {}
        }
        try {
            val settings = SettingsDataStore(dataStore, storage)
            assertFalse(settings.getShowThinking())
            settings.setShowThinking(true)
            assertTrue(settings.showThinking.first())
            settings.setShowThinking(false)
            settings.setMaxTokens(32)
            val restored = SettingsDataStore(dataStore, storage)
            assertFalse(restored.getShowThinking())
            assertFalse(restored.showThinking.first())
            assertEquals(32, restored.getMaxTokens())
        } finally { scope.cancel() }
    }

}
