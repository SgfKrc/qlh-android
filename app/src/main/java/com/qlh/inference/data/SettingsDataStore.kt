package com.qlh.inference.data

import com.qlh.inference.network.httpBaseUrl

import android.content.Context
import androidx.datastore.core.DataStore
import androidx.datastore.preferences.core.Preferences
import androidx.datastore.preferences.core.booleanPreferencesKey
import androidx.datastore.preferences.core.edit
import androidx.datastore.preferences.core.floatPreferencesKey
import androidx.datastore.preferences.core.intPreferencesKey
import androidx.datastore.preferences.core.stringPreferencesKey
import androidx.datastore.preferences.preferencesDataStore
import com.qlh.inference.security.ClusterCredentialStorage
import com.qlh.inference.security.ClusterCredentialStore
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.map
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import java.util.UUID

private val Context.dataStore: DataStore<Preferences> by preferencesDataStore(name = "qlh_settings")

class SettingsDataStore internal constructor(
    private val dataStore: DataStore<Preferences>,
    private val credentialStore: ClusterCredentialStorage,
) {
    constructor(context: Context) : this(
        dataStore = context.applicationContext.dataStore,
        credentialStore = ClusterCredentialStore(context.applicationContext),
    )

    private val credentialMigrationMutex = Mutex()

    companion object {
        // ---- 主节点连接 ----
        val KEY_SERVER_HOST = stringPreferencesKey("server_host")
        val KEY_SERVER_PORT = intPreferencesKey("server_port")
        val KEY_ANDROID_NODE_ID = stringPreferencesKey("android_node_id")
        val KEY_BOOTSTRAPPED = booleanPreferencesKey("bootstrapped")
        val KEY_CLUSTER_ID = stringPreferencesKey("cluster_id")
        val KEY_MASTER_TCP_HOST = stringPreferencesKey("master_tcp_host")
        val KEY_MASTER_TCP_PORT = intPreferencesKey("master_tcp_port")
        val KEY_CLUSTER_SECRET = stringPreferencesKey("cluster_secret")
        val KEY_CLUSTER_SECRET_EPOCH = intPreferencesKey("cluster_secret_epoch")
        val KEY_MODEL_MANIFEST_URL = stringPreferencesKey("model_manifest_url")

        // ---- 推理模式 ----
        val KEY_INFERENCE_MODE = stringPreferencesKey("inference_mode")  // "local" | "distributed" | "fallback"

        // ---- 推理参数 ----
        val KEY_MAX_TOKENS = intPreferencesKey("max_tokens")
        val KEY_TEMPERATURE = floatPreferencesKey("temperature")
        val KEY_TOP_P = floatPreferencesKey("top_p")
        val KEY_SHOW_THINKING = booleanPreferencesKey("show_thinking")
        val KEY_CONTEXT_SIZE = intPreferencesKey("context_size")

        // ---- 全有模式 ----
        val KEY_MODEL_PATH = stringPreferencesKey("model_path")
        val KEY_MODEL_TREE_URI = stringPreferencesKey("model_tree_uri")
        val KEY_SELECTED_MODEL_URI = stringPreferencesKey("selected_model_uri")
        val KEY_MODEL_STORAGE_MODE = stringPreferencesKey("model_storage_mode")
        val KEY_THEME_MODE = stringPreferencesKey("theme_mode") // "system" | "light" | "dark"

        // ---- task-worker 启动配置（系统重建恢复用）★ 2026-10-07（DIST-NEXT-5） ----
        val KEY_TASK_WORKER_STARTUP = stringPreferencesKey("task_worker_startup")

        /** ★ 2026-10-07（DIST-NEXT-5b）：电池优化豁免引导是否已问过（一次性，不反复打扰）。 */
        val KEY_BATTERY_EXEMPTION_ASKED = booleanPreferencesKey("battery_exemption_asked")

        // ---- 默认值 ----
        const val DEFAULT_HOST = "100.90.76.108"
        const val DEFAULT_PORT = 8000
        const val MODE_LOCAL = "local"
        const val MODE_DISTRIBUTED = "distributed"
        const val MODE_FALLBACK = "fallback"
        const val DEFAULT_MODE = MODE_DISTRIBUTED

        fun normalizeInferenceMode(value: String): String = when (value.lowercase()) {
            "full" -> MODE_LOCAL
            "thin" -> MODE_DISTRIBUTED
            MODE_LOCAL, MODE_DISTRIBUTED, MODE_FALLBACK -> value.lowercase()
            else -> DEFAULT_MODE
        }
        const val DEFAULT_MAX_TOKENS = 1024
        const val DEFAULT_TEMPERATURE = 0.7f
        const val DEFAULT_TOP_P = 0.9f
        const val DEFAULT_CONTEXT_SIZE = 2048
        const val DEFAULT_MODEL_STORAGE_MODE = "saf_fd"
        const val DEFAULT_THEME_MODE = "system"
    }

    // ==================== 流式读取 ====================

    val serverHost: Flow<String> = dataStore.data.map { prefs ->
        prefs[KEY_SERVER_HOST] ?: DEFAULT_HOST
    }

    val serverPort: Flow<Int> = dataStore.data.map { prefs ->
        prefs[KEY_SERVER_PORT] ?: DEFAULT_PORT
    }

    val bootstrapped: Flow<Boolean> = dataStore.data.map { prefs ->
        prefs[KEY_BOOTSTRAPPED] ?: false
    }

    val inferenceMode: Flow<String> = dataStore.data.map { prefs ->
        normalizeInferenceMode(prefs[KEY_INFERENCE_MODE] ?: DEFAULT_MODE)
    }

    val maxTokens: Flow<Int> = dataStore.data.map { prefs ->
        prefs[KEY_MAX_TOKENS] ?: DEFAULT_MAX_TOKENS
    }

    val temperature: Flow<Float> = dataStore.data.map { prefs ->
        prefs[KEY_TEMPERATURE] ?: DEFAULT_TEMPERATURE
    }

    val topP: Flow<Float> = dataStore.data.map { prefs ->
        prefs[KEY_TOP_P] ?: DEFAULT_TOP_P
    }

    val showThinking: Flow<Boolean> = dataStore.data.map { prefs ->
        prefs[KEY_SHOW_THINKING] ?: false
    }

    val modelPath: Flow<String> = dataStore.data.map { prefs ->
        prefs[KEY_MODEL_PATH] ?: ""
    }

    val modelTreeUri: Flow<String> = dataStore.data.map { prefs ->
        prefs[KEY_MODEL_TREE_URI] ?: ""
    }

    val selectedModelUri: Flow<String> = dataStore.data.map { prefs ->
        prefs[KEY_SELECTED_MODEL_URI] ?: ""
    }

    val contextSize: Flow<Int> = dataStore.data.map { prefs ->
        prefs[KEY_CONTEXT_SIZE] ?: DEFAULT_CONTEXT_SIZE
    }

    val modelStorageMode: Flow<String> = dataStore.data.map { prefs ->
        prefs[KEY_MODEL_STORAGE_MODE] ?: DEFAULT_MODEL_STORAGE_MODE
    }

    val themeMode: Flow<String> = dataStore.data.map { prefs ->
        prefs[KEY_THEME_MODE] ?: DEFAULT_THEME_MODE
    }

    /** 获取完整的服务器 base URL */
    val baseUrl: Flow<String> = dataStore.data.map { prefs ->
        val host = prefs[KEY_SERVER_HOST] ?: DEFAULT_HOST
        val port = prefs[KEY_SERVER_PORT] ?: DEFAULT_PORT
        httpBaseUrl(host, port)
    }

    // ==================== 一次性读取 ====================

    suspend fun getServerHost(): String = dataStore.data.first()[KEY_SERVER_HOST] ?: DEFAULT_HOST
    suspend fun getServerPort(): Int = dataStore.data.first()[KEY_SERVER_PORT] ?: DEFAULT_PORT
    suspend fun isBootstrapped(): Boolean = dataStore.data.first()[KEY_BOOTSTRAPPED] ?: false
    suspend fun getClusterId(): String = dataStore.data.first()[KEY_CLUSTER_ID] ?: ""
    suspend fun getMasterTcpHost(): String = dataStore.data.first()[KEY_MASTER_TCP_HOST] ?: ""
    suspend fun getMasterTcpPort(): Int = dataStore.data.first()[KEY_MASTER_TCP_PORT] ?: 8888
    suspend fun getClusterSecretEpoch(): Int =
        dataStore.data.first()[KEY_CLUSTER_SECRET_EPOCH]?.takeIf { it >= 1 } ?: 1

    /**
     * Reads the only supported credential store and migrates the legacy DataStore
     * plaintext exactly once. The plaintext is removed only after encryption succeeds.
     */
    suspend fun getClusterSecret(): String = credentialMigrationMutex.withLock {
        credentialStore.read()?.takeIf { it.isNotBlank() }?.also {
            removeLegacyClusterSecret()
        } ?: run {
            val legacy = dataStore.data.first()[KEY_CLUSTER_SECRET].orEmpty()
            if (legacy.isBlank()) {
                removeLegacyClusterSecret()
                ""
            } else {
                credentialStore.save(legacy)
                removeLegacyClusterSecret()
                legacy
            }
        }
    }

    suspend fun getModelManifestUrl(): String = dataStore.data.first()[KEY_MODEL_MANIFEST_URL] ?: ""
    suspend fun getInferenceMode(): String = normalizeInferenceMode(
        dataStore.data.first()[KEY_INFERENCE_MODE] ?: DEFAULT_MODE,
    )
    suspend fun getMaxTokens(): Int = dataStore.data.first()[KEY_MAX_TOKENS] ?: DEFAULT_MAX_TOKENS
    suspend fun getTemperature(): Float = dataStore.data.first()[KEY_TEMPERATURE] ?: DEFAULT_TEMPERATURE
    suspend fun getTopP(): Float = dataStore.data.first()[KEY_TOP_P] ?: DEFAULT_TOP_P
    suspend fun getContextSize(): Int = dataStore.data.first()[KEY_CONTEXT_SIZE] ?: DEFAULT_CONTEXT_SIZE
    suspend fun getModelPath(): String = dataStore.data.first()[KEY_MODEL_PATH] ?: ""
    suspend fun getModelTreeUri(): String = dataStore.data.first()[KEY_MODEL_TREE_URI] ?: ""
    suspend fun getSelectedModelUri(): String = dataStore.data.first()[KEY_SELECTED_MODEL_URI] ?: ""
    suspend fun getModelStorageMode(): String =
        dataStore.data.first()[KEY_MODEL_STORAGE_MODE] ?: DEFAULT_MODEL_STORAGE_MODE
    suspend fun getThemeMode(): String = dataStore.data.first()[KEY_THEME_MODE] ?: DEFAULT_THEME_MODE

    suspend fun getOrCreateAndroidNodeId(): String {
        val existing = dataStore.data.first()[KEY_ANDROID_NODE_ID]
        if (!existing.isNullOrBlank()) return existing
        val generated = "android-${UUID.randomUUID().toString().take(8)}"
        dataStore.edit { it[KEY_ANDROID_NODE_ID] = generated }
        return generated
    }

    suspend fun setAndroidNodeId(nodeId: String) {
        if (nodeId.isBlank()) return
        dataStore.edit { it[KEY_ANDROID_NODE_ID] = nodeId }
    }

    // ==================== 写入 ====================

    suspend fun setServerHost(host: String) {
        dataStore.edit { it[KEY_SERVER_HOST] = host }
    }

    suspend fun setServerPort(port: Int) {
        dataStore.edit { it[KEY_SERVER_PORT] = port }
    }

    /** Saves a cluster credential without ever writing it to DataStore. */
    suspend fun saveClusterSecret(clusterSecret: String) = credentialMigrationMutex.withLock {
        if (clusterSecret.isBlank()) return@withLock
        credentialStore.save(clusterSecret)
        removeLegacyClusterSecret()
    }

    /**
     * ★ 2026-10-07（DIST-NEXT-5）：task-worker 的启动配置（`TaskWorkerStartupConfig.toJson()`）。
     *
     * 系统回收后 `START_STICKY` 重建时 `TaskWorkerService` 读回它恢复 worker；
     * 用户主动停止（`ACTION_STOP`）或配置不足时清除，避免把坏配置反复拉起。
     */
    suspend fun getTaskWorkerStartupConfig(): String =
        dataStore.data.first()[KEY_TASK_WORKER_STARTUP] ?: ""

    suspend fun setTaskWorkerStartupConfig(json: String) {
        dataStore.edit { it[KEY_TASK_WORKER_STARTUP] = json }
    }

    suspend fun clearTaskWorkerStartupConfig() {
        dataStore.edit { it.remove(KEY_TASK_WORKER_STARTUP) }
    }

    /**
     * ★ 2026-10-07（DIST-NEXT-5b）：本次安装内是否已经问过「忽略电池优化」。
     */
    suspend fun hasAskedBatteryExemption(): Boolean =
        dataStore.data.first()[KEY_BATTERY_EXEMPTION_ASKED] ?: false

    /**
     * ★ 2026-10-07（DIST-NEXT-5b）：标记已问过 —— 系统对话框只在首次出现一次，
     * 用户拒绝后不再重复打扰（撤销豁免要用户自己去系统设置）。
     */
    suspend fun markBatteryExemptionAsked() {
        dataStore.edit { it[KEY_BATTERY_EXEMPTION_ASKED] = true }
    }

    suspend fun saveBootstrapConfig(
        serverHost: String,
        serverPort: Int,
        masterTcpHost: String,
        masterTcpPort: Int,
        clusterId: String,
        clusterSecret: String,
        nodeId: String,
        modelManifestUrl: String,
        clusterSecretEpoch: Int = 1,
    ) {
        if (clusterSecret.isNotBlank()) saveClusterSecret(clusterSecret)
        dataStore.edit {
            if (serverHost.isNotBlank()) it[KEY_SERVER_HOST] = serverHost
            it[KEY_SERVER_PORT] = serverPort
            if (masterTcpHost.isNotBlank()) it[KEY_MASTER_TCP_HOST] = masterTcpHost
            it[KEY_MASTER_TCP_PORT] = masterTcpPort
            it[KEY_CLUSTER_ID] = clusterId
            it.remove(KEY_CLUSTER_SECRET)
            it[KEY_CLUSTER_SECRET_EPOCH] = clusterSecretEpoch.coerceAtLeast(1)
            if (nodeId.isNotBlank()) it[KEY_ANDROID_NODE_ID] = nodeId
            if (modelManifestUrl.isNotBlank()) it[KEY_MODEL_MANIFEST_URL] = modelManifestUrl
            it[KEY_BOOTSTRAPPED] = true
        }
    }

    suspend fun clearBootstrapConfig() = credentialMigrationMutex.withLock {
        credentialStore.clear()
        dataStore.edit {
            it.remove(KEY_BOOTSTRAPPED)
            it.remove(KEY_CLUSTER_ID)
            it.remove(KEY_MASTER_TCP_HOST)
            it.remove(KEY_MASTER_TCP_PORT)
            it.remove(KEY_CLUSTER_SECRET)
            it.remove(KEY_CLUSTER_SECRET_EPOCH)
            it.remove(KEY_MODEL_MANIFEST_URL)
            it.remove(KEY_TASK_WORKER_STARTUP)
        }
    }

    private suspend fun removeLegacyClusterSecret() {
        if (!dataStore.data.first().contains(KEY_CLUSTER_SECRET)) return
        dataStore.edit { prefs ->
            if (prefs.contains(KEY_CLUSTER_SECRET)) prefs.remove(KEY_CLUSTER_SECRET)
        }
    }

    suspend fun setInferenceMode(mode: String) {
        dataStore.edit { it[KEY_INFERENCE_MODE] = mode }
    }

    suspend fun setMaxTokens(tokens: Int) {
        dataStore.edit { it[KEY_MAX_TOKENS] = tokens }
    }

    suspend fun setTemperature(temp: Float) {
        dataStore.edit { it[KEY_TEMPERATURE] = temp }
    }

    suspend fun setTopP(topP: Float) {
        dataStore.edit { it[KEY_TOP_P] = topP }
    }

    suspend fun setModelPath(path: String) {
        dataStore.edit { it[KEY_MODEL_PATH] = path }
    }

    suspend fun setModelTreeUri(uri: String) {
        dataStore.edit { it[KEY_MODEL_TREE_URI] = uri }
    }

    suspend fun setSelectedModelUri(uri: String) {
        dataStore.edit { it[KEY_SELECTED_MODEL_URI] = uri }
    }

    suspend fun setContextSize(size: Int) {
        dataStore.edit { it[KEY_CONTEXT_SIZE] = size.coerceIn(512, 4096) }
    }

    suspend fun setModelStorageMode(mode: String) {
        dataStore.edit { it[KEY_MODEL_STORAGE_MODE] = mode }
    }

    suspend fun setThemeMode(mode: String) {
        val normalized = when (mode) {
            "light", "dark" -> mode
            else -> DEFAULT_THEME_MODE
        }
        dataStore.edit { it[KEY_THEME_MODE] = normalized }
    }

    suspend fun clearSelectedModelUri() {
        dataStore.edit { it.remove(KEY_SELECTED_MODEL_URI) }
    }

    suspend fun clearModelPath() {
        dataStore.edit { it.remove(KEY_MODEL_PATH) }
    }
}
