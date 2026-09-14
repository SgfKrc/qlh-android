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
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.map
import java.util.UUID

private val Context.dataStore: DataStore<Preferences> by preferencesDataStore(name = "qlh_settings")

class SettingsDataStore(private val context: Context) {

    companion object {
        // ---- 主节点连接 ----
        val KEY_SERVER_HOST = stringPreferencesKey("server_host")
        val KEY_SERVER_PORT = intPreferencesKey("server_port")
        val KEY_ANDROID_NODE_ID = stringPreferencesKey("android_node_id")
        val KEY_BOOTSTRAPPED = booleanPreferencesKey("bootstrapped")
        val KEY_CLUSTER_ID = stringPreferencesKey("cluster_id")
        val KEY_MASTER_TCP_PORT = intPreferencesKey("master_tcp_port")
        val KEY_MODEL_MANIFEST_URL = stringPreferencesKey("model_manifest_url")

        // ---- 推理模式 ----
        val KEY_INFERENCE_MODE = stringPreferencesKey("inference_mode")  // "thin" | "full"

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

        // ---- 默认值 ----
        const val DEFAULT_HOST = "100.90.76.108"
        const val DEFAULT_PORT = 8000
        const val DEFAULT_MODE = "thin"
        const val DEFAULT_MAX_TOKENS = 1024
        const val DEFAULT_TEMPERATURE = 0.7f
        const val DEFAULT_TOP_P = 0.9f
        const val DEFAULT_CONTEXT_SIZE = 2048
        const val DEFAULT_MODEL_STORAGE_MODE = "saf_fd"
        const val DEFAULT_THEME_MODE = "system"
    }

    // ==================== 流式读取 ====================

    val serverHost: Flow<String> = context.dataStore.data.map { prefs ->
        prefs[KEY_SERVER_HOST] ?: DEFAULT_HOST
    }

    val serverPort: Flow<Int> = context.dataStore.data.map { prefs ->
        prefs[KEY_SERVER_PORT] ?: DEFAULT_PORT
    }

    val bootstrapped: Flow<Boolean> = context.dataStore.data.map { prefs ->
        prefs[KEY_BOOTSTRAPPED] ?: false
    }

    val inferenceMode: Flow<String> = context.dataStore.data.map { prefs ->
        prefs[KEY_INFERENCE_MODE] ?: DEFAULT_MODE
    }

    val maxTokens: Flow<Int> = context.dataStore.data.map { prefs ->
        prefs[KEY_MAX_TOKENS] ?: DEFAULT_MAX_TOKENS
    }

    val temperature: Flow<Float> = context.dataStore.data.map { prefs ->
        prefs[KEY_TEMPERATURE] ?: DEFAULT_TEMPERATURE
    }

    val topP: Flow<Float> = context.dataStore.data.map { prefs ->
        prefs[KEY_TOP_P] ?: DEFAULT_TOP_P
    }

    val showThinking: Flow<Boolean> = context.dataStore.data.map { prefs ->
        prefs[KEY_SHOW_THINKING] ?: false
    }

    val modelPath: Flow<String> = context.dataStore.data.map { prefs ->
        prefs[KEY_MODEL_PATH] ?: ""
    }

    val modelTreeUri: Flow<String> = context.dataStore.data.map { prefs ->
        prefs[KEY_MODEL_TREE_URI] ?: ""
    }

    val selectedModelUri: Flow<String> = context.dataStore.data.map { prefs ->
        prefs[KEY_SELECTED_MODEL_URI] ?: ""
    }

    val contextSize: Flow<Int> = context.dataStore.data.map { prefs ->
        prefs[KEY_CONTEXT_SIZE] ?: DEFAULT_CONTEXT_SIZE
    }

    val modelStorageMode: Flow<String> = context.dataStore.data.map { prefs ->
        prefs[KEY_MODEL_STORAGE_MODE] ?: DEFAULT_MODEL_STORAGE_MODE
    }

    val themeMode: Flow<String> = context.dataStore.data.map { prefs ->
        prefs[KEY_THEME_MODE] ?: DEFAULT_THEME_MODE
    }

    /** 获取完整的服务器 base URL */
    val baseUrl: Flow<String> = context.dataStore.data.map { prefs ->
        val host = prefs[KEY_SERVER_HOST] ?: DEFAULT_HOST
        val port = prefs[KEY_SERVER_PORT] ?: DEFAULT_PORT
        httpBaseUrl(host, port)
    }

    // ==================== 一次性读取 ====================

    suspend fun getServerHost(): String = context.dataStore.data.first()[KEY_SERVER_HOST] ?: DEFAULT_HOST
    suspend fun getServerPort(): Int = context.dataStore.data.first()[KEY_SERVER_PORT] ?: DEFAULT_PORT
    suspend fun isBootstrapped(): Boolean = context.dataStore.data.first()[KEY_BOOTSTRAPPED] ?: false
    suspend fun getClusterId(): String = context.dataStore.data.first()[KEY_CLUSTER_ID] ?: ""
    suspend fun getMasterTcpPort(): Int = context.dataStore.data.first()[KEY_MASTER_TCP_PORT] ?: 8888
    suspend fun getModelManifestUrl(): String = context.dataStore.data.first()[KEY_MODEL_MANIFEST_URL] ?: ""
    suspend fun getInferenceMode(): String = context.dataStore.data.first()[KEY_INFERENCE_MODE] ?: DEFAULT_MODE
    suspend fun getMaxTokens(): Int = context.dataStore.data.first()[KEY_MAX_TOKENS] ?: DEFAULT_MAX_TOKENS
    suspend fun getTemperature(): Float = context.dataStore.data.first()[KEY_TEMPERATURE] ?: DEFAULT_TEMPERATURE
    suspend fun getTopP(): Float = context.dataStore.data.first()[KEY_TOP_P] ?: DEFAULT_TOP_P
    suspend fun getContextSize(): Int = context.dataStore.data.first()[KEY_CONTEXT_SIZE] ?: DEFAULT_CONTEXT_SIZE
    suspend fun getModelPath(): String = context.dataStore.data.first()[KEY_MODEL_PATH] ?: ""
    suspend fun getModelTreeUri(): String = context.dataStore.data.first()[KEY_MODEL_TREE_URI] ?: ""
    suspend fun getSelectedModelUri(): String = context.dataStore.data.first()[KEY_SELECTED_MODEL_URI] ?: ""
    suspend fun getModelStorageMode(): String =
        context.dataStore.data.first()[KEY_MODEL_STORAGE_MODE] ?: DEFAULT_MODEL_STORAGE_MODE
    suspend fun getThemeMode(): String = context.dataStore.data.first()[KEY_THEME_MODE] ?: DEFAULT_THEME_MODE

    suspend fun getOrCreateAndroidNodeId(): String {
        val existing = context.dataStore.data.first()[KEY_ANDROID_NODE_ID]
        if (!existing.isNullOrBlank()) return existing
        val generated = "android-${UUID.randomUUID().toString().take(8)}"
        context.dataStore.edit { it[KEY_ANDROID_NODE_ID] = generated }
        return generated
    }

    suspend fun setAndroidNodeId(nodeId: String) {
        if (nodeId.isBlank()) return
        context.dataStore.edit { it[KEY_ANDROID_NODE_ID] = nodeId }
    }

    // ==================== 写入 ====================

    suspend fun setServerHost(host: String) {
        context.dataStore.edit { it[KEY_SERVER_HOST] = host }
    }

    suspend fun setServerPort(port: Int) {
        context.dataStore.edit { it[KEY_SERVER_PORT] = port }
    }

    suspend fun saveBootstrapConfig(
        serverHost: String,
        serverPort: Int,
        masterTcpPort: Int,
        clusterId: String,
        nodeId: String,
        modelManifestUrl: String,
    ) {
        context.dataStore.edit {
            if (serverHost.isNotBlank()) it[KEY_SERVER_HOST] = serverHost
            it[KEY_SERVER_PORT] = serverPort
            it[KEY_MASTER_TCP_PORT] = masterTcpPort
            it[KEY_CLUSTER_ID] = clusterId
            if (nodeId.isNotBlank()) it[KEY_ANDROID_NODE_ID] = nodeId
            if (modelManifestUrl.isNotBlank()) it[KEY_MODEL_MANIFEST_URL] = modelManifestUrl
            it[KEY_BOOTSTRAPPED] = true
        }
    }

    suspend fun clearBootstrapConfig() {
        context.dataStore.edit {
            it.remove(KEY_BOOTSTRAPPED)
            it.remove(KEY_CLUSTER_ID)
            it.remove(KEY_MASTER_TCP_PORT)
            it.remove(KEY_MODEL_MANIFEST_URL)
        }
    }

    suspend fun setInferenceMode(mode: String) {
        context.dataStore.edit { it[KEY_INFERENCE_MODE] = mode }
    }

    suspend fun setMaxTokens(tokens: Int) {
        context.dataStore.edit { it[KEY_MAX_TOKENS] = tokens }
    }

    suspend fun setTemperature(temp: Float) {
        context.dataStore.edit { it[KEY_TEMPERATURE] = temp }
    }

    suspend fun setTopP(topP: Float) {
        context.dataStore.edit { it[KEY_TOP_P] = topP }
    }

    suspend fun setModelPath(path: String) {
        context.dataStore.edit { it[KEY_MODEL_PATH] = path }
    }

    suspend fun setModelTreeUri(uri: String) {
        context.dataStore.edit { it[KEY_MODEL_TREE_URI] = uri }
    }

    suspend fun setSelectedModelUri(uri: String) {
        context.dataStore.edit { it[KEY_SELECTED_MODEL_URI] = uri }
    }

    suspend fun setContextSize(size: Int) {
        context.dataStore.edit { it[KEY_CONTEXT_SIZE] = size.coerceIn(512, 4096) }
    }

    suspend fun setModelStorageMode(mode: String) {
        context.dataStore.edit { it[KEY_MODEL_STORAGE_MODE] = mode }
    }

    suspend fun setThemeMode(mode: String) {
        val normalized = when (mode) {
            "light", "dark" -> mode
            else -> DEFAULT_THEME_MODE
        }
        context.dataStore.edit { it[KEY_THEME_MODE] = normalized }
    }

    suspend fun clearSelectedModelUri() {
        context.dataStore.edit { it.remove(KEY_SELECTED_MODEL_URI) }
    }

    suspend fun clearModelPath() {
        context.dataStore.edit { it.remove(KEY_MODEL_PATH) }
    }
}
