package com.qlh.inference.worker

import com.google.gson.Gson
import com.google.gson.JsonParser
import com.google.gson.annotations.SerializedName

/**
 * ★ 2026-10-07（DIST-NEXT-5）：task-worker 的**启动配置**（可持久化）与**重建决策**。
 *
 * 背景（审计 P1-3）：`TaskWorkerService` 返回 `START_STICKY`，但只有带完整 extras 的
 * `ACTION_START` 才能建 client —— 系统回收后重建时 intent 为空，服务既不恢复也拿不到
 * 配置，worker 静默消失。本文件把「配置解析 / 校验 / 序列化」与「重建时该做什么」
 * 抽成**纯逻辑**，让 JVM 单测锁住重建路径；真正的 DataStore 读写与 Intent extras
 * 解析留在 `TaskWorkerService` 里。
 */
data class TaskWorkerStartupConfig(
    val coordinatorHost: String,
    val coordinatorPort: Int,
    val nodeId: String,
    @SerializedName(value = "clusterSecretEpoch", alternate = ["secretEpoch"])
    val clusterSecretEpoch: Int = 1,
    val hostname: String = "",
    val networkType: String = "unknown",
    val deviceInfo: Map<String, Any?> = emptyMap(),
    val modelId: String = "",
    val modelFormat: String = "gguf",
    val modelRevision: String = "local",
    val modelSha256: String = "",
    val resourceAdmitted: Boolean = false,
    val resourceReason: String = "resource_gate_not_confirmed",
    val fullInferenceAvailable: Boolean = true,
) {
    /**
     * 配置是否足以建 client —— **唯一**的校验判据。
     *
     * 调用方（`TaskWorkerService.startWorker`）不再各自检查 host/port/node；凭据在独立的
     * AndroidKeyStore 边界校验。系统重建读回的旧配置也必须过这一关，否则应明确停止。
     */
    val usable: Boolean
        get() = coordinatorHost.isNotBlank() &&
            coordinatorPort in 1..65535 &&
            nodeId.isNotBlank() &&
            clusterSecretEpoch >= 1

    fun toJson(): String = GSON.toJson(this)

    companion object {
        private val GSON = Gson()

        /** 从持久化 JSON 读回；损坏或缺失时返回 `null`（调用方按「无配置」处理）。 */
        fun fromJson(raw: String?): TaskWorkerStartupConfig? {
            val text = raw?.trim().orEmpty()
            if (text.isEmpty()) return null
            return runCatching {
                GSON.fromJson(text, TaskWorkerStartupConfig::class.java)
                    ?.let { parsed ->
                        if (parsed.clusterSecretEpoch >= 1) {
                            parsed
                        } else {
                            parsed.copy(clusterSecretEpoch = 1)
                        }
                    }
            }.getOrNull()
        }

        /** Reads the old persisted field only long enough to migrate it to secure storage. */
        fun legacyClusterSecretFromJson(raw: String?): String? {
            val text = raw?.trim().orEmpty()
            if (text.isEmpty()) return null
            return runCatching {
                JsonParser.parseString(text)
                    .asJsonObject
                    .get("clusterSecret")
                    ?.takeUnless { it.isJsonNull }
                    ?.asString
                    ?.takeIf { it.isNotBlank() }
            }.getOrNull()
        }

        /** True even for a blank old field so the caller can scrub the obsolete JSON key. */
        fun containsLegacyClusterSecret(raw: String?): Boolean {
            val text = raw?.trim().orEmpty()
            if (text.isEmpty()) return false
            return runCatching {
                JsonParser.parseString(text).asJsonObject.has("clusterSecret")
            }.getOrDefault(false)
        }
    }
}

/** ★ 2026-10-07（DIST-NEXT-5）：`onStartCommand` 该做什么。 */
sealed class TaskWorkerStartupDecision {
    /** 带配置的 `ACTION_START`：建 client（并持久化配置）。 */
    data object Start : TaskWorkerStartupDecision()

    /** 系统重建（intent 无 action / null）：用持久化配置恢复 worker。 */
    data object Resume : TaskWorkerStartupDecision()

    /** 明确停止，或重建时既无配置也无在跑 client。 */
    data object Stop : TaskWorkerStartupDecision()

    /** `ACTION_CANCEL`：本地取消当前 attempt。 */
    data object Cancel : TaskWorkerStartupDecision()

    /** 无关启动（例如已在跑且没有新配置）：不动。 */
    data object Ignore : TaskWorkerStartupDecision()
}

/**
 * ★ 2026-10-07（DIST-NEXT-5）：由 `(action, 是否有持久化配置, 是否已有 client)` 决定动作。
 *
 * `null` action 是**系统重建**的信号（`START_STICKY` 后台被回收后重新拉起）——
 * 此前它落进 `when` 的默认分支，返回 `START_NOT_STICKY` 且不恢复任何东西。
 */
fun decideTaskWorkerStartup(
    action: String?,
    hasPersistedConfig: Boolean,
    hasActiveClient: Boolean,
): TaskWorkerStartupDecision = when (action) {
    TaskWorkerService.ACTION_START -> TaskWorkerStartupDecision.Start
    TaskWorkerService.ACTION_STOP -> TaskWorkerStartupDecision.Stop
    TaskWorkerService.ACTION_CANCEL -> TaskWorkerStartupDecision.Cancel
    else -> when {
        hasPersistedConfig -> TaskWorkerStartupDecision.Resume
        hasActiveClient -> TaskWorkerStartupDecision.Ignore
        else -> TaskWorkerStartupDecision.Stop
    }
}
