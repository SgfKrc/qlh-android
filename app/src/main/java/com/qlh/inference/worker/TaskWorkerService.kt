package com.qlh.inference.worker

import android.annotation.SuppressLint
import android.app.Notification
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.os.Binder
import android.os.Build
import android.os.IBinder
import androidx.core.app.NotificationCompat
import androidx.core.content.ContextCompat
import com.qlh.inference.BuildConfig
import com.qlh.inference.MainActivity
import com.qlh.inference.QlhApplication
import com.qlh.inference.R
import com.qlh.inference.data.SettingsDataStore
import com.qlh.inference.logging.QlhLogger
import com.qlh.inference.service.InferenceService
import com.qlh.inference.service.ModelManager
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.cancel
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import java.util.concurrent.atomic.AtomicReference

/** Foreground lifecycle shell for the Android Full Worker client. */
class TaskWorkerService : Service() {
    private val binder = LocalBinder()
    private var client: TaskWorkerClient? = null
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)

    /** ★ 2026-10-07（DIST-NEXT-5）：启动配置的持久化载体（系统重建时读回）。 */
    private val settings by lazy { SettingsDataStore(applicationContext) }

    inner class LocalBinder : Binder() {
        fun getService(): TaskWorkerService = this@TaskWorkerService
        fun snapshot(): TaskWorkerSnapshot = this@TaskWorkerService.snapshot()
    }

    override fun onCreate() {
        super.onCreate()
        startForegroundNotification()
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        // ★ 2026-10-07（DIST-NEXT-5）：决策抽成纯函数（`decideTaskWorkerStartup`，可单测），
        //   这里只做 IO 与分派。关键变化：`intent == null`（系统 `START_STICKY` 回收后重建）
        //   不再落进默认分支什么都不做，而是用**持久化配置**恢复 worker。
        scope.launch { handleStartCommand(intent, startId) }
        return if (BuildConfig.IS_LITE) START_NOT_STICKY else START_STICKY
    }

    private suspend fun handleStartCommand(intent: Intent?, startId: Int) {
        val action = intent?.action
        val persisted = TaskWorkerStartupConfig.fromJson(
            settings.getTaskWorkerStartupConfig(),
        )
        when (decideTaskWorkerStartup(
            action = action,
            hasPersistedConfig = persisted?.usable == true,
            hasActiveClient = client != null,
        )) {
            TaskWorkerStartupDecision.Start -> {
                val config = startupConfigFromIntent(intent)
                if (config == null || !config.usable) {
                    // 配置不足以建 client ⇒ 明确停止，并把坏配置清掉（免得重建时再撞一次）。
                    QlhLogger.w(
                        "TaskWorkerService",
                        "ACTION_START rejected: incomplete worker configuration",
                    )
                    settings.clearTaskWorkerStartupConfig()
                    stopSelf(startId)
                    return
                }
                // 持久化：系统回收后凭它恢复（`START_STICKY` 重建路径的前提）。
                settings.setTaskWorkerStartupConfig(config.toJson())
                startWorker(config)
            }
            TaskWorkerStartupDecision.Resume -> {
                QlhLogger.i(
                    "TaskWorkerService",
                    "system recreated the worker without an intent; resuming from " +
                        "persisted config (node=${persisted?.nodeId} host=${persisted?.coordinatorHost})",
                )
                startWorker(persisted!!)
            }
            TaskWorkerStartupDecision.Stop -> {
                // 用户主动停止 ⇒ 连同持久化配置一起清掉，否则系统重建会把它拉起来。
                settings.clearTaskWorkerStartupConfig()
                stopWorker()
                stopSelf(startId)
            }
            TaskWorkerStartupDecision.Cancel ->
                client?.cancelActive(intent?.getStringExtra(EXTRA_REASON) ?: "user_cancelled")
            TaskWorkerStartupDecision.Ignore -> Unit
        }
    }

    /** 把 `ACTION_START` 的 extras 解析成可持久化的启动配置（校验交给 `usable`）。 */
    @Suppress("UNCHECKED_CAST")
    private fun startupConfigFromIntent(intent: Intent?): TaskWorkerStartupConfig? {
        if (intent == null) return null
        val deviceInfo = intent.getStringExtra(EXTRA_DEVICE_INFO_JSON).orEmpty()
            .let { raw ->
                runCatching {
                    com.google.gson.Gson().fromJson(raw, Map::class.java) as? Map<String, Any?>
                }.getOrNull().orEmpty()
            }
        return TaskWorkerStartupConfig(
            coordinatorHost = intent.getStringExtra(EXTRA_COORDINATOR_HOST).orEmpty().trim(),
            coordinatorPort = intent.getIntExtra(EXTRA_COORDINATOR_PORT, 0),
            nodeId = intent.getStringExtra(EXTRA_NODE_ID).orEmpty().trim(),
            clusterSecret = intent.getStringExtra(EXTRA_CLUSTER_SECRET).orEmpty(),
            hostname = intent.getStringExtra(EXTRA_HOSTNAME).orEmpty(),
            networkType = intent.getStringExtra(EXTRA_NETWORK_TYPE).orEmpty(),
            deviceInfo = deviceInfo,
            modelId = intent.getStringExtra(EXTRA_MODEL_ID).orEmpty().trim(),
            modelFormat = intent.getStringExtra(EXTRA_MODEL_FORMAT).orEmpty(),
            modelRevision = intent.getStringExtra(EXTRA_MODEL_REVISION).orEmpty(),
            modelSha256 = intent.getStringExtra(EXTRA_MODEL_SHA256).orEmpty().trim(),
            resourceAdmitted = intent.getBooleanExtra(EXTRA_RESOURCE_ADMITTED, false),
            resourceReason = intent.getStringExtra(EXTRA_RESOURCE_REASON).orEmpty().trim(),
            fullInferenceAvailable = intent.getBooleanExtra(EXTRA_FULL_INFERENCE_AVAILABLE, true),
        )
    }

    override fun onBind(intent: Intent?): IBinder = binder

    override fun onDestroy() {
        stopWorker()
        scope.cancel()
        super.onDestroy()
    }

    fun snapshot(): TaskWorkerSnapshot = client?.snapshot?.value ?: TaskWorkerSnapshot()

    fun cancelActive(reasonCode: String = "user_cancelled"): Boolean = client?.cancelActive(reasonCode) == true

    private suspend fun startWorker(config: TaskWorkerStartupConfig) {
        if (BuildConfig.IS_LITE) {
            stopSelf()
            return
        }
        // ★ 2026-10-07（DIST-NEXT-5）：参数来自 `TaskWorkerStartupConfig`（可持久化）
        //   而不是直接读 intent —— 系统重建路径因此与首次启动走**同一条**代码。
        //   校验已由 `config.usable` 完成（调用点判定）。
        val host = config.coordinatorHost
        val port = config.coordinatorPort
        val nodeId = config.nodeId
        val clusterSecret = config.clusterSecret
        val hostname = config.hostname.ifBlank { nodeId }
        val networkType = config.networkType.ifBlank { "unknown" }
        val deviceInfo = config.deviceInfo
        val modelId = config.modelId
        val modelFormat = config.modelFormat.ifBlank { "gguf" }
        val modelRevision = config.modelRevision.ifBlank { "local" }
        val modelSha256 = config.modelSha256
        val resourceAdmitted = config.resourceAdmitted
        val resourceReason = config.resourceReason
        val fullInferenceAvailable = config.fullInferenceAvailable
        val runtimeProfile = AndroidWorkerCapabilities.normalizeRuntimeProfile(
            deviceInfo["runtime_profile"]?.toString(),
        )
        // The distributed route must be able to advertise crop artifacts before
        // the local inference service has loaded anything (or even exists yet).
        val modelManager = QlhApplication.instance.inferenceService?.modelManager
            ?: ModelManager(this)
        // Capability construction is the integrity boundary: it hashes each
        // artifact once. Decode offers only consult this immutable verified
        // snapshot; re-hashing multi-GB GGUF files for every token would make
        // the stage path unusable.
        val verifiedLayerArtifacts = AtomicReference<List<ModelManager.LayerArtifact>>(emptyList())
        if (QlhApplication.instance.inferenceService == null && !BuildConfig.IS_LITE) {
            runCatching {
                ContextCompat.startForegroundService(this, Intent(this, InferenceService::class.java))
            }.onFailure { error ->
                QlhLogger.w("TaskWorkerService", "inference service start failed: ${error.message}")
            }
        }
        val expectedModelIdentity = {
            AndroidWorkerCapabilities.modelIdentity(
                modelId, modelFormat, modelRevision, modelSha256, resourceAdmitted,
            )
        }
        val buildCapabilities: suspend () -> Map<String, Any?> = suspend {
            val inventory = modelManager.scanLayerArtifacts(
                expectedModelSha256 = modelSha256,
                verifyArtifactDigest = true,
            ).getOrNull()
            val layerArtifacts = inventory?.artifacts.orEmpty()
            verifiedLayerArtifacts.set(layerArtifacts.toList())
            val layerRanges = layerArtifacts
                .map { listOf(it.startLayer, it.endLayerExclusive) }
            val advertisedLayerArtifacts = layerArtifacts.map {
                AndroidWorkerCapabilities.LayerArtifactCapability(
                    startLayer = it.startLayer,
                    endLayerExclusive = it.endLayerExclusive,
                    segmentMode = it.mode,
                    modelId = AndroidWorkerCapabilities.artifactModelId(
                        it.document.name,
                        it.artifactSha256,
                    ),
                    artifactSha256 = it.artifactSha256,
                    sourceModelSha256 = it.sourceModelSha256.ifBlank { null },
                )
            }
            val layerForwardInfo = QlhApplication.instance.inferenceService
                ?.engine
                ?.layerForwardInfo()
                ?.getOrNull()
                .orEmpty()
            // ★ 2026-10-03：设备自荐层容量 —— 按可用内存与已选工件推算"能承载多少层"。
            //   `localCut` 先恒为 false：设备侧本地裁层执行体尚未落地，不虚报能力。
            // ★ 每层字节必须来自**同一个工件**：用它自己的文件字节 ÷ 它覆盖的层数。
            //   此前用"目录里最大工件 ÷ 当前工件层数"，两者会错配（拿整模的字节除以
            //   中段的层数）⇒ 每层字节偏大、可承载层数被低估。
            val readyArtifact = layerArtifacts.firstOrNull { it.document.sizeBytes > 0 }
            // A single node-level budget cannot truthfully describe several
            // pre-cut artifacts with different bytes/layer.  Until local
            // cutting exists, exact per-artifact ranges are the contract; only
            // retain the legacy budget hint when there is one artifact.
            val layerBudget = if (layerArtifacts.size == 1) {
                AndroidWorkerCapabilities.computeLayerBudget(
                    availableBytes = (deviceInfo["memory"] as? Map<*, *>)
                        ?.get("available_bytes")
                        ?.let { (it as? Number)?.toLong() }
                        ?: 0L,
                    modelFileBytes = readyArtifact?.document?.sizeBytes ?: 0L,
                    coveredLayers = readyArtifact
                        ?.let { it.endLayerExclusive - it.startLayer } ?: 0,
                )
            } else {
                null
            }
            val capabilities = AndroidWorkerCapabilities.build(
                modelId = modelId,
                modelFormat = modelFormat,
                modelRevision = modelRevision,
                modelSha256 = modelSha256,
                resourceAdmitted = resourceAdmitted,
                resourceReason = resourceReason,
                layerRanges = layerRanges,
                fullInferenceAvailable = fullInferenceAvailable,
                layerWorker = !fullInferenceAvailable && layerRanges.isNotEmpty(),
                runtimeProfile = runtimeProfile,
                middleChannel = layerForwardInfo["middle_channel"],
                nPosPerEmbd = layerForwardInfo["n_pos_per_embd"]?.toIntOrNull(),
                layerBudget = layerBudget,
                layerArtifacts = advertisedLayerArtifacts,
                // ★ 2026-10-07（DIST-NEXT-6）：不可用工件的结构化原因（无本地路径）。
                layerArtifactDiagnostics = inventory?.failures
                    ?.map { it.toAdvertisement() }
                    .orEmpty(),
                // ★ 2026-10-07（DIST-NEXT-2b）：本构建已接线分片输入（`stage_chunk`
                //   接收 + `hidden_ref` 装配）⇒ 声明给主节点。
                stageChunkedInput = true,
            )
            QlhLogger.i(
                "TaskWorkerService",
                "hello capabilities: stages=${capabilities["stage_types"]} " +
                    "ranges=${capabilities["layer_ranges"]} model=${modelId.ifBlank { "<none>" }} " +
                    "sha=${modelSha256.take(12)} profile=$runtimeProfile " +
                    "budget=${layerBudget?.maxLayers} " +
                    "admitted=$resourceAdmitted full=$fullInferenceAvailable",
            )
            capabilities
        }
        stopWorker()
        client = TaskWorkerClient(
            host = host,
            port = port,
            nodeId = nodeId,
            registration = TaskWorkerRegistration(
                nodeId = nodeId,
                clusterSecret = clusterSecret,
                hostname = hostname,
                networkType = networkType,
                deviceInfo = deviceInfo,
                modelSha256 = modelSha256,
            ),
            capabilities = buildCapabilities,
            stageHandler = AndroidFullWorkerStageExecutor(
                expectedModelIdentity = expectedModelIdentity,
                allowLayerIdentityAlias = { !fullInferenceAvailable },
                ensureModelLoaded = { contextSize ->
                    awaitInferenceService()?.ensureModelLoaded(contextSize)
                        ?: Result.failure(IllegalStateException("inference_service_unavailable"))
                },
                generate = { prompt, maxTokens, temperature, topP ->
                    awaitInferenceService()?.generate(
                        prompt, maxTokens, temperature, topP,
                    ) ?: Result.failure(IllegalStateException("inference_service_unavailable"))
                },
                // ★ 2026-09-20（层段）：接上 layer_forward。
                //   中间段需要隐藏态导出 ⇒ 用 ensureModelLoadedForLayer 单独加载。
                //   ⚠️ 「是否导出隐藏态」是两种不同的加载方式，二者**不能共存**
                //      （引擎会按需卸载重载）。这是刻意的 fail-closed：
                //      宁可重载一次，也不静默降级成取不到 hidden。
                layerForward = { req ->
                    awaitInferenceService()?.engine?.let { engine ->
                        engine.layerForward(
                            hidden = req.hidden,
                            nTokens = req.nTokens,
                            posBase = req.posBase,
                            wantHidden = req.wantHidden,
                            // ★ 2026-09-23：中间段通道由 stage_offer 的 `middle_channel` 决定 ——
                            //   `keep_head_layer_out` ⇒ keep-head（**末层输出**，`output_norm` 之前），
                            //   其余 ⇒ 旧的 `extract_hidden`（`output_norm(H)`，多一次归一化）。
                            keepHead = req.middleChannel == "keep_head_layer_out",
                            // ★ 2026-09-23（A12）：多序列显式位置（协议层已校验；null = 单序列）。
                            seqIds = req.seqIds,
                            positions = req.positions,
                        ).map { out ->
                            LayerForwardResult(
                                tokenArgmax = out.tokenArgmax,
                                hiddenOut = out.hiddenOut,
                            )
                        }
                    } ?: Result.failure(IllegalStateException("inference_service_unavailable"))
                },
                // ★ 2026-10-07（DIST-NEXT-1）：取消合同 —— 把中止请求传到 native。
                //   `nativeLayerForward*` 因此在下一个可分割的张量边界退出并返回 -4，
                //   使「取消」不必等整个 stage 跑完；worker 客户端据此把
                //   `execution_in_flight` 升级为 `execution_stopped`。
                requestExecutionAbort = {
                    QlhApplication.instance.inferenceService?.engine
                        ?.requestLayerForwardAbort()
                },
                ensureModelLoadedForLayer = { range, contextSize, embeddingWidth, wantHidden, sha256 ->
                    awaitInferenceService()
                        ?.ensureLayerModelLoaded(
                            layerRange = range,
                            contextSize = contextSize,
                            extractHidden = wantHidden,
                            expectedModelSha256 = sha256,
                            expectedEmbeddingWidth = embeddingWidth,
                        )
                        ?: Result.failure(IllegalStateException("inference_service_unavailable"))
                },
                resolveLayerModelIdentity = { range, requested ->
                    verifiedLayerArtifacts.get().firstOrNull {
                        it.startLayer == range.getOrNull(0) &&
                            it.endLayerExclusive == range.getOrNull(1) &&
                            AndroidWorkerCapabilities.artifactModelId(
                                it.document.name,
                                it.artifactSha256,
                            ) == requested["model_id"] &&
                            it.artifactSha256 == requested["sha256"]
                    }?.let {
                        AndroidWorkerCapabilities.LayerArtifactCapability(
                            startLayer = it.startLayer,
                            endLayerExclusive = it.endLayerExclusive,
                            segmentMode = it.mode,
                            modelId = AndroidWorkerCapabilities.artifactModelId(
                                it.document.name,
                                it.artifactSha256,
                            ),
                            artifactSha256 = it.artifactSha256,
                            sourceModelSha256 = it.sourceModelSha256.ifBlank { null },
                        ).modelIdentity()
                    }
                },
            ),
        ).also { it.start() }
    }

    /**
     * ★ 2026-10-07（DIST-NEXT-5）：InferenceService 的**就绪屏障**。
     *
     * 此前固定等 5s（50 × 100ms），慢启动时 stage offer 会直接吃到
     * `inference_service_unavailable` —— 那不是「服务不可用」，而是「还没起完」。
     * 现在等待窗口由常量给出（默认 30s，与主仓 task-worker 控制面 health 超时同量级），
     * 超时仍返回 `null`（fail-closed），但会留下具名告警，便于与真实缺服务区分。
     */
    private suspend fun awaitInferenceService(
        timeoutMs: Long = INFERENCE_READY_TIMEOUT_MS,
    ): InferenceService? {
        val deadline = System.currentTimeMillis() + timeoutMs
        while (System.currentTimeMillis() < deadline) {
            QlhApplication.instance.inferenceService?.let { return it }
            delay(INFERENCE_READY_POLL_MS)
        }
        QlhApplication.instance.inferenceService?.let { return it }
        QlhLogger.w(
            "TaskWorkerService",
            "inference_service_unavailable after ${timeoutMs}ms wait " +
                "(service still starting or not admitted)",
        )
        return null
    }

    private fun stopWorker() {
        val old = client ?: return
        client = null
        old.stop()
    }

    @SuppressLint("ForegroundServiceType")
    private fun startForegroundNotification() {
        val pendingIntent = PendingIntent.getActivity(
            this,
            1,
            Intent(this, MainActivity::class.java),
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE,
        )
        val notification: Notification = NotificationCompat.Builder(
            this,
            QlhApplication.NOTIFICATION_CHANNEL_INFERENCE,
        )
            .setContentTitle("QLH Worker")
            .setContentText("Android Worker client is running")
            .setSmallIcon(android.R.drawable.ic_media_play)
            .setOngoing(true)
            .setContentIntent(pendingIntent)
            .setPriority(NotificationCompat.PRIORITY_LOW)
            .build()
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.UPSIDE_DOWN_CAKE) {
            startForeground(
                QlhApplication.NOTIFICATION_ID_WORKER,
                notification,
                android.content.pm.ServiceInfo.FOREGROUND_SERVICE_TYPE_SPECIAL_USE,
            )
        } else {
            startForeground(QlhApplication.NOTIFICATION_ID_WORKER, notification)
        }
    }

    companion object {
        const val ACTION_START = "com.qlh.inference.worker.START"
        const val ACTION_STOP = "com.qlh.inference.worker.STOP"
        const val ACTION_CANCEL = "com.qlh.inference.worker.CANCEL"

        /**
         * ★ 2026-10-07（DIST-NEXT-5）：InferenceService 就绪屏障的等待窗口。
         *
         * 与主仓 `scheduler_types.TASK_WORKER_HEALTH_TIMEOUT_FLOOR_SECONDS`（30s）同量级：
         * 设备侧等模型/引擎起完，而不是把「还没起完」报成「服务不可用」。
         */
        const val INFERENCE_READY_TIMEOUT_MS: Long = 30_000L
        const val INFERENCE_READY_POLL_MS: Long = 100L
        const val EXTRA_COORDINATOR_HOST = "coordinator_host"
        const val EXTRA_COORDINATOR_PORT = "coordinator_port"
        const val EXTRA_NODE_ID = "node_id"
        const val EXTRA_CLUSTER_SECRET = "cluster_secret"
        const val EXTRA_HOSTNAME = "hostname"
        const val EXTRA_NETWORK_TYPE = "network_type"
        const val EXTRA_DEVICE_INFO_JSON = "device_info_json"
        const val EXTRA_MODEL_ID = "model_id"
        const val EXTRA_MODEL_FORMAT = "model_format"
        const val EXTRA_MODEL_REVISION = "model_revision"
        const val EXTRA_MODEL_SHA256 = "model_sha256"
        const val EXTRA_RESOURCE_ADMITTED = "resource_admitted"
        const val EXTRA_RESOURCE_REASON = "resource_reason"
        const val EXTRA_FULL_INFERENCE_AVAILABLE = "full_inference_available"
        const val EXTRA_REASON = "reason"

        fun startIntent(
            context: Context,
            host: String,
            port: Int,
            nodeId: String,
            clusterSecret: String,
            hostname: String,
            networkType: String,
            deviceInfo: Map<String, Any?>,
            modelId: String = "",
            modelFormat: String = "gguf",
            modelRevision: String = "local",
            modelSha256: String = "",
            resourceAdmitted: Boolean = false,
            resourceReason: String = "resource_gate_not_confirmed",
            fullInferenceAvailable: Boolean = true,
        ): Intent = Intent(
            context,
            TaskWorkerService::class.java,
        ).setAction(ACTION_START)
            .putExtra(EXTRA_COORDINATOR_HOST, host)
            .putExtra(EXTRA_COORDINATOR_PORT, port)
            .putExtra(EXTRA_NODE_ID, nodeId)
            .putExtra(EXTRA_CLUSTER_SECRET, clusterSecret)
            .putExtra(EXTRA_HOSTNAME, hostname)
            .putExtra(EXTRA_NETWORK_TYPE, networkType)
            .putExtra(EXTRA_DEVICE_INFO_JSON, com.google.gson.Gson().toJson(deviceInfo))
            .putExtra(EXTRA_MODEL_ID, modelId)
            .putExtra(EXTRA_MODEL_FORMAT, modelFormat)
            .putExtra(EXTRA_MODEL_REVISION, modelRevision)
            .putExtra(EXTRA_MODEL_SHA256, modelSha256)
            .putExtra(EXTRA_RESOURCE_ADMITTED, resourceAdmitted)
            .putExtra(EXTRA_RESOURCE_REASON, resourceReason)
            .putExtra(EXTRA_FULL_INFERENCE_AVAILABLE, fullInferenceAvailable)

        fun stopIntent(context: Context): Intent = Intent(context, TaskWorkerService::class.java)
            .setAction(ACTION_STOP)

        fun cancelIntent(context: Context, reasonCode: String = "user_cancelled"): Intent = Intent(
            context,
            TaskWorkerService::class.java,
        ).setAction(ACTION_CANCEL).putExtra(EXTRA_REASON, reasonCode)
    }
}
