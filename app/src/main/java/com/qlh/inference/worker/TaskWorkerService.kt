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
import com.qlh.inference.BuildConfig
import com.qlh.inference.MainActivity
import com.qlh.inference.QlhApplication
import com.qlh.inference.R
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.cancel
import kotlinx.coroutines.launch

/** Foreground lifecycle shell for the Android Full Worker client. */
class TaskWorkerService : Service() {
    private val binder = LocalBinder()
    private var client: TaskWorkerClient? = null
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)

    inner class LocalBinder : Binder() {
        fun getService(): TaskWorkerService = this@TaskWorkerService
        fun snapshot(): TaskWorkerSnapshot = this@TaskWorkerService.snapshot()
    }

    override fun onCreate() {
        super.onCreate()
        startForegroundNotification()
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        val starting = intent?.action == ACTION_START
        when (intent?.action) {
            ACTION_START -> scope.launch { startWorker(intent) }
            ACTION_STOP -> {
                stopWorker()
                stopSelf(startId)
            }
            ACTION_CANCEL -> client?.cancelActive(intent.getStringExtra(EXTRA_REASON) ?: "user_cancelled")
        }
        return if (starting || client != null) START_STICKY else START_NOT_STICKY
    }

    override fun onBind(intent: Intent?): IBinder = binder

    override fun onDestroy() {
        stopWorker()
        scope.cancel()
        super.onDestroy()
    }

    fun snapshot(): TaskWorkerSnapshot = client?.snapshot?.value ?: TaskWorkerSnapshot()

    fun cancelActive(reasonCode: String = "user_cancelled"): Boolean = client?.cancelActive(reasonCode) == true

    private suspend fun startWorker(intent: Intent) {
        if (BuildConfig.IS_LITE) {
            stopSelf()
            return
        }
        val host = intent.getStringExtra(EXTRA_COORDINATOR_HOST).orEmpty().trim()
        val port = intent.getIntExtra(EXTRA_COORDINATOR_PORT, 0)
        val nodeId = intent.getStringExtra(EXTRA_NODE_ID).orEmpty().trim()
        val clusterSecret = intent.getStringExtra(EXTRA_CLUSTER_SECRET).orEmpty()
        val hostname = intent.getStringExtra(EXTRA_HOSTNAME).orEmpty().ifBlank { nodeId }
        val networkType = intent.getStringExtra(EXTRA_NETWORK_TYPE).orEmpty().ifBlank { "unknown" }
        val deviceInfo = intent.getStringExtra(EXTRA_DEVICE_INFO_JSON).orEmpty()
            .let { raw ->
                runCatching {
                    @Suppress("UNCHECKED_CAST")
                    com.google.gson.Gson().fromJson(raw, Map::class.java) as? Map<String, Any?>
                }.getOrNull().orEmpty()
            }
        if (host.isEmpty() || port !in 1..65535 || nodeId.isEmpty()) {
            stopSelf()
            return
        }
        if (clusterSecret.isBlank()) {
            stopSelf()
            return
        }
        val modelId = intent.getStringExtra(EXTRA_MODEL_ID).orEmpty().trim()
        val modelFormat = intent.getStringExtra(EXTRA_MODEL_FORMAT).orEmpty().ifBlank { "gguf" }
        val modelRevision = intent.getStringExtra(EXTRA_MODEL_REVISION).orEmpty().ifBlank { "local" }
        val modelSha256 = intent.getStringExtra(EXTRA_MODEL_SHA256).orEmpty().trim()
        val resourceAdmitted = intent.getBooleanExtra(EXTRA_RESOURCE_ADMITTED, false)
        val resourceReason = intent.getStringExtra(EXTRA_RESOURCE_REASON).orEmpty().trim()
        val layerRanges = QlhApplication.instance.inferenceService
            ?.modelManager
            ?.listLayerArtifacts(
                expectedModelSha256 = modelSha256,
                verifyArtifactDigest = true,
            )
            ?.getOrNull()
            ?.map { listOf(it.startLayer, it.endLayerExclusive) }
            .orEmpty()
        // ★ 2026-09-23：native 的中间段能力（`layerForwardInfo()` 是 suspend，且要求模型已加载）
        //   ⇒ 这里预取一次，capabilities lambda（非 suspend）后续直接读；模型未加载时留空
        //   （协议侧这两个键都是可选的，缺失即不写）。
        val layerForwardInfo = QlhApplication.instance.inferenceService
            ?.engine
            ?.layerForwardInfo()
            ?.getOrNull()
            .orEmpty()
        val expectedModelIdentity = {
            AndroidWorkerCapabilities.modelIdentity(
                modelId, modelFormat, modelRevision, modelSha256, resourceAdmitted,
            )
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
            capabilities = {
                AndroidWorkerCapabilities.build(
                    modelId = modelId,
                    modelFormat = modelFormat,
                    modelRevision = modelRevision,
                    modelSha256 = modelSha256,
                    resourceAdmitted = resourceAdmitted,
                    resourceReason = resourceReason,
                    layerRanges = layerRanges,
                    middleChannel = layerForwardInfo["middle_channel"],
                    nPosPerEmbd = layerForwardInfo["n_pos_per_embd"]?.toIntOrNull(),
                )
            },
            stageHandler = AndroidFullWorkerStageExecutor(
                expectedModelIdentity = expectedModelIdentity,
                ensureModelLoaded = { contextSize ->
                    QlhApplication.instance.inferenceService?.ensureModelLoaded(contextSize)
                        ?: Result.failure(IllegalStateException("inference_service_unavailable"))
                },
                generate = { prompt, maxTokens, temperature, topP ->
                    QlhApplication.instance.inferenceService?.generate(
                        prompt, maxTokens, temperature, topP,
                    ) ?: Result.failure(IllegalStateException("inference_service_unavailable"))
                },
                // ★ 2026-09-20（层段）：接上 layer_forward。
                //   中间段需要隐藏态导出 ⇒ 用 ensureModelLoadedForLayer 单独加载。
                //   ⚠️ 「是否导出隐藏态」是两种不同的加载方式，二者**不能共存**
                //      （引擎会按需卸载重载）。这是刻意的 fail-closed：
                //      宁可重载一次，也不静默降级成取不到 hidden。
                layerForward = { req ->
                    QlhApplication.instance.inferenceService?.engine?.let { engine ->
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
                ensureModelLoadedForLayer = { range, contextSize, embeddingWidth, wantHidden, sha256 ->
                    QlhApplication.instance.inferenceService
                        ?.ensureLayerModelLoaded(
                            layerRange = range,
                            contextSize = contextSize,
                            extractHidden = wantHidden,
                            expectedModelSha256 = sha256,
                            expectedEmbeddingWidth = embeddingWidth,
                        )
                        ?: Result.failure(IllegalStateException("inference_service_unavailable"))
                },
            ),
        ).also { it.start() }
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

        fun stopIntent(context: Context): Intent = Intent(context, TaskWorkerService::class.java)
            .setAction(ACTION_STOP)

        fun cancelIntent(context: Context, reasonCode: String = "user_cancelled"): Intent = Intent(
            context,
            TaskWorkerService::class.java,
        ).setAction(ACTION_CANCEL).putExtra(EXTRA_REASON, reasonCode)
    }
}
