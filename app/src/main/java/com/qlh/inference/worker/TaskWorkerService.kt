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

/** Foreground lifecycle shell for the Android Full Worker client. */
class TaskWorkerService : Service() {
    private val binder = LocalBinder()
    private var client: TaskWorkerClient? = null

    inner class LocalBinder : Binder() {
        fun getService(): TaskWorkerService = this@TaskWorkerService
        fun snapshot(): TaskWorkerSnapshot = this@TaskWorkerService.snapshot()
    }

    override fun onCreate() {
        super.onCreate()
        startForegroundNotification()
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        when (intent?.action) {
            ACTION_START -> startWorker(intent)
            ACTION_STOP -> {
                stopWorker()
                stopSelf(startId)
            }
            ACTION_CANCEL -> client?.cancelActive(intent.getStringExtra(EXTRA_REASON) ?: "user_cancelled")
        }
        return if (client != null) START_STICKY else START_NOT_STICKY
    }

    override fun onBind(intent: Intent?): IBinder = binder

    override fun onDestroy() {
        stopWorker()
        super.onDestroy()
    }

    fun snapshot(): TaskWorkerSnapshot = client?.snapshot?.value ?: TaskWorkerSnapshot()

    fun cancelActive(reasonCode: String = "user_cancelled"): Boolean = client?.cancelActive(reasonCode) == true

    private fun startWorker(intent: Intent) {
        if (BuildConfig.IS_LITE) {
            stopSelf()
            return
        }
        val host = intent.getStringExtra(EXTRA_COORDINATOR_HOST).orEmpty().trim()
        val port = intent.getIntExtra(EXTRA_COORDINATOR_PORT, 0)
        val nodeId = intent.getStringExtra(EXTRA_NODE_ID).orEmpty().trim()
        if (host.isEmpty() || port !in 1..65535 || nodeId.isEmpty()) {
            stopSelf()
            return
        }
        val modelId = intent.getStringExtra(EXTRA_MODEL_ID).orEmpty().trim()
        val modelFormat = intent.getStringExtra(EXTRA_MODEL_FORMAT).orEmpty().ifBlank { "gguf" }
        val modelRevision = intent.getStringExtra(EXTRA_MODEL_REVISION).orEmpty().ifBlank { "local" }
        val modelSha256 = intent.getStringExtra(EXTRA_MODEL_SHA256).orEmpty().trim()
        val resourceAdmitted = intent.getBooleanExtra(EXTRA_RESOURCE_ADMITTED, false)
        val resourceReason = intent.getStringExtra(EXTRA_RESOURCE_REASON).orEmpty().trim()
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
            capabilities = {
                AndroidWorkerCapabilities.build(
                    modelId = modelId,
                    modelFormat = modelFormat,
                    modelRevision = modelRevision,
                    modelSha256 = modelSha256,
                    resourceAdmitted = resourceAdmitted,
                    resourceReason = resourceReason,
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
