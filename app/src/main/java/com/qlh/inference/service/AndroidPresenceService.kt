package com.qlh.inference.service

import android.annotation.SuppressLint
import com.google.gson.Gson
import android.app.Notification
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.os.Build
import android.os.IBinder
import androidx.core.app.NotificationCompat
import com.qlh.inference.BuildConfig
import com.qlh.inference.MainActivity
import com.qlh.inference.QlhApplication
import com.qlh.inference.network.AndroidPresenceHeartbeatRequest
import com.qlh.inference.network.AndroidPresenceSnapshot
import com.qlh.inference.network.AndroidPresenceStateMachine
import com.qlh.inference.network.ApiClient
import com.qlh.inference.network.ApiClientHttpException
import com.qlh.inference.network.RegisterNodeRequest
import com.qlh.inference.network.httpBaseUrl
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.cancel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch

/** Foreground Android HTTP presence lease owner. No model data crosses this channel. */
class AndroidPresenceService : Service() {
    private val serviceScope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
    private var loopJob: Job? = null
    private var machine = AndroidPresenceStateMachine()

    override fun onCreate() {
        super.onCreate()
        startForegroundNotification()
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        if (BuildConfig.IS_LITE || intent?.action == ACTION_STOP) {
            stopPresence()
            stopSelf(startId)
            return START_NOT_STICKY
        }
        val host = intent?.getStringExtra(EXTRA_HOST).orEmpty().trim()
        val port = intent?.getIntExtra(EXTRA_PORT, 0) ?: 0
        val nodeId = intent?.getStringExtra(EXTRA_NODE_ID).orEmpty().trim()
        if (host.isEmpty() || port !in 1..65535 || nodeId.isEmpty()) {
            publish(AndroidPresenceSnapshot(lastErrorCode = "invalid_presence_config", lastErrorMessage = "presence 配置无效"))
            stopSelf(startId)
            return START_NOT_STICKY
        }
        startPresence(
            host,
            port,
            nodeId,
            intent?.getStringExtra(EXTRA_HOSTNAME).orEmpty(),
            intent?.getStringExtra(EXTRA_NETWORK_TYPE).orEmpty(),
            intent?.getStringExtra(EXTRA_DEVICE_INFO_JSON).orEmpty(),
        )
        return START_STICKY
    }

    override fun onDestroy() {
        stopPresence()
        serviceScope.cancel()
        super.onDestroy()
    }

    override fun onBind(intent: Intent?): IBinder? = null

    private fun startPresence(
        host: String,
        port: Int,
        nodeId: String,
        hostname: String,
        networkType: String,
        deviceInfoJson: String,
    ) {
        loopJob?.cancel()
        machine = AndroidPresenceStateMachine()
        val generation = runGeneration.incrementAndGet()
        loopJob = serviceScope.launch {
            val client = ApiClient(httpBaseUrl(host, port))
            machine.start(System.currentTimeMillis())
            publish(machine.snapshot())
            while (isActive && generation == runGeneration.get()) {
                val now = System.currentTimeMillis()
                val current = machine.snapshot()
                when (current.state) {
                    com.qlh.inference.network.AndroidPresenceState.REGISTERING -> {
                        if (machine.beginRegistration(now)) {
                            try {
                                val response = client.registerAndroidNode(
                                    RegisterNodeRequest(
                                        nodeId = nodeId,
                                        hostname = hostname.ifBlank { nodeId },
                                        networkType = networkType.ifBlank { "unknown" },
                                        deviceInfo = decodeDeviceInfo(deviceInfoJson),
                                        clientMode = "thin",
                                        appVariant = if (BuildConfig.IS_LITE) "lite" else "full",
                                        appVersion = BuildConfig.VERSION_NAME,
                                    )
                                ).getOrThrow()
                                machine.onRegistered(
                                    generation = response.presenceGeneration,
                                    leaseId = response.presenceLeaseId,
                                    leaseExpiresAtMs = response.leaseExpiresAtMs,
                                    heartbeatIntervalSeconds = response.heartbeatIntervalSeconds,
                                    nowMs = System.currentTimeMillis(),
                                    serverTimeMs = response.serverTimeMs,
                                )
                            } catch (e: Exception) {
                                machine.onFailure(errorCode(e, "presence_register_failed"), errorMessage(e), now)
                            }
                            publish(machine.snapshot())
                        }
                    }
                    com.qlh.inference.network.AndroidPresenceState.ONLINE -> {
                        if (machine.heartbeatDue(now)) {
                            val lease = machine.snapshot()
                            try {
                                val response = client.heartbeatAndroidNode(
                                    AndroidPresenceHeartbeatRequest(nodeId, lease.generation, lease.leaseId)
                                ).getOrThrow()
                                machine.onHeartbeatSuccess(
                                    generation = response.presenceGeneration,
                                    leaseId = response.presenceLeaseId,
                                    leaseExpiresAtMs = response.leaseExpiresAtMs,
                                    heartbeatIntervalSeconds = response.heartbeatIntervalSeconds,
                                    nowMs = System.currentTimeMillis(),
                                    serverTimeMs = response.serverTimeMs,
                                )
                            } catch (e: Exception) {
                                machine.onFailure(errorCode(e, "presence_heartbeat_failed"), errorMessage(e), now)
                            }
                            publish(machine.snapshot())
                        }
                    }
                    com.qlh.inference.network.AndroidPresenceState.BACKING_OFF -> {
                        machine.retryIfDue(now)
                        publish(machine.snapshot())
                    }
                    else -> Unit
                }
                delay(1_000L)
            }
        }
    }

    private fun stopPresence() {
        runGeneration.incrementAndGet()
        loopJob?.cancel()
        loopJob = null
        machine.stop()
        publish(machine.snapshot())
    }

    private fun publish(snapshot: AndroidPresenceSnapshot) {
        _snapshot.value = snapshot
    }

    private fun basicDeviceInfo(): Map<String, Any?> = mapOf(
        "connection_type" to "http_thin",
        "pipeline_worker" to false,
        "android" to mapOf(
            "manufacturer" to Build.MANUFACTURER.orEmpty(),
            "model" to Build.MODEL.orEmpty(),
            "sdk_int" to Build.VERSION.SDK_INT,
        ),
    )

    @Suppress("UNCHECKED_CAST")
    private fun decodeDeviceInfo(json: String): Map<String, Any?> =
        if (json.isBlank()) basicDeviceInfo()
        else runCatching { Gson().fromJson(json, Map::class.java) as Map<String, Any?> }
            .getOrElse { basicDeviceInfo() }

    private fun errorCode(error: Throwable, fallback: String): String =
        (error as? ApiClientHttpException)?.errorCode?.ifBlank { fallback } ?: fallback

    private fun errorMessage(error: Throwable): String = error.message ?: error.javaClass.simpleName

    @SuppressLint("ForegroundServiceType")
    private fun startForegroundNotification() {
        val pendingIntent = PendingIntent.getActivity(
            this,
            3,
            Intent(this, MainActivity::class.java),
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE,
        )
        val notification: Notification = NotificationCompat.Builder(
            this,
            QlhApplication.NOTIFICATION_CHANNEL_PRESENCE,
        )
            .setContentTitle("QLH presence")
            .setContentText("集群在线状态同步中")
            .setSmallIcon(android.R.drawable.ic_menu_info_details)
            .setOngoing(true)
            .setContentIntent(pendingIntent)
            .setPriority(NotificationCompat.PRIORITY_LOW)
            .build()
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.UPSIDE_DOWN_CAKE) {
            startForeground(
                QlhApplication.NOTIFICATION_ID_PRESENCE,
                notification,
                android.content.pm.ServiceInfo.FOREGROUND_SERVICE_TYPE_SPECIAL_USE,
            )
        } else {
            startForeground(QlhApplication.NOTIFICATION_ID_PRESENCE, notification)
        }
    }

    companion object {
        private val runGeneration = java.util.concurrent.atomic.AtomicLong(0L)
        private val _snapshot = MutableStateFlow(AndroidPresenceSnapshot())
        val snapshot: StateFlow<AndroidPresenceSnapshot> = _snapshot.asStateFlow()
        const val ACTION_START = "com.qlh.inference.presence.START"
        const val ACTION_STOP = "com.qlh.inference.presence.STOP"
        const val EXTRA_HOST = "master_host"
        const val EXTRA_PORT = "master_port"
        const val EXTRA_NODE_ID = "node_id"
        const val EXTRA_HOSTNAME = "hostname"
        const val EXTRA_NETWORK_TYPE = "network_type"
        const val EXTRA_DEVICE_INFO_JSON = "device_info_json"

        fun startIntent(
            context: Context,
            host: String,
            port: Int,
            nodeId: String,
            hostname: String,
            networkType: String = "unknown",
            deviceInfoJson: String = "",
        ): Intent =
            Intent(context, AndroidPresenceService::class.java)
                .setAction(ACTION_START)
                .putExtra(EXTRA_HOST, host)
                .putExtra(EXTRA_PORT, port)
                .putExtra(EXTRA_NODE_ID, nodeId)
                .putExtra(EXTRA_HOSTNAME, hostname)
                .putExtra(EXTRA_NETWORK_TYPE, networkType)
                .putExtra(EXTRA_DEVICE_INFO_JSON, deviceInfoJson)

        fun stopIntent(context: Context): Intent =
            Intent(context, AndroidPresenceService::class.java).setAction(ACTION_STOP)
    }
}
