package com.qlh.inference.service

import android.annotation.SuppressLint
import android.app.Notification
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.os.Build
import android.os.IBinder
import androidx.core.app.NotificationCompat
import com.google.gson.JsonParser
import com.qlh.inference.MainActivity
import com.qlh.inference.QlhApplication
import com.qlh.inference.R
import com.qlh.inference.control.AndroidVoterSet
import com.qlh.inference.data.AndroidQuorumVoter
import com.qlh.inference.data.QuorumVoterLedger
import com.qlh.inference.data.SettingsDataStore
import com.qlh.inference.logging.QlhLogger
import com.qlh.inference.network.QuorumVoterRpcServer
import com.qlh.inference.security.QuorumEd25519
import com.qlh.inference.security.QuorumVoterKeyStore
import com.qlh.inference.security.QuorumVoterKeyMaterial
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock

object QuorumVoterServiceStatus {
    private val mutableMessage = MutableStateFlow("未运行")
    val message = mutableMessage.asStateFlow()
    internal fun publish(message: String) { mutableMessage.value = message }
}

class QuorumVoterService : Service() {
    private val serviceScope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
    private val startupMutex = Mutex()
    private var startup: Job? = null
    @Volatile private var rpcServer: QuorumVoterRpcServer? = null
    @Volatile private var ledger: QuorumVoterLedger? = null
    @Volatile private var voter: AndroidQuorumVoter? = null

    override fun onCreate() {
        super.onCreate()
        startForegroundNotification()
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        if (intent?.action == ACTION_STOP) {
            QuorumVoterServiceStatus.publish("已停止")
            stopSelf(startId)
            return START_NOT_STICKY
        }
        QuorumVoterServiceStatus.publish("启动中")
        startup?.cancel()
        startup = serviceScope.launch {
            runCatching { startVoter() }
                .onFailure { error ->
                    QuorumVoterServiceStatus.publish("启动失败: ${error.message ?: error.javaClass.simpleName}")
                    QlhLogger.e("QuorumVoterService", "voter startup rejected: ${error.javaClass.simpleName}", error)
                    stopSelf(startId)
                }
        }
        return START_STICKY
    }

    private suspend fun startVoter() = startupMutex.withLock {
        stopVoter()
        var material: QuorumVoterKeyMaterial? = null
        var store: QuorumVoterLedger? = null
        var localVoter: AndroidQuorumVoter? = null
        var server: QuorumVoterRpcServer? = null
        try {
            check(Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
                "Ed25519 voter requires Android API 33 or newer"
            }
            val settings = SettingsDataStore(applicationContext)
            check(settings.isQuorumVoterEnabled()) { "quorum voter is disabled" }
            val voterId = settings.getQuorumVoterId().trim()
            val voterSetJson = settings.getQuorumVoterSet()
            require(voterId.isNotBlank() && voterSetJson.isNotBlank()) { "quorum voter configuration is incomplete" }
            val voterSet = AndroidVoterSet.parse(JsonParser.parseString(voterSetJson).asJsonObject)
            require(voterSet.voters.size >= 3) { "automatic quorum requires at least three configured voters" }
            require(voterSet.clusterId == settings.getClusterId()) { "quorum voter cluster does not match bootstrap" }
            val identity = voterSet.voters[voterId] ?: error("voter id is absent from voter set")
            material = QuorumVoterKeyStore(applicationContext).readOrCreate()
            require(QuorumEd25519.matches(requireNotNull(material).seed, identity.publicKey)) {
                "voter public key does not match this device key"
            }
            val secret = settings.getClusterSecret()
            require(secret.isNotBlank()) { "cluster credential is unavailable" }

            store = QuorumVoterLedger(applicationContext, voterId, voterSet)
            localVoter = AndroidQuorumVoter(voterId, voterSet, requireNotNull(material).seed, requireNotNull(store))
            requireNotNull(material).seed.fill(0)
            server = QuorumVoterRpcServer(requireNotNull(localVoter), secret)
            requireNotNull(server).start()
            ledger = store
            voter = localVoter
            rpcServer = server
            QuorumVoterServiceStatus.publish("运行中，RPC 端口 ${requireNotNull(server).boundPort}")
            QlhLogger.i("QuorumVoterService", "voter ready: id=$voterId port=${QuorumVoterRpcServer.DEFAULT_PORT}")
        } catch (error: Exception) {
            server?.close()
            localVoter?.close()
            material?.seed?.fill(0)
            store?.close()
            throw error
        }
    }

    private fun stopVoter() {
        rpcServer?.close()
        rpcServer = null
        voter?.close()
        voter = null
        ledger?.close()
        ledger = null
    }

    override fun onDestroy() {
        startup?.cancel()
        stopVoter()
        if (QuorumVoterServiceStatus.message.value.startsWith("运行中")) {
            QuorumVoterServiceStatus.publish("已停止")
        }
        serviceScope.cancel()
        super.onDestroy()
    }

    override fun onBind(intent: Intent?): IBinder? = null

    @SuppressLint("ForegroundServiceType")
    private fun startForegroundNotification() {
        val pending = PendingIntent.getActivity(
            this, 5, Intent(this, MainActivity::class.java),
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE,
        )
        val notification: Notification = NotificationCompat.Builder(this, QlhApplication.NOTIFICATION_CHANNEL_PRESENCE)
            .setContentTitle("QLH quorum voter")
            .setContentText("Android 投票节点正在待命")
            .setSmallIcon(android.R.drawable.ic_menu_info_details)
            .setOngoing(true)
            .setContentIntent(pending)
            .setPriority(NotificationCompat.PRIORITY_LOW)
            .build()
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.UPSIDE_DOWN_CAKE) {
            startForeground(
                QlhApplication.NOTIFICATION_ID_PRESENCE + 1,
                notification,
                android.content.pm.ServiceInfo.FOREGROUND_SERVICE_TYPE_SPECIAL_USE,
            )
        } else {
            startForeground(QlhApplication.NOTIFICATION_ID_PRESENCE + 1, notification)
        }
    }

    companion object {
        const val ACTION_START = "com.qlh.inference.quorum.START"
        const val ACTION_STOP = "com.qlh.inference.quorum.STOP"

        fun startIntent(context: Context): Intent = Intent(context, QuorumVoterService::class.java)
            .setAction(ACTION_START)

        fun stopIntent(context: Context): Intent = Intent(context, QuorumVoterService::class.java)
            .setAction(ACTION_STOP)
    }
}
