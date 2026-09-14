package com.qlh.inference

import android.app.Application
import android.app.NotificationChannel
import android.app.NotificationManager
import android.os.Build
import com.qlh.inference.data.AppDatabase
import com.qlh.inference.logging.QlhLogger
import com.qlh.inference.service.InferenceService

class QlhApplication : Application() {

    lateinit var database: AppDatabase
        private set

    /** 本地推理引擎 Service（全有模式下绑定后设置，全无模式为 null） */
    @Volatile
    var inferenceService: InferenceService? = null

    override fun onCreate() {
        super.onCreate()
        instance = this

        // 初始化 Room 数据库
        database = AppDatabase.getInstance(this)

        // 创建通知渠道（全有模式推理引擎前台 Service）
        createNotificationChannels()

        // 初始化文件日志（输出到 filesDir/logs/）
        QlhLogger.init(this)
        val previousHandler = Thread.getDefaultUncaughtExceptionHandler()
        Thread.setDefaultUncaughtExceptionHandler { thread, throwable ->
            QlhLogger.crash("Uncaught", "${thread.name} crashed", throwable)
            previousHandler?.uncaughtException(thread, throwable)
            if (previousHandler == null) {
                android.os.Process.killProcess(android.os.Process.myPid())
                System.exit(10)
            }
        }
    }

    private fun createNotificationChannels() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channel = NotificationChannel(
                NOTIFICATION_CHANNEL_INFERENCE,
                getString(R.string.notification_channel_inference),
                NotificationManager.IMPORTANCE_LOW
            ).apply {
                description = "推理引擎运行状态"
                setShowBadge(false)
            }
            val manager = getSystemService(NotificationManager::class.java)
            manager.createNotificationChannel(channel)
            manager.createNotificationChannel(
                NotificationChannel(
                    NOTIFICATION_CHANNEL_PRESENCE,
                    "QLH 集群在线状态",
                    NotificationManager.IMPORTANCE_LOW,
                ).apply {
                    description = "Android 主节点 presence lease"
                    setShowBadge(false)
                }
            )
        }
    }

    companion object {
        const val NOTIFICATION_CHANNEL_INFERENCE = "inference_engine"
        const val NOTIFICATION_CHANNEL_PRESENCE = "cluster_presence"
        const val NOTIFICATION_ID_INFERENCE = 1001
        const val NOTIFICATION_ID_WORKER = 1002
        const val NOTIFICATION_ID_PRESENCE = 1003

        lateinit var instance: QlhApplication
            private set
    }
}
