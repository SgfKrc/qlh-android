package com.qlh.inference.system

/**
 * ★ 2026-10-07（DIST-NEXT-5b）：worker 的**电源策略**（纯逻辑，可 JVM 单测）。
 *
 * 审计 P1-3 的两条结论：① 三个前台服务没有电池优化豁免引导，厂商省电策略会杀后台；
 * ② `InferenceService` 的 WakeLock 固定 30 分钟，既不随任务续、也不在空闲释放。
 * 本文件把这两件事的**判定**抽出来，真机 soak 与 UI 引导接线留在各自的位置。
 */
object WorkerBatteryPolicy {
    /** 系统设置里的「忽略电池优化」申请入口（弹出系统对话框）。 */
    const val ACTION_REQUEST_IGNORE_BATTERY_OPTIMIZATIONS =
        "android.settings.REQUEST_IGNORE_BATTERY_OPTIMIZATIONS"

    /**
     * 是否应当发起「忽略电池优化」引导。
     *
     * 只在**尚未豁免**且**本次安装内还没问过**时为真 —— 反复弹系统对话框会被视为骚扰，
     * 而引导本身必须可以拒绝（拒绝后服务照常跑，只是可能被厂商策略杀）。
     */
    fun shouldRequestExemption(
        isIgnoringBatteryOptimizations: Boolean,
        alreadyAskedThisInstall: Boolean,
    ): Boolean = !isIgnoringBatteryOptimizations && !alreadyAskedThisInstall
}

object WorkerWakeLockPolicy {
    /**
     * 续租窗口：每次获取/续租后的最长持有时间（10 分钟）。
     *
     * 比起固定 30 分钟，它是「**持有期间必须能被续租**」的语义：有任务时会不断续，
     * 任务结束后不再续 ⇒ 窗口自然到期释放，而不是靠一个长超时硬撑。
     */
    const val RENEWAL_TIMEOUT_MS: Long = 10 * 60 * 1000L

    /**
     * 本时刻是否应当持有 WakeLock：**有活动任务才持有**。
     *
     * 空闲时释放让系统按自身省电策略休眠（前台通知仍然在），而不是固定 30 分钟硬撑 ——
     * 审计点名的正是「固定 30 分钟」这条。
     */
    fun shouldHold(activeTasks: Int): Boolean = activeTasks > 0
}
