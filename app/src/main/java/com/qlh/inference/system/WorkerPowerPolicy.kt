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

/**
 * ★ 2026-10-08：worker 的 **WiFi lock 策略**（纯逻辑，可 JVM 单测）。
 *
 * 由来（实测）：Y700 的局域网 ICMP 延迟是 `9 / 23 / 39 / 71 / 79 ms`（抖动 3–8×），
 * 而 decode **每步要一个网络往返** ⇒ 息屏/空闲时 WiFi 进入省电后的唤醒延迟，会直接变成
 * 每一步的固定税（对照：同网段 Surface 的 TCP RTT 稳定 6.3ms）。带宽不是问题
 * （11ax / RSSI −28 / Link 154Mbps）⇒ 这里要保的是**延迟**，不是吞吐。
 *
 * 与 [WorkerWakeLockPolicy] **同生命周期**：有任务才持有、空闲即释放 ——
 * 常驻持有 WiFi lock 会反向耗电，且与"空闲让系统休眠"的既有取向冲突。
 */
object WorkerWifiLockPolicy {
    /** 锁标签（`WifiManager.createWifiLock(mode, tag)`）。 */
    const val LOCK_TAG: String = "QLH:LayerWorkerWifi"

    /**
     * 空闲释放窗口（60 秒）：最后一次层段执行之后再无任务，才释放 WiFi lock。
     *
     * ★ 为什么**不能**与 WakeLock 一样"任务计数归零即释放"（实测踩到）：
     * 单次层段执行只持续几十毫秒，而 WiFi 的省电模式切换（PS ⇄ CAM）本身要花时间；
     * 更关键的是**下一个 offer 的到达时刻才决定"网络活跃期"** —— 步与步之间的等待
     * 同样依赖 WiFi 低延迟。若按"每次执行成对开关"，锁在 99% 的时间里都是松的
     * （现场验证：请求进行中 `dumpsys power` 的 `Wake Locks: size=0`，抓不到锁）。
     * 因此 WiFi 采用"**续租 + 空闲超时**"：连续工作时一直持有，真空闲才放手。
     */
    const val IDLE_RELEASE_MS: Long = 60_000L

    /**
     * 本时刻是否应当持有 WiFi lock（获取门槛）：与 WakeLock 同一判据（有活动任务）。
     *
     * 与 WakeLock **分叉点**：这里只决定"是否获取"，**释放**由 [IDLE_RELEASE_MS] 超时驱动，
     * 而不是任务计数归零 —— 见上面的实测理由。
     */
    fun shouldHold(activeTasks: Int): Boolean = activeTasks > 0
}
