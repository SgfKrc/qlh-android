package com.qlh.inference.system

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * ★ 2026-10-07（DIST-NEXT-5b）：worker 电源策略（电池优化引导 + WakeLock 持有规则）。
 *
 * 审计 P1-3：三个前台服务没有电池优化引导；`InferenceService` 的 WakeLock 固定 30 分钟。
 */
class WorkerPowerPolicyTest {
    @Test
    fun `battery exemption is requested once and never when already exempt`() {
        // 还没问过、且未豁免 ⇒ 申请一次
        assertTrue(
            WorkerBatteryPolicy.shouldRequestExemption(
                isIgnoringBatteryOptimizations = false,
                alreadyAskedThisInstall = false,
            ),
        )
        // 已经豁免 ⇒ 不再打扰
        assertFalse(
            WorkerBatteryPolicy.shouldRequestExemption(
                isIgnoringBatteryOptimizations = true,
                alreadyAskedThisInstall = false,
            ),
        )
        // 本次安装内已问过（用户拒绝也算）⇒ 不再弹系统对话框
        assertFalse(
            WorkerBatteryPolicy.shouldRequestExemption(
                isIgnoringBatteryOptimizations = false,
                alreadyAskedThisInstall = true,
            ),
        )
        assertFalse(
            WorkerBatteryPolicy.shouldRequestExemption(
                isIgnoringBatteryOptimizations = true,
                alreadyAskedThisInstall = true,
            ),
        )
    }

    @Test
    fun `exemption action is the platform one`() {
        assertEquals(
            "android.settings.REQUEST_IGNORE_BATTERY_OPTIMIZATIONS",
            WorkerBatteryPolicy.ACTION_REQUEST_IGNORE_BATTERY_OPTIMIZATIONS,
        )
    }

    @Test
    fun `wake lock is held only while there is active work`() {
        // 空闲不持有（让系统按自身策略休眠；前台通知仍在）
        assertFalse(WorkerWakeLockPolicy.shouldHold(0))
        assertTrue(WorkerWakeLockPolicy.shouldHold(1))
        assertTrue(WorkerWakeLockPolicy.shouldHold(7))
    }

    @Test
    fun `renewal window replaces the fixed thirty minute hold`() {
        // 续租窗口必须存在且明显短于旧的固定 30 分钟 —— 它靠"有任务才续租"而不是长超时硬撑
        assertTrue(WorkerWakeLockPolicy.RENEWAL_TIMEOUT_MS > 0L)
        assertTrue(WorkerWakeLockPolicy.RENEWAL_TIMEOUT_MS <= 15L * 60 * 1000)
        assertTrue(WorkerWakeLockPolicy.RENEWAL_TIMEOUT_MS < 30L * 60 * 1000)
    }

    // ------------------------------------------------------------ WiFi lock（2026-10-08）

    @Test
    fun `wifi lock is held only while there is active work`() {
        // 与 WakeLock 同一取向：空闲释放，常驻持有会反向耗电
        assertFalse(WorkerWifiLockPolicy.shouldHold(0))
        assertTrue(WorkerWifiLockPolicy.shouldHold(1))
        assertTrue(WorkerWifiLockPolicy.shouldHold(7))
    }

    @Test
    fun `wifi lock and wake lock share the same hold predicate`() {
        // 两者必须同进同出：任何分叉都会留下"锁在跑但网络被省电"或"反向耗电"的状态
        for (tasks in listOf(0, 1, 3, 12)) {
            assertEquals(
                WorkerWakeLockPolicy.shouldHold(tasks),
                WorkerWifiLockPolicy.shouldHold(tasks),
            )
        }
    }

    @Test
    fun `wifi lock tag is stable and namespaced`() {
        // 标签进 dumpsys power；改名会让"是否真持有"的现场核对失效
        assertEquals("QLH:LayerWorkerWifi", WorkerWifiLockPolicy.LOCK_TAG)
    }

    @Test
    fun `wifi lock releases only after an idle window`() {
        // ★ 2026-10-08 实测教训：单次层段执行只持续几十毫秒，若与 WakeLock 一样"计数归零即释放"，
        //   锁在请求进行中也抓不到（dumpsys power 显示 Wake Locks: size=0）⇒ 必须留空闲窗口。
        assertTrue(
            "空闲窗口必须显著长于单次层段执行（数十毫秒）",
            WorkerWifiLockPolicy.IDLE_RELEASE_MS >= 30_000L,
        )
        assertTrue(
            "但也不能常驻持有（否则反向耗电）",
            WorkerWifiLockPolicy.IDLE_RELEASE_MS <= 5L * 60 * 1000,
        )
    }
}
