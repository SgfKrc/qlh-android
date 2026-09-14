package com.qlh.inference.system

import android.app.ActivityManager
import android.content.Context
import android.opengl.EGL14
import android.opengl.EGLConfig
import android.opengl.EGLContext
import android.opengl.EGLDisplay
import android.opengl.EGLSurface
import android.opengl.GLES20
import android.os.Build
import android.os.PowerManager
import android.os.StatFs
import com.qlh.inference.status.GpuStatus
import com.qlh.inference.status.MemoryStatus
import com.qlh.inference.status.StorageStatus
import com.qlh.inference.status.SystemStatus

class AndroidDeviceInfoProvider(private val context: Context) {

    fun getSystemStatus(): SystemStatus {
        val powerManager = context.getSystemService(Context.POWER_SERVICE) as? PowerManager
        return SystemStatus(
            manufacturer = Build.MANUFACTURER.orEmpty(),
            brand = Build.BRAND.orEmpty(),
            model = Build.MODEL.orEmpty(),
            device = Build.DEVICE.orEmpty(),
            hardware = Build.HARDWARE.orEmpty(),
            socManufacturer = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
                Build.SOC_MANUFACTURER.orEmpty()
            } else {
                ""
            },
            socModel = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
                Build.SOC_MODEL.orEmpty()
            } else {
                ""
            },
            sdkInt = Build.VERSION.SDK_INT,
            androidRelease = Build.VERSION.RELEASE.orEmpty(),
            abis = Build.SUPPORTED_ABIS?.toList().orEmpty(),
            cpuCores = Runtime.getRuntime().availableProcessors(),
            powerSaveMode = powerManager?.isPowerSaveMode == true,
            thermalStatus = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                thermalStatusName(powerManager?.currentThermalStatus ?: PowerManager.THERMAL_STATUS_NONE)
            } else {
                "unsupported"
            }
        )
    }

    fun getMemoryStatus(): MemoryStatus {
        val am = context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
        val mem = ActivityManager.MemoryInfo()
        am.getMemoryInfo(mem)
        val runtime = Runtime.getRuntime()
        return MemoryStatus(
            availableBytes = mem.availMem,
            totalBytes = mem.totalMem,
            thresholdBytes = mem.threshold,
            lowMemory = mem.lowMemory,
            heapMaxBytes = runtime.maxMemory(),
            heapTotalBytes = runtime.totalMemory(),
            heapFreeBytes = runtime.freeMemory(),
            lowRamDevice = am.isLowRamDevice
        )
    }

    fun getStorageStatus(): StorageStatus {
        val files = statFs(context.filesDir.absolutePath)
        val cache = statFs(context.cacheDir.absolutePath)
        return StorageStatus(
            filesAvailableBytes = files.first,
            filesTotalBytes = files.second,
            cacheAvailableBytes = cache.first,
            cacheTotalBytes = cache.second
        )
    }

    fun getGpuStatus(): GpuStatus {
        return try {
            val gl = probeOpenGl()
            GpuStatus(
                vendor = gl.vendor,
                renderer = gl.renderer,
                version = gl.version,
                probeError = null
            )
        } catch (e: Exception) {
            GpuStatus(probeError = e.message ?: e.javaClass.simpleName)
        }
    }

    private fun statFs(path: String): Pair<Long, Long> {
        return try {
            val stat = StatFs(path)
            stat.availableBytes to stat.totalBytes
        } catch (_: Exception) {
            0L to 0L
        }
    }

    private fun probeOpenGl(): GlInfo {
        val display = EGL14.eglGetDisplay(EGL14.EGL_DEFAULT_DISPLAY)
        if (display == EGL14.EGL_NO_DISPLAY) throw IllegalStateException("EGL display unavailable")
        val versions = IntArray(2)
        if (!EGL14.eglInitialize(display, versions, 0, versions, 1)) {
            throw IllegalStateException("EGL init failed")
        }

        var contextHandle: EGLContext = EGL14.EGL_NO_CONTEXT
        var surface: EGLSurface = EGL14.EGL_NO_SURFACE
        try {
            val configs = arrayOfNulls<EGLConfig>(1)
            val numConfigs = IntArray(1)
            val configAttribs = intArrayOf(
                EGL14.EGL_RENDERABLE_TYPE, EGL14.EGL_OPENGL_ES2_BIT,
                EGL14.EGL_RED_SIZE, 8,
                EGL14.EGL_GREEN_SIZE, 8,
                EGL14.EGL_BLUE_SIZE, 8,
                EGL14.EGL_NONE
            )
            if (!EGL14.eglChooseConfig(display, configAttribs, 0, configs, 0, 1, numConfigs, 0)) {
                throw IllegalStateException("EGL config unavailable")
            }
            val config = configs[0] ?: throw IllegalStateException("EGL config is null")
            val contextAttribs = intArrayOf(EGL14.EGL_CONTEXT_CLIENT_VERSION, 2, EGL14.EGL_NONE)
            contextHandle = EGL14.eglCreateContext(display, config, EGL14.EGL_NO_CONTEXT, contextAttribs, 0)
            if (contextHandle == EGL14.EGL_NO_CONTEXT) throw IllegalStateException("EGL context failed")

            val surfaceAttribs = intArrayOf(EGL14.EGL_WIDTH, 1, EGL14.EGL_HEIGHT, 1, EGL14.EGL_NONE)
            surface = EGL14.eglCreatePbufferSurface(display, config, surfaceAttribs, 0)
            if (surface == EGL14.EGL_NO_SURFACE) throw IllegalStateException("EGL surface failed")
            if (!EGL14.eglMakeCurrent(display, surface, surface, contextHandle)) {
                throw IllegalStateException("EGL make current failed")
            }

            return GlInfo(
                vendor = GLES20.glGetString(GLES20.GL_VENDOR).orEmpty(),
                renderer = GLES20.glGetString(GLES20.GL_RENDERER).orEmpty(),
                version = GLES20.glGetString(GLES20.GL_VERSION).orEmpty()
            )
        } finally {
            EGL14.eglMakeCurrent(display, EGL14.EGL_NO_SURFACE, EGL14.EGL_NO_SURFACE, EGL14.EGL_NO_CONTEXT)
            if (surface != EGL14.EGL_NO_SURFACE) EGL14.eglDestroySurface(display, surface)
            if (contextHandle != EGL14.EGL_NO_CONTEXT) EGL14.eglDestroyContext(display, contextHandle)
            EGL14.eglTerminate(display)
        }
    }

    private fun thermalStatusName(status: Int): String {
        return when (status) {
            PowerManager.THERMAL_STATUS_NONE -> "none"
            PowerManager.THERMAL_STATUS_LIGHT -> "light"
            PowerManager.THERMAL_STATUS_MODERATE -> "moderate"
            PowerManager.THERMAL_STATUS_SEVERE -> "severe"
            PowerManager.THERMAL_STATUS_CRITICAL -> "critical"
            PowerManager.THERMAL_STATUS_EMERGENCY -> "emergency"
            PowerManager.THERMAL_STATUS_SHUTDOWN -> "shutdown"
            else -> "unknown"
        }
    }

    private data class GlInfo(
        val vendor: String,
        val renderer: String,
        val version: String,
    )
}
