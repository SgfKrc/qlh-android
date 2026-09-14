package com.qlh.inference.logging

import android.content.Context
import android.util.Log
import java.io.File
import java.io.FileWriter
import java.io.IOException
import java.io.RandomAccessFile
import java.time.Instant
import java.time.LocalDate
import java.time.LocalDateTime
import java.time.ZoneId
import java.time.format.DateTimeFormatter
import java.util.Date
import java.util.concurrent.Executors

/**
 * 应用内文件日志单例。
 *
 * 同时输出到 Logcat（android.util.Log）和 filesDir/logs/ 下的日期滚动文件。
 * 写文件在单线程 executor 中完成，不阻塞调用线程。
 *
 * ## 日志级别约定（与 PC 端 Python logging 对齐）
 *
 * | Android     | PC Python        | 说明 |
 * |-------------|------------------|------|
 * | `V` (VERBOSE) | `DEBUG` (10)   | 详细诊断信息，仅开发时启用 |
 * | `D` (DEBUG)   | `DEBUG` (10)   | 调试信息 |
 * | `I` (INFO)    | `INFO` (20)    | 常规运行时信息 |
 * | `W` (WARNING)  | `WARNING` (30) | 可恢复的异常情况 |
 * | `E` (ERROR)   | `ERROR` (40)   | 错误，需要关注 |
 * | `crash()`     | `CRITICAL` (50)| 崩溃级，同步写入 |
 *
 * PC 端 `CRITICAL` (50) 在 Android 端无直接对应，使用 `e()` + `crash()` 兜底。
 *
 * 使用方式：
 *   QlhLogger.init(context)  // Application.onCreate() 中调用一次
 *   QlhLogger.i(TAG, "message")
 *   QlhLogger.e(TAG, "error", throwable)
 */
object QlhLogger {

    private const val MAX_LOG_SIZE = 5 * 1024 * 1024L   // 5 MB 滚动
    private const val BACKUP_COUNT = 5
    const val READ_MAX_BYTES = 500 * 1024L              // 读取上限 500 KB（public 供 UI 截断提示用）
    private const val TAG = "QlhLogger"
    private val logFileRegex = Regex("^[^/\\\\]+\\.log(?:\\.\\d+)?$")

    private var logDir: File? = null
    private var writer: FileWriter? = null
    @Volatile
    private var currentDateStr: String = ""

    private val executor = Executors.newSingleThreadExecutor { r ->
        Thread(r, "qlh-logger").apply { isDaemon = true }
    }

    private fun isLogFile(file: File): Boolean = logFileRegex.matches(file.name)

    // DateTimeFormatter 是不可变对象，天然线程安全（替代 SimpleDateFormat）
    private val timestampFmt = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss")
    private val dateFmt = DateTimeFormatter.ofPattern("yyyy-MM-dd")

    // ──────────────────────────────────────────────
    // 初始化
    // ──────────────────────────────────────────────

    /** 必须在 Application.onCreate() 中调用一次。 */
    @Synchronized
    fun init(context: Context) {
        if (logDir != null) return
        logDir = File(context.filesDir, "logs").also { it.mkdirs() }
        openCurrentFile()
    }

    private fun getCurrentLogFile(): File {
        val date = dateFmt.format(LocalDate.now())
        currentDateStr = date
        return File(requireNotNull(logDir), "qlh-$date.log")
    }

    @Synchronized
    private fun openCurrentFile() {
        if (logDir == null) return
        try {
            writer?.close()
        } catch (_: IOException) {
        }
        writer = null

        val file = getCurrentLogFile()
        if (file.exists() && file.length() > MAX_LOG_SIZE) {
            // 大小滚动：shift 旧备份
            for (i in BACKUP_COUNT downTo 2) {
                val from = File(logDir, "qlh-${currentDateStr}.${i - 1}.log")
                val to = File(logDir, "qlh-${currentDateStr}.$i.log")
                if (to.exists()) to.delete()
                if (from.exists()) from.renameTo(to)
            }
            val first = File(logDir, "qlh-${currentDateStr}.1.log")
            if (first.exists()) first.delete()
            file.renameTo(first)
        }

        try {
            writer = FileWriter(file, true) // append
        } catch (e: IOException) {
            Log.e(TAG, "无法打开日志文件: ${file.absolutePath}", e)
        }
    }

    // ──────────────────────────────────────────────
    // 公开日志 API
    // ──────────────────────────────────────────────

    fun v(tag: String, msg: String) {
        log('V', tag, msg)
        Log.v(tag, msg)
    }

    fun d(tag: String, msg: String) {
        log('D', tag, msg)
        Log.d(tag, msg)
    }

    fun i(tag: String, msg: String) {
        log('I', tag, msg)
        Log.i(tag, msg)
    }

    fun w(tag: String, msg: String) {
        log('W', tag, msg)
        Log.w(tag, msg)
    }

    fun e(tag: String, msg: String) {
        log('E', tag, msg)
        Log.e(tag, msg)
    }

    fun e(tag: String, msg: String, tr: Throwable) {
        log('E', tag, "$msg: ${tr.message}\n${Log.getStackTraceString(tr)}")
        Log.e(tag, msg, tr)
    }

    /** 崩溃兜底日志：同步写文件，避免进程退出前 executor 尚未 flush。 */
    @Synchronized
    fun crash(tag: String, msg: String, tr: Throwable) {
        try {
            if (logDir != null && writer == null) openCurrentFile()
            writer?.write(
                "[${timestampFmt.format(LocalDateTime.now())}] [E/$tag] " +
                    "$msg: ${tr.message}\n${Log.getStackTraceString(tr)}\n"
            )
            writer?.flush()
        } catch (_: Exception) {
        }
        Log.e(tag, msg, tr)
    }

    private fun log(level: Char, tag: String, msg: String) {
        executor.execute {
            try {
                val line = "[${timestampFmt.format(LocalDateTime.now())}] [$level/$tag] $msg\n"
                writer?.write(line)
                writer?.flush()

                // 跨日检查
                val today = dateFmt.format(LocalDate.now())
                if (today != currentDateStr) {
                    openCurrentFile()
                }
            } catch (_: IOException) {
            }
        }
    }

    // ──────────────────────────────────────────────
    // 日志文件管理
    // ──────────────────────────────────────────────

    data class LogFileInfo(
        val name: String,
        val size: Long,
        val modified: Date,
    )

    /** 日志读取结果，包含内容与是否截断信息。 */
    data class LogReadResult(
        val content: String?,
        val truncated: Boolean,
        val fileSize: Long,
    )

    /** 列出所有 .log 文件，按修改时间降序。 */
    fun getLogFiles(): List<LogFileInfo> {
        return logDir?.listFiles()
            ?.filter { isLogFile(it) }
            ?.sortedByDescending { it.lastModified() }
            ?.map { LogFileInfo(it.name, it.length(), Date(it.lastModified())) }
            ?: emptyList()
    }

    /** 读取日志文件内容（返回末 500 KB）。失败返回 null。 */
    fun readLogFile(name: String): String? {
        return readLogFileWithInfo(name).content
    }

    /**
     * 读取日志文件内容并返回截断信息。
     *
     * @param name 日志文件名
     * @return [LogReadResult] — content 为 null 表示读取失败；
     *         truncated=true 表示文件超过 [READ_MAX_BYTES]（500 KB），仅返回末尾部分。
     */
    fun readLogFileWithInfo(name: String): LogReadResult {
        val file = File(logDir, name)
        if (!file.exists() || !isLogFile(file)) {
            return LogReadResult(null, false, 0L)
        }
        val fileSize = file.length()
        return try {
            if (fileSize > READ_MAX_BYTES) {
                val raf = RandomAccessFile(file, "r")
                raf.seek(fileSize - READ_MAX_BYTES)
                val bytes = ByteArray(READ_MAX_BYTES.toInt())
                val n = raf.read(bytes)
                raf.close()
                LogReadResult(
                    content = String(bytes, 0, n, Charsets.UTF_8),
                    truncated = true,
                    fileSize = fileSize,
                )
            } else {
                LogReadResult(
                    content = file.readText(),
                    truncated = false,
                    fileSize = fileSize,
                )
            }
        } catch (_: Exception) {
            LogReadResult(null, false, fileSize)
        }
    }

    /** Return a bounded, credential-redacted bundle suitable for copy/share/upload. */
    fun readRedactedLogBundle(maxBytes: Long = 1_000_000L): String {
        val limit = maxBytes.coerceIn(4_096L, 2_000_000L)
        val out = StringBuilder()
        for (file in getLogFiles()) {
            val result = readLogFileWithInfo(file.name)
            val section = buildString {
                appendLine("===== ${file.name} (${file.size} bytes) =====")
                if (result.truncated) appendLine("[truncated: ${result.fileSize} bytes]")
                appendLine(result.content ?: "(read failed)")
                appendLine()
            }
            val redacted = redactDiagnosticText(section)
            if (out.length + redacted.length > limit) {
                val remaining = (limit - out.length).coerceAtLeast(0L).toInt()
                if (remaining > 0) out.append(redacted.take(remaining))
                break
            }
            out.append(redacted)
        }
        return out.toString()
    }

    /** Redact a log preview before it is rendered or shared outside the app. */
    fun redactDiagnosticText(value: String): String =
        com.qlh.inference.logging.redactDiagnosticText(value)

    /** 获取单个日志文件的 [LogFileInfo]（含 size 信息），不存在则返回 null。 */
    fun getLogFileInfo(name: String): LogFileInfo? {
        val file = File(logDir, name)
        return if (file.exists() && isLogFile(file)) {
            LogFileInfo(file.name, file.length(), Date(file.lastModified()))
        } else {
            null
        }
    }

    /** 删除所有 .log 文件并重新打开当前日志。 */
    @Synchronized
    fun clearLogs(): Boolean {
        return try {
            logDir?.listFiles()?.filter { isLogFile(it) }?.forEach { it.delete() }
            openCurrentFile()
            true
        } catch (_: Exception) {
            false
        }
    }
}
