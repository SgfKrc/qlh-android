package com.qlh.inference.ui

import androidx.compose.ui.test.junit4.createAndroidComposeRule
import androidx.compose.ui.test.onAllNodesWithTag
import androidx.compose.ui.test.onNodeWithTag
import androidx.compose.ui.test.performClick
import androidx.compose.ui.test.performTextInput
import androidx.compose.ui.test.printToString
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.qlh.inference.MainActivity
import com.qlh.inference.data.SettingsDataStore
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.runBlocking
import org.json.JSONObject
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Rule
import org.junit.Test
import org.junit.rules.RuleChain
import org.junit.rules.TestRule
import org.junit.runner.RunWith
import org.junit.runners.model.Statement
import java.net.HttpURLConnection
import java.net.URL

/**
 * ★ 安卓侧「从**聊天输入框**发起真实请求」的端到端档（此前为空缺）。
 *
 * 缺口背景（实测核实）：`ChatScreenTest` 只把 `onSendMessage` 换成记录用的 lambda，**没有
 * HTTP、没有回答**；JVM 侧最高只覆盖到 `ApiClientContractTest` 的本机 mock socket。即
 * 「APK 内 输入框 → ChatRepository → POST /api/chat → 真回答」这一段**从无自动化证据**，
 * 既有的安卓分布式实测都是**主机侧**发起的（安卓只被证明是 worker，未被证明是 chat client）。
 *
 * 本档补的正是这一段：真机 + 真后端 + 真集群，操作**只走 UI**（不绕过输入框直调 API），
 * 断言口径与 TUI 档 `tests/test_tui_e2e_flow_distributed.py` 对齐：
 *   1. 状态行必须给出**分布式承层证据**（`分布式 ✓（N 段·层 x-y）`）—— 它是
 *      `distributed_used=true` 经 `formatMetrics` 的呈现，同时意味着该回答已回来；
 *   2. 该承层区间必须与 `/api/cluster/pipeline-capacity` 规划的**远端（非 master）段有交叠**
 *      —— 专治"本地假装分布式"（若实际是本地整模，区间会落在 master 那段上、不相交）。
 *
 * ## 运行方式（**默认 skip**，需显式开关）
 *
 * 前置：主机 `api_server` 已加载模型、集群 ≥2 节点 online、capacity 可解，且设备能访问
 * App 里配置的 `serverHost:serverPort`（默认 `SettingsDataStore.DEFAULT_HOST:8000`）。
 *
 * ```
 * .\gradlew.bat :app:connectedDebugAndroidTest `
 *   -Pandroid.testInstrumentationRunnerArguments.class=com.qlh.inference.ui.ChatE2EInstrumentedTest `
 *   -Pandroid.testInstrumentationRunnerArguments.qlhE2E=1
 * ```
 *
 * 纪律（与仓库既有 TUI 档一致）：缺开关 / 缺集群 / 缺后端 ⇒ **一律 skip，不用 xfail、
 * 不静默 pass**；不虚构"跑了分布式"。
 */
@RunWith(AndroidJUnit4::class)
class ChatE2EInstrumentedTest {

    private val targetContext = InstrumentationRegistry.getInstrumentation().targetContext

    private fun settings(): SettingsDataStore = SettingsDataStore(targetContext)

    /** 设备上要访问的后端（取 App 当前配置，尊重既有设置而不是硬编码）。 */
    private fun configuredBackend(): Pair<String, Int> = runBlocking {
        val store = settings()
        store.serverHost.first() to store.serverPort.first()
    }

    /**
     * Activity 启动**之前**：确保推理模式走网络（否则 `apiClient()` 为空、请求走本地
     * llama.cpp，本档就测不到分布式），并把生成长度压小（真分布式实测约 2 tok/s，
     * 默认 1024 会让本档等数分钟）。**结束后恢复原值**，不把测试偏好留在设备上。
     */
    private val prepareDeviceRule = TestRule { base, _ ->
        object : Statement() {
            override fun evaluate() {
                val store = settings()
                val originalMode = runBlocking { store.inferenceMode.first() }
                val originalMaxTokens = runBlocking { store.maxTokens.first() }
                runBlocking {
                    if (originalMode != SettingsDataStore.MODE_DISTRIBUTED) {
                        store.setInferenceMode(SettingsDataStore.MODE_DISTRIBUTED)
                    }
                    store.setMaxTokens(E2E_MAX_TOKENS)
                }
                try {
                    base.evaluate()
                } finally {
                    runCatching {
                        runBlocking {
                            store.setInferenceMode(originalMode)
                            store.setMaxTokens(originalMaxTokens)
                        }
                    }
                }
            }
        }
    }

    private val composeRule = createAndroidComposeRule<MainActivity>()

    @get:Rule
    val rules: RuleChain = RuleChain.outerRule(prepareDeviceRule).around(composeRule)

    @Test
    fun chatFromInputBoxReachesDistributedPipeline() {
        assumeTrue(
            "需显式开启：-Pandroid.testInstrumentationRunnerArguments.qlhE2E=1（真后端 + 真集群档）",
            InstrumentationRegistry.getArguments().getString("qlhE2E") == "1",
        )

        val (host, port) = configuredBackend()
        val base = "http://$host:$port"

        // ---- 前置（只读探测）：后端可达 / 集群就绪 / capacity 可解；任一不满足即 skip ----
        val health = httpGet("$base/api/health")
        assumeTrue("后端不可达：$base/api/health -> HTTP ${health.first}", health.first == 200)

        val capacityRaw = httpGet("$base/api/cluster/pipeline-capacity")
        assumeTrue(
            "capacity 端点不可用：$base/api/cluster/pipeline-capacity -> HTTP ${capacityRaw.first}",
            capacityRaw.first == 200,
        )
        val capacity = JSONObject(capacityRaw.second)
        assumeTrue(
            "capacity 未给出可用计划（reason=${capacity.optString("reason_code")}）" +
                "⇒ 集群未就绪，本档 skip（不当失败）",
            capacity.optBoolean("admitted", false),
        )
        val remoteRanges = remoteRangesOf(capacity)
        assumeTrue(
            "capacity 没有任何**远端**（非 master）段 ⇒ 无从验证'不是本地假装分布式'",
            remoteRanges.isNotEmpty(),
        )

        // ---- 操作：只走 UI（聊天输入框 + 发送按钮）----
        composeRule.waitUntil(TIMEOUT_MS) {
            composeRule.onAllNodesWithTag("chat_input").fetchSemanticsNodes().isNotEmpty()
        }
        composeRule.onNodeWithTag("chat_screen").assertExists()
        composeRule.onNodeWithTag("chat_input").performTextInput(PROMPT)
        composeRule.onNodeWithTag("chat_send").performClick()

        // ---- 判据：状态行必须出现分布式承层证据（= 回答已回来且 distributed_used=true）----
        composeRule.waitUntil(TIMEOUT_MS) {
            semanticsDump().contains(DISTRIBUTED_MARK)
        }
        val dump = semanticsDump()
        val evidence = EVIDENCE_PATTERN.find(dump)
        assertTrue(
            "聊天屏语义树里未出现带承层证据的分布式状态行（形如 '分布式 ✓（2 段·层 0-24）'）" +
                "；已渲染内容（前 600 字）: ${dump.take(600)}",
            evidence != null,
        )
        val segStart = evidence!!.groupValues[1].toInt()
        val segEnd = evidence.groupValues[2].toInt()
        assertTrue("承层区间非法：$segStart-$segEnd", segStart < segEnd)

        // ---- 交叉核验：承层区间必须与 capacity 的远端段有交叠（防本地假装分布式）----
        assertTrue(
            "状态行承层区间 $segStart-$segEnd 与 capacity 远端段 $remoteRanges 无交叠 " +
                "⇒ UI 声称的分布式与规划不符（疑似本地整模在跑）",
            remoteRanges.any { (start, end) -> !(segEnd <= start || segStart >= end) },
        )
    }

    /**
     * 聊天屏的语义树 dump（**未合并树**，含各处 `Text`）。
     *
     * ⚠️ 为什么不用 `SemanticsConfiguration.getOrNull` 取文本：它不是公开 API（编译不过）。
     * `printToString()` 是公开扩展，拿到的 dump 足以用正则精确匹配状态行片段。
     */
    private fun semanticsDump(): String =
        composeRule.onNodeWithTag("chat_screen", useUnmergedTree = true).printToString()

    private fun remoteRangesOf(capacity: JSONObject): List<Pair<Int, Int>> {
        val assignments = capacity.optJSONArray("assignments") ?: return emptyList()
        return (0 until assignments.length()).mapNotNull { index ->
            val item = assignments.optJSONObject(index) ?: return@mapNotNull null
            if (item.optString("node_id") == "master") return@mapNotNull null
            if (!item.has("start_layer") || !item.has("end_layer")) return@mapNotNull null
            item.optInt("start_layer") to item.optInt("end_layer")
        }
    }

    private fun httpGet(url: String): Pair<Int, String> {
        val conn = (URL(url).openConnection() as HttpURLConnection).apply {
            requestMethod = "GET"
            connectTimeout = 8_000
            readTimeout = 15_000
        }
        return try {
            val code = conn.responseCode
            val stream = if (code in 200..299) conn.inputStream else conn.errorStream
            val body = stream?.bufferedReader(Charsets.UTF_8)?.use { it.readText() }.orEmpty()
            code to body
        } catch (error: Exception) {
            -1 to (error.message ?: error.javaClass.simpleName)
        } finally {
            runCatching { conn.disconnect() }
        }
    }

    private companion object {
        const val PROMPT = "用一句话说明分布式推理的意义。"
        const val DISTRIBUTED_MARK = "分布式 ✓"
        const val E2E_MAX_TOKENS = 24
        const val TIMEOUT_MS = 300_000L
        /** 完整状态行片段：`分布式 ✓（2 段·层 0-24）`。 */
        val EVIDENCE_PATTERN = Regex("""分布式\s*✓\s*（\s*(\d+)\s*段·层\s*(\d+)-(\d+)\s*）""")
    }
}
