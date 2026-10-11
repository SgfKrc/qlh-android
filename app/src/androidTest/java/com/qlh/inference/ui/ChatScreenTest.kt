package com.qlh.inference.ui

import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.ui.test.assertIsDisplayed
import androidx.compose.ui.test.assertIsEnabled
import androidx.compose.ui.test.assertIsNotEnabled
import androidx.compose.ui.test.assertTextEquals
import androidx.compose.ui.test.junit4.createEmptyComposeRule
import androidx.compose.ui.test.onNodeWithTag
import androidx.compose.ui.test.onNodeWithText
import androidx.compose.ui.test.performClick
import androidx.compose.ui.test.performTextInput
import androidx.test.core.app.ActivityScenario
import com.qlh.inference.ui.theme.QlhTheme
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Rule
import org.junit.Test

class ChatScreenTest {
    /**
     * ★ 2026-10-11：改用 `createEmptyComposeRule()` + 自建宿主，**不**用 `createComposeRule()`。
     * 原因：后者经 `ActivityScenarioRule`，其 `after()` 调 `ActivityScenario.close()`，而 close
     * 会经 `InstrumentationActivityInvoker` 启动 `EmptyActivity` —— 本设备上恒抛
     * `ActivityNotFoundException` 并**替换掉测试体的断言**（详见
     * `docs/彻查后修复与重构排票计划-2026-10-09.md` 票12 的设备门小节）。
     */
    @get:Rule
    val composeRule = createEmptyComposeRule()

    private lateinit var scenario: ActivityScenario<ComponentActivity>

    @Before
    fun launchHost() {
        scenario = ActivityScenario.launch(ComponentActivity::class.java)
    }

    @After
    fun finishHost() {
        // 不调 close()：那正是触发 invoker activity 的那一步；直接 finish() 达到同样清理效果。
        if (::scenario.isInitialized) {
            runCatching { scenario.onActivity { it.finish() } }
        }
    }

    @Test
    fun inputTrimsAndSendsMessageThenClearsDraft() {
        val sentMessages = mutableListOf<String>()
        setChatContent(onSendMessage = { sentMessages += it })

        composeRule.onNodeWithTag("chat_input").performTextInput("  你好 QLH  ")
        composeRule.onNodeWithTag("chat_send").assertIsEnabled().performClick()

        composeRule.runOnIdle {
            assertEquals(listOf("你好 QLH"), sentMessages)
        }
        composeRule.onNodeWithTag("chat_input").assertTextEquals("")
    }

    @Test
    fun loadingDisablesInputAndSendAction() {
        setChatContent(isLoading = true)

        composeRule.onNodeWithTag("chat_screen").assertIsDisplayed()
        composeRule.onNodeWithTag("chat_input").assertIsNotEnabled()
        composeRule.onNodeWithTag("chat_send").assertIsNotEnabled()
        composeRule.onNodeWithText("思考中…").assertIsDisplayed()
    }

    @Test
    fun retryInvokesRetryAndClearsVisibleError() {
        var retried = false
        var cleared = false
        setChatContent(
            error = "连接超时",
            onRetry = { retried = true },
            onClearError = { cleared = true },
        )

        composeRule.onNodeWithText("重试").assertIsDisplayed().performClick()

        composeRule.runOnIdle {
            assertTrue(retried)
            assertTrue(cleared)
        }
    }

    @Test
    fun imagePickerIsEnabledOnlyInThinMode() {
        setChatContent(inferenceMode = "thin")
        composeRule.onNodeWithTag("chat_image_pick").assertIsEnabled()

        setChatContent(inferenceMode = "full")
        composeRule.onNodeWithTag("chat_image_pick").assertIsNotEnabled()
    }

    private fun setChatContent(
        isLoading: Boolean = false,
        error: String? = null,
        onSendMessage: (String) -> Unit = {},
        onRetry: () -> Unit = {},
        onClearError: () -> Unit = {},
        inferenceMode: String = "thin",
    ) {
        scenario.onActivity { activity ->
            activity.setContent {
                QlhTheme(darkTheme = false) {
                    ChatScreen(
                        sessionId = 1L,
                        sessionTitle = "测试会话",
                        messages = emptyList(),
                        isLoading = isLoading,
                        error = error,
                        onSendMessage = onSendMessage,
                        onRetry = onRetry,
                        onClearError = onClearError,
                        inferenceMode = inferenceMode,
                    )
                }
            }
        }
    }
}
