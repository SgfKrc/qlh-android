package com.qlh.inference.ui

import androidx.compose.ui.test.assertIsDisplayed
import androidx.compose.ui.test.assertIsEnabled
import androidx.compose.ui.test.assertIsNotEnabled
import androidx.compose.ui.test.assertTextEquals
import androidx.compose.ui.test.junit4.createComposeRule
import androidx.compose.ui.test.onNodeWithTag
import androidx.compose.ui.test.onNodeWithText
import androidx.compose.ui.test.performClick
import androidx.compose.ui.test.performTextInput
import com.qlh.inference.ui.theme.QlhTheme
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test

class ChatScreenTest {
    @get:Rule
    val composeRule = createComposeRule()

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
        composeRule.setContent {
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
