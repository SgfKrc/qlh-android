package com.qlh.inference.ui

import androidx.compose.ui.test.assertIsDisplayed
import androidx.compose.ui.test.assertIsNotSelected
import androidx.compose.ui.test.assertIsSelected
import androidx.compose.ui.test.junit4.createComposeRule
import androidx.compose.ui.test.onNodeWithTag
import androidx.compose.ui.test.onNodeWithText
import androidx.compose.ui.test.performClick
import androidx.compose.ui.unit.Density
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.runtime.CompositionLocalProvider
import com.qlh.inference.data.SessionEntity
import com.qlh.inference.ui.theme.QlhTheme
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test

class SessionListScreenTest {
    @get:Rule
    val composeRule = createComposeRule()

    private val sessions = listOf(
        SessionEntity(id = 1L, title = "当前会话", messageCount = 2),
        SessionEntity(id = 2L, title = "待处理会话", messageCount = 1),
    )

    @Test
    fun selectedStateAndSessionActionsEmitExpectedIds() {
        val selectedIds = mutableListOf<Long>()
        var created = false
        setSessionContent(
            onSessionClick = { selectedIds += it },
            onCreateSession = { created = true },
        )

        composeRule.onNodeWithTag("session_list_screen").assertIsDisplayed()
        composeRule.onNodeWithTag("session_card_1").assertIsSelected()
        composeRule.onNodeWithTag("session_card_2").assertIsNotSelected().performClick()
        composeRule.onNodeWithTag("session_create").performClick()

        composeRule.runOnIdle {
            assertEquals(listOf(2L), selectedIds)
            assertTrue(created)
        }
    }

    @Test
    fun deletingSessionRequiresConfirmationAndEmitsTargetId() {
        val deletedIds = mutableListOf<Long>()
        setSessionContent(onDeleteSession = { deletedIds += it })

        composeRule.onNodeWithTag("session_delete_2").performClick()
        composeRule.onNodeWithText("删除会话").assertIsDisplayed()
        composeRule.onNodeWithText("删除").performClick()

        composeRule.runOnIdle {
            assertEquals(listOf(2L), deletedIds)
        }
    }

    @Test
    fun largeFontScaleKeepsSessionActionsReachable() {
        composeRule.setContent {
            CompositionLocalProvider(LocalDensity provides Density(1f, 1.6f)) {
                QlhTheme(darkTheme = false) {
                    SessionListScreen(
                        sessions = sessions,
                        currentSessionId = 1L,
                        onSessionClick = {},
                        onCreateSession = {},
                        onDeleteSession = {},
                    )
                }
            }
        }

        composeRule.onNodeWithTag("session_list_screen").assertIsDisplayed()
        composeRule.onNodeWithTag("session_card_1").assertIsDisplayed()
        composeRule.onNodeWithTag("session_create").assertIsDisplayed()
    }

    private fun setSessionContent(
        onSessionClick: (Long) -> Unit = {},
        onCreateSession: () -> Unit = {},
        onDeleteSession: (Long) -> Unit = {},
    ) {
        composeRule.setContent {
            QlhTheme(darkTheme = false) {
                SessionListScreen(
                    sessions = sessions,
                    currentSessionId = 1L,
                    onSessionClick = onSessionClick,
                    onCreateSession = onCreateSession,
                    onDeleteSession = onDeleteSession,
                )
            }
        }
    }
}
