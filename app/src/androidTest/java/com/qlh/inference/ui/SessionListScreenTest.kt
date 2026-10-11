package com.qlh.inference.ui

import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.ui.test.assertIsDisplayed
import androidx.compose.ui.test.assertIsNotSelected
import androidx.compose.ui.test.assertIsSelected
import androidx.compose.ui.test.junit4.createEmptyComposeRule
import androidx.compose.ui.test.onNodeWithTag
import androidx.compose.ui.test.onNodeWithText
import androidx.compose.ui.test.performClick
import androidx.compose.ui.unit.Density
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.runtime.CompositionLocalProvider
import androidx.test.core.app.ActivityScenario
import com.qlh.inference.data.SessionEntity
import com.qlh.inference.ui.theme.QlhTheme
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Rule
import org.junit.Test

class SessionListScreenTest {
    /**
     * ★ 2026-10-11：改用 `createEmptyComposeRule()` + 自建宿主，**不**用 `createComposeRule()`；
     * 后者经 `ActivityScenarioRule`，其 `close()` 会在本设备上抛 `ActivityNotFoundException`
     * 并替换测试体断言（详见 `docs/彻查后修复与重构排票计划-2026-10-09.md` 票12 设备门小节）。
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
        if (::scenario.isInitialized) {
            runCatching { scenario.onActivity { it.finish() } }
        }
    }

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
        scenario.onActivity { activity ->
            activity.setContent {
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
        scenario.onActivity { activity ->
            activity.setContent {
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
}
