package com.qlh.inference.ui

import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.ui.Modifier
import androidx.compose.ui.test.assertIsDisplayed
import androidx.compose.ui.test.assertIsEnabled
import androidx.compose.ui.test.junit4.createEmptyComposeRule
import androidx.compose.ui.test.onNodeWithTag
import androidx.compose.ui.test.onNodeWithText
import androidx.compose.ui.test.performClick
import androidx.compose.ui.test.performScrollTo
import androidx.compose.ui.test.performTextInput
import androidx.compose.ui.unit.dp
import androidx.test.core.app.ActivityScenario
import com.qlh.inference.ModelIntentUiState
import com.qlh.inference.network.ModelIntentBinding
import com.qlh.inference.network.ModelIntentCatalog
import com.qlh.inference.network.ModelIntentNode
import com.qlh.inference.network.ModelIntentOperation
import com.qlh.inference.network.ModelIntentRequest
import com.qlh.inference.network.ServerModelSummary
import com.qlh.inference.ui.theme.QlhTheme
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Before
import org.junit.Rule
import org.junit.Test

class ModelIntentPanelTest {
    @get:Rule val composeRule = createEmptyComposeRule()
    private lateinit var scenario: ActivityScenario<ComponentActivity>

    @Before fun launchHost() {
        scenario = ActivityScenario.launch(ComponentActivity::class.java)
    }

    @After fun finishHost() {
        // Match the existing SettingsScreenTest host cleanup without invoking EmptyActivity.
        if (::scenario.isInitialized) scenario.onActivity { it.finish() }
    }

    private fun state(status: String, loadedReady: Boolean = false): ModelIntentUiState {
        val intent = ModelIntentRequest("qwen3-5-9b", "llama_cpp")
        return ModelIntentUiState(
            catalog = ModelIntentCatalog(
                models = (0 until 20).map {
                    if (it == 9) ServerModelSummary(modelId = "qwen3-5-9b", name = "Qwen3.5-9B", isAvailable = false)
                    else ServerModelSummary(modelId = "model-$it", name = "Model $it", isAvailable = true)
                },
                binding = ModelIntentBinding("connected", true),
            ),
            operation = ModelIntentOperation(
                operationId = "supply_panel", intent = intent, status = status, loadedReady = loadedReady,
                reasonCode = if (status == "failed") "INTENT_SOURCE_UNAVAILABLE" else null,
                reason = if (status == "failed") "源模型获取失败" else null,
                nodes = mapOf("y700" to ModelIntentNode(status = "prepared", bytesReceived = 10, totalBytes = 10)),
            ),
            requestedIntent = intent,
        )
    }

    private fun setPanel(
        state: ModelIntentUiState,
        onSelect: (String, String) -> Unit = { _, _ -> },
        onRetry: () -> Unit = {},
        onChat: () -> Unit = {},
    ) {
        scenario.onActivity { activity ->
            activity.setContent {
                QlhTheme(darkTheme = false) {
                    Column(Modifier.fillMaxWidth().height(420.dp).verticalScroll(rememberScrollState())) {
                        ModelIntentPanel(state, discovering = false, onDiscoverMaster = {}, onRefresh = {},
                            onSelectModel = onSelect, onRetry = onRetry, onCancel = {}, onChat = onChat)
                    }
                }
            }
        }
    }

    @Test fun prepareProgressAndNodeAreVisibleBeforeTwentyModelButtons() {
        setPanel(state("preparing"))
        composeRule.onNodeWithText("qwen3-5-9b · 准备推理").assertIsDisplayed()
        composeRule.onNodeWithTag("intent_node_y700").assertIsDisplayed()
        composeRule.onNodeWithTag("intent_cancel").assertIsDisplayed().assertIsEnabled()
        composeRule.onNodeWithTag("intent_open_chat").assertDoesNotExist()
        composeRule.onNodeWithTag("intent_model_model-19").assertExists()
    }

    @Test fun failedOperationRetryIsVisibleAnd9BSearchStillSubmitsUncachedLLModel() {
        var retries = 0
        val choices = mutableListOf<Pair<String, String>>()
        setPanel(state("failed"), onRetry = { retries++ }, onSelect = { model, engine -> choices += model to engine })
        composeRule.onNodeWithText("INTENT_SOURCE_UNAVAILABLE · 源模型获取失败").assertIsDisplayed()
        composeRule.onNodeWithTag("intent_retry").assertIsDisplayed().assertIsEnabled().performClick()
        composeRule.onNodeWithTag("intent_model_search").performScrollTo().performTextInput(" 9B ")
        composeRule.onNodeWithTag("intent_model_model-0").assertDoesNotExist()
        composeRule.onNodeWithTag("intent_engine_llama_cpp").performScrollTo().performClick()
        composeRule.onNodeWithTag("intent_model_qwen3-5-9b").performScrollTo().assertIsEnabled().performClick()
        composeRule.runOnIdle {
            assertEquals(1, retries)
            assertEquals(listOf("qwen3-5-9b" to "llama_cpp"), choices)
        }
    }

    @Test fun loadedReadyChatActionIsVisibleBeforeCatalog() {
        var chats = 0
        setPanel(state("ready", loadedReady = true), onChat = { chats++ })
        composeRule.onNodeWithTag("intent_open_chat").assertIsDisplayed().assertIsEnabled().performClick()
        composeRule.runOnIdle { assertEquals(1, chats) }
    }
}
