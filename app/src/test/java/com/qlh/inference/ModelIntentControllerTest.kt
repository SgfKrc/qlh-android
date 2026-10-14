package com.qlh.inference

import com.qlh.inference.network.*
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.first
import org.junit.After
import org.junit.Assert.*
import org.junit.Test

class ModelIntentControllerTest {
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Unconfined)
    private val endpoint = IntentMasterEndpoint("100.64.0.1", 8000, "cluster", 1)
    @After fun stop() { scope.cancel() }

    private fun operation(status: String = "supplying", loaded: Boolean = false, id: String = "supply_one") =
        ModelIntentOperation(operationId = id, intent = ModelIntentRequest("qwen"), status = status,
            generation = 4, loadedReady = loaded, nodes = mapOf("phone" to ModelIntentNode(
                status = "supplying", bytesReceived = 10, totalBytes = 20)))

    private class FakeApi : ModelIntentApi {
        var catalog = ModelIntentCatalog(models = listOf(ServerModelSummary(modelId = "qwen", isAvailable = true)),
            binding = ModelIntentBinding("connected", true))
        var readCatalog: suspend () -> Result<ModelIntentCatalog> = { Result.success(catalog) }
        var catalogCalls = 0
        var current: suspend () -> Result<ModelIntentOperation?> = { Result.success(null) }
        var post: suspend (ModelIntentRequest) -> Result<ModelIntentOperation> = {
            Result.success(ModelIntentOperation("supply_new", it, status = "queued"))
        }
        var get: suspend (String) -> Result<ModelIntentOperation> = { error("unexpected poll") }
        var delete: suspend (String) -> Result<ModelIntentOperation> = { error("unexpected delete") }
        var currentCalls = 0
        val submissions = mutableListOf<ModelIntentRequest>()
        val cancellations = mutableListOf<String>()
        override suspend fun getModelIntentCatalog(): Result<ModelIntentCatalog> { catalogCalls++; return readCatalog() }
        override suspend fun getCurrentModelIntent(): Result<ModelIntentOperation?> { currentCalls++; return current() }
        override suspend fun submitModelIntent(intent: ModelIntentRequest): Result<ModelIntentOperation> {
            submissions += intent; return post(intent)
        }
        override suspend fun getModelIntent(operationId: String) = get(operationId)
        override suspend fun cancelModelIntent(operationId: String): Result<ModelIntentOperation> {
            cancellations += operationId; return delete(operationId)
        }
    }

    @Test fun `null current keeps a connected selector but cannot enable remote chat`() {
        val api = FakeApi()
        val controller = ModelIntentController(scope, { api }, polling = false)
        controller.updateContext("distributed", endpoint)
        assertTrue(controller.state.value.connected)
        assertNull(controller.state.value.operation)
        assertFalse(modelIntentAllowsChat("distributed", controller.state.value))
        assertTrue(modelIntentAllowsChat("local", controller.state.value))
    }

    @Test fun `supply node ready and preparing never impersonate loaded ready`() {
        for (status in listOf("supplying", "verifying", "preparing", "ready")) {
            val state = ModelIntentUiState(catalog = ModelIntentCatalog(binding = ModelIntentBinding("connected")),
                operation = operation(status, loaded = false).copy(nodes = mapOf("master" to ModelIntentNode(status = "ready"))))
            assertFalse(state.readyForChat)
            assertFalse(modelIntentAllowsChat("distributed", state))
        }
        val ready = ModelIntentUiState(catalog = ModelIntentCatalog(binding = ModelIntentBinding("connected")),
            operation = operation("ready", true))
        assertTrue(ready.readyForChat)
        assertFalse(ready.copy(error = ModelIntentError("offline", "offline")).readyForChat)
        assertFalse(ready.copy(suspended = true).readyForChat)
        assertFalse(ready.copy(busy = true).readyForChat)
    }

    @Test fun `current recovery is retired immediately on user local selection`() = runBlocking {
        val response = CompletableDeferred<ModelIntentOperation?>()
        val api = FakeApi().apply { current = { withContext(NonCancellable) { Result.success(response.await()) } } }
        val controller = ModelIntentController(scope, { api }, polling = false)
        controller.updateContext("distributed", endpoint)
        controller.suspendForSelection() // Before the real unload/persist gate has finished.
        response.complete(operation("ready", true))
        yield()
        assertNull(controller.state.value.operation)
        assertTrue(controller.state.value.suspended)
        controller.updateContext("local", endpoint)
        controller.refresh()
        assertEquals(1, api.currentCalls)
        assertNull(controller.state.value.operation)
    }

    @Test fun `host A to B to A cannot restore a request from the earlier host generation`() = runBlocking {
        val delayed = CompletableDeferred<ModelIntentOperation?>()
        val a = FakeApi().apply { current = { withContext(NonCancellable) { Result.success(delayed.await()) } } }
        val b = FakeApi().apply { current = { Result.success(operation(id = "supply_b")) } }
        val controller = ModelIntentController(scope, { if (it.host == endpoint.host) a else b }, polling = false)
        controller.updateContext("distributed", endpoint)
        controller.updateContext("distributed", endpoint.copy(host = "100.64.0.2"))
        assertEquals("supply_b", controller.state.value.operation?.operationId)
        a.current = { Result.success(operation(id = "supply_a_new")) }
        controller.updateContext("distributed", endpoint)
        delayed.complete(operation("ready", true, "supply_a_old"))
        yield()
        assertEquals("supply_a_new", controller.state.value.operation?.operationId)
        assertFalse(controller.state.value.readyForChat)
    }

    @Test fun `same endpoint secret epoch change retires old current recovery`() = runBlocking {
        val delayed = CompletableDeferred<ModelIntentOperation?>()
        val api = FakeApi().apply { current = { withContext(NonCancellable) { Result.success(delayed.await()) } } }
        val controller = ModelIntentController(scope, { api }, polling = false)
        controller.updateContext("distributed", endpoint)
        api.current = { Result.success(operation(id = "supply_new_binding")) }
        controller.updateContext("distributed", endpoint.copy(secretEpoch = 2))
        delayed.complete(operation("ready", true))
        yield()
        assertEquals("supply_new_binding", controller.state.value.operation?.operationId)
    }

    @Test fun `older refresh cannot replace a later intent submission`() = runBlocking {
        val delayed = CompletableDeferred<ModelIntentOperation?>()
        val api = FakeApi().apply { current = { withContext(NonCancellable) { Result.success(delayed.await()) } } }
        val controller = ModelIntentController(scope, { api }, polling = false)
        controller.updateContext("distributed", endpoint)
        controller.submit(ModelIntentRequest("new-model", "llama_cpp"))
        delayed.complete(operation("ready", true))
        yield()
        assertEquals("new-model", controller.state.value.operation?.intent?.modelId)
        assertEquals("supply_new", controller.state.value.operation?.operationId)
        assertFalse(controller.state.value.readyForChat)
    }

    @Test fun `late POST and DELETE cannot republish after a local choice`() = runBlocking {
        for (cancel in listOf(false, true)) {
            val response = CompletableDeferred<ModelIntentOperation>()
            val api = FakeApi().apply {
                current = { Result.success(operation()) }
                post = { withContext(NonCancellable) { Result.success(response.await()) } }
                delete = { withContext(NonCancellable) { Result.success(response.await()) } }
            }
            val controller = ModelIntentController(scope, { api }, polling = false)
            controller.updateContext("distributed", endpoint)
            if (cancel) controller.cancel() else controller.submit(ModelIntentRequest("qwen"))
            assertTrue(controller.state.value.busy)
            controller.updateContext("local", endpoint)
            response.complete(operation("ready", true))
            yield()
            assertNull(controller.state.value.operation)
            assertFalse(controller.state.value.readyForChat)
        }
    }

    @Test fun `cancel and retry use exact operation id and full original intent`() {
        val intent = ModelIntentRequest("qwen", "pytorch", "Q4", "auto", 3600)
        val api = FakeApi().apply {
            current = { Result.success(operation().copy(intent = intent)) }
            delete = { Result.success(operation("cancelled").copy(intent = intent)) }
        }
        val controller = ModelIntentController(scope, { api }, polling = false)
        controller.updateContext("distributed", endpoint)
        controller.cancel()
        assertEquals(listOf("supply_one"), api.cancellations)
        assertEquals("cancelled", controller.state.value.operation?.status)
        assertTrue(controller.state.value.canRetry)
        controller.retry()
        assertEquals(listOf(intent), api.submissions)
        assertEquals("queued", controller.state.value.operation?.status)
    }

    @Test fun `poll follows authoritative current through prepare and loaded ready`() = runBlocking {
        var reads = 0
        val api = FakeApi().apply {
            current = {
                reads++
                Result.success(when (reads) {
                    1 -> operation()
                    2 -> operation("preparing")
                    else -> operation("ready", true)
                })
            }
        }
        val controller = ModelIntentController(scope, { api }, pollIntervalMillis = 10)
        controller.updateContext("distributed", endpoint)
        withTimeout(1500) { controller.state.first { it.readyForChat } }
        assertTrue(api.currentCalls >= 3)
        assertTrue(api.submissions.isEmpty())
    }

    @Test fun `unbound discovery is named and no current request is made`() {
        val api = FakeApi().apply { catalog = ModelIntentCatalog(binding = ModelIntentBinding("not_found", reasonCode = "INTENT_MASTER_NOT_FOUND")) }
        val controller = ModelIntentController(scope, { api }, polling = false)
        controller.updateContext("distributed", endpoint)
        assertEquals("INTENT_MASTER_NOT_FOUND", controller.state.value.error?.code)
        assertEquals(0, api.currentCalls)
        controller.updateContext("distributed", null)
        assertEquals("INTENT_MASTER_NOT_FOUND", controller.state.value.error?.code)
        assertNull(controller.state.value.catalog)
    }

    @Test fun `HTTP named failure and byte progress remain bounded`() {
        assertEquals("模型段已供给，等待准备", modelIntentStatusLabel("available"))
        assertEquals("模型段已加载，等待提交", modelIntentStatusLabel("prepared"))
        assertEquals("model_intent_api_unavailable", modelIntentError(ApiClientHttpException(404, "", "{}")).code)
        assertEquals("master_auth_required", modelIntentError(ApiClientHttpException(403, "", "{}")).code)
        assertEquals(1f, ModelIntentNode(bytesReceived = 30, totalBytes = 20).progress!!, 0f)
        assertEquals(0f, ModelIntentNode(bytesReceived = -1, totalBytes = 20).progress!!, 0f)
        assertNull(ModelIntentNode(bytesReceived = 1).progress)
        assertEquals("模型意图 supply_one · 模型配置 cfg-model · 模型代际 4",
            modelIntentMetricsLabel("supply_one", "cfg-model", 4))
        assertEquals("", modelIntentMetricsLabel("route_a:task-id", "cfg-task", 4))
    }
    @Test fun `uncached catalog model can be selected for master owned LL supply`() {
        val api = FakeApi().apply {
            catalog = catalog.copy(models = listOf(ServerModelSummary(
                modelId = "qwen", name = "Qwen", isAvailable = false, unavailableReason = "source_not_downloaded",
            )))
        }
        val controller = ModelIntentController(scope, { api }, polling = false)
        controller.updateContext("distributed", endpoint)
        assertTrue(controller.state.value.canChooseModel)
        controller.submit(ModelIntentRequest("qwen", "llama_cpp"))
        assertEquals(listOf(ModelIntentRequest("qwen", "llama_cpp")), api.submissions)
        assertEquals("queued", controller.state.value.operation?.status)
        assertFalse(controller.state.value.readyForChat)
    }

    @Test fun `model selection stays disabled during request refresh unbinding and user mode selection`() {
        val readyToChoose = ModelIntentUiState(catalog = ModelIntentCatalog(binding = ModelIntentBinding("connected")))
        assertTrue(readyToChoose.canChooseModel)
        assertFalse(readyToChoose.copy(loading = true).canChooseModel)
        assertFalse(readyToChoose.copy(busy = true).canChooseModel)
        assertFalse(readyToChoose.copy(suspended = true).canChooseModel)
        assertFalse(readyToChoose.copy(catalog = null).canChooseModel)
        assertFalse(readyToChoose.copy(catalog = ModelIntentCatalog(binding = ModelIntentBinding("local"))).canChooseModel)
    }


    @Test fun `external device selection and cancellation replace terminal current without a refresh or writes`() = runBlocking {
        var master = operation("failed", id = "supply_first")
        val api = FakeApi().apply { current = { Result.success(master) } }
        val controller = ModelIntentController(scope, { api }, pollIntervalMillis = 10)
        controller.updateContext("distributed", endpoint)
        assertEquals("supply_first", controller.state.value.operation?.operationId)
        master = operation("preparing", id = "supply_external").copy(intent = ModelIntentRequest("9b", "llama_cpp"))
        withTimeout(1500) { controller.state.first { it.operation?.operationId == "supply_external" } }
        assertEquals(master.intent, controller.state.value.requestedIntent)
        assertFalse(controller.state.value.readyForChat)
        master = master.copy(status = "ready", loadedReady = true)
        withTimeout(1500) { controller.state.first { it.readyForChat } }
        master = master.copy(status = "cancelled", loadedReady = false)
        withTimeout(1500) { controller.state.first { it.operation?.status == "cancelled" } }
        assertFalse(controller.state.value.readyForChat)
        assertTrue(controller.state.value.canRetry)
        assertTrue(api.submissions.isEmpty())
        assertTrue(api.cancellations.isEmpty())
    }

    @Test fun `late old current GET cannot replace a recovered external selection in the same context`() = runBlocking {
        val oldRead = CompletableDeferred<ModelIntentOperation?>()
        val entered = CompletableDeferred<Unit>()
        var reads = 0
        val api = FakeApi().apply {
            current = {
                if (++reads == 2) {
                    entered.complete(Unit)
                    withContext(NonCancellable) { Result.success(oldRead.await()) }
                } else Result.success(operation("preparing", id = if (reads == 1) "supply_old" else "supply_external"))
            }
        }
        val controller = ModelIntentController(scope, { api }, pollIntervalMillis = 10)
        controller.updateContext("distributed", endpoint)
        withTimeout(1500) { entered.await() }
        controller.refresh()
        assertEquals("supply_external", controller.state.value.operation?.operationId)
        oldRead.complete(operation("ready", true, "supply_old"))
        yield()
        assertEquals("supply_external", controller.state.value.operation?.operationId)
        assertFalse(controller.state.value.readyForChat)
    }

    @Test fun `local write in flight retires a delayed cross device GET`() = runBlocking {
        val oldRead = CompletableDeferred<ModelIntentOperation?>()
        val posted = CompletableDeferred<ModelIntentOperation>()
        val entered = CompletableDeferred<Unit>()
        var reads = 0
        val api = FakeApi().apply {
            current = {
                if (++reads == 2) {
                    entered.complete(Unit)
                    withContext(NonCancellable) { Result.success(oldRead.await()) }
                } else Result.success(operation("failed"))
            }
            post = { withContext(NonCancellable) { Result.success(posted.await()) } }
        }
        val controller = ModelIntentController(scope, { api }, pollIntervalMillis = 10)
        controller.updateContext("distributed", endpoint)
        withTimeout(1500) { entered.await() }
        val chosen = ModelIntentRequest("user-choice", "llama_cpp")
        controller.submit(chosen)
        oldRead.complete(operation("ready", true, "supply_external"))
        yield()
        assertTrue(controller.state.value.busy)
        assertEquals(chosen, controller.state.value.requestedIntent)
        assertNull(controller.state.value.operation)
        posted.complete(operation("queued", id = "supply_user").copy(intent = chosen))
        yield()
        assertEquals("supply_user", controller.state.value.operation?.operationId)
    }

    @Test fun `late periodic GET cannot restore remote state after local choice or endpoint generation change`() = runBlocking {
        for (selectLocal in listOf(true, false)) {
            val oldRead = CompletableDeferred<ModelIntentOperation?>()
            val entered = CompletableDeferred<Unit>()
            var reads = 0
            val a = FakeApi().apply {
                current = {
                    if (++reads == 2) {
                        entered.complete(Unit)
                        withContext(NonCancellable) { Result.success(oldRead.await()) }
                    } else Result.success(operation("failed", id = "supply_a"))
                }
            }
            val b = FakeApi().apply { current = { Result.success(operation("preparing", id = "supply_b")) } }
            val controller = ModelIntentController(scope, { if (it.host == endpoint.host) a else b }, pollIntervalMillis = 10)
            controller.updateContext("distributed", endpoint)
            withTimeout(1500) { entered.await() }
            if (selectLocal) controller.suspendForSelection()
            else controller.updateContext("distributed", endpoint.copy(host = "100.64.0.2"))
            oldRead.complete(operation("ready", true, "supply_a"))
            yield()
            assertFalse(controller.state.value.readyForChat)
            if (selectLocal) {
                assertNull(controller.state.value.operation)
                assertTrue(controller.state.value.suspended)
                controller.updateContext("local", endpoint)
            } else assertEquals("supply_b", controller.state.value.operation?.operationId)
        }
    }


    @Test fun `startup source gate 403 automatically retries catalog then current after admission`() = runBlocking {
        val admitted = CompletableDeferred<Unit>()
        val api = FakeApi().apply {
            readCatalog = {
                if (catalogCalls == 1) Result.failure(ApiClientHttpException(403, "", """{
                    "detail":{"reason_code":"MODEL_API_SOURCE_UNTRUSTED","reason":"worker not registered"}
                }"""))
                else { admitted.await(); Result.success(catalog) }
            }
            current = { Result.success(operation("preparing", id = "supply_recovered")) }
        }
        val controller = ModelIntentController(scope, { api }, pollIntervalMillis = 10)
        controller.updateContext("distributed", endpoint)
        assertEquals("MODEL_API_SOURCE_UNTRUSTED", controller.state.value.error?.code)
        assertNull(controller.state.value.catalog)
        assertFalse(controller.state.value.loading)
        assertEquals(0, api.currentCalls)
        admitted.complete(Unit)
        withTimeout(1500) { controller.state.first { it.operation?.operationId == "supply_recovered" } }
        assertTrue(api.catalogCalls >= 2)
        assertTrue(controller.state.value.connected)
        assertNull(controller.state.value.error)
        assertFalse(controller.state.value.readyForChat)
        assertTrue(api.submissions.isEmpty())
        assertTrue(api.cancellations.isEmpty())
    }

    @Test fun `startup unconnected catalog is retried until connected before polling current`() = runBlocking {
        for (binding in listOf("not_found", "local")) {
            val admitted = CompletableDeferred<Unit>()
            val api = FakeApi().apply {
                readCatalog = {
                    if (catalogCalls == 1) Result.success(catalog.copy(binding = ModelIntentBinding(
                        binding, reasonCode = "INTENT_MASTER_NOT_FOUND")))
                    else { admitted.await(); Result.success(catalog) }
                }
                current = { Result.success(operation("preparing", id = "supply_connected")) }
            }
            val controller = ModelIntentController(scope, { api }, pollIntervalMillis = 10)
            controller.updateContext("distributed", endpoint)
            assertFalse(controller.state.value.connected)
            assertEquals("INTENT_MASTER_NOT_FOUND", controller.state.value.error?.code)
            assertEquals(0, api.currentCalls)
            admitted.complete(Unit)
            withTimeout(1500) { controller.state.first { it.operation?.operationId == "supply_connected" } }
            assertTrue(api.catalogCalls >= 2)
            assertTrue(controller.state.value.connected)
            assertNull(controller.state.value.error)
            assertTrue(api.submissions.isEmpty())
            assertTrue(api.cancellations.isEmpty())
            controller.suspendForSelection()
        }
    }

    @Test fun `late startup recovery catalog is rejected after local selection or endpoint epoch change`() = runBlocking {
        for (selectLocal in listOf(true, false)) {
            val retryEntered = CompletableDeferred<Unit>()
            val oldCatalog = CompletableDeferred<ModelIntentCatalog>()
            val a = FakeApi().apply {
                readCatalog = {
                    if (catalogCalls == 1) Result.failure(ApiClientHttpException(403, "", "{}"))
                    else {
                        retryEntered.complete(Unit)
                        withContext(NonCancellable) { Result.success(oldCatalog.await()) }
                    }
                }
                current = { fail("retired catalog must not trigger current"); Result.success(null) }
            }
            val b = FakeApi().apply { current = { Result.success(operation("preparing", id = "supply_new_binding")) } }
            val controller = ModelIntentController(scope, { if (it.secretEpoch == endpoint.secretEpoch) a else b }, pollIntervalMillis = 10)
            controller.updateContext("distributed", endpoint)
            withTimeout(1500) { retryEntered.await() }
            if (selectLocal) controller.suspendForSelection()
            else controller.updateContext("distributed", endpoint.copy(secretEpoch = 2))
            oldCatalog.complete(a.catalog.copy(models = listOf(ServerModelSummary(modelId = "retired-catalog"))))
            yield()
            assertEquals(0, a.currentCalls)
            if (selectLocal) {
                assertNull(controller.state.value.catalog)
                assertNull(controller.state.value.operation)
                assertTrue(controller.state.value.suspended)
                controller.updateContext("local", endpoint)
            } else {
                assertEquals(b.catalog, controller.state.value.catalog)
                assertEquals("supply_new_binding", controller.state.value.operation?.operationId)
            }
            controller.suspendForSelection()
        }
    }

    @Test fun `late recovered current cannot override a local write and background reads wait for that write`() = runBlocking {
        val oldCurrent = CompletableDeferred<ModelIntentOperation?>()
        val retryEntered = CompletableDeferred<Unit>()
        val posted = CompletableDeferred<ModelIntentOperation>()
        val api = FakeApi().apply {
            readCatalog = {
                if (catalogCalls == 1) Result.failure(ApiClientHttpException(403, "", "{}"))
                else Result.success(catalog)
            }
            current = {
                if (currentCalls == 1) {
                    retryEntered.complete(Unit)
                    withContext(NonCancellable) { Result.success(oldCurrent.await()) }
                } else Result.success(operation("queued", id = "supply_user").copy(
                    intent = ModelIntentRequest("user-choice", "llama_cpp")))
            }
            post = { withContext(NonCancellable) { Result.success(posted.await()) } }
        }
        val controller = ModelIntentController(scope, { api }, pollIntervalMillis = 10)
        controller.updateContext("distributed", endpoint)
        withTimeout(1500) { retryEntered.await() }
        val chosen = ModelIntentRequest("user-choice", "llama_cpp")
        controller.submit(chosen)
        oldCurrent.complete(operation("ready", true, "supply_old_startup"))
        yield()
        assertTrue(controller.state.value.busy)
        assertEquals(chosen, controller.state.value.requestedIntent)
        assertNull(controller.state.value.operation)
        // Allow multiple polling intervals while POST is held; the read lane remains retired.
        delay(35)
        assertEquals(1, api.currentCalls)
        assertEquals(2, api.catalogCalls)
        assertEquals(listOf(chosen), api.submissions)
        assertTrue(api.cancellations.isEmpty())
        posted.complete(operation("queued", id = "supply_user").copy(intent = chosen))
        withTimeout(1500) { controller.state.first { it.operation?.operationId == "supply_user" } }
        assertFalse(controller.state.value.busy)
        assertFalse(controller.state.value.readyForChat)
    }

}
