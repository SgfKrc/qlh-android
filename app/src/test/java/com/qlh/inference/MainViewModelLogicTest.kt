package com.qlh.inference

import com.qlh.inference.data.SettingsDataStore
import com.qlh.inference.status.AndroidRuntimeStatus
import com.qlh.inference.status.BackendStatus
import com.qlh.inference.status.GpuStatus
import com.qlh.inference.status.MemoryStatus
import com.qlh.inference.status.SystemStatus
import com.qlh.inference.network.ClusterNode
import com.qlh.inference.network.ClusterStatus
import com.qlh.inference.network.ClusterTask
import com.qlh.inference.network.CurrentModelStatus
import com.qlh.inference.network.AuditAttemptSummary
import com.qlh.inference.network.AuditData
import com.qlh.inference.network.AuditReviewTicketsResponse
import com.qlh.inference.network.AuditStageSummary
import com.qlh.inference.network.AuditWorkflowSummary
import com.qlh.inference.network.AuditWorkflowsResponse
import com.qlh.inference.network.AuthCapabilityResponse
import com.qlh.inference.network.GgufModelInfo
import com.qlh.inference.network.LocalModelAssetSummary
import com.qlh.inference.network.LocalModelAssetsResponse
import com.qlh.inference.network.LocalModelAssetsSummary
import com.qlh.inference.network.MasterModelFleetData
import com.qlh.inference.network.ServerModelSummary
import com.qlh.inference.network.ServerModelsResponse
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertSame
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * MainViewModel 纯逻辑与 UI 状态默认值测试（JVM，无需设备）。
 */
class MainViewModelLogicTest {

    // ---- networkTypeFromTransports ----

    @Test
    fun `network type wifi wins over others`() {
        assertEquals("wifi", networkTypeFromTransports(true, true, true, true))
        assertEquals("wifi", networkTypeFromTransports(true, false, false, false))
    }

    @Test
    fun `network type falls through cellular ethernet vpn`() {
        assertEquals("mobile", networkTypeFromTransports(false, true, false, false))
        assertEquals("ethernet", networkTypeFromTransports(false, false, true, false))
        assertEquals("vpn", networkTypeFromTransports(false, false, false, true))
    }

    @Test
    fun `network type other when no transport`() {
        assertEquals("other", networkTypeFromTransports(false, false, false, false))
    }

    @Test
    fun `management projection is bounded and only allows manager roles`() {
        val user = com.qlh.inference.network.AuthUser(
            userId = "u-1",
            username = "alice",
            displayName = "Alice",
            role = "admin",
            status = "active",
            aggregateVersion = 4,
        )
        val binding = com.qlh.inference.network.TailscaleBinding(
            bindingId = "b-1",
            userId = "u-1",
            tailnetId = "tailnet-a",
            tailscaleUserId = "ts-a",
            nodeId = "node-a",
            state = "active",
        )
        val snapshot = user.toManagedUserSnapshot(listOf(binding.toManagedBindingSnapshot("u-1")))
        assertEquals("Alice", snapshot.displayName)
        assertEquals(4, snapshot.aggregateVersion)
        assertEquals("tailnet-a", snapshot.bindings.single().tailnetId)
        assertTrue(managementRoleAllowed("owner"))
        assertTrue(managementRoleAllowed("admin"))
        assertFalse(managementRoleAllowed("member"))
        assertFalse(managementRoleAllowed(null))
    }

    // ---- buildAndroidPresencePayload ----

    private fun sampleSystem() = SystemStatus(
        manufacturer = "TestMaker",
        brand = "TestBrand",
        model = "TestModel",
        device = "test-device",
        hardware = "test-hw",
        socManufacturer = "SoC",
        socModel = "SoC-1",
        sdkInt = 34,
        androidRelease = "14",
        abis = listOf("arm64-v8a", "x86_64"),
        cpuCores = 8,
        thermalStatus = "normal",
    )

    private fun sampleMemory() = MemoryStatus(
        availableBytes = 1024L * 1024 * 1024,
        totalBytes = 8L * 1024 * 1024 * 1024,
        lowMemory = false,
        lowRamDevice = false,
    )

    private fun sampleGpu() = GpuStatus(
        vendor = "Qualcomm",
        renderer = "Adreno",
        version = "1.0",
        probeError = null,
        supportsGpuOffload = false,
        backendDevices = "",
        note = "probe note",
    )

    @Test
    fun `payload has fixed envelope fields`() {
        val payload = buildAndroidPresencePayload(
            inferenceMode = "thin",
            appVariant = "full",
            appVersion = "0.1.0",
            system = sampleSystem(),
            memory = sampleMemory(),
            gpu = sampleGpu(),
            runtime = null,
        )
        assertEquals("http_thin", payload["connection_type"])
        assertEquals(false, payload["pipeline_worker"])
        assertEquals("thin", payload["client_mode"])
        assertEquals("full", payload["app_variant"])
        assertEquals("0.1.0", payload["app_version"])
    }

    @Test
    fun `payload nests android memory gpu backend sections`() {
        val payload = buildAndroidPresencePayload(
            inferenceMode = "full",
            appVariant = "lite",
            appVersion = "0.2.0",
            system = sampleSystem(),
            memory = sampleMemory(),
            gpu = sampleGpu(),
            runtime = null,
        )
        @Suppress("UNCHECKED_CAST")
        val android = payload["android"] as Map<String, Any?>
        assertEquals("TestMaker", android["manufacturer"])
        assertEquals("TestModel", android["model"])
        assertEquals(listOf("arm64-v8a", "x86_64"), android["abis"])
        assertEquals(8, android["cpu_cores"])
        assertEquals("normal", android["thermal_status"])

        @Suppress("UNCHECKED_CAST")
        val memory = payload["memory"] as Map<String, Any?>
        assertEquals(1024L * 1024 * 1024, memory["available_bytes"])
        assertEquals(false, memory["low_ram_device"])

        @Suppress("UNCHECKED_CAST")
        val gpu = payload["gpu"] as Map<String, Any?>
        assertEquals("Qualcomm", gpu["vendor"])
        assertEquals(false, gpu["supports_gpu_offload"])
        assertEquals("probe note", gpu["note"])

        @Suppress("UNCHECKED_CAST")
        val backend = payload["backend"] as Map<String, Any?>
        assertEquals("", backend["engine"])
        assertEquals(false, backend["supports_gpu_offload"])

        @Suppress("UNCHECKED_CAST")
        val multimodal = payload["multimodal"] as Map<String, Any?>
        assertEquals(false, multimodal["vision_supported"])
        assertEquals(false, multimodal["assets_present"])
        assertEquals(false, multimodal["ready"])
    }

    @Test
    fun `payload reflects runtime gpu and backend status when present`() {
        val runtime = AndroidRuntimeStatus(
            gpu = GpuStatus(supportsGpuOffload = true, backendDevices = "gpu:0"),
            backend = BackendStatus(engine = "llama.cpp", supportsGpuOffload = true),
        )
        val payload = buildAndroidPresencePayload(
            inferenceMode = "full",
            appVariant = "full",
            appVersion = "0.1.0",
            system = sampleSystem(),
            memory = sampleMemory(),
            gpu = sampleGpu(),
            runtime = runtime,
        )
        @Suppress("UNCHECKED_CAST")
        val gpu = payload["gpu"] as Map<String, Any?>
        assertEquals(true, gpu["supports_gpu_offload"])
        assertEquals("gpu:0", gpu["backend_devices"])

        @Suppress("UNCHECKED_CAST")
        val backend = payload["backend"] as Map<String, Any?>
        assertEquals("llama.cpp", backend["engine"])
        assertEquals(true, backend["supports_gpu_offload"])
    }

    @Test
    fun `payload never contains absolute paths or credentials`() {
        val payload = buildAndroidPresencePayload(
            inferenceMode = "full",
            appVariant = "full",
            appVersion = "0.1.0",
            system = sampleSystem(),
            memory = sampleMemory(),
            gpu = sampleGpu(),
            runtime = AndroidRuntimeStatus(),
        )
        val json = payload.toString()
        assertFalse(json.contains("/storage/"))
        assertFalse(json.contains("token"))
        assertFalse(json.contains("password"))
        assertFalse(json.contains("secret"))
    }

    // ---- MainUiState 默认值 ----

    @Test
    fun `MainUiState defaults match settings defaults`() {
        val state = MainUiState()
        assertEquals("chat", state.currentTab)
        assertEquals(0L, state.currentSessionId)
        assertEquals("新对话", state.currentSessionTitle)
        assertTrue(state.messages.isEmpty())
        assertFalse(state.isLoading)
        assertEquals(SettingsDataStore.DEFAULT_HOST, state.serverHost)
        assertEquals(SettingsDataStore.DEFAULT_PORT, state.serverPort)
        assertEquals(SettingsDataStore.DEFAULT_MODE, state.inferenceMode)
        assertEquals(SettingsDataStore.DEFAULT_MAX_TOKENS, state.maxTokens)
        assertEquals(SettingsDataStore.DEFAULT_TEMPERATURE, state.temperature, 0.001f)
        assertEquals(SettingsDataStore.DEFAULT_TOP_P, state.topP, 0.001f)
        assertEquals(SettingsDataStore.DEFAULT_CONTEXT_SIZE, state.contextSize)
        assertEquals("", state.modelTreeUri)
        assertEquals(SettingsDataStore.DEFAULT_MODEL_STORAGE_MODE, state.modelStorageMode)
        assertEquals(SettingsDataStore.DEFAULT_THEME_MODE, state.themeMode)
        assertTrue(state.availableModels.isEmpty())
        assertTrue(state.sessions.isEmpty())
    }

    @Test
    fun `cluster overview normalizes node map and keeps busy workers reachable`() {
        val overview = toClusterOverviewSnapshot(
            ClusterStatus(
                running = true,
                runMode = "distributed",
                nodesReady = true,
                nodes = mapOf(
                    "worker-b" to ClusterNode(
                        nodeId = "worker-b",
                        role = "client",
                        nodeType = "android",
                        state = "busy",
                        hostname = "phone",
                        networkType = "wifi",
                        taskCount = 1,
                    ),
                    "master" to ClusterNode(
                        nodeId = "master",
                        role = "master",
                        state = "online",
                        hostname = "main",
                        isAvailable = true,
                    ),
                    "worker-a" to ClusterNode(
                        nodeId = "worker-a",
                        role = "client",
                        state = "offline",
                    ),
                ),
                currentTask = ClusterTask(taskId = "task-1", state = "running", elapsed = 7.9),
            ),
        )

        assertEquals(listOf("master", "worker-a", "worker-b"), overview.nodes.map { it.nodeId })
        assertEquals(2, overview.reachableNodes)
        assertEquals(3, overview.totalNodes)
        assertEquals("task-1", overview.currentTaskId)
        assertEquals(7L, overview.currentTaskElapsedSeconds)
        assertTrue(overview.nodes.last().reachable)
    }

    @Test
    fun `cluster overview errors do not disclose endpoint details`() {
        assertEquals(
            "无法连接主节点",
            formatClusterOverviewError(java.net.ConnectException("failed to connect to 100.90.76.108")),
        )
        assertEquals(
            "读取主节点状态超时",
            formatClusterOverviewError(java.net.SocketTimeoutException("http://[fd7a::1]:8000")),
        )
        assertEquals(
            "读取集群状态失败",
            formatClusterOverviewError(IllegalStateException("Bearer secret")),
        )
    }

    @Test
    fun `model fleet merges active available missing unverified and local selection`() {
        val snapshot = toModelFleetSnapshot(
            data = MasterModelFleetData(
                current = CurrentModelStatus(
                    loaded = true,
                    modelId = "qwen-1_8b",
                    modelName = "Qwen 1.8B",
                    engine = "llama_cpp",
                    quantType = "Q4_K_M",
                ),
                registry = ServerModelsResponse(
                    models = listOf(
                        ServerModelSummary(
                            modelId = "qwen-1-8b",
                            name = "Qwen 1.8B",
                            modelType = "both",
                            isAvailable = true,
                            availableFormats = listOf("gguf", "safetensors"),
                        ),
                        ServerModelSummary(modelId = "unverified", name = "Unverified"),
                        ServerModelSummary(modelId = "missing", name = "Missing"),
                    ),
                ),
                localAssets = LocalModelAssetsResponse(
                    assets = listOf(
                        LocalModelAssetSummary(
                            modelId = "unverified",
                            name = "Unverified",
                            totalBytes = 512L,
                            integrity = "filesystem_discovered",
                        ),
                        LocalModelAssetSummary(
                            modelId = "asset-only",
                            name = "Asset only",
                            totalBytes = 1024L,
                            integrity = "manifest_verified",
                        ),
                    ),
                    summary = LocalModelAssetsSummary(total = 2, totalBytes = 1536L),
                ),
                verifiedGguf = listOf(
                    GgufModelInfo(
                        filename = "downloaded.gguf",
                        sizeBytes = 2048L,
                        sha256 = "a".repeat(64),
                    ),
                ),
            ),
            androidSelectedModelName = "phone.gguf",
            androidSelectedModelSizeBytes = 4096L,
        )

        assertEquals("qwen-1_8b", snapshot.currentModelId)
        assertTrue(snapshot.currentLoaded)
        assertEquals("phone.gguf", snapshot.androidSelectedModelName)
        assertEquals(4096L, snapshot.androidSelectedModelSizeBytes)
        assertEquals(3, snapshot.registryCount)
        assertEquals(2, snapshot.localAssetCount)
        assertEquals(1, snapshot.verifiedGgufCount)
        assertEquals(ModelFleetStatus.ACTIVE, snapshot.entries.first().status)
        assertEquals(
            ModelFleetStatus.UNVERIFIED,
            snapshot.entries.first { it.modelId == "unverified" }.status,
        )
        assertEquals(
            ModelFleetStatus.MISSING,
            snapshot.entries.first { it.modelId == "missing" }.status,
        )
        assertEquals(
            ModelFleetStatus.AVAILABLE,
            snapshot.entries.first { it.modelId == "asset-only" }.status,
        )
        assertTrue(snapshot.entries.none { it.name.contains("C:/") })
    }

    @Test
    fun `prepared pipeline is not reported as an active model`() {
        val snapshot = toModelFleetSnapshot(
            MasterModelFleetData(
                current = CurrentModelStatus(
                    loaded = false,
                    pipelinePrepared = true,
                    modelId = "qwen_1_8b",
                    modelName = "Qwen 1.8B",
                ),
                registry = ServerModelsResponse(
                    models = listOf(
                        ServerModelSummary(
                            modelId = "qwen-1-8b",
                            name = "Qwen 1.8B",
                            isAvailable = true,
                        ),
                    ),
                ),
            ),
        )

        assertTrue(snapshot.pipelinePrepared)
        assertFalse(snapshot.currentLoaded)
        assertEquals(ModelFleetStatus.AVAILABLE, snapshot.entries.single().status)
    }

    @Test
    fun `model fleet errors do not disclose endpoint details`() {
        assertEquals(
            "无法连接主节点",
            formatModelFleetError(java.net.ConnectException("failed to connect to 100.90.76.108")),
        )
        assertEquals(
            "读取模型舰队超时",
            formatModelFleetError(java.net.SocketTimeoutException("http://[fd7a::1]:8000")),
        )
        assertEquals(
            "读取模型舰队失败",
            formatModelFleetError(IllegalStateException("Bearer secret")),
        )
    }

    // ---- 聊天与会话状态机 ----

    @Test
    fun `audit projection is bounded and content free`() {
        val data = AuditData(
            workflows = AuditWorkflowsResponse(
                enabled = true,
                available = true,
                role = "master",
                workflows = (1..12).map { index ->
                    AuditWorkflowSummary(
                        workflowId = "wf-$index",
                        template = "chat",
                        state = "completed",
                        createdAt = index.toDouble(),
                        stageCount = 20,
                        completedStageCount = 20,
                        attemptCount = 20,
                        stages = (1..20).map { stage ->
                            AuditStageSummary(
                                stageId = "stage-$stage",
                                stageType = "llm",
                                attempts = (1..10).map { attempt ->
                                    AuditAttemptSummary(
                                        attemptId = "attempt-$attempt",
                                        providerKind = "local",
                                        providerNodeId = "node-$attempt",
                                        state = "completed",
                                    )
                                },
                            )
                        },
                    )
                },
            ),
            reviewTickets = AuditReviewTicketsResponse(
                tickets = (1..12).map { index ->
                    com.qlh.inference.network.ReviewTicketSummary(
                        ticketId = "review-$index",
                        status = "pending",
                        targetNodeId = "node-$index",
                        score = 300,
                        voteCount = 500,
                    )
                },
            ),
        )

        val snapshot = toAuditSnapshot(data)
        assertEquals(MAX_AUDIT_WORKFLOWS, snapshot.workflows.size)
        assertEquals(MAX_AUDIT_REVIEWS, snapshot.reviews.size)
        assertEquals(MAX_AUDIT_STAGES, snapshot.workflows.first().stages.size)
        assertEquals(MAX_AUDIT_ATTEMPTS, snapshot.workflows.first().stages.first().attempts.size)
        assertEquals(100, snapshot.reviews.first().score)
        assertEquals(100, snapshot.reviews.first().voteCount)
    }

    @Test
    fun `audit errors do not disclose endpoint details`() {
        assertEquals("无法连接主节点", formatAuditError(java.net.ConnectException("100.90.76.108")))
        assertEquals("读取审计资料超时", formatAuditError(java.net.SocketTimeoutException("Bearer secret")))
        assertEquals("读取审计资料失败", formatAuditError(IllegalStateException("C:/secret")))
    }

    @Test
    fun `auth capability normalizes gateway and standalone shapes`() {
        val gateway = AuthCapabilityResponse(
            required = true,
            enforced = true,
            mode = "local_totp",
            bootstrapAvailable = true,
        ).toSnapshot()
        assertTrue(gateway.canAuthenticate)
        assertEquals("local_totp", gateway.mode)

        val standalone = AuthCapabilityResponse(
            required = false,
            available = false,
            mode = "local_primary_node",
            reasonCode = "auth_control_plane_unavailable",
        ).toSnapshot()
        assertFalse(standalone.canAuthenticate)
        assertEquals("auth_control_plane_unavailable", standalone.reasonCode)
    }

    @Test
    fun `auth errors never disclose response details`() {
        assertEquals("无法连接认证控制面", formatAuthError(java.net.ConnectException("100.90.76.108")))
        assertEquals("认证控制面响应超时", formatAuthError(java.net.SocketTimeoutException("Bearer secret")))
        assertEquals("认证会话无效或已过期", formatAuthError(IllegalStateException("HTTP 401: secret")))
        assertEquals("认证服务不可用", formatAuthError(IllegalStateException("C:/secret")))
    }

    @Test
    fun `selecting a session returns to chat and preserves unrelated state`() {
        val state = MainUiState(
            currentTab = "sessions",
            serverHost = "100.64.0.2",
            error = "旧错误",
        )

        val selected = selectUiSession(state, sessionId = 42L, sessionTitle = "部署讨论")

        assertEquals("chat", selected.currentTab)
        assertEquals(42L, selected.currentSessionId)
        assertEquals("部署讨论", selected.currentSessionTitle)
        assertEquals("100.64.0.2", selected.serverHost)
        assertEquals("旧错误", selected.error)
    }

    @Test
    fun `message submission lifecycle records retry payload and clears transient error`() {
        val initial = MainUiState(currentSessionId = 9L, error = "上次失败")
        val sending = startMessageSubmission(initial, "重新发送这条消息")
        val failed = failMessageSubmission(sending, "连接超时")
        val retrying = startMessageRetry(failed)
        val completed = completeMessageSubmission(retrying)

        assertTrue(sending.isLoading)
        assertEquals("重新发送这条消息", sending.lastSentMessage)
        assertEquals(null, sending.error)
        assertFalse(failed.isLoading)
        assertEquals("连接超时", failed.error)
        assertTrue(retrying.isLoading)
        assertEquals(null, retrying.error)
        assertFalse(completed.isLoading)
        assertEquals(null, completed.error)
        assertEquals("重新发送这条消息", completed.lastSentMessage)
    }

    @Test
    fun `retry without a prior message is a no-op and clear only removes error`() {
        val state = MainUiState(currentTab = "settings", error = "发送失败")

        assertSame(state, startMessageRetry(state))
        val cleared = clearMessageError(state)
        assertEquals("settings", cleared.currentTab)
        assertEquals(null, cleared.error)
        assertFalse(cleared.isLoading)
    }

    @Test
    fun `send error formatting keeps actionable network and fallback messages`() {
        assertEquals("远程推理需要先在账户页完成 Auth App 登录", formatMessageSendError(com.qlh.inference.network.ApiClientHttpException(401, "", "{}")))
        assertEquals("主节点暂时拒绝远程推理，请检查分布式节点连接与模型准入", formatMessageSendError(com.qlh.inference.network.ApiClientHttpException(503, "", "{}")))
        assertEquals(
            "无法连接主节点，请检查地址和网络",
            formatMessageSendError(java.net.ConnectException()),
        )
        assertEquals(
            "连接超时，请检查主节点是否运行",
            formatMessageSendError(java.net.SocketTimeoutException()),
        )
        assertEquals(
            "当前模式暂不支持",
            formatMessageSendError(UnsupportedOperationException()),
        )
        assertEquals(
            "发送失败: bad gateway",
            formatMessageSendError(IllegalStateException("bad gateway")),
        )
    }

    // ---- 聊天/会话 UI 状态机（MainViewModelLogic 纯函数） ----

    @Test
    fun `select tab switches tab without touching other fields`() {
        val base = MainUiState(currentTab = "chat", serverHost = "1.2.3.4")
        val next = selectUiTab(base, "settings")
        assertEquals("settings", next.currentTab)
        assertEquals("1.2.3.4", next.serverHost)
    }

    @Test
    fun `select session sets id title and jumps to chat tab`() {
        val base = MainUiState(currentTab = "sessions", currentSessionId = 0)
        val next = selectUiSession(base, sessionId = 7L, sessionTitle = "网络排查")
        assertEquals(7L, next.currentSessionId)
        assertEquals("网络排查", next.currentSessionTitle)
        assertEquals("chat", next.currentTab)
    }

    @Test
    fun `start submission marks loading clears error and records last message`() {
        val base = MainUiState(isLoading = false, error = "旧错误")
        val next = startMessageSubmission(base, "你好")
        assertTrue(next.isLoading)
        assertNull(next.error)
        assertEquals("你好", next.lastSentMessage)
    }

    @Test
    fun `complete submission stops loading and clears error`() {
        val base = MainUiState(isLoading = true, error = "旧错误")
        val next = completeMessageSubmission(base)
        assertFalse(next.isLoading)
        assertNull(next.error)
        assertEquals("旧错误", base.error) // 纯函数不改原状态
    }

    @Test
    fun `fail submission stops loading and records error`() {
        val next = failMessageSubmission(MainUiState(isLoading = true), "后端 500")
        assertFalse(next.isLoading)
        assertEquals("后端 500", next.error)
    }

    @Test
    fun `retry only starts when a last message exists`() {
        // 无 lastSentMessage 时保持原状态（不进入加载）
        val idle = MainUiState(isLoading = false, lastSentMessage = null)
        assertSame(idle, startMessageRetry(idle))
        // 有 lastSentMessage：进入加载并清错误（保留原文）
        val next = startMessageRetry(MainUiState(isLoading = false, error = "旧错", lastSentMessage = "原文"))
        assertTrue(next.isLoading)
        assertNull(next.error)
        assertEquals("原文", next.lastSentMessage)
    }

    @Test
    fun `clear error nulls the error only`() {
        val next = clearMessageError(MainUiState(error = "x", isLoading = true))
        assertNull(next.error)
        assertTrue(next.isLoading)
    }

    @Test
    fun `image submission keeps a bounded normalized payload`() {
        val image = "data:image/png;base64,iVBORw0KGgo="
        val urls = normalizeChatImageDataUrls(listOf("  $image  ", "", image))

        assertEquals(listOf(image, image), urls)
        assertNull(validateChatImageSubmission("thin", urls))
    }

    @Test
    fun `image submission rejects invalid format but allows full mode gate`() {
        assertEquals(
            "图像仅支持 PNG/JPEG/WebP base64 data URL",
            validateChatImageSubmission("thin", listOf("https://example/image.png")),
        )
        assertNull(
            validateChatImageSubmission("full", listOf("data:image/png;base64,iVBORw0KGgo=")),
        )
    }

    @Test
    fun `image submission rejects more than four non-empty urls`() {
        val images = (1..5).map { "data:image/png;base64,$it" }
        assertEquals(
            "一次最多附加 4 张图片",
            validateChatImageSubmission("thin", images),
        )
    }

    @Test
    fun `retry state retains image payload`() {
        val image = "data:image/png;base64,iVBORw0KGgo="
        val state = startMessageSubmission(MainUiState(), "看图", listOf(image))
        val retry = startMessageRetry(failMessageSubmission(state, "超时"))

        assertEquals(listOf(image), retry.lastSentImageDataUrls)
        assertEquals("看图", retry.lastSentMessage)
    }

    // ---- T12：validateChatImageSubmission 分支补齐（测试修复票排期） ----

    private fun legalPngUrl(miB: Double): String {
        val prefix = byteArrayOf(
            0x89.toByte(), 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a,
        )
        val target = (miB * 1024.0 * 1024.0).toInt()
        val bytes = ByteArray(target)
        System.arraycopy(prefix, 0, bytes, 0, prefix.size)
        val b64 = java.util.Base64.getEncoder().encodeToString(bytes)
        return "data:image/png;base64,$b64"
    }

    @Test
    fun `T12 single image over 8 MiB is rejected`() {
        val tooBig = legalPngUrl(8.1)  // > MAX_CHAT_IMAGE_BYTES
        assertEquals(
            "单张图片不得超过 8 MiB",
            validateChatImageSubmission("thin", listOf(tooBig)),
        )
    }

    @Test
    fun `T12 invalid base64 payload is rejected`() {
        // 格式匹配（合法 base64 字符）但长度非法 -> decode 抛
        assertEquals(
            "图像 data URL 的 base64 数据无效",
            validateChatImageSubmission("thin", listOf("data:image/png;base64,A")),
        )
        assertEquals(
            "图像 data URL 的 base64 数据无效",
            validateChatImageSubmission("thin", listOf("data:image/png;base64,=")),
        )
        // 非 base64 字符（!）-> 格式分支拒绝
        assertEquals(
            "图像仅支持 PNG/JPEG/WebP base64 data URL",
            validateChatImageSubmission(
                "thin",
                listOf("data:image/png;base64,!!!invalid!!!not-base64"),
            ),
        )
    }

    @Test
    fun `T12 minimal decoded payload is rejected by signature`() {
        // 单字符 payload 解码后 1 字节，PNG 签名不全 -> MIME 不一致
        val oneByte = "data:image/png;base64,AA=="
        assertEquals(
            "图像 MIME 类型与文件签名不一致",
            validateChatImageSubmission("thin", listOf(oneByte)),
        )
    }

    @Test
    fun `T12 mime signature mismatch is rejected`() {
        // data URL 声明 png 但内容非 PNG 签名
        val mismatch = "data:image/png;base64," +
            java.util.Base64.getEncoder().encodeToString("not-a-png".toByteArray())
        assertEquals(
            "图像 MIME 类型与文件签名不一致",
            validateChatImageSubmission("thin", listOf(mismatch)),
        )
    }

    @Test
    fun `T12 total size over 16 MiB is rejected`() {
        // 三张 ~5.6 MiB 合法 PNG = 16.8 MiB > 16 MiB 总上限
        val three = listOf(legalPngUrl(5.6), legalPngUrl(5.6), legalPngUrl(5.6))
        assertEquals(
            "图像总大小不得超过 16 MiB",
            validateChatImageSubmission("thin", three),
        )
    }
}
