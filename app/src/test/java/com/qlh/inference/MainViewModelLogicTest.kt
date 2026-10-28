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
import com.qlh.inference.network.CurrentModelDescriptor
import com.qlh.inference.network.AuditRequestsResponse
import com.qlh.inference.network.AuditRequestSummary
import com.qlh.inference.network.AuditData
import com.qlh.inference.network.GgufModelInfo
import com.qlh.inference.network.LocalModelAssetSummary
import com.qlh.inference.network.LocalModelAssetsResponse
import com.qlh.inference.network.LocalModelAssetsSummary
import com.qlh.inference.network.MasterModelFleetData
import com.qlh.inference.network.ServerModelSummary
import com.qlh.inference.network.ServerModelsResponse
import com.qlh.inference.service.LayerArtifactCandidate
import com.qlh.inference.service.LayerArtifactDescriptor
import com.qlh.inference.service.ModelDirectoryGrantState
import com.qlh.inference.ui.modelDirectoryStatusLabel
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertSame
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * MainViewModel 纯逻辑与 UI 状态默认值测试（JVM，无需设备）。
 */
class MainViewModelLogicTest {

    @Test
    fun `SAF status reports the persisted grant instead of the saved URI alone`() {
        val selected = "content://com.android.externalstorage.documents/tree/primary%3ADownload"
        assertEquals(
            "默认 Download/QLH/models（无授权时可能不可读）",
            modelDirectoryStatusLabel("", ModelDirectoryGrantState()),
        )
        assertEquals(
            "SAF 外部目录（授权已失效）",
            modelDirectoryStatusLabel(selected, ModelDirectoryGrantState(selected = true)),
        )
        assertEquals(
            "SAF 外部目录（仅有读取授权）",
            modelDirectoryStatusLabel(
                selected,
                ModelDirectoryGrantState(selected = true, readGranted = true),
            ),
        )
        assertEquals(
            "SAF 外部目录（读写授权有效）",
            modelDirectoryStatusLabel(
                selected,
                ModelDirectoryGrantState(
                    selected = true,
                    readGranted = true,
                    writeGranted = true,
                ),
            ),
        )
    }

    @Test
    fun `successful SAF grant clears only stale authorization failure`() {
        val granted = ModelDirectoryGrantState(
            selected = true,
            readGranted = true,
            writeGranted = true,
        )
        assertNull(reconcileRemoteModelMessage("模型下载失败: 模型目录授权已失效", granted))
        assertEquals(
            "模型下载失败: checksum mismatch",
            reconcileRemoteModelMessage("模型下载失败: checksum mismatch", granted),
        )
        assertEquals(
            "模型下载失败: 模型目录授权已失效",
            reconcileRemoteModelMessage(
                "模型下载失败: 模型目录授权已失效",
                ModelDirectoryGrantState(selected = true, readGranted = true),
            ),
        )
    }

    @Test
    fun `coordinator source identity wins over registry alias`() {
        val status = CurrentModelStatus(
            modelId = "top-level-alias",
            descriptor = CurrentModelDescriptor(
                sourceModelId = "qwen3-5-9b",
                modelId = "descriptor-alias",
            ),
        )
        assertEquals("qwen3-5-9b", coordinatorSourceModelId(status))
        assertEquals(
            "descriptor-alias",
            coordinatorSourceModelId(
                status.copy(descriptor = status.descriptor?.copy(sourceModelId = "")),
            ),
        )
        assertEquals(
            "top-level-alias",
            coordinatorSourceModelId(status.copy(descriptor = null)),
        )
    }

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
        assertEquals("tcp_task_worker", payload["connection_type"])
        assertEquals(false, payload["pipeline_worker"])
        assertEquals("distributed", payload["client_mode"])
        assertEquals("full", payload["app_variant"])
        assertEquals("0.1.0", payload["app_version"])
        assertEquals("unspecified", payload["runtime_profile"])
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
            nativeRuntimeAvailable = true,
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
        assertEquals("llama_cpp_only", payload["runtime_profile"])
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

    // ---- ★ 2026-09-30：能力字段必须同源（防「能力撒谎」）----

    /**
     * `pipeline_worker` / `backend_id` / `capabilities` 三者**必须同源**。
     *
     * 真机实测（Y700 / Android 15）曾出现 `pipeline_worker:true` +
     * `backend_id:"llama_cpp"` + `capabilities:[]` 的自相矛盾上报 ——
     * 根因是 task worker 的 device_info 组装**无条件硬写**前两者，
     * 覆盖了按 `nativeRuntimeAvailable` 算出的值。
     * 调度侧 `_client_supports_forward_layers` / `_node_supports_forward_layers`
     * 优先读 `capabilities`、读不到才用 `backend_id` 兜底 ⇒ 字段打架时会一起说谎。
     */
    private fun assertCapabilityFieldsAreConsistent(payload: Map<String, Any?>) {
        val pipelineWorker = payload["pipeline_worker"] as? Boolean
        val backendId = payload["backend_id"] as? String
        @Suppress("UNCHECKED_CAST")
        val capabilities = payload["capabilities"] as? List<String>
        assertNotNull(pipelineWorker)
        assertNotNull(backendId)
        assertNotNull(capabilities)
        if (pipelineWorker == true) {
            assertEquals("声明 pipeline_worker=true 就必须同时给出 backend_id",
                "llama_cpp", backendId)
            assertTrue("声明 pipeline_worker=true 就必须同时声明 forward_layers",
                capabilities!!.contains("forward_layers"))
        } else {
            assertEquals("未声明 pipeline_worker 时 backend_id 必须为空", "", backendId)
            assertTrue("未声明 pipeline_worker 时能力列表必须为空", capabilities!!.isEmpty())
        }
    }

    private fun presencePayload(nativeRuntimeAvailable: Boolean): Map<String, Any?> =
        buildAndroidPresencePayload(
            inferenceMode = "distributed",
            appVariant = "full",
            appVersion = "0.1.0",
            system = sampleSystem(),
            memory = sampleMemory(),
            gpu = sampleGpu(),
            runtime = AndroidRuntimeStatus(
                nativeRuntimeAvailable = nativeRuntimeAvailable, isLite = false),
        )

    @Test
    fun `capability fields stay consistent without native runtime`() {
        val payload = presencePayload(nativeRuntimeAvailable = false)
        assertEquals(false, payload["pipeline_worker"])
        assertCapabilityFieldsAreConsistent(payload)
    }

    @Test
    fun `capability fields stay consistent with native runtime`() {
        val payload = presencePayload(nativeRuntimeAvailable = true)
        assertEquals(true, payload["pipeline_worker"])
        assertEquals("llama_cpp", payload["backend_id"])
        assertCapabilityFieldsAreConsistent(payload)
    }

    /**
     * ★ 该红必须红守卫：`buildTaskWorkerDeviceInfo` **不得**改写能力三项。
     *
     * 修复前它 `put("pipeline_worker", true)` + `put("backend_id", "llama_cpp")`，
     * 在 native runtime 未就绪时就会造出 `pipeline_worker:true` + `capabilities:[]`
     * 的矛盾上报（真机实测形态）。本用例断言 worker 组装后的三项与 presence 真源**逐项相同**。
     */
    @Test
    fun `task worker device info must not override capability fields`() {
        val presence = presencePayload(nativeRuntimeAvailable = false)
        val worker = buildTaskWorkerDeviceInfo(presence)

        assertEquals("worker 专属字段应被补上",
            "tcp_task_worker", worker["connection_type"])
        assertEquals(true, worker["task_worker"])
        for (key in listOf("pipeline_worker", "backend_id", "capabilities")) {
            assertEquals(
                "buildTaskWorkerDeviceInfo 不得改写 $key（能力三项必须同源）",
                presence[key], worker[key],
            )
        }
        assertCapabilityFieldsAreConsistent(worker)

        // 就绪情形同样不得被改写。
        val readyPresence = presencePayload(nativeRuntimeAvailable = true)
        val readyWorker = buildTaskWorkerDeviceInfo(readyPresence)
        for (key in listOf("pipeline_worker", "backend_id", "capabilities")) {
            assertEquals(readyPresence[key], readyWorker[key])
        }
        assertCapabilityFieldsAreConsistent(readyWorker)
    }

    // ---- prepare-only 冷启动工件身份 ----

    private val cropSourceSha = "a".repeat(64)
    private val cropArtifactSha = "b".repeat(64)
    private val cropUri = "content://models/segment.gguf"

    private fun cropCandidate(): LayerArtifactCandidate = LayerArtifactCandidate(
        descriptor = LayerArtifactDescriptor(
            artifactName = "segment.gguf",
            startLayer = 4,
            endLayerExclusive = 16,
            artifactSha256 = cropArtifactSha,
            sourceModelSha256 = cropSourceSha,
            sourceModelId = "qwen3-5-2b",
            hiddenSize = 2048,
            tokenizerSha256 = "e".repeat(64),
            architecture = "qwen35",
            mode = "middle",
        ),
        reference = cropUri,
        manifestName = "segment.manifest.json",
        root = LayerArtifactCandidate.Root.SAF,
    )

    private fun cropIdentity(
        coordinatorModel: CurrentModelStatus? = null,
        selectedUri: String = cropUri,
        selectedName: String = "segment.gguf",
        selectedSizeBytes: Long = 1024L,
        selectedSha256: String = cropArtifactSha,
        verifiedArtifacts: List<LayerArtifactCandidate> = listOf(cropCandidate()),
    ): StandaloneLayerIdentity? = standaloneLayerIdentity(
        coordinatorModel = coordinatorModel,
        selectedModelUri = selectedUri,
        selectedModelName = selectedName,
        selectedModelSizeBytes = selectedSizeBytes,
        selectedSha256 = selectedSha256,
        verifiedArtifacts = verifiedArtifacts,
    )

    @Test
    fun `unloaded master without descriptor can advertise selected verified crop source identity`() {
        val expected = StandaloneLayerIdentity("qwen3-5-2b", cropSourceSha)
        assertEquals(expected, cropIdentity())
        assertEquals(expected, cropIdentity(coordinatorModel = CurrentModelStatus(loaded = false)))
        assertFalse(expected.sourceSha256 == cropArtifactSha)
    }

    @Test
    fun `standalone crop requires explicit matching gguf selection and artifact bytes`() {
        assertNull(cropIdentity(selectedUri = ""))
        assertNull(cropIdentity(selectedUri = "content://models/other.gguf"))
        assertNull(cropIdentity(selectedName = "other.gguf"))
        assertNull(cropIdentity(selectedName = "segment.bin"))
        assertNull(cropIdentity(selectedSizeBytes = 0L))
        assertNull(cropIdentity(selectedSha256 = ""))
        assertNull(cropIdentity(selectedSha256 = "c".repeat(64)))
        assertNull(cropIdentity(verifiedArtifacts = emptyList()))
    }

    @Test
    fun `standalone crop rejects incomplete or invalid manifest preflight identity`() {
        val candidate = cropCandidate()
        val invalid = listOf(
            candidate.copy(descriptor = candidate.descriptor.copy(sourceModelId = null)),
            candidate.copy(descriptor = candidate.descriptor.copy(sourceModelId = "bad/id")),
            candidate.copy(descriptor = candidate.descriptor.copy(sourceModelSha256 = "")),
            candidate.copy(descriptor = candidate.descriptor.copy(hiddenSize = null)),
            candidate.copy(descriptor = candidate.descriptor.copy(hiddenSize = 0)),
            candidate.copy(descriptor = candidate.descriptor.copy(tokenizerSha256 = null)),
            candidate.copy(descriptor = candidate.descriptor.copy(tokenizerSha256 = "invalid")),
            candidate.copy(descriptor = candidate.descriptor.copy(artifactSha256 = "invalid")),
        )
        invalid.forEach { assertNull(cropIdentity(verifiedArtifacts = listOf(it))) }
    }

    @Test
    fun `standalone crop accepts selected artifact alongside other distinct crop`() {
        val selected = cropCandidate().copy(
            descriptor = cropCandidate().descriptor.copy(startLayer = 24, endLayerExclusive = 32),
        )
        val other = cropCandidate().copy(
            descriptor = cropCandidate().descriptor.copy(
                artifactName = "other.gguf",
                artifactSha256 = "c".repeat(64),
                startLayer = 20,
                endLayerExclusive = 24,
            ),
            reference = "content://models/other.gguf",
        )
        assertEquals(
            StandaloneLayerIdentity("qwen3-5-2b", cropSourceSha),
            cropIdentity(verifiedArtifacts = listOf(other, selected)),
        )
    }

    @Test
    fun `standalone crop rejects duplicate selected reference and conflicting interval`() {
        val candidate = cropCandidate()
        val sameFileOtherManifest = candidate.copy(
            descriptor = candidate.descriptor.copy(sourceModelId = "other-source"),
        )
        val sameFileOtherRange = candidate.copy(
            descriptor = candidate.descriptor.copy(startLayer = 16, endLayerExclusive = 24),
        )
        val otherFile = candidate.copy(
            descriptor = candidate.descriptor.copy(artifactName = "other.gguf"),
            reference = "content://models/other.gguf",
        )
        assertNull(cropIdentity(verifiedArtifacts = listOf(candidate, sameFileOtherManifest)))
        assertNull(cropIdentity(verifiedArtifacts = listOf(candidate, sameFileOtherRange)))
        assertNull(cropIdentity(verifiedArtifacts = listOf(candidate, otherFile)))
    }

    @Test
    fun `master identity or active status never falls back to selected crop identity`() {
        val descriptor = CurrentModelDescriptor(
            modelId = "other-model",
            sourceModelSha256 = "c".repeat(64),
        )
        listOf(
            CurrentModelStatus(loaded = true),
            CurrentModelStatus(pipelinePrepared = true),
            CurrentModelStatus(modelId = "registry-id"),
            CurrentModelStatus(descriptor = descriptor),
            CurrentModelStatus(descriptor = CurrentModelDescriptor()),
        ).forEach { assertNull(cropIdentity(coordinatorModel = it)) }
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
    fun `audit projection is bounded to content-free workflow requests`() {
        val data = AuditData(
            requests = AuditRequestsResponse(schemaVersion = 2, revision = 1, validity = "valid",
                requests = (1..12).map { index -> AuditRequestSummary(
                    requestId = "req-$index", deploymentId = "deployment", phase = "finished",
                    executionState = "stopped", terminal = "completed", outcome = "completed") }),
        )

        val snapshot = toAuditSnapshot(data)
        assertEquals(MAX_AUDIT_REQUESTS, snapshot.requests.size)
    }

    @Test
    fun `audit errors do not disclose endpoint details`() {
        assertEquals("无法连接主节点", formatAuditError(java.net.ConnectException("100.90.76.108")))
        assertEquals("读取审计资料超时", formatAuditError(java.net.SocketTimeoutException("Bearer secret")))
        assertEquals("读取审计资料失败", formatAuditError(IllegalStateException("C:/secret")))
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
        assertEquals("主节点拒绝远程推理，请检查服务端访问策略", formatMessageSendError(com.qlh.inference.network.ApiClientHttpException(401, "", "{}")))
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
