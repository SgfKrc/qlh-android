package com.qlh.inference.worker

import com.google.gson.JsonObject
import com.google.gson.JsonParser

import android.annotation.SuppressLint
import android.app.Notification
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.os.Binder
import android.os.Build
import android.os.IBinder
import androidx.core.app.NotificationCompat
import androidx.core.content.ContextCompat
import com.qlh.inference.BuildConfig
import com.qlh.inference.control.AndroidQuorumCertificate
import com.qlh.inference.control.AndroidVoterSet
import com.qlh.inference.control.validateAndroidCertificate
import com.qlh.inference.MainActivity
import com.qlh.inference.QlhApplication
import com.qlh.inference.R
import com.qlh.inference.data.SettingsDataStore
import com.qlh.inference.data.WorkerParticipationSnapshot
import com.qlh.inference.logging.QlhLogger
import com.qlh.inference.service.InferenceService
import com.qlh.inference.service.inferenceModes
import com.qlh.inference.service.ModelManager
import com.qlh.inference.service.ModelSupplyManifest
import com.qlh.inference.service.ModelSupplyCurrent
import com.qlh.inference.service.ModelSupplyException
import com.qlh.inference.service.GgufSegmentCutter
import com.qlh.inference.network.canonicalHost
import com.qlh.inference.service.LayerArtifactInventory
import com.qlh.inference.service.toCandidate
import com.qlh.inference.service.unambiguousLayerArtifactCandidates
import com.qlh.inference.system.AndroidDeviceInfoProvider
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.NonCancellable
import kotlinx.coroutines.flow.distinctUntilChanged
import kotlinx.coroutines.launch
import kotlinx.coroutines.cancel
import kotlinx.coroutines.delay
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicReference

/** Foreground lifecycle shell for the Android Full Worker client. */
class TaskWorkerService : Service() {
    private val runtimeIncarnation = java.util.UUID.randomUUID().toString()

    private val binder = LocalBinder()
    private val lifecycleLock = Any()
    private var destroyed = false
    @Volatile
    private var client: TaskWorkerClient? = null
    private var snapshotJob: Job? = null
    private var activeWorkerKey: TaskWorkerStartKey? = null
    private var activeInferenceStartAccepted = false
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
    private data class WorkerCommand(val intent: Intent?, val startId: Int)
    private val commandQueue = TaskWorkerCommandQueue<WorkerCommand>(
        scope = scope,
        handle = { command -> handleStartCommand(command.intent, command.startId) },
        onFailure = { command, error ->
            QlhLogger.e("TaskWorkerService", "worker command failed; stopping worker", error)
            runCatching { settings.clearTaskWorkerStartupConfig() }
                .onFailure { failure ->
                    QlhLogger.e("TaskWorkerService", "could not clear worker startup", failure)
                }
            inferenceModes.exclusive { stopWorker() }
            stopSelf(command.startId)
        },
    )

    /** ★ 2026-10-07（DIST-NEXT-5）：启动配置的持久化载体（系统重建时读回）。 */
    private val settings by lazy { SettingsDataStore(applicationContext) }

    inner class LocalBinder : Binder() {
        fun getService(): TaskWorkerService = this@TaskWorkerService
        fun snapshot(): TaskWorkerSnapshot = this@TaskWorkerService.snapshot()
    }

    override fun onCreate() {
        super.onCreate()
        startForegroundNotification()
        scope.launch {
            settings.inferenceMode.distinctUntilChanged().collect { mode ->
                if (mode != SettingsDataStore.MODE_DISTRIBUTED) {
                    inferenceModes.exclusive {
                        if (settings.getInferenceMode() != SettingsDataStore.MODE_DISTRIBUTED) {
                            settings.clearTaskWorkerStartupConfig()
                            stopWorker()
                            stopSelf()
                        }
                    }
                }
            }
        }
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        val queuedIntent = intent?.let(::Intent)
        intent?.removeExtra(EXTRA_CLUSTER_SECRET)
        if (!commandQueue.enqueue(WorkerCommand(queuedIntent, startId))) {
            QlhLogger.e("TaskWorkerService", "worker command queue unavailable")
            stopSelf(startId)
        }
        return if (BuildConfig.IS_LITE) START_NOT_STICKY else START_STICKY
    }

    private suspend fun handleStartCommand(intent: Intent?, startId: Int) = inferenceModes.exclusive {
        handleStartCommandLocked(intent, startId)
    }

    private suspend fun handleStartCommandLocked(intent: Intent?, startId: Int) {
        if (settings.getInferenceMode() != SettingsDataStore.MODE_DISTRIBUTED) {
            settings.clearTaskWorkerStartupConfig()
            stopWorker()
            stopSelf(startId)
            return
        }
        val action = intent?.action
        val persistedRaw = settings.getTaskWorkerStartupConfig()
        val persisted = TaskWorkerStartupConfig.fromJson(persistedRaw)
        val legacyWorkerSecret = TaskWorkerStartupConfig.legacyClusterSecretFromJson(persistedRaw)
        var clusterSecret = runCatching { settings.getClusterSecret() }
            .onFailure { error ->
                QlhLogger.e(
                    "TaskWorkerService",
                    "cluster credential unavailable; worker startup will fail closed",
                    error,
                )
            }
            .getOrDefault("")
        if (clusterSecret.isBlank() && !legacyWorkerSecret.isNullOrBlank()) {
            clusterSecret = runCatching {
                settings.saveClusterSecret(legacyWorkerSecret)
                settings.getClusterSecret()
            }.onFailure { error ->
                QlhLogger.e(
                    "TaskWorkerService",
                    "legacy worker credential migration failed; worker startup will fail closed",
                    error,
                )
            }.getOrDefault("")
        }
        if (
            TaskWorkerStartupConfig.containsLegacyClusterSecret(persistedRaw) &&
            (legacyWorkerSecret.isNullOrBlank() || clusterSecret.isNotBlank())
        ) {
            if (persisted == null) {
                settings.clearTaskWorkerStartupConfig()
            } else {
                settings.setTaskWorkerStartupConfig(persisted.toJson())
            }
        }
        when (decideTaskWorkerStartup(
            action = action,
            hasPersistedConfig = persisted?.usable == true && clusterSecret.isNotBlank(),
            hasActiveClient = client?.isRunning == true,
        )) {
            TaskWorkerStartupDecision.Start -> {
                val config = startupConfigFromIntent(intent)
                val incomingSecret = intent?.getStringExtra(EXTRA_CLUSTER_SECRET).orEmpty()
                intent?.removeExtra(EXTRA_CLUSTER_SECRET)
                if (config == null || !config.usable || incomingSecret.isBlank()) {
                    // 配置不足以建 client ⇒ 明确停止，并把坏配置清掉（免得重建时再撞一次）。
                    QlhLogger.w(
                        "TaskWorkerService",
                        "ACTION_START rejected: incomplete worker configuration",
                    )
                    settings.clearTaskWorkerStartupConfig()
                    stopSelf(startId)
                    return
                }
                val securedSecret = runCatching {
                    settings.saveClusterSecret(incomingSecret)
                    settings.getClusterSecret()
                }.onFailure { error ->
                    QlhLogger.e(
                        "TaskWorkerService",
                        "ACTION_START rejected: cluster credential could not be secured",
                        error,
                    )
                }.getOrDefault("")
                if (securedSecret.isBlank()) {
                    settings.clearTaskWorkerStartupConfig()
                    stopSelf(startId)
                    return
                }
                // 持久化：系统回收后凭它恢复（`START_STICKY` 重建路径的前提）。
                settings.setTaskWorkerStartupConfig(config.toJson())
                startWorker(config, securedSecret)
            }
            TaskWorkerStartupDecision.Resume -> {
                QlhLogger.i(
                    "TaskWorkerService",
                    "system recreated the worker without an intent; resuming from " +
                        "persisted config (node=${persisted?.nodeId} host=${persisted?.coordinatorHost})",
                )
                startWorker(persisted!!, clusterSecret)
            }
            TaskWorkerStartupDecision.Stop -> {
                // 用户主动停止 ⇒ 连同持久化配置一起清掉，否则系统重建会把它拉起来。
                settings.clearTaskWorkerStartupConfig()
                stopWorker()
                stopSelf(startId)
            }
            TaskWorkerStartupDecision.Cancel ->
                client?.cancelActive(intent?.getStringExtra(EXTRA_REASON) ?: "user_cancelled")
            TaskWorkerStartupDecision.Ignore -> Unit
        }
    }

    /** 把 `ACTION_START` 的 extras 解析成可持久化的启动配置（校验交给 `usable`）。 */
    @Suppress("UNCHECKED_CAST")
    private fun startupConfigFromIntent(intent: Intent?): TaskWorkerStartupConfig? {
        if (intent == null) return null
        val deviceInfo = intent.getStringExtra(EXTRA_DEVICE_INFO_JSON).orEmpty()
            .let { raw ->
                runCatching {
                    com.google.gson.Gson().fromJson(raw, Map::class.java) as? Map<String, Any?>
                }.getOrNull().orEmpty()
            }
        return TaskWorkerStartupConfig(
            coordinatorHost = intent.getStringExtra(EXTRA_COORDINATOR_HOST).orEmpty().trim(),
            coordinatorPort = intent.getIntExtra(EXTRA_COORDINATOR_PORT, 0),
            nodeId = intent.getStringExtra(EXTRA_NODE_ID).orEmpty().trim(),
            clusterSecretEpoch = intent.getIntExtra(EXTRA_CLUSTER_SECRET_EPOCH, 1),
            hostname = intent.getStringExtra(EXTRA_HOSTNAME).orEmpty(),
            networkType = intent.getStringExtra(EXTRA_NETWORK_TYPE).orEmpty(),
            deviceInfo = deviceInfo,
            modelId = intent.getStringExtra(EXTRA_MODEL_ID).orEmpty().trim(),
            modelFormat = intent.getStringExtra(EXTRA_MODEL_FORMAT).orEmpty(),
            modelRevision = intent.getStringExtra(EXTRA_MODEL_REVISION).orEmpty(),
            modelSha256 = intent.getStringExtra(EXTRA_MODEL_SHA256).orEmpty().trim(),
            resourceAdmitted = intent.getBooleanExtra(EXTRA_RESOURCE_ADMITTED, false),
            resourceReason = intent.getStringExtra(EXTRA_RESOURCE_REASON).orEmpty().trim(),
            fullInferenceAvailable = intent.getBooleanExtra(EXTRA_FULL_INFERENCE_AVAILABLE, true),
        )
    }

    override fun onBind(intent: Intent?): IBinder = binder

    override fun onDestroy() {
        commandQueue.close()
        val old = synchronized(lifecycleLock) {
            destroyed = true
            client.also { it?.stop() }
        }
        if (old != null) {
            // Keep ownership registered until even non-cancellable JNI work has exited.
            scope.launch(NonCancellable) { inferenceModes.retireWorker(old) }
        }
        snapshotJob?.cancel()
        mutableWorkerSnapshot.value = old?.snapshot?.value ?: TaskWorkerSnapshot()
        scope.cancel()
        super.onDestroy()
    }

    fun snapshot(): TaskWorkerSnapshot = client?.snapshot?.value ?: TaskWorkerSnapshot()

    fun cancelActive(reasonCode: String = "user_cancelled"): Boolean = client?.cancelActive(reasonCode) == true

    private suspend fun startWorker(config: TaskWorkerStartupConfig, clusterSecret: String) {
        if (BuildConfig.IS_LITE) {
            stopSelf()
            return
        }
        val modelTreeUri = settings.getModelTreeUri()
        val selectedModelUri = settings.getSelectedModelUri()
        val treePermissionGranted = modelTreeUri.isBlank() ||
            contentResolver.persistedUriPermissions.any {
                it.uri.toString() == modelTreeUri && it.isReadPermission
            }
        val existing = client
        if (canReuseTaskWorker(
                processRunning = existing?.isRunning == true,
                connectionAlive = existing?.isConnectionLive == true,
                configurationMatches = activeWorkerKey?.matches(
                    config, modelTreeUri, selectedModelUri, clusterSecret, treePermissionGranted,
                ) == true,
                inferenceAvailable = activeInferenceStartAccepted || QlhApplication.instance.inferenceService != null,
            )
        ) {
            QlhLogger.i("TaskWorkerService", "worker_start_reused=true connection_alive=true")
            return
        }
        QlhLogger.i("TaskWorkerService", "worker_start_reused=false " +
            "process_running=${existing?.isRunning == true} connection_alive=${existing?.isConnectionLive == true} " +
            "previous_connection=${existing?.snapshot?.value?.connection}")
        // A recreated service must release any previous local full model before hello/capabilities.
        if (!inferenceModes.prepareWorkerLocked(
                readMode = { settings.getInferenceMode() },
                releaseModel = {
                    QlhApplication.instance.inferenceService?.unloadModelLocked()
                        ?: Result.success(Unit)
                },
            )) return
        // ★ 2026-10-07（DIST-NEXT-5）：参数来自 `TaskWorkerStartupConfig`（可持久化）
        //   而不是直接读 intent —— 系统重建路径因此与首次启动走**同一条**代码。
        //   校验已由 `config.usable` 完成（调用点判定）。
        val host = config.coordinatorHost
        val port = config.coordinatorPort
        val nodeId = config.nodeId
        val hostname = config.hostname.ifBlank { nodeId }
        val networkType = config.networkType.ifBlank { "unknown" }
        val deviceInfo = config.deviceInfo
        val suppliedIdentity = AtomicReference<Pair<String, String>?>(null)
        val modelId = config.modelId
        val modelFormat = config.modelFormat.ifBlank { "gguf" }
        val modelRevision = config.modelRevision.ifBlank { "local" }
        val modelSha256 = config.modelSha256
        val resourceAdmitted = config.resourceAdmitted
        val resourceReason = config.resourceReason
        val fullInferenceAvailable = false // Distributed workers never accept full-model loads.
        val runtimeProfile = AndroidWorkerCapabilities.normalizeRuntimeProfile(
            deviceInfo["runtime_profile"]?.toString(),
        )
        // The distributed route must be able to advertise crop artifacts before
        // the local inference service has loaded anything (or even exists yet).
        val modelManager = QlhApplication.instance.inferenceService?.modelManager
            ?: ModelManager(this)
        val supplyBinding = modelManager.currentSupplyBinding()
        val supplyCurrent = ModelSupplyCurrent(modelManager.suppliedSegmentsDir)
        val currentSupply = { supplyCurrent.read(supplyBinding)?.takeIf { it.nodeId == nodeId } }
        val currentModelId = { currentSupply()?.sourceModelId ?: suppliedIdentity.get()?.first ?: config.modelId }
        val currentModelSha = { currentSupply()?.sourceSha256 ?: suppliedIdentity.get()?.second ?: config.modelSha256 }
        // Capability construction is the integrity boundary: it hashes each
        // artifact once. Decode offers only consult this immutable verified
        // snapshot; re-hashing multi-GB GGUF files for every token would make
        // the stage path unusable.
        val verifiedLayerArtifacts = AtomicReference<Pair<String?, List<ModelManager.LayerArtifact>>>(null to emptyList())
        val verifiedInventory = AtomicReference<LayerArtifactInventory?>(null)
        val layerArtifactCapability = { artifact: ModelManager.LayerArtifact,
                                        scan: GgufLayerWeightReader.ScanResult? ->
            val hasModelPreflightContract =
                artifact.sourceModelId != null &&
                    artifact.hiddenSize != null &&
                    artifact.tokenizerSha256 != null
            AndroidWorkerCapabilities.LayerArtifactCapability(
                startLayer = artifact.startLayer,
                endLayerExclusive = artifact.endLayerExclusive,
                segmentMode = artifact.mode,
                modelId = AndroidWorkerCapabilities.artifactModelId(
                    artifact.document.name,
                    artifact.artifactSha256,
                ),
                artifactSha256 = artifact.artifactSha256,
                sourceModelSha256 = artifact.sourceModelSha256.ifBlank { null },
                sourceModelId = artifact.sourceModelId.takeIf { hasModelPreflightContract },
                hiddenSize = artifact.hiddenSize.takeIf { hasModelPreflightContract },
                tokenizerSha256 = artifact.tokenizerSha256.takeIf {
                    hasModelPreflightContract
                },
                layerWeightBytes = scan?.layerWeightBytes,
            )
        }
        var inferenceStartAccepted = true
        if (QlhApplication.instance.inferenceService == null && !BuildConfig.IS_LITE) {
            runCatching {
                ContextCompat.startForegroundService(this, Intent(this, InferenceService::class.java))
            }.onFailure { error ->
                inferenceStartAccepted = false
                QlhLogger.w("TaskWorkerService", "inference service start failed: ${error.message}")
            }
        }
        val verifiedResourceAdmitted = AtomicBoolean(false)
        val expectedModelIdentity = {
            AndroidWorkerCapabilities.modelIdentity(
                currentModelId(), modelFormat, modelRevision, currentModelSha(),
                verifiedResourceAdmitted.get(),
            )
        }
        val buildCapabilities: suspend () -> Map<String, Any?> = suspend {
            val supplyManifest = currentSupply()?.manifestSha256
            var modelId = currentModelId()
            var modelSha256 = currentModelSha()
            val inferenceReady = QlhApplication.instance.inferenceService != null ||
                (inferenceStartAccepted && awaitInferenceService() != null)
            val scanStartedAtMs = android.os.SystemClock.elapsedRealtime()
            val inventory = if (inferenceReady) {
                modelManager.scanLayerArtifacts(
                    expectedModelSha256 = modelSha256,
                    verifyArtifactDigest = true,
                ).getOrNull()
            } else null
            verifiedInventory.set(inventory)
            if (modelId.isBlank() || !modelSha256.matches(Regex("[0-9a-fA-F]{64}"))) {
                inventory?.artifacts.orEmpty().mapNotNull { artifact ->
                    artifact.sourceModelId?.let { it to artifact.sourceModelSha256 }
                }.distinct().singleOrNull()?.let { identity ->
                    suppliedIdentity.set(identity); modelId = identity.first; modelSha256 = identity.second
                }
            }
            val eligibleArtifacts = inventory?.artifacts.orEmpty().filter {
                it.sourceModelSha256.equals(modelSha256, ignoreCase = true) &&
                    (it.sourceModelId == null || it.sourceModelId == modelId)
            }
            val scansByArtifact = mutableMapOf<ModelManager.LayerArtifact, GgufLayerWeightReader.ScanResult?>()
            val checkedArtifacts = eligibleArtifacts.filter { artifact ->
                val scan = try {
                    contentResolver.openInputStream(artifact.document.uri)?.use { input ->
                        GgufLayerWeightReader.scan(
                            input, artifact.endLayerExclusive - artifact.startLayer,
                        )
                    }
                } catch (cancelled: CancellationException) {
                    throw cancelled
                } catch (error: Exception) {
                    QlhLogger.w("TaskWorkerService", "layer tensor scan failed: ${error.message}")
                    null
                }
                scansByArtifact[artifact] = scan
                GgufLayerWeightReader.matchesTokenizerSha256(artifact.tokenizerSha256, scan)
                    .also { matches ->
                        if (!matches) {
                            QlhLogger.w("TaskWorkerService", "layer artifact tokenizer mismatch: " +
                                "manifest=${artifact.manifest.name} " +
                                "range=[${artifact.startLayer},${artifact.endLayerExclusive})")
                        }
                    }
            }
            val candidates = checkedArtifacts.map { it.toCandidate() }
            val allowedReferences = unambiguousLayerArtifactCandidates(candidates)
                .map { it.reference }.toSet()
            val layerArtifacts = checkedArtifacts
                .filter { it.document.uri.toString() in allowedReferences }
            if (currentSupply()?.manifestSha256 != supplyManifest) {
                throw ModelSupplyException("SUPPLY_CURRENT_CHANGED", "current changed during capability scan")
            }
            verifiedLayerArtifacts.set(supplyManifest to layerArtifacts.toList())
            QlhLogger.i(
                "TaskWorkerService",
                "hello_scan_ms=${android.os.SystemClock.elapsedRealtime() - scanStartedAtMs} " +
                    "scanned=${inventory != null} eligible_ranges=${layerArtifacts.size}",
            )
            val verifiedAdmission = AndroidWorkerCapabilities.verifiedAdmission(
                layerArtifacts.isNotEmpty(), fullInferenceAvailable, layerArtifacts.size, inferenceReady,
            )
            verifiedResourceAdmitted.set(verifiedAdmission)
            val verifiedReason = if (verifiedAdmission) "" else if (resourceAdmitted) {
                if (inferenceReady) "layer_artifact_unavailable" else "inference_service_unavailable"
            } else resourceReason
            val layerRanges = layerArtifacts
                .map { listOf(it.startLayer, it.endLayerExclusive) }
            // ★ 2026-10-23（#117 对称修复）：持有完整 GGUF 源、能本地裁层的节点，即使
            //   没有任何段工件，也应广告 `layer_forward` 并以 [0, total) 声明"可承担任意
            //   连续层段"。否则 master 永不分配新段，历史上乱切的旧段就永远清不掉
            //   （鸡生蛋）—— 与 Python 侧 `scheduler_task_worker` 的修复对称。
            //   只读 GGUF 头（`GgufSegmentCutter.inspect` 有 HEADER_LIMIT），不解析权重。
            val selfCutRanges: List<List<Int>>? = runCatching {
                // Self-cutting supersedes any single pre-cut segment as the *dispatch*
                // contract: an old segment stays a reusable artefact, but must not pin
                // the layout forever. Only a loaded full model opts out.
                if (fullInferenceAvailable) return@runCatching null
                val dir = java.io.File(filesDir.canonicalFile, "local-cut")
                if (!dir.isDirectory) return@runCatching null
                dir.listFiles { f -> f.isFile && f.extension.equals("gguf", true) }
                    ?.sortedBy { it.name }
                    ?.firstNotNullOfOrNull { f ->
                        runCatching {
                            f.inputStream().use { GgufSegmentCutter.inspect(it).layers }
                        }.getOrNull()?.takeIf { it > 0 }?.let { total -> listOf(listOf(0, total)) }
                    }
            }.getOrNull()
            if (selfCutRanges != null) {
                QlhLogger.i(
                    "TaskWorkerService",
                    "local_cut_source_available ranges=$selfCutRanges (no pre-cut segment)",
                )
            }
            val effectiveLayerRanges = selfCutRanges ?: layerRanges
            val advertisedLayerArtifacts = layerArtifacts.map {
                layerArtifactCapability(it, scansByArtifact[it])
            }
            val layerForwardInfo = QlhApplication.instance.inferenceService
                ?.engine
                ?.layerForwardInfo()
                ?.getOrNull()
                .orEmpty()
            // This legacy budget describes an existing artifact. Source-specific local cutting
            // is verified separately by model_supply inventory, never inferred from this estimate.
            val readyArtifact = layerArtifacts.firstOrNull { it.document.sizeBytes > 0 }
            // A single node-level budget cannot truthfully describe several
            // artifacts with different bytes/layer. Exact per-artifact ranges remain
            // the admission contract; only
            // retain the legacy budget hint when there is one artifact.
            val layerBudget = if (layerArtifacts.size == 1) {
                val availableBytes = runCatching {
                    AndroidDeviceInfoProvider(this@TaskWorkerService).getMemoryStatus().availableBytes
                }.getOrDefault(
                    (deviceInfo["memory"] as? Map<*, *>)
                        ?.get("available_bytes")
                        ?.let { (it as? Number)?.toLong() }
                        ?: 0L,
                )
                AndroidWorkerCapabilities.computeLayerBudget(
                    availableBytes = availableBytes,
                    modelFileBytes = readyArtifact?.document?.sizeBytes ?: 0L,
                    coveredLayers = readyArtifact
                        ?.let { it.endLayerExclusive - it.startLayer } ?: 0,
                    // ★ 2026-10-23（#117）：此前这里**从未传** `localCut`，于是
                    //   `layer_budget.local_cut` 恒为 false，master 的求解器把
                    //   `layer_ranges` 当硬约束（`node_layer_range_not_advertised`）⇒
                    //   预置段永久钉死切点、容量不足也无从重排。有整模源即声明
                    //   "可按分配本地裁层"，让求解器把任意连续段分给本节点。
                    localCut = selfCutRanges != null,
                )
            } else {
                null
            }
            val capabilities = AndroidWorkerCapabilities.build(
                modelId = modelId,
                modelFormat = modelFormat,
                modelRevision = modelRevision,
                modelSha256 = modelSha256,
                resourceAdmitted = verifiedAdmission,
                resourceReason = verifiedReason,
                layerRanges = effectiveLayerRanges,
                fullInferenceAvailable = fullInferenceAvailable,
                layerWorker = !fullInferenceAvailable && effectiveLayerRanges.isNotEmpty(),
                runtimeProfile = runtimeProfile,
                middleChannel = layerForwardInfo["middle_channel"],
                nPosPerEmbd = layerForwardInfo["n_pos_per_embd"]?.toIntOrNull(),
                layerBudget = layerBudget,
                layerArtifacts = advertisedLayerArtifacts,
                // ★ 2026-10-07（DIST-NEXT-6）：不可用工件的结构化原因（无本地路径）。
                layerArtifactDiagnostics = inventory?.failures
                    ?.map { it.toAdvertisement() }
                    .orEmpty(),
                // ★ 2026-10-07（DIST-NEXT-2b）：本构建已接线分片输入（`stage_chunk`
                //   接收 + `hidden_ref` 装配）⇒ 声明给主节点。
                stageChunkedInput = true,
                nodeIncarnation = runtimeIncarnation,
                // ★ 票 21a（COMM-WIRE-01）：本构建已接线二进制 hidden 帧
                //   （`TaskWorkerBinaryHidden` 抽取/放回 + 帧层数据段）⇒ 声明给主节点。
                binaryHidden = true,
                multiPositionArgmax = QlhApplication.instance.inferenceService
                    ?.engine?.supportsMultiPositionArgmax() == true,
            )
            QlhLogger.i(
                "TaskWorkerService",
                "hello capabilities: stages=${capabilities["stage_types"]} " +
                    "ranges=${capabilities["layer_ranges"]} model=${modelId.ifBlank { "<none>" }} " +
                    "sha=${modelSha256.take(12)} profile=$runtimeProfile " +
                    "budget=${layerBudget?.maxLayers} " +
                    "layer_byte_counts=${advertisedLayerArtifacts.map { it.layerWeightBytes?.size ?: 0 }} " +
                "admitted=$verifiedAdmission full=$fullInferenceAvailable",
            )
            capabilities
        }
        val boundClusterId = settings.getClusterId()
        fun bindingCurrent(snapshot: WorkerParticipationSnapshot): Boolean =
            snapshot.bindingCurrent(boundClusterId, host, port, config.clusterSecretEpoch)
        suspend fun participationCurrent(): Boolean = settings.getWorkerParticipationSnapshot()
            .participationCurrent(boundClusterId, host, port, config.clusterSecretEpoch)
        val supply = ModelSupplyController(
            nodeId = nodeId, boundHost = host, root = modelManager.suppliedSegmentsDir,
            distributed = { settings.getInferenceMode() == SettingsDataStore.MODE_DISTRIBUTED },
            bindingCurrent = { bindingCurrent(settings.getWorkerParticipationSnapshot()) },
            memoryAvailable = { AndroidDeviceInfoProvider(this@TaskWorkerService).getMemoryStatus().availableBytes },
            publicationGate = { action ->
                // Read inside the existing mode lock, then use that same generation for binding.
                lateinit var snapshot: WorkerParticipationSnapshot
                inferenceModes.worker({
                    settings.getWorkerParticipationSnapshot().also { snapshot = it }.inferenceMode
                }) {
                    if (!bindingCurrent(snapshot)) Result.failure(ModelSupplyException("SUPPLY_MASTER_BINDING_CHANGED", "master binding changed"))
                    else { action(); Result.success(Unit) }
                }.getOrThrow()
            },
            available = { _ -> },
            bindingSha256 = supplyBinding,
            participationCurrent = { participationCurrent() },
            localSources = modelManager.localCutSources,
        )
        val prepare = WorkerPrepareController(nodeId, object : WorkerPrepareRuntime {
            override suspend fun loadWithSpec(identity: WorkerPrepareIdentity, spec: JsonObject) = loadFrozen(identity, spec)
            override suspend fun receipt(identity: WorkerPrepareIdentity, spec: JsonObject): JsonObject =
                requireNotNull(QlhApplication.instance.inferenceService?.engine).deploymentReceipt(identity, spec)
            override suspend fun load(identity: WorkerPrepareIdentity) = loadFrozen(identity, null)
            private suspend fun loadFrozen(identity: WorkerPrepareIdentity, spec: JsonObject?) {
                val service = awaitInferenceService() ?: throw IllegalStateException("inference_service_unavailable")
                service.beginForegroundWork()
                try {
                    service.ensureLayerModelLoaded(
                        layerRange = identity.range, contextSize = 4096, extractHidden = identity.mode != "tail",
                        expectedModelSha256 = identity.text("artifact_sha256"),
                        expectedSourceModelId = identity.text("source_model_id"),
                        expectedSourceModelSha256 = identity.text("source_model_sha256"),
                        expectedEmbeddingWidth = identity.hiddenSize,
                        expectedArtifactModelId = identity.stageIdentity["model_id"] as String,
                        expectedSupplyManifestSha256 = identity.text("manifest_sha256"),
                        prepareIdentity = identity, frozenCpuPolicy = spec != null,
                    ).getOrThrow()
                } finally { service.endForegroundWork() }
            }
            // No mode-lock acquisition here: stopAndJoin can be called while exclusive is held.
            override suspend fun unload() { QlhApplication.instance.inferenceService?.retirePreparedLayer()?.getOrThrow() }
            override fun loaded(identity: WorkerPrepareIdentity): Boolean =
                QlhApplication.instance.inferenceService?.isPreparedLayerLoaded(identity) == true
            override fun hasRuntime(): Boolean = QlhApplication.instance.inferenceService?.engine?.isLoaded == true
            override fun currentPreparedIdentity(): WorkerPrepareIdentity? =
                QlhApplication.instance.inferenceService?.engine?.loadedPreparedIdentity()
        }, participationCurrent = { participationCurrent() })
        val controlVoterSet = runCatching {
            settings.getQuorumVoterSet().trim().takeIf { it.isNotEmpty() }?.let { raw ->
                AndroidVoterSet.parse(JsonParser.parseString(raw).asJsonObject).also { voterSet ->
                    check(voterSet.clusterId == settings.getClusterId()) {
                        "quorum voter set does not match the worker cluster"
                    }
                }
            }
        }.onFailure { error ->
            QlhLogger.e(
                "TaskWorkerService",
                "control certificate authority unavailable; fenced worker traffic will fail closed",
                error,
            )
        }.getOrNull()
        val controlCertificateValidator: ((JsonObject) -> Unit)? = controlVoterSet?.let { voterSet ->
            { value ->
                validateAndroidCertificate(
                    AndroidQuorumCertificate.parse(value),
                    voterSet,
                    System.currentTimeMillis(),
                )
            }
        }
        val registration = TaskWorkerRegistration(
            nodeId = nodeId,
            clusterSecret = clusterSecret,
            clusterSecretEpoch = config.clusterSecretEpoch,
            hostname = hostname,
            networkType = networkType,
            deviceInfo = deviceInfo,
            modelSha256 = modelSha256,
        )
        val worker = TaskWorkerClient(
            host = host,
            port = port,
            nodeId = nodeId,
            registration = registration,
            transportFactory = SocketTaskWorkerTransportFactory(
                registration = registration,
                controlCertificateValidator = controlCertificateValidator,
            ),
            capabilities = buildCapabilities,
            modelSupplyHandler = supply,
            modelPrepareHandler = prepare,
            stageHandler = AndroidFullWorkerStageExecutor(
                expectedModelIdentity = expectedModelIdentity,
                allowLayerIdentityAlias = { !fullInferenceAvailable },
                ensureModelLoaded = { contextSize ->
                    awaitInferenceService()?.ensureWorkerModelLoaded(contextSize)
                        ?: Result.failure(IllegalStateException("inference_service_unavailable"))
                },
                generate = { prompt, maxTokens, temperature, topP ->
                    val service = awaitInferenceService()
                    if (service == null) {
                        Result.failure(IllegalStateException("inference_service_unavailable"))
                    } else {
                        // ★ 2026-10-07（DIST-NEXT-5b）：stage 执行期间按任务持有 WakeLock
                        //   （空闲即释放，不再固定 30 分钟硬撑）。
                        service.beginForegroundWork()
                        try {
                            service.generateWorker(prompt, maxTokens, temperature, topP)
                        } finally {
                            service.endForegroundWork()
                        }
                    }
                },
                // ★ 2026-09-20（层段）：接上 layer_forward。
                //   中间段需要隐藏态导出 ⇒ 用 ensureModelLoadedForLayer 单独加载。
                //   ⚠️ 「是否导出隐藏态」是两种不同的加载方式，二者**不能共存**
                //      （引擎会按需卸载重载）。这是刻意的 fail-closed：
                //      宁可重载一次，也不静默降级成取不到 hidden。
                layerForward = { req ->
                    val service = awaitInferenceService()
                    val engine = service?.engine
                    if (engine == null) {
                        Result.failure(IllegalStateException("inference_service_unavailable"))
                    } else {
                        // ★ 2026-10-07（DIST-NEXT-5b）：层段前向同样按任务持有 WakeLock
                        //   （这是最容易被厂商省电策略打断的长任务）。
                        service.beginForegroundWork()
                        try {
                            engine.layerForward(
                                hidden = req.hidden,
                                nTokens = req.nTokens,
                                posBase = req.posBase,
                                wantHidden = req.wantHidden,
                                // ★ 2026-09-23：中间段通道由 stage_offer 的 `middle_channel` 决定 ——
                                //   `keep_head_layer_out` ⇒ keep-head（**末层输出**，`output_norm` 之前），
                                //   其余 ⇒ 旧的 `extract_hidden`（`output_norm(H)`，多一次归一化）。
                                keepHead = req.middleChannel == "keep_head_layer_out",
                                // ★ 2026-09-23（A12）：多序列显式位置（协议层已校验；null = 单序列）。
                                seqIds = req.seqIds,
                                positions = req.positions,
                                tokenOutput = req.tokenOutput,
                            ).map { out ->
                                LayerForwardResult(
                                    tokenArgmax = out.tokenArgmax,
                                    hiddenOut = out.hiddenOut,
                                    tokenArgmaxByPosition = out.tokenArgmaxByPosition,
                                    positions = out.positions,
                                    nVocab = out.nVocab,
                                )
                            }
                        } finally {
                            service.endForegroundWork()
                        }
                    }
                },
                // ★ 2026-10-07（DIST-NEXT-1）：取消合同 —— 把中止请求传到 native。
                //   `nativeLayerForward*` 因此在下一个可分割的张量边界退出并返回 -4，
                //   使「取消」不必等整个 stage 跑完；worker 客户端据此把
                //   `execution_in_flight` 升级为 `execution_stopped`。
                requestExecutionAbort = {
                    QlhApplication.instance.inferenceService?.engine
                        ?.requestLayerForwardAbort()
                },
                // ★ 2026-10-07（真机 P0）：与上面成对 —— 新 stage 开始前复位中止标志，
                //   否则「取消后的第一次执行」会被上一次取消的标志直接判失败。
                clearExecutionAbort = {
                    QlhApplication.instance.inferenceService?.engine
                        ?.clearLayerForwardAbort()
                },
                clearLayerKv = {
                    QlhApplication.instance.inferenceService?.engine?.clearLayerKv()
                        ?: Result.failure(IllegalStateException("prepare_runtime_lost"))
                },
                // Stages only inspect a committed native runtime. ensureLayerModelLoaded belongs to prepare.
                ensureModelLoadedForLayer = { range, _, embeddingWidth, wantHidden, sha256 ->
                    val committed = prepare.committedIdentity()
                    if (committed == null || range != committed.range ||
                        embeddingWidth != committed.hiddenSize || wantHidden != (committed.mode != "tail") ||
                        sha256 != committed.text("artifact_sha256") ||
                        QlhApplication.instance.inferenceService?.isPreparedLayerLoaded(committed) != true) {
                        Result.failure(WorkerPrepareException("prepare_runtime_lost", "committed runtime differs"))
                    } else Result.success(Unit)
                },
                resolveLayerModelIdentity = { range, requested ->
                    prepare.committedIdentity()?.takeIf { it.range == range && it.stageIdentity == requested }?.stageIdentity
                },
            ),
        )
        synchronized(lifecycleLock) {
            if (destroyed) throw CancellationException("task worker service destroyed")
            inferenceModes.registerWorker(worker) {
                worker.stopAndJoin()
                if (client === worker) {
                    snapshotJob?.cancel()
                    mutableWorkerSnapshot.value = worker.snapshot.value
                    releaseRuntimeOwner.get()?.takeIf { it.client === worker }?.let { releaseRuntimeOwner.compareAndSet(it, null) }
                    client = null
                    activeWorkerKey = null
                    activeInferenceStartAccepted = false
                }
            }
            client = worker
            releaseRuntimeOwner.set(RuntimeOwner(worker, prepare))
            activeWorkerKey = TaskWorkerStartKey(
                config, modelTreeUri, selectedModelUri, clusterSecret, treePermissionGranted,
            )
            activeInferenceStartAccepted = inferenceStartAccepted
            worker.start()
            snapshotJob?.cancel()
            snapshotJob = scope.launch {
                worker.snapshot.collect { snapshot ->
                    if (client === worker) {
                        val previous = mutableWorkerSnapshot.value
                        mutableWorkerSnapshot.value = snapshot
                        if (previous.connection != snapshot.connection || previous.lastErrorCode != snapshot.lastErrorCode) {
                            startForegroundNotification(snapshot)
                        }
                    }
                }
            }
        }
    }

    /**
     * ★ 2026-10-07（DIST-NEXT-5）：InferenceService 的**就绪屏障**。
     *
     * 此前固定等 5s（50 × 100ms），慢启动时 stage offer 会直接吃到
     * `inference_service_unavailable` —— 那不是「服务不可用」，而是「还没起完」。
     * 现在等待窗口由常量给出（默认 30s，与主仓 task-worker 控制面 health 超时同量级），
     * 超时仍返回 `null`（fail-closed），但会留下具名告警，便于与真实缺服务区分。
     */
    private suspend fun awaitInferenceService(
        timeoutMs: Long = INFERENCE_READY_TIMEOUT_MS,
    ): InferenceService? {
        val deadline = System.currentTimeMillis() + timeoutMs
        while (System.currentTimeMillis() < deadline) {
            QlhApplication.instance.inferenceService?.let { return it }
            delay(INFERENCE_READY_POLL_MS)
        }
        QlhApplication.instance.inferenceService?.let { return it }
        QlhLogger.w(
            "TaskWorkerService",
            "inference_service_unavailable after ${timeoutMs}ms wait " +
                "(service still starting or not admitted)",
        )
        return null
    }

    private suspend fun stopWorker() {
        inferenceModes.stopWorkerLocked()
        snapshotJob?.cancel()
        mutableWorkerSnapshot.value = TaskWorkerSnapshot()
        activeWorkerKey = null
        activeInferenceStartAccepted = false
        client = null
    }

    @SuppressLint("ForegroundServiceType")
    private fun startForegroundNotification(snapshot: TaskWorkerSnapshot = TaskWorkerSnapshot()) {
        val pendingIntent = PendingIntent.getActivity(
            this,
            1,
            Intent(this, MainActivity::class.java),
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE,
        )
        val notification: Notification = NotificationCompat.Builder(
            this,
            QlhApplication.NOTIFICATION_CHANNEL_INFERENCE,
        )
            .setContentTitle("QLH Worker")
            .setContentText(taskWorkerConnectionLabel(snapshot))
            .setSmallIcon(android.R.drawable.ic_media_play)
            .setOngoing(true)
            .setContentIntent(pendingIntent)
            .setPriority(NotificationCompat.PRIORITY_LOW)
            .build()
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.UPSIDE_DOWN_CAKE) {
            startForeground(
                QlhApplication.NOTIFICATION_ID_WORKER,
                notification,
                android.content.pm.ServiceInfo.FOREGROUND_SERVICE_TYPE_SPECIAL_USE,
            )
        } else {
            startForeground(QlhApplication.NOTIFICATION_ID_WORKER, notification)
        }
    }

    companion object {
        private val mutableWorkerSnapshot = MutableStateFlow(TaskWorkerSnapshot())
        val workerSnapshot = mutableWorkerSnapshot.asStateFlow()
        private data class RuntimeOwner(val client: TaskWorkerClient, val prepare: WorkerPrepareController)
        private val releaseRuntimeOwner = AtomicReference<RuntimeOwner?>(null)
        internal fun committedRuntimeIdentity(): WorkerPrepareIdentity? = releaseRuntimeOwner.get()?.prepare?.committedIdentity()

        const val ACTION_START = "com.qlh.inference.worker.START"
        const val ACTION_STOP = "com.qlh.inference.worker.STOP"
        const val ACTION_CANCEL = "com.qlh.inference.worker.CANCEL"

        /**
         * ★ 2026-10-07（DIST-NEXT-5）：InferenceService 就绪屏障的等待窗口。
         *
         * 与主仓 `scheduler_types.TASK_WORKER_HEALTH_TIMEOUT_FLOOR_SECONDS`（30s）同量级：
         * 设备侧等模型/引擎起完，而不是把「还没起完」报成「服务不可用」。
         */
        const val INFERENCE_READY_TIMEOUT_MS: Long = 30_000L
        const val INFERENCE_READY_POLL_MS: Long = 100L
        const val EXTRA_COORDINATOR_HOST = "coordinator_host"
        const val EXTRA_COORDINATOR_PORT = "coordinator_port"
        const val EXTRA_NODE_ID = "node_id"
        const val EXTRA_CLUSTER_SECRET = "cluster_secret"
        const val EXTRA_CLUSTER_SECRET_EPOCH = "cluster_secret_epoch"
        const val EXTRA_HOSTNAME = "hostname"
        const val EXTRA_NETWORK_TYPE = "network_type"
        const val EXTRA_DEVICE_INFO_JSON = "device_info_json"
        const val EXTRA_MODEL_ID = "model_id"
        const val EXTRA_MODEL_FORMAT = "model_format"
        const val EXTRA_MODEL_REVISION = "model_revision"
        const val EXTRA_MODEL_SHA256 = "model_sha256"
        const val EXTRA_RESOURCE_ADMITTED = "resource_admitted"
        const val EXTRA_RESOURCE_REASON = "resource_reason"
        const val EXTRA_FULL_INFERENCE_AVAILABLE = "full_inference_available"
        const val EXTRA_REASON = "reason"

        fun startIntent(
            context: Context,
            host: String,
            port: Int,
            nodeId: String,
            clusterSecret: String,
            clusterSecretEpoch: Int = 1,
            hostname: String,
            networkType: String,
            deviceInfo: Map<String, Any?>,
            modelId: String = "",
            modelFormat: String = "gguf",
            modelRevision: String = "local",
            modelSha256: String = "",
            resourceAdmitted: Boolean = false,
            resourceReason: String = "resource_gate_not_confirmed",
            fullInferenceAvailable: Boolean = true,
        ): Intent = Intent(
            context,
            TaskWorkerService::class.java,
        ).setAction(ACTION_START)
            .putExtra(EXTRA_COORDINATOR_HOST, host)
            .putExtra(EXTRA_COORDINATOR_PORT, port)
            .putExtra(EXTRA_NODE_ID, nodeId)
            .putExtra(EXTRA_CLUSTER_SECRET, clusterSecret)
            .putExtra(EXTRA_CLUSTER_SECRET_EPOCH, clusterSecretEpoch.coerceAtLeast(1))
            .putExtra(EXTRA_HOSTNAME, hostname)
            .putExtra(EXTRA_NETWORK_TYPE, networkType)
            .putExtra(EXTRA_DEVICE_INFO_JSON, com.google.gson.Gson().toJson(deviceInfo))
            .putExtra(EXTRA_MODEL_ID, modelId)
            .putExtra(EXTRA_MODEL_FORMAT, modelFormat)
            .putExtra(EXTRA_MODEL_REVISION, modelRevision)
            .putExtra(EXTRA_MODEL_SHA256, modelSha256)
            .putExtra(EXTRA_RESOURCE_ADMITTED, resourceAdmitted)
            .putExtra(EXTRA_RESOURCE_REASON, resourceReason)
            .putExtra(EXTRA_FULL_INFERENCE_AVAILABLE, fullInferenceAvailable)

        fun stopIntent(context: Context): Intent = Intent(context, TaskWorkerService::class.java)
            .setAction(ACTION_STOP)

        fun requestStop(context: Context) {
            val appContext = context.applicationContext
            ContextCompat.startForegroundService(appContext, stopIntent(appContext))
        }

        fun cancelIntent(context: Context, reasonCode: String = "user_cancelled"): Intent = Intent(
            context,
            TaskWorkerService::class.java,
        ).setAction(ACTION_CANCEL).putExtra(EXTRA_REASON, reasonCode)
    }
}
