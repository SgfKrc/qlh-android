package com.qlh.inference.service

import android.content.Context
import android.content.Intent
import android.net.Uri
import android.os.Environment
import android.os.ParcelFileDescriptor
import android.system.Os
import android.system.OsConstants
import android.provider.DocumentsContract
import android.provider.OpenableColumns
import android.util.Log
import com.qlh.inference.data.SettingsDataStore
import com.qlh.inference.model.Gemma4AssetFile
import com.qlh.inference.model.Gemma4NativeAssetStatus
import com.qlh.inference.model.Gemma4NativeAssets
import com.qlh.inference.network.GgufModelInfo
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.delay
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.flowOn
import kotlinx.coroutines.withContext
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.HttpUrl.Companion.toHttpUrl
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.security.MessageDigest
import java.util.Locale
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.atomic.AtomicLong
import java.util.concurrent.TimeUnit

internal suspend fun verifyExpectedArtifactDigest(
    expectedSha256: String,
    readDigest: suspend (refresh: Boolean) -> String,
): Boolean {
    for (refresh in listOf(false, true)) {
        try {
            if (readDigest(refresh).equals(expectedSha256, ignoreCase = true)) return true
        } catch (cancelled: CancellationException) {
            throw cancelled
        } catch (_: Exception) {
            // A fresh, full-byte pass below is the only recovery path.
        }
    }
    return false
}

data class ModelDirectoryGrantState(
    val selected: Boolean = false,
    val readGranted: Boolean = false,
    val writeGranted: Boolean = false,
) {
    val canScan: Boolean get() = selected && readGranted
    val canDownload: Boolean get() = canScan && writeGranted
}

/**
 * GGUF 模型文件管理器。
 *
 * 正式 Android 全有模式优先使用 SAF 用户授权目录保存模型，使模型文件独立于 APK
 * 生命周期；内部目录仍保留为测试和 SAF fallback 的普通文件加载路径。
 */
class ModelManager(private val context: Context) {

    companion object {
        private const val TAG = "ModelManager"
        private const val MODELS_DIR = "models"
        private const val MODEL_CACHE_DIR = "model_cache"
        const val DEFAULT_MODEL_FILENAME = "qwen-1.8b-Q4_K_M.gguf"
        const val SHA256_SUFFIX = ".sha256"
        const val DOWNLOAD_TMP_SUFFIX = ".tmp"

        const val STORAGE_MODE_SAF_FD = "saf_fd"
        const val STORAGE_MODE_SAF_CACHE = "saf_cache"
        const val STORAGE_MODE_INTERNAL_TEST = "internal_test"
        const val STORAGE_MODE_FALLBACK_FILE = "fallback_file"

        private const val GGUF_EXTENSION = ".gguf"
        private const val JSON_EXTENSION = ".json"
        private val verifiedModelDigests = VerifiedModelDigestCache(
            clockMs = { android.os.SystemClock.elapsedRealtime() },
        )
        private val verifiedModelDigestEpochs = ConcurrentHashMap<String, AtomicLong>()
    }

    private val settings = SettingsDataStore(context)

    /**
     * ★ 2026-10-07（DIST-NEXT-6）：最近一次摘要校验扫描的结构化结果（含失败原因）。
     *
     * 供本地诊断 / UI 查询；跨端只上报 [LayerArtifactDiagnostic.toAdvertisement]
     * 那一行（无本地路径）。
     */
    @Volatile
    var lastLayerArtifactInventory: LayerArtifactInventory? = null
        private set

    private val httpClient = OkHttpClient.Builder()
        .connectTimeout(15, TimeUnit.SECONDS)
        .readTimeout(300, TimeUnit.SECONDS)
        .writeTimeout(60, TimeUnit.SECONDS)
        .build()

    /** 模型存储根目录（内部存储，仅用于测试和 SAF fallback 缓存）。 */
    val modelsDir: File
        get() {
            val dir = File(context.filesDir, MODELS_DIR)
            if (!dir.exists()) dir.mkdirs()
            return dir
        }

    // Resolve only the OS-owned app directory alias; private models/supply children reject reparse paths.
    internal val suppliedSegmentsDir: File get() = File(File(context.filesDir.canonicalFile, MODELS_DIR), "supply")

    internal suspend fun currentSupplyBinding(): String = supplyBindingSha256(
        settings.getClusterId(), settings.getMasterTcpHost(), settings.getMasterTcpPort(), settings.getClusterSecretEpoch(),
    )

    internal suspend fun currentSupply(): ModelSupplyManifest? =
        if (settings.isBootstrapped()) ModelSupplyCurrent(suppliedSegmentsDir).read(currentSupplyBinding()) else null

    /** 默认内部测试模型路径。 */
    val defaultModelPath: String
        get() = File(modelsDir, DEFAULT_MODEL_FILENAME).absolutePath

    private val fallbackModelsDir: File
        get() = File(
            Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS),
            "QLH/models",
        )

    private fun listPhysicalModels(
        directory: File,
        source: ModelSource,
        extensions: Set<String> = setOf(GGUF_EXTENSION),
    ): List<ModelDocument> = scanModelFiles(directory, extensions).map { file ->
        ModelDocument(file.name, Uri.fromFile(file), file.length(), source, directory.canonicalPath)
    }

    private fun physicalDocumentKey(document: ModelDocument): String {
        val uri = document.uri
        if (uri.scheme == "file") return File(uri.path.orEmpty()).canonicalPath
        if (uri.scheme == "content" && uri.authority == "com.android.externalstorage.documents") {
            val documentId = runCatching { DocumentsContract.getDocumentId(uri) }.getOrNull()
            if (documentId?.startsWith("primary:", ignoreCase = true) == true) {
                return File(
                    Environment.getExternalStorageDirectory(),
                    documentId.substringAfter(':'),
                ).canonicalPath
            }
        }
        return uri.toString()
    }

    private fun physicalModelSource(file: File): ModelSource =
        if (file.canonicalPath.startsWith(fallbackModelsDir.canonicalPath + File.separator)) {
            ModelSource.FALLBACK
        } else {
            ModelSource.INTERNAL
        }

    private val cacheDir: File
        get() {
            val dir = File(context.cacheDir, MODEL_CACHE_DIR)
            if (!dir.exists()) dir.mkdirs()
            return dir
        }

    // ================================================================
    // SAF 目录与模型选择
    // ================================================================

    suspend fun selectModelDirectory(treeUri: Uri): Result<List<ModelDocument>> =
        withContext(Dispatchers.IO) {
            try {
                val flags = Intent.FLAG_GRANT_READ_URI_PERMISSION or
                    Intent.FLAG_GRANT_WRITE_URI_PERMISSION
                context.contentResolver.takePersistableUriPermission(treeUri, flags)
                settings.setModelTreeUri(treeUri.toString())
                settings.setModelStorageMode(STORAGE_MODE_SAF_FD)
                // 切换目录时旧模型 URI 不再可靠，必须清空，避免 UI/加载仍指向旧目录。
                settings.clearSelectedModelUri()
                settings.clearModelPath()

                val authorizedModels = listSafModels(treeUri)
                if (authorizedModels.size == 1) {
                    settings.setSelectedModelUri(authorizedModels.first().uri.toString())
                }
                listModels()
            } catch (e: Exception) {
                Log.e(TAG, "保存 SAF 模型目录权限失败", e)
                Result.failure(e)
            }
        }

    suspend fun listModels(): Result<List<ModelDocument>> = withContext(Dispatchers.IO) {
        try {
            val treeUri = getSavedTreeUri()
            val saf = if (treeUri != null && hasPersistedPermission(treeUri)) {
                runCatching { listSafModels(treeUri) }
                    .onFailure { Log.w(TAG, "authorized model directory is unavailable", it) }
                    .getOrDefault(emptyList())
            } else emptyList()
            val fallback = runCatching { listPhysicalModels(fallbackModelsDir, ModelSource.FALLBACK) }
                .onFailure { Log.w(TAG, "default model directory is unavailable", it) }
                .getOrDefault(emptyList())
            val internal = listPhysicalModels(modelsDir, ModelSource.INTERNAL)
            val supplied = currentSupply()?.let { current ->
                listPhysicalModels(ModelSupplyCurrent(suppliedSegmentsDir).directory(current), ModelSource.INTERNAL)
            }.orEmpty()
            Result.success((saf + fallback + internal + supplied).distinctBy(::physicalDocumentKey))
        } catch (e: Exception) {
            Log.e(TAG, "扫描模型失败", e)
            Result.failure(e)
        }
    }

    suspend fun selectModel(modelUri: Uri): Result<Unit> = withContext(Dispatchers.IO) {
        try {
            settings.setSelectedModelUri(modelUri.toString())
            if (modelUri.scheme == "content") {
                settings.setModelStorageMode(STORAGE_MODE_SAF_FD)
            } else {
                val file = File(modelUri.path.orEmpty())
                settings.setModelStorageMode(
                    if (physicalModelSource(file) == ModelSource.FALLBACK) {
                        STORAGE_MODE_FALLBACK_FILE
                    } else STORAGE_MODE_INTERNAL_TEST,
                )
                settings.setModelPath(file.absolutePath)
            }
            Result.success(Unit)
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    suspend fun getSelectedModel(): ModelDocument? = withContext(Dispatchers.IO) {
        val selected = settings.getSelectedModelUri()
        if (selected.isBlank()) {
            return@withContext null
        }

        val uri = Uri.parse(selected)
        if (uri.scheme == "content") {
            return@withContext queryContentModel(uri)
        }

        val file = File(uri.path ?: selected)
        if (!file.exists()) {
            return@withContext null
        }
        ModelDocument(
            name = file.name,
            uri = Uri.fromFile(file),
            sizeBytes = file.length(),
            source = physicalModelSource(file)
        )
    }

    suspend fun getSelectedModelUri(): String = settings.getSelectedModelUri()

    /** Compute the selected GGUF identity used by the authenticated task worker. */
    suspend fun getSelectedModelSha256(): String = withContext(Dispatchers.IO) {
        val selected = settings.getSelectedModelUri()
        if (selected.isBlank()) return@withContext ""
        try {
            verifiedSha256(Uri.parse(selected), getSavedTreeUri()?.toString().orEmpty())
        } catch (cancelled: CancellationException) {
            throw cancelled
        } catch (_: Exception) {
            ""
        }
    }

    suspend fun isModelReady(): Boolean = withContext(Dispatchers.IO) {
        val selected = getSelectedModel()
        if (selected != null && selected.sizeBytes != 0L) {
            return@withContext true
        }

        val modelFile = File(defaultModelPath)
        if (!modelFile.exists() || modelFile.length() == 0L) return@withContext false

        val sha256File = File(defaultModelPath + SHA256_SUFFIX)
        if (sha256File.exists()) {
            val expected = sha256File.readText().trim().split("\\s+".toRegex()).firstOrNull() ?: ""
            if (expected.isNotEmpty()) {
                val actual = computeSha256(modelFile)
                return@withContext actual == expected
            }
        }
        true
    }

    suspend fun openModelForLlama(preferFd: Boolean = true): Result<ModelOpenHandle> =
        withContext(Dispatchers.IO) {
            try {
                val selectedUri = settings.getSelectedModelUri()
                if (selectedUri.isNotBlank()) {
                    val uri = Uri.parse(selectedUri)
                    if (uri.scheme == "content") {
                        if (preferFd && settings.getModelStorageMode() != STORAGE_MODE_SAF_CACHE) {
                            openSafFd(uri).onSuccess {
                                return@withContext Result.success(it)
                            }.onFailure { e ->
                                Log.w(TAG, "SAF fd 加载路径不可用，准备缓存副本 fallback: ${e.message}")
                            }
                        }
                        return@withContext openSafCachedCopy(uri)
                    }

                    val file = File(uri.path ?: selectedUri)
                    if (file.exists() && file.canRead()) {
                        return@withContext Result.success(
                            ModelOpenHandle(
                                loadPath = file.absolutePath,
                                displayName = file.name,
                                sourceUri = Uri.fromFile(file),
                                mode = if (physicalModelSource(file) == ModelSource.FALLBACK) {
                                    STORAGE_MODE_FALLBACK_FILE
                                } else STORAGE_MODE_INTERNAL_TEST,
                                pfd = null
                            )
                        )
                    }
                }

                val internalFile = File(defaultModelPath)
                if (internalFile.exists() && internalFile.canRead()) {
                    return@withContext Result.success(
                        ModelOpenHandle(
                            loadPath = internalFile.absolutePath,
                            displayName = internalFile.name,
                            sourceUri = Uri.fromFile(internalFile),
                            mode = STORAGE_MODE_INTERNAL_TEST,
                            pfd = null
                        )
                    )
                }

                Result.failure(
                    IOException(
                        "未找到可加载的 GGUF 模型。请在「设置 → 模型管理」选择 SAF 模型目录并选中模型。"
                    )
                )
            } catch (e: Exception) {
                Result.failure(e)
            }
        }

    /**
     * Resolve and verify the GGUF artifact for one layer stage.
     *
     * Layer stages never fall back to the selected full model. A manifest is
     * required so the loaded GGUF geometry cannot silently disagree with the
     * coordinator's half-open layer range.
     */
    suspend fun openLayerModelForLlama(
        layerRange: List<Int>,
        expectedSourceModelId: String = "",
        expectedSourceModelSha256: String = "",
        expectedArtifactSha256: String = "",
        preferFd: Boolean = true,
        expectedArtifactModelId: String = "",
    ): Result<LayerArtifactHandle> = withContext(Dispatchers.IO) {
        val pendingSupply = ModelSupplyStorage.pending(suppliedSegmentsDir)
        try {
            if (layerRange.size != 2 || layerRange[1] <= layerRange[0] || layerRange[0] < 0) {
                return@withContext Result.failure(IOException("invalid layer range"))
            }
            if (expectedSourceModelId.isBlank() ||
                !expectedSourceModelSha256.matches(Regex("[0-9a-fA-F]{64}")) ||
                !expectedArtifactSha256.matches(Regex("[0-9a-fA-F]{64}"))
            ) {
                return@withContext Result.failure(IOException("source identity and artifact digest are required"))
            }
            val artifacts = listLayerArtifacts(
                expectedModelSha256 = expectedSourceModelSha256,
                verifyArtifactDigest = true,
            ).getOrElse { return@withContext Result.failure(it) }
            val choice = chooseLayerArtifact(
                candidates = artifacts.map { it.toCandidate() },
                masterModelId = expectedSourceModelId,
                masterSourceSha256 = expectedSourceModelSha256,
                startLayer = layerRange[0],
                endLayerExclusive = layerRange[1],
                offeredArtifactSha256 = expectedArtifactSha256,
                offeredArtifactModelId = expectedArtifactModelId,
            )
            val artifact = when (choice) {
                is LayerArtifactChoice.Selected -> artifacts.first {
                    it.document.uri.toString() == choice.artifact.reference
                }
                is LayerArtifactChoice.ManualRequired -> return@withContext Result.failure(
                    IOException("${choice.reason}: ${choice.prompt}"),
                )
            }

            openModelHandle(artifact.document.uri, preferFd).map { handle ->
                LayerArtifactHandle(
                    handle = handle,
                    startLayer = artifact.startLayer,
                    endLayerExclusive = artifact.endLayerExclusive,
                    sourceModelSha256 = artifact.sourceModelSha256,
                    artifactSha256 = artifact.artifactSha256,
                )
            }
        } finally { pendingSupply.close() }
    }

    /** Resolve the exact prepare identity before a native reuse/load; inventory alone is insufficient. */
    internal suspend fun verifyWorkerPrepare(identity: com.qlh.inference.worker.WorkerPrepareIdentity) = withContext(Dispatchers.IO) {
        val pendingSupply = ModelSupplyStorage.pending(suppliedSegmentsDir)
        try {
            val manifest = currentSupply()
            val expectedManifest = identity.text("manifest_sha256")
            if (expectedManifest.isNotEmpty()) {
                com.qlh.inference.worker.checkPrepare(manifest != null, "prepare_manifest_missing", "no selected supply")
                identity.verifyManifest(manifest!!)
            } else {
                com.qlh.inference.worker.checkPrepare(manifest == null, "prepare_manifest_mismatch", "supply cannot be treated as legacy SAF")
            }
            val artifacts = listLayerArtifacts(identity.text("source_model_sha256"), verifyArtifactDigest = true).getOrThrow()
            val matching = artifacts.filter { artifact ->
                identity.range == listOf(artifact.startLayer, artifact.endLayerExclusive) &&
                    artifact.artifactSha256 == identity.text("artifact_sha256") &&
                    artifact.sourceModelSha256 == identity.text("source_model_sha256") &&
                    artifact.sourceModelId == identity.text("source_model_id") &&
                    artifact.tokenizerSha256 == identity.text("tokenizer_sha256") &&
                    artifact.hiddenSize == identity.hiddenSize && artifact.mode == identity.mode &&
                    com.qlh.inference.worker.AndroidWorkerCapabilities.artifactModelId(artifact.document.name, artifact.artifactSha256) ==
                        identity.stageIdentity["model_id"]
            }
            com.qlh.inference.worker.checkPrepare(matching.isNotEmpty(), "prepare_artifact_identity_mismatch", "no verified exact artifact")
            val artifact = matching.first()
            val native = com.google.gson.JsonParser.parseString(readDocument(artifact.manifest.uri)
                ?: throw IOException("prepare native manifest missing")).asJsonObject
            com.qlh.inference.worker.checkPrepare(native["n_layer"]?.asInt == identity.totalLayers,
                "prepare_manifest_mismatch", "source total layers differ")
            if (manifest != null) {
                val expectedReference = Uri.fromFile(File(ModelSupplyCurrent(suppliedSegmentsDir).directory(manifest), manifest.weight.path)).toString()
                com.qlh.inference.worker.checkPrepare(artifact.document.uri.toString() == expectedReference,
                    "prepare_manifest_mismatch", "exact selected supply required")
            }
            val scan = context.contentResolver.openInputStream(artifact.document.uri)?.use {
                com.qlh.inference.worker.GgufLayerWeightReader.scan(it, identity.range[1] - identity.range[0])
            }
            com.qlh.inference.worker.checkPrepare(scan != null &&
                com.qlh.inference.worker.GgufLayerWeightReader.matchesTokenizerSha256(identity.text("tokenizer_sha256"), scan),
                "prepare_gguf_shape_mismatch", "proper decoder segment and tokenizer required before native load")
            com.qlh.inference.worker.checkPrepare(currentSupply()?.manifestSha256.orEmpty() == expectedManifest,
                "prepare_manifest_mismatch", "supply selection changed during verification")
        } finally { pendingSupply.close() }
    }

    /**
     * Return manifests whose artifact files are present in the selected model root.
     *
     * ★ 2026-10-07（DIST-NEXT-6）：fail-closed 语义不变（不可用工件绝不返回），
     * 但**不再静默** —— 需要原因的调用方用 [scanLayerArtifacts] 拿结构化诊断。
     */
    suspend fun listLayerArtifacts(
        expectedModelSha256: String = "",
        verifyArtifactDigest: Boolean = false,
    ): Result<List<LayerArtifact>> =
        scanLayerArtifacts(expectedModelSha256, verifyArtifactDigest)
            .map { it.artifacts }

    /**
     * ★ 2026-10-07（DIST-NEXT-6）：扫描层段工件并**保留每条 manifest 的判定**。
     *
     * 旧实现把「读不到 manifest」「manifest 非法」「工件文件不存在」「摘要不符」
     * 一律 `continue`，返回空/部分列表 ⇒ 主节点只看到 `layer_range_not_advertised`
     * 或没有可用 worker，看不出是哪一条、为什么。这里为每条 manifest 产出
     * [LayerArtifactDiagnostic]（稳定 error code + 已解析出的身份字段），
     * 并逐条写日志；本地路径只留在日志与 [lastLayerArtifactInventory]。
     */
    suspend fun scanLayerArtifacts(
        expectedModelSha256: String = "",
        verifyArtifactDigest: Boolean = false,
    ): Result<LayerArtifactInventory> = withContext(Dispatchers.IO) {
        try {
            val scanStartedAtMs = android.os.SystemClock.elapsedRealtime()
            var digestReused = 0
            var digestStreamed = 0
            val treeUri = getSavedTreeUri()
            val roots = mutableListOf<Triple<String, List<ModelDocument>, List<ModelDocument>>>()
            if (treeUri != null && hasPersistedPermission(treeUri)) {
                runCatching {
                    Triple("saf", listSafModels(treeUri), listSafFiles(treeUri, setOf(JSON_EXTENSION)))
                }.onSuccess { roots += it }
                    .onFailure { Log.w(TAG, "authorized layer directory is unavailable", it) }
            }
            runCatching {
                Triple(
                    "fallback",
                    listPhysicalModels(fallbackModelsDir, ModelSource.FALLBACK),
                    listPhysicalModels(fallbackModelsDir, ModelSource.FALLBACK, setOf(JSON_EXTENSION)),
                )
            }.onSuccess { roots += it }
                .onFailure { Log.w(TAG, "default layer directory is unavailable", it) }
            roots += Triple(
                "internal",
                listPhysicalModels(modelsDir, ModelSource.INTERNAL),
                listPhysicalModels(modelsDir, ModelSource.INTERNAL, setOf(JSON_EXTENSION)),
            )
            val current = currentSupply()
            val currentDirectory = current?.let { ModelSupplyCurrent(suppliedSegmentsDir).directory(it) }
            val currentReference = if (current != null && currentDirectory != null) {
                Uri.fromFile(File(currentDirectory, current.weight.path)).toString()
            } else ""
            var currentFailure: LayerArtifactDiagnostic? = null
            if (current != null && currentDirectory != null) {
                try {
                    val marker = File(currentDirectory, ModelSupplyManifest.MARKER)
                    checkSupply(!java.nio.file.Files.isSymbolicLink(marker.toPath()) && marker.isFile &&
                        marker.length() <= 4 * 1024 * 1024 &&
                        ModelSupplyManifest.parse(marker.readText(Charsets.UTF_8)).json == current.json,
                        "SUPPLY_MANIFEST_DIGEST_MISMATCH", "current marker differs")
                    // Recovery uses the same complete SHA/native/GGUF/tokenizer checks as receipt.
                    if (verifyArtifactDigest) ModelSupplyReceiver(suppliedSegmentsDir, "unused").validateFiles(currentDirectory, current)
                    val documents = listPhysicalModels(currentDirectory, ModelSource.INTERNAL)
                    val native = listPhysicalModels(currentDirectory, ModelSource.INTERNAL, setOf(JSON_EXTENSION))
                        .filter { it.name == current.native.path }
                    roots += Triple("internal-supply", documents, native)
                } catch (cancelled: CancellationException) { throw cancelled }
                catch (error: Exception) {
                    currentFailure = LayerArtifactDiagnostic(
                        manifestName = current.native.path,
                        errorCode = LayerArtifactDiagnostic.ARTIFACT_DIGEST_MISMATCH,
                        detail = (error as? ModelSupplyException)?.reasonCode ?: "current supply unavailable",
                        startLayer = current.start, endLayerExclusive = current.end,
                    )
                }
            }
            val expected = expectedModelSha256.trim().lowercase()
            val storageScope = treeUri?.toString().orEmpty()
            val output = mutableListOf<LayerArtifact>()
            val diagnostics = mutableListOf<LayerArtifactDiagnostic>()
            currentFailure?.let { diagnostics += it }
            if (roots.none { it.first == "saf" } &&
                roots.none { it.first == "fallback" && (it.second.isNotEmpty() || it.third.isNotEmpty()) } &&
                roots.none { it.third.isNotEmpty() }
            ) {
                diagnostics += LayerArtifactDiagnostic(
                    manifestName = "default-storage",
                    errorCode = LayerArtifactDiagnostic.FALLBACK_UNAVAILABLE,
                    detail = "请在设置中授权 Download/QLH/models 子目录",
                )
            }
            for ((_, modelDocuments, manifests) in roots) {
                for (manifest in manifests) {
                    var selectedDocument: ModelDocument? = null
                    val raw = readDocument(manifest.uri)
                    val preflightDescriptor = raw?.let {
                        LayerArtifactManifestParser.parse(it, manifest.name).getOrNull()
                    }
                    val paired = preflightDescriptor?.takeIf {
                        expected.isEmpty() || it.sourceModelSha256 == expected ||
                            it.artifactSha256 == expected
                    }?.let { parsed ->
                        pairLayerArtifactVerified(
                            manifestParentId = manifest.parentId,
                            artifactName = parsed.artifactName,
                            candidates = modelDocuments,
                            parentId = { it.parentId },
                            name = { it.name },
                            verifyArtifactDigest = verifyArtifactDigest,
                            verify = { candidate ->
                                verifyExpectedArtifactDigest(parsed.artifactSha256) { refresh ->
                                    if (refresh) delay(100L)
                                    verifiedSha256(
                                        candidate.uri,
                                        storageScope,
                                        forceRefresh = refresh,
                                    ) { reused ->
                                        if (reused) digestReused++ else digestStreamed++
                                    }
                                }
                            },
                        )
                    }
                    val classification = classifyLayerArtifact(
                        manifestName = manifest.name,
                        raw = raw,
                        expectedModelSha256 = expected,
                        probeArtifact = {
                            val (probed, digestMatches) = paired
                                ?: return@classifyLayerArtifact null
                            selectedDocument = probed
                            LayerArtifactProbe(
                                sizeBytes = probed.sizeBytes,
                                digestMatches = digestMatches,
                            )
                        },
                    )
                    diagnostics += classification.diagnostic
                    val descriptor = classification.descriptor
                    if (!classification.usable || descriptor == null) {
                        Log.w(
                            TAG,
                            "layer artifact unusable: manifest=${manifest.name} " +
                                "code=${classification.diagnostic.errorCode} " +
                                "detail=${classification.diagnostic.detail}",
                        )
                        continue
                    }
                    val document = selectedDocument ?: continue
                    val artifact = LayerArtifact(
                        document = document,
                        manifest = manifest,
                        startLayer = descriptor.startLayer,
                        endLayerExclusive = descriptor.endLayerExclusive,
                        artifactSha256 = descriptor.artifactSha256,
                        sourceModelSha256 = descriptor.sourceModelSha256,
                        sourceModelId = descriptor.sourceModelId,
                        hiddenSize = descriptor.hiddenSize,
                        tokenizerSha256 = descriptor.tokenizerSha256,
                        architecture = descriptor.architecture,
                        // ★ 2026-10-05（DIST-3）：段类型透传（`head`/`middle`/`tail`）。
                        mode = descriptor.mode,
                    )
                    output += artifact
                }
            }
            val allowedReferences = preferCurrentSupplyCandidates(output.map { it.toCandidate() }, current, currentReference)
                .map { it.reference }.toSet()
            val distinctArtifacts = output.filter { it.document.uri.toString() in allowedReferences }.distinctBy {
                listOf(
                    it.sourceModelSha256.lowercase(), it.sourceModelId,
                    it.startLayer, it.endLayerExclusive, it.artifactSha256.lowercase(),
                    it.document.name, it.mode, it.architecture, it.hiddenSize,
                    it.tokenizerSha256,
                )
            }
            distinctArtifacts.groupBy { it.startLayer to it.endLayerExclusive }
                .values.filter { it.size > 1 }
                .forEach { alternatives ->
                    val first = alternatives.first()
                    diagnostics += LayerArtifactDiagnostic(
                        manifestName = first.manifest.name,
                        errorCode = LayerArtifactDiagnostic.ARTIFACT_AMBIGUOUS,
                        detail = "同一区间存在多个不同工件，需指定工件摘要或移走冲突工件",
                        startLayer = first.startLayer,
                        endLayerExclusive = first.endLayerExclusive,
                        artifactPresent = true,
                    )
                }
            checkSupply(currentSupply()?.manifestSha256 == current?.manifestSha256,
                "SUPPLY_CURRENT_CHANGED", "current changed during artifact scan")
            val inventory = LayerArtifactInventory(
                artifacts = distinctArtifacts,
                diagnostics = diagnostics,
                root = roots.filter { it.second.isNotEmpty() || it.third.isNotEmpty() }
                    .joinToString("+") { it.first }.ifBlank { "internal" },
                manifestCount = roots.sumOf { it.third.size },
                artifactFileCount = roots.sumOf { it.second.size },
            )
            lastLayerArtifactInventory = if (verifyArtifactDigest) inventory else null
            for (diagnostic in diagnostics) {
                Log.i(
                    TAG,
                    "layer artifact: manifest=${diagnostic.manifestName} " +
                        "code=${diagnostic.errorCode.ifBlank { "ok" }} " +
                        "range=[${diagnostic.startLayer},${diagnostic.endLayerExclusive}) " +
                        "artifact=${diagnostic.artifactName} " +
                        "mode=${diagnostic.mode} arch=${diagnostic.architecture} " +
                        "source_sha=${diagnostic.sourceModelSha256.take(12)} " +
                        "artifact_sha=${diagnostic.artifactSha256.take(12)} " +
                        "detail=${diagnostic.detail}",
                )
            }
            Log.i(
                TAG,
                "layer artifact scan: verified=$verifyArtifactDigest " +
                    "elapsed_ms=${android.os.SystemClock.elapsedRealtime() - scanStartedAtMs} " +
                    "digest_reused=$digestReused digest_streamed=$digestStreamed " +
                    "root=${inventory.root} " +
                    "manifests=${inventory.manifestCount} files=${inventory.artifactFileCount} " +
                    "usable=${inventory.artifacts.size} failures=${inventory.failureCounts()}",
            )
            Result.success(inventory)
        } catch (cancelled: CancellationException) {
            throw cancelled
        } catch (error: Exception) {
            Log.e(TAG, "layer artifact scan failed", error)
            Result.failure(error)
        }
    }

    private suspend fun verifiedSha256(
        uri: Uri,
        storageScope: String,
        forceRefresh: Boolean = false,
        onLookup: (Boolean) -> Unit = {},
    ): String {
        val documentIdentity = "$storageScope|$uri"
        val epoch = verifiedModelDigestEpochs.computeIfAbsent(documentIdentity) { AtomicLong() }
        if (forceRefresh) epoch.incrementAndGet()
        val version = fileVersionToken(uri) ?: "unversioned"
        return verifiedModelDigests.digest(
            "$documentIdentity|$version|${epoch.get()}",
            { fileFingerprint(uri) },
            onLookup,
        ) {
            computeSha256(uri)
        }
    }

    private fun fileFingerprint(uri: Uri): ModelFileFingerprint? = runCatching {
        val stat = statDigestFile(uri)
        stat?.takeIf {
            OsConstants.S_ISREG(it.st_mode) && it.st_ino > 0L &&
                it.st_size >= 0L && it.st_ctime > 0L
        }?.let { valid ->
            ModelFileFingerprint(
                device = valid.st_dev,
                inode = valid.st_ino,
                size = valid.st_size,
                modifiedSeconds = valid.st_mtime,
                changedSeconds = valid.st_ctime,
            )
        }
    }.getOrNull()

    private fun fileVersionToken(uri: Uri): String? = runCatching {
        statDigestFile(uri)?.takeIf {
            OsConstants.S_ISREG(it.st_mode) && it.st_ino > 0L && it.st_size >= 0L
        }?.let(::statVersionToken)
    }.getOrNull()

    private fun statDigestFile(uri: Uri): android.system.StructStat? = when {
        uri.scheme == "file" -> {
            val file = File(uri.path.orEmpty())
            if (file.isFile && file.canRead()) Os.stat(file.canonicalPath) else null
        }
        uri.scheme == "content" &&
            uri.authority == "com.android.externalstorage.documents" -> {
            context.contentResolver.openFileDescriptor(uri, "r")?.use { descriptor ->
                Os.fstat(descriptor.fileDescriptor)
            }
        }
        else -> null
    }

    private fun statVersionToken(stat: android.system.StructStat): String = listOf(
        stat.st_dev,
        stat.st_ino,
        stat.st_size,
        stat.st_mtim.tv_sec,
        stat.st_mtim.tv_nsec,
        stat.st_ctim.tv_sec,
        stat.st_ctim.tv_nsec,
    ).joinToString(":")

    private fun openDigestDescriptor(uri: Uri): ParcelFileDescriptor {
        if (uri.scheme == "file") {
            val file = File(uri.path.orEmpty())
            if (!file.isFile || !file.canRead()) {
                throw IOException("model artifact is not readable: ${file.name}")
            }
            return ParcelFileDescriptor.open(file, ParcelFileDescriptor.MODE_READ_ONLY)
        }
        return context.contentResolver.openFileDescriptor(uri, "r")
            ?: throw IOException("cannot open model artifact for SHA-256 verification")
    }

    private fun descriptorVersionToken(descriptor: ParcelFileDescriptor): String? = runCatching {
        val stat = Os.fstat(descriptor.fileDescriptor)
        stat.takeIf {
            OsConstants.S_ISREG(it.st_mode) && it.st_ino > 0L && it.st_size >= 0L
        }?.let(::statVersionToken)
    }.getOrNull()

    private suspend fun openModelHandle(uri: Uri, preferFd: Boolean): Result<ModelOpenHandle> {
        if (uri.scheme == "content") {
            if (preferFd && settings.getModelStorageMode() != STORAGE_MODE_SAF_CACHE) {
                val fdResult = openSafFd(uri)
                if (fdResult.isSuccess) return fdResult
            }
            return openSafCachedCopy(uri)
        }
        val file = File(uri.path ?: uri.toString())
        if (!file.exists() || !file.canRead()) {
            return Result.failure(IOException("model artifact is not readable: ${file.name}"))
        }
        return Result.success(
            ModelOpenHandle(
                loadPath = file.absolutePath,
                displayName = file.name,
                sourceUri = Uri.fromFile(file),
                mode = if (physicalModelSource(file) == ModelSource.FALLBACK) {
                    STORAGE_MODE_FALLBACK_FILE
                } else STORAGE_MODE_INTERNAL_TEST,
                pfd = null,
            )
        )
    }

    /** Open the fixed Gemma4 mmproj from the same user-owned storage root. */
    suspend fun openGemma4MmprojForLlama(preferFd: Boolean = true): Result<ModelOpenHandle> =
        withContext(Dispatchers.IO) {
            try {
                val treeUri = getSavedTreeUri()
                if (treeUri != null && hasPersistedPermission(treeUri)) {
                    val document = listSafModels(treeUri).firstOrNull {
                        it.name.equals(Gemma4NativeAssets.MMPROJ_FILENAME, ignoreCase = true)
                    } ?: return@withContext Result.failure(
                        IOException("未找到 Gemma4 mmproj: ${Gemma4NativeAssets.MMPROJ_FILENAME}")
                    )
                    if (preferFd && settings.getModelStorageMode() != STORAGE_MODE_SAF_CACHE) {
                        openSafFd(document.uri).onSuccess {
                            return@withContext Result.success(it)
                        }
                    }
                    return@withContext openSafCachedCopy(document.uri)
                }

                val file = File(modelsDir, Gemma4NativeAssets.MMPROJ_FILENAME)
                if (!file.exists() || !file.canRead()) {
                    return@withContext Result.failure(
                        IOException("未找到 Gemma4 mmproj: ${file.name}")
                    )
                }
                Result.success(
                    ModelOpenHandle(
                        loadPath = file.absolutePath,
                        displayName = file.name,
                        sourceUri = Uri.fromFile(file),
                        mode = STORAGE_MODE_INTERNAL_TEST,
                        pfd = null,
                    )
                )
            } catch (error: Exception) {
                Result.failure(error)
            }
        }

    suspend fun deleteSelectedModel(): Result<Unit> = withContext(Dispatchers.IO) {
        try {
            val selectedUri = settings.getSelectedModelUri()
            if (selectedUri.isBlank()) {
                return@withContext Result.success(Unit)
            }

            val uri = Uri.parse(selectedUri)
            if (uri.scheme == "content") {
                if (!DocumentsContract.deleteDocument(context.contentResolver, uri)) {
                    throw IOException("删除模型失败: ${getDisplayName(uri).ifBlank { uri.toString() }}")
                }
            } else {
                val file = File(uri.path ?: selectedUri)
                if (file.exists() && !file.delete()) {
                    throw IOException("删除模型失败: ${file.absolutePath}")
                }
            }

            settings.clearSelectedModelUri()
            settings.clearModelPath()
            clearCachedCopies()
            Result.success(Unit)
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    suspend fun modelDirectoryGrantState(): ModelDirectoryGrantState =
        modelDirectoryGrantState(settings.getModelTreeUri())

    fun modelDirectoryGrantState(treeUri: String): ModelDirectoryGrantState {
        if (treeUri.isBlank()) return ModelDirectoryGrantState()
        val selectedUri = runCatching { Uri.parse(treeUri) }.getOrNull()
            ?: return ModelDirectoryGrantState(selected = true)
        val matching = context.contentResolver.persistedUriPermissions.filter {
            sameSafTree(it.uri, selectedUri)
        }
        return ModelDirectoryGrantState(
            selected = true,
            readGranted = matching.any { it.isReadPermission },
            writeGranted = matching.any { it.isWritePermission },
        )
    }

    suspend fun hasModelDirectoryPermission(): Boolean = modelDirectoryGrantState().canScan

    /**
     * Inspect the fixed Gemma4 native model/mmproj pair without opening either file.
     * This is intentionally a registration gate; JNI capability and SHA-256 checks
     * are reported separately by the runtime/import paths.
     */
    suspend fun inspectGemma4NativeAssets(): Gemma4NativeAssetStatus =
        withContext(Dispatchers.IO) {
            try {
                val treeUri = getSavedTreeUri()
                val files = if (treeUri != null && hasPersistedPermission(treeUri)) {
                    listSafModels(treeUri).map { Gemma4AssetFile(it.name, it.sizeBytes) }
                } else {
                    modelsDir.listFiles { file ->
                        file.isFile && file.name.endsWith(GGUF_EXTENSION, ignoreCase = true)
                    }?.map { Gemma4AssetFile(it.name, it.length()) }.orEmpty()
                }
                Gemma4NativeAssets.inspect(files)
            } catch (error: Exception) {
                Gemma4NativeAssets.inspect(emptyList()).copy(
                    reason = "asset scan failed: ${error.message ?: error.javaClass.simpleName}"
                )
            }
        }

    private fun listSafModels(treeUri: Uri): List<ModelDocument> {
        val treeDocumentId = DocumentsContract.getTreeDocumentId(treeUri)
        val models = mutableListOf<ModelDocument>()
        collectSafModels(
            treeUri = treeUri,
            parentDocumentId = treeDocumentId,
            output = models,
            depth = 0,
            maxDepth = 2
        )
        return models
            .distinctBy { it.uri.toString() }
            .sortedBy { it.name.lowercase() }
    }

    private fun listSafFiles(treeUri: Uri, extensions: Set<String>): List<ModelDocument> {
        val treeDocumentId = DocumentsContract.getTreeDocumentId(treeUri)
        val files = mutableListOf<ModelDocument>()
        collectSafModels(
            treeUri = treeUri,
            parentDocumentId = treeDocumentId,
            output = files,
            depth = 0,
            maxDepth = 2,
            extensions = extensions,
        )
        return files.distinctBy { it.uri.toString() }.sortedBy { it.name.lowercase() }
    }

    /**
     * 递归扫描 SAF 目录。MVP 限制为 2 层，覆盖常见 Download/QLH/models 场景，
     * 避免用户误选 Download 根目录时完全扫不到模型。
     */
    private fun collectSafModels(
        treeUri: Uri,
        parentDocumentId: String,
        output: MutableList<ModelDocument>,
        depth: Int,
        maxDepth: Int,
        extensions: Set<String> = setOf(GGUF_EXTENSION),
    ) {
        val childrenUri = DocumentsContract.buildChildDocumentsUriUsingTree(treeUri, parentDocumentId)
        val projection = arrayOf(
            DocumentsContract.Document.COLUMN_DOCUMENT_ID,
            DocumentsContract.Document.COLUMN_DISPLAY_NAME,
            DocumentsContract.Document.COLUMN_SIZE,
            DocumentsContract.Document.COLUMN_MIME_TYPE
        )

        context.contentResolver.query(childrenUri, projection, null, null, null)?.use { cursor ->
            val idIndex = cursor.getColumnIndexOrThrow(DocumentsContract.Document.COLUMN_DOCUMENT_ID)
            val nameIndex = cursor.getColumnIndexOrThrow(DocumentsContract.Document.COLUMN_DISPLAY_NAME)
            val sizeIndex = cursor.getColumnIndex(DocumentsContract.Document.COLUMN_SIZE)
            val mimeIndex = cursor.getColumnIndexOrThrow(DocumentsContract.Document.COLUMN_MIME_TYPE)

            while (cursor.moveToNext()) {
                val documentId = cursor.getString(idIndex)
                val mimeType = cursor.getString(mimeIndex).orEmpty()
                val name = cursor.getString(nameIndex).orEmpty()

                if (mimeType == DocumentsContract.Document.MIME_TYPE_DIR) {
                    if (depth < maxDepth) {
                        collectSafModels(
                            treeUri = treeUri,
                            parentDocumentId = documentId,
                            output = output,
                            depth = depth + 1,
                            maxDepth = maxDepth,
                            extensions = extensions,
                        )
                    }
                    continue
                }

                if (extensions.none { name.endsWith(it, ignoreCase = true) }) continue

                val documentUri = DocumentsContract.buildDocumentUriUsingTree(treeUri, documentId)
                val sizeBytes = if (sizeIndex >= 0 && !cursor.isNull(sizeIndex)) {
                    cursor.getLong(sizeIndex)
                } else {
                    -1L
                }
                output += ModelDocument(
                    name = name.ifBlank { "unknown.gguf" },
                    uri = documentUri,
                    sizeBytes = sizeBytes,
                    source = ModelSource.SAF,
                    parentId = parentDocumentId,
                )
            }
        } ?: throw IOException("无法读取模型目录，请重新授权")
    }

    private suspend fun openSafFd(uri: Uri): Result<ModelOpenHandle> {
        return try {
            val pfd = context.contentResolver.openFileDescriptor(uri, "r")
                ?: throw IOException("无法打开模型文件 fd: $uri")
            val loadPath = "/proc/self/fd/${pfd.fd}"
            settings.setModelStorageMode(STORAGE_MODE_SAF_FD)
            Result.success(
                ModelOpenHandle(
                    loadPath = loadPath,
                    displayName = getDisplayName(uri),
                    sourceUri = uri,
                    mode = STORAGE_MODE_SAF_FD,
                    pfd = pfd
                )
            )
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    private suspend fun openSafCachedCopy(uri: Uri): Result<ModelOpenHandle> {
        return try {
            val displayName = getDisplayName(uri).ifBlank { DEFAULT_MODEL_FILENAME }
            val safeName = displayName.replace(Regex("[^A-Za-z0-9._-]"), "_")
            val outputFile = File(cacheDir, "${shortHash(uri.toString())}_$safeName")
            copyUriToFileIfNeeded(uri, outputFile)
            settings.setModelStorageMode(STORAGE_MODE_SAF_CACHE)
            Result.success(
                ModelOpenHandle(
                    loadPath = outputFile.absolutePath,
                    displayName = displayName,
                    sourceUri = uri,
                    mode = STORAGE_MODE_SAF_CACHE,
                    pfd = null
                )
            )
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    private fun copyUriToFileIfNeeded(uri: Uri, outputFile: File) {
        val sourceSize = getSize(uri)
        if (outputFile.exists() && outputFile.length() > 0L) {
            if (sourceSize < 0L || outputFile.length() == sourceSize) {
                return
            }
        }

        val tmpFile = File(outputFile.absolutePath + DOWNLOAD_TMP_SUFFIX)
        context.contentResolver.openInputStream(uri).use { input ->
            if (input == null) throw IOException("无法读取模型文件: $uri")
            FileOutputStream(tmpFile, false).use { output ->
                input.copyTo(output, bufferSize = 1024 * 1024)
            }
        }
        if (outputFile.exists() && !outputFile.delete()) {
            throw IOException("无法替换旧缓存文件: ${outputFile.absolutePath}")
        }
        if (!tmpFile.renameTo(outputFile)) {
            throw IOException("缓存模型重命名失败: ${tmpFile.absolutePath}")
        }
    }

    private fun clearCachedCopies() {
        cacheDir.listFiles()?.forEach { it.delete() }
    }

    private fun shortHash(value: String): String {
        val digest = MessageDigest.getInstance("SHA-256")
            .digest(value.toByteArray(Charsets.UTF_8))
        return digest
            .take(8)
            .joinToString("") { "%02x".format(Locale.US, it) }
    }

    private suspend fun getSavedTreeUri(): Uri? {
        val value = settings.getModelTreeUri()
        return value.takeIf { it.isNotBlank() }?.let(Uri::parse)
    }

    private fun hasPersistedPermission(uri: Uri): Boolean =
        modelDirectoryGrantState(uri.toString()).canScan

    private fun sameSafTree(left: Uri, right: Uri): Boolean {
        if (left == right) return true
        if (left.scheme != right.scheme || left.authority != right.authority) return false
        val leftId = runCatching { DocumentsContract.getTreeDocumentId(left) }.getOrNull()
        val rightId = runCatching { DocumentsContract.getTreeDocumentId(right) }.getOrNull()
        return leftId != null && leftId == rightId
    }

    private fun queryContentModel(uri: Uri): ModelDocument? {
        return try {
            context.contentResolver.openFileDescriptor(uri, "r")?.close()
            ModelDocument(
                name = getDisplayName(uri),
                uri = uri,
                sizeBytes = getSize(uri),
                source = ModelSource.SAF
            )
        } catch (e: Exception) {
            null
        }
    }

    private fun getDisplayName(uri: Uri): String {
        if (uri.scheme != "content") {
            return File(uri.path.orEmpty()).name
        }
        context.contentResolver.query(uri, arrayOf(OpenableColumns.DISPLAY_NAME), null, null, null)
            ?.use { cursor ->
                val index = cursor.getColumnIndex(OpenableColumns.DISPLAY_NAME)
                if (index >= 0 && cursor.moveToFirst()) {
                    return cursor.getString(index).orEmpty()
                }
            }
        return uri.lastPathSegment.orEmpty()
    }

    private fun getSize(uri: Uri): Long {
        if (uri.scheme != "content") {
            return File(uri.path.orEmpty()).length()
        }
        context.contentResolver.query(uri, arrayOf(OpenableColumns.SIZE), null, null, null)
            ?.use { cursor ->
                val index = cursor.getColumnIndex(OpenableColumns.SIZE)
                if (index >= 0 && cursor.moveToFirst() && !cursor.isNull(index)) {
                    return cursor.getLong(index)
                }
            }
        return -1L
    }

    private fun readDocument(uri: Uri): String? = try {
        if (uri.scheme == "file") {
            File(uri.path.orEmpty()).readText()
        } else {
            context.contentResolver.openInputStream(uri)?.bufferedReader()?.use { it.readText() }
        }
    } catch (error: Exception) {
        Log.w(TAG, "cannot read layer manifest: ${error.message}")
        null
    }

    // ================================================================
    // Remote PC GGUF distribution to user-owned SAF storage
    // ================================================================

    fun downloadRemoteModel(
        model: GgufModelInfo,
        baseUrl: String,
    ): Flow<DownloadProgress> = flow {
        validateRemoteModel(model)
        val treeUri = getSavedTreeUri()
            ?: throw IOException("请先选择模型存储目录")
        if (!modelDirectoryGrantState(treeUri.toString()).canDownload) {
            throw IOException("模型目录授权已失效，请重新选择目录")
        }

        val existingFinal = findDirectSafChild(treeUri, model.filename)
        if (existingFinal != null && getSize(existingFinal) == model.sizeBytes) {
            emit(DownloadProgress(model.sizeBytes, model.sizeBytes, 100, phase = "verifying"))
            if (computeSha256(existingFinal) == model.sha256.lowercase()) {
                selectDownloadedSafModel(existingFinal)
                emit(
                    DownloadProgress(
                        model.sizeBytes,
                        model.sizeBytes,
                        100,
                        phase = "completed",
                        verified = true,
                    ),
                )
                return@flow
            }
        }

        val partialName = "${model.filename}.part"
        var partialUri = findDirectSafChild(treeUri, partialName)
            ?: createSafDocument(treeUri, partialName)
        var existingBytes = getSize(partialUri).coerceAtLeast(0L)
        if (existingBytes >= model.sizeBytes) {
            if (
                existingBytes == model.sizeBytes &&
                computeSha256(partialUri) == model.sha256.lowercase()
            ) {
                partialUri = promoteSafDownload(treeUri, partialUri, model.filename, existingFinal)
                selectDownloadedSafModel(partialUri)
                emit(
                    DownloadProgress(
                        model.sizeBytes,
                        model.sizeBytes,
                        100,
                        phase = "completed",
                        verified = true,
                    ),
                )
                return@flow
            }
            existingBytes = 0L
        }

        val downloadUrl = baseUrl.trimEnd('/').toHttpUrl().newBuilder()
            .addPathSegments("api/models/download")
            .addPathSegment(model.filename)
            .build()
        val request = Request.Builder()
            .url(downloadUrl)
            .get()
            .apply {
                if (existingBytes > 0L) header("Range", "bytes=$existingBytes-")
            }
            .build()

        httpClient.newCall(request).execute().use { response ->
            if (!response.isSuccessful) {
                throw IOException("模型下载失败: HTTP ${response.code}")
            }
            val body = response.body ?: throw IOException("模型下载响应体为空")
            val plan = try {
                planModelDownloadWrite(
                    existingBytes = existingBytes,
                    statusCode = response.code,
                    contentLength = body.contentLength(),
                    contentRange = response.header("Content-Range"),
                    expectedTotalBytes = model.sizeBytes,
                )
            } catch (error: IllegalArgumentException) {
                throw IOException("模型续传响应无效: ${error.message}", error)
            }
            val descriptor = context.contentResolver.openFileDescriptor(partialUri, "rw")
                ?: throw IOException("无法写入模型临时文件")
            ParcelFileDescriptor.AutoCloseOutputStream(descriptor).use { output ->
                if (plan.append) {
                    output.channel.position(plan.startBytes)
                } else {
                    output.channel.truncate(0L)
                    output.channel.position(0L)
                }
                body.byteStream().use { input ->
                    val buffer = ByteArray(1024 * 1024)
                    var downloaded = plan.startBytes
                    while (true) {
                        currentCoroutineContext().ensureActive()
                        val count = input.read(buffer)
                        if (count < 0) break
                        output.write(buffer, 0, count)
                        downloaded += count
                        if (downloaded > model.sizeBytes) {
                            throw IOException("模型下载大小超过清单声明")
                        }
                        emit(
                            DownloadProgress(
                                downloadedBytes = downloaded,
                                totalBytes = model.sizeBytes,
                                percent = (downloaded * 100.0 / model.sizeBytes)
                                    .toInt().coerceIn(0, 100),
                                phase = "downloading",
                            ),
                        )
                    }
                    output.flush()
                    if (downloaded != model.sizeBytes) {
                        throw IOException("模型下载不完整: $downloaded/${model.sizeBytes} bytes")
                    }
                }
            }
        }

        emit(DownloadProgress(model.sizeBytes, model.sizeBytes, 100, phase = "verifying"))
        val actualSha256 = computeSha256(partialUri)
        if (actualSha256 != model.sha256.lowercase()) {
            runCatching { DocumentsContract.deleteDocument(context.contentResolver, partialUri) }
            throw IOException("模型 SHA-256 校验失败")
        }

        partialUri = promoteSafDownload(treeUri, partialUri, model.filename, existingFinal)
        selectDownloadedSafModel(partialUri)
        emit(
            DownloadProgress(
                model.sizeBytes,
                model.sizeBytes,
                100,
                phase = "completed",
                verified = true,
            ),
        )
    }.flowOn(Dispatchers.IO)

    private fun validateRemoteModel(model: GgufModelInfo) {
        require(model.filename.endsWith(GGUF_EXTENSION, ignoreCase = true)) {
            "远端模型必须是 GGUF 文件"
        }
        require(
            model.filename.length in 1..255 &&
                '/' !in model.filename && '\\' !in model.filename && ".." !in model.filename
        ) { "远端模型文件名无效" }
        require(model.sizeBytes > 0L) { "远端模型大小无效" }
        require(model.sha256.matches(Regex("[a-fA-F0-9]{64}"))) {
            "远端模型缺少有效 SHA-256"
        }
    }

    private fun findDirectSafChild(treeUri: Uri, displayName: String): Uri? {
        val rootId = DocumentsContract.getTreeDocumentId(treeUri)
        val childrenUri = DocumentsContract.buildChildDocumentsUriUsingTree(treeUri, rootId)
        val projection = arrayOf(
            DocumentsContract.Document.COLUMN_DOCUMENT_ID,
            DocumentsContract.Document.COLUMN_DISPLAY_NAME,
            DocumentsContract.Document.COLUMN_MIME_TYPE,
        )
        context.contentResolver.query(childrenUri, projection, null, null, null)?.use { cursor ->
            val idIndex = cursor.getColumnIndexOrThrow(DocumentsContract.Document.COLUMN_DOCUMENT_ID)
            val nameIndex = cursor.getColumnIndexOrThrow(DocumentsContract.Document.COLUMN_DISPLAY_NAME)
            val mimeIndex = cursor.getColumnIndexOrThrow(DocumentsContract.Document.COLUMN_MIME_TYPE)
            while (cursor.moveToNext()) {
                if (
                    cursor.getString(nameIndex) == displayName &&
                    cursor.getString(mimeIndex) != DocumentsContract.Document.MIME_TYPE_DIR
                ) {
                    return DocumentsContract.buildDocumentUriUsingTree(
                        treeUri,
                        cursor.getString(idIndex),
                    )
                }
            }
        }
        return null
    }

    private fun createSafDocument(treeUri: Uri, displayName: String): Uri {
        val parent = DocumentsContract.buildDocumentUriUsingTree(
            treeUri,
            DocumentsContract.getTreeDocumentId(treeUri),
        )
        return DocumentsContract.createDocument(
            context.contentResolver,
            parent,
            "application/octet-stream",
            displayName,
        ) ?: throw IOException("无法在模型目录创建 $displayName")
    }

    private fun promoteSafDownload(
        treeUri: Uri,
        partialUri: Uri,
        finalName: String,
        previousFinal: Uri?,
    ): Uri {
        if (previousFinal != null && previousFinal != partialUri) {
            if (!DocumentsContract.deleteDocument(context.contentResolver, previousFinal)) {
                throw IOException("无法替换旧模型文件: $finalName")
            }
        }
        return DocumentsContract.renameDocument(
            context.contentResolver,
            partialUri,
            finalName,
        ) ?: findDirectSafChild(treeUri, finalName)
        ?: throw IOException("模型已校验，但无法将临时文件重命名为 $finalName")
    }

    private suspend fun computeSha256(uri: Uri): String {
        val digest = MessageDigest.getInstance("SHA-256")
        val descriptor = openDigestDescriptor(uri)
        val input = ParcelFileDescriptor.AutoCloseInputStream(descriptor)
        try {
            val before = descriptorVersionToken(descriptor)
            val buffer = ByteArray(1024 * 1024)
            while (true) {
                currentCoroutineContext().ensureActive()
                val count = input.read(buffer)
                if (count < 0) break
                digest.update(buffer, 0, count)
            }
            val after = descriptorVersionToken(descriptor)
            if (before != null && before != after) {
                throw IOException("model artifact changed during SHA-256 verification")
            }
        } finally {
            input.close()
        }
        return digest.digest().joinToString("") { "%02x".format(it) }
    }

    private suspend fun selectDownloadedSafModel(uri: Uri) {
        settings.setSelectedModelUri(uri.toString())
        settings.setModelStorageMode(STORAGE_MODE_SAF_FD)
        settings.clearModelPath()
    }

    // ================================================================
    // 旧内部目录下载/校验 API（保留给 PC 下载和测试模式）
    // ================================================================

    fun downloadModel(
        downloadUrl: String,
        localPath: String = defaultModelPath
    ): Flow<DownloadProgress> = flow {
        val tmpFile = File(localPath + DOWNLOAD_TMP_SUFFIX)
        val existingBytes = if (tmpFile.exists()) tmpFile.length() else 0L

        Log.i(TAG, "开始下载模型: $downloadUrl -> $localPath (已有 $existingBytes bytes)")

        val requestBuilder = Request.Builder()
            .url(downloadUrl)
            .get()

        if (existingBytes > 0) {
            requestBuilder.header("Range", "bytes=$existingBytes-")
        }

        val request = requestBuilder.build()
        val response = httpClient.newCall(request).execute()

        if (!response.isSuccessful && response.code != 206) {
            throw IOException("下载失败: HTTP ${response.code} ${response.message}")
        }

        val responseBody = response.body ?: throw IOException("响应体为空")
        val contentLength = responseBody.contentLength()
        val append = existingBytes > 0L && response.code == 206 &&
            response.header("Content-Range")?.startsWith("bytes $existingBytes-") == true
        if (existingBytes > 0L && response.code == 206 && !append) {
            response.close()
            throw IOException("续传响应的 Content-Range 与本地临时文件不一致")
        }
        val startBytes = if (append) existingBytes else 0L
        val totalExpected = if (contentLength > 0) startBytes + contentLength else -1L
        val inputStream = responseBody.byteStream()
        val outputStream = FileOutputStream(tmpFile, append)

        try {
            val buffer = ByteArray(8192)
            var downloaded = startBytes
            var bytesRead: Int

            while (inputStream.read(buffer).also { bytesRead = it } != -1) {
                outputStream.write(buffer, 0, bytesRead)
                downloaded += bytesRead

                emit(
                    DownloadProgress(
                        downloadedBytes = downloaded,
                        totalBytes = totalExpected,
                        percent = if (totalExpected > 0) {
                            (downloaded * 100.0 / totalExpected).toInt().coerceIn(0, 100)
                        } else {
                            -1
                        }
                    )
                )
            }

            outputStream.flush()
        } finally {
            inputStream.close()
            outputStream.close()
            response.close()
        }

        val finalFile = File(localPath)
        if (finalFile.exists()) finalFile.delete()
        if (!tmpFile.renameTo(finalFile)) {
            throw IOException("临时文件重命名失败: ${tmpFile.absolutePath} -> ${finalFile.absolutePath}")
        }

        Log.i(TAG, "模型下载完成: $localPath (${finalFile.length()} bytes)")
        settings.setModelPath(finalFile.absolutePath)
        settings.setSelectedModelUri(Uri.fromFile(finalFile).toString())
        settings.setModelStorageMode(STORAGE_MODE_INTERNAL_TEST)
        emit(
            DownloadProgress(
                downloadedBytes = finalFile.length(),
                totalBytes = finalFile.length(),
                percent = 100
            )
        )
    }.flowOn(Dispatchers.IO)

    suspend fun verifyModel(
        localPath: String = defaultModelPath,
        sha256Url: String = ""
    ): Result<Boolean> = withContext(Dispatchers.IO) {
        try {
            val modelFile = File(localPath)
            if (!modelFile.exists()) {
                return@withContext Result.failure(IOException("模型文件不存在: $localPath"))
            }

            val sha256Content: String = if (sha256Url.isNotEmpty()) {
                val request = Request.Builder().url(sha256Url).get().build()
                val response = httpClient.newCall(request).execute()
                if (!response.isSuccessful) {
                    return@withContext Result.failure(
                        IOException("SHA256 文件下载失败: HTTP ${response.code}")
                    )
                }
                response.body?.string()?.trim() ?: ""
            } else {
                val sha256File = File(localPath + SHA256_SUFFIX)
                if (sha256File.exists()) sha256File.readText().trim() else ""
            }

            if (sha256Content.isEmpty()) {
                Log.w(TAG, "无 SHA256 校验文件，跳过校验")
                return@withContext Result.success(true)
            }

            val expected = sha256Content.split("\\s+".toRegex()).firstOrNull()?.lowercase() ?: ""
            if (expected.length != 64 || !expected.all { it in '0'..'9' || it in 'a'..'f' }) {
                return@withContext Result.failure(IOException("SHA256 格式无效: $expected"))
            }

            val actual = computeSha256(modelFile)
            val ok = actual == expected

            if (ok) {
                File(localPath + SHA256_SUFFIX).writeText("$expected  $DEFAULT_MODEL_FILENAME")
                Log.i(TAG, "模型校验通过: $expected")
            } else {
                Log.e(TAG, "模型校验失败! expected=$expected actual=$actual")
            }

            Result.success(ok)
        } catch (e: Exception) {
            Log.e(TAG, "模型校验异常", e)
            Result.failure(e)
        }
    }

    private fun computeSha256(file: File): String {
        val digest = MessageDigest.getInstance("SHA-256")
        file.inputStream().use { input ->
            val buffer = ByteArray(8192)
            var bytesRead: Int
            while (input.read(buffer).also { bytesRead = it } != -1) {
                digest.update(buffer, 0, bytesRead)
            }
        }
        return digest.digest().joinToString("") { "%02x".format(it) }
    }

    fun getLocalModelInfo(): LocalModelInfo {
        val modelFile = File(defaultModelPath)
        val sha256File = File(defaultModelPath + SHA256_SUFFIX)
        return LocalModelInfo(
            exists = modelFile.exists(),
            path = defaultModelPath,
            sizeBytes = if (modelFile.exists()) modelFile.length() else 0L,
            sha256Verified = sha256File.exists(),
        )
    }

    suspend fun deleteModel(): Result<Unit> = withContext(Dispatchers.IO) {
        try {
            File(defaultModelPath).delete()
            File(defaultModelPath + SHA256_SUFFIX).delete()
            File(defaultModelPath + DOWNLOAD_TMP_SUFFIX).delete()
            val selected = settings.getSelectedModelUri()
            if (selected == Uri.fromFile(File(defaultModelPath)).toString()) {
                settings.clearSelectedModelUri()
            }
            settings.clearModelPath()
            Log.i(TAG, "内部模型文件已删除")
            Result.success(Unit)
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    // ================================================================
    // 数据类
    // ================================================================

    enum class ModelSource {
        SAF,
        FALLBACK,
        INTERNAL
    }

    data class ModelDocument(
        val name: String,
        val uri: Uri,
        val sizeBytes: Long,
        val source: ModelSource,
        val parentId: String = "",
    )

    data class LayerArtifact(
        val document: ModelDocument,
        val manifest: ModelDocument,
        val startLayer: Int,
        val endLayerExclusive: Int,
        val artifactSha256: String,
        val sourceModelSha256: String,
        val sourceModelId: String?,
        val hiddenSize: Int?,
        val tokenizerSha256: String?,
        val architecture: String,
        /**
         * ★ 2026-10-05（DIST-3 实测缺口）：工件段类型（`head`/`middle`/`tail`），
         * 来自 manifest 的 `mode`。随能力快照广告给主仓，避免「中间段被分到末段」
         * —— 中间段的职责与末段位次不符，接末段会执行失败。
         * Missing or invalid modes make the manifest unusable.
         */
        val mode: String,
    )

    data class LayerArtifactHandle(
        val handle: ModelOpenHandle,
        val startLayer: Int,
        val endLayerExclusive: Int,
        val sourceModelSha256: String,
        val artifactSha256: String,
    )

    data class ModelOpenHandle(
        val loadPath: String,
        val displayName: String,
        val sourceUri: Uri,
        val mode: String,
        val pfd: ParcelFileDescriptor?
    ) : AutoCloseable {
        private val supplyReference = ModelSupplyStorage.artifact(File(loadPath))
        @Volatile
        private var closed = false

        @Synchronized
        override fun close() {
            if (closed) return
            try {
                pfd?.close()
                supplyReference?.close()
                closed = true
            } catch (e: Exception) {
                Log.w(TAG, "关闭模型文件句柄失败: ${e.message}")
            }
        }
    }

    data class DownloadProgress(
        val downloadedBytes: Long,
        val totalBytes: Long,
        val percent: Int,
        val phase: String = "downloading",
        val verified: Boolean = false,
    )

    data class LocalModelInfo(
        val exists: Boolean,
        val path: String,
        val sizeBytes: Long,
        val sha256Verified: Boolean
    )
}
