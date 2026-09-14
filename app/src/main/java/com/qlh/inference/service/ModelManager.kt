package com.qlh.inference.service

import android.content.Context
import android.content.Intent
import android.net.Uri
import android.os.ParcelFileDescriptor
import android.provider.DocumentsContract
import android.provider.OpenableColumns
import android.util.Log
import com.qlh.inference.data.SettingsDataStore
import com.qlh.inference.model.Gemma4AssetFile
import com.qlh.inference.model.Gemma4NativeAssetStatus
import com.qlh.inference.model.Gemma4NativeAssets
import com.qlh.inference.network.GgufModelInfo
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.currentCoroutineContext
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
import java.util.concurrent.TimeUnit

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

        private const val GGUF_EXTENSION = ".gguf"
    }

    private val settings = SettingsDataStore(context)

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

    /** 默认内部测试模型路径。 */
    val defaultModelPath: String
        get() = File(modelsDir, DEFAULT_MODEL_FILENAME).absolutePath

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

                val models = listSafModels(treeUri)
                if (models.size == 1) {
                    settings.setSelectedModelUri(models.first().uri.toString())
                }
                Result.success(models)
            } catch (e: Exception) {
                Log.e(TAG, "保存 SAF 模型目录权限失败", e)
                Result.failure(e)
            }
        }

    suspend fun listModels(): Result<List<ModelDocument>> = withContext(Dispatchers.IO) {
        try {
            val treeUri = getSavedTreeUri()
            if (treeUri != null && hasPersistedPermission(treeUri)) {
                return@withContext Result.success(listSafModels(treeUri))
            }

            val internalModels = modelsDir
                .listFiles { file -> file.isFile && file.name.endsWith(GGUF_EXTENSION, true) }
                ?.sortedBy { it.name.lowercase() }
                ?.map {
                    ModelDocument(
                        name = it.name,
                        uri = Uri.fromFile(it),
                        sizeBytes = it.length(),
                        source = ModelSource.INTERNAL
                    )
                }
                ?: emptyList()
            Result.success(internalModels)
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
                settings.setModelStorageMode(STORAGE_MODE_INTERNAL_TEST)
                settings.setModelPath(modelUri.path.orEmpty())
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
            source = ModelSource.INTERNAL
        )
    }

    suspend fun getSelectedModelUri(): String = settings.getSelectedModelUri()

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
                                mode = STORAGE_MODE_INTERNAL_TEST,
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

    suspend fun hasModelDirectoryPermission(): Boolean {
        val treeUri = getSavedTreeUri() ?: return false
        return hasPersistedPermission(treeUri)
    }

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

    /**
     * 递归扫描 SAF 目录。MVP 限制为 2 层，覆盖常见 Download/QLH/models 场景，
     * 避免用户误选 Download 根目录时完全扫不到模型。
     */
    private fun collectSafModels(
        treeUri: Uri,
        parentDocumentId: String,
        output: MutableList<ModelDocument>,
        depth: Int,
        maxDepth: Int
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
                            maxDepth = maxDepth
                        )
                    }
                    continue
                }

                if (!name.endsWith(GGUF_EXTENSION, ignoreCase = true)) continue

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
                    source = ModelSource.SAF
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

    private fun hasPersistedPermission(uri: Uri): Boolean {
        return context.contentResolver.persistedUriPermissions.any {
            it.uri == uri && it.isReadPermission
        }
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
        if (!hasPersistedPermission(treeUri)) {
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

    private fun computeSha256(uri: Uri): String {
        val digest = MessageDigest.getInstance("SHA-256")
        context.contentResolver.openInputStream(uri)?.use { input ->
            val buffer = ByteArray(1024 * 1024)
            while (true) {
                val count = input.read(buffer)
                if (count < 0) break
                digest.update(buffer, 0, count)
            }
        } ?: throw IOException("无法读取模型文件进行 SHA-256 校验")
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
        INTERNAL
    }

    data class ModelDocument(
        val name: String,
        val uri: Uri,
        val sizeBytes: Long,
        val source: ModelSource
    )

    data class ModelOpenHandle(
        val loadPath: String,
        val displayName: String,
        val sourceUri: Uri,
        val mode: String,
        val pfd: ParcelFileDescriptor?
    ) : AutoCloseable {
        @Volatile
        private var closed = false

        override fun close() {
            if (closed) return
            closed = true
            try {
                pfd?.close()
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
