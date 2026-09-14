package com.qlh.inference.network

import android.util.Log
import com.google.gson.Gson
import com.qlh.inference.BuildConfig
import com.qlh.inference.data.MessageDao
import com.qlh.inference.logging.QlhLogger
import com.qlh.inference.data.MessageEntity
import com.qlh.inference.data.SessionDao
import com.qlh.inference.data.SessionEntity
import com.qlh.inference.service.InferenceService
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.first
import java.util.Base64

/**
 * 聊天数据仓库 — 统一管理本地 Room 数据库 + 远程 API + 本地推理。
 *
 * 双模式路由：
 * - 「全无」模式 (thin)：消息通过 HTTP 发送到 PC 主节点 → 回复存入 Room
 * - 「全有」模式 (full)：本地 llama.cpp 引擎直接推理 → 回复存入 Room
 *
 * 消息始终持久化在 Room，离线可查看历史。
 */
class ChatRepository(
    private val sessionDao: SessionDao,
    private val messageDao: MessageDao,
    private val apiClient: () -> ApiClient?,          // 全无模式：HTTP 客户端
    private val inferenceService: () -> InferenceService?  // 全有模式：本地推理引擎
) {
    companion object {
        private const val TAG = "ChatRepository"
    }

    private val gson = Gson()
    private var clientNodeIdProvider: (suspend () -> String?)? = null
    private var thinPresenceHook: (suspend () -> Unit)? = null

    fun setThinClientMetadataProvider(provider: suspend () -> String?) {
        clientNodeIdProvider = provider
    }

    fun setThinPresenceHook(hook: suspend () -> Unit) {
        thinPresenceHook = hook
    }

    // ==================== 会话 ====================

    fun getAllSessions(): Flow<List<SessionEntity>> = sessionDao.getAllSessions()

    suspend fun getSession(sessionId: Long): SessionEntity? = sessionDao.getById(sessionId)

    suspend fun createSession(title: String = "新对话"): Long {
        return sessionDao.insert(SessionEntity(title = title))
    }

    suspend fun deleteSession(sessionId: Long) {
        sessionDao.deleteById(sessionId)
        // 同步删除服务端会话（如果连接可用）
        apiClient()?.deleteSession(sessionId.toString())
    }

    // ==================== 消息 ====================

    fun getMessages(sessionId: Long): Flow<List<MessageEntity>> =
        messageDao.getMessagesBySession(sessionId)

    /**
     * 发送用户消息并获取 AI 回复。
     *
     * 路由逻辑：
     * - apiClient() != null → 全无模式：HTTP → PC 主节点
     * - apiClient() == null → 全有模式：本地 llama.cpp 推理
     *
     * @param sessionId 本地会话 ID
     * @param message 用户输入
     * @param maxTokens 最大 token 数
     * @param temperature 温度
     * @param topP top_p
     * @param showThinking 深度思考（仅全无模式生效）
     * @param skipUserSave 跳过用户消息保存（重试时使用）
     * @return 发送是否成功
     */
    suspend fun sendMessage(
        sessionId: Long,
        message: String,
        maxTokens: Int = 1024,
        temperature: Float = 0.7f,
        topP: Float = 0.9f,
        showThinking: Boolean = false,
        skipUserSave: Boolean = false,
        imageDataUrls: List<String> = emptyList(),
    ): Result<String> {
        val routeClient = apiClient()
        val localService = if (routeClient == null) inferenceService() else null

        // 1. 保存用户消息到本地（重试时跳过，避免重复）
        if (!skipUserSave) {
            val userMsg = MessageEntity(
                sessionId = sessionId,
                role = "user",
                content = message
            )
            messageDao.insert(userMsg)

            // 2. 更新会话标题（首条消息）
            val session = sessionDao.getById(sessionId)
            if (session != null && session.messageCount == 0) {
                val title = if (message.length > 30) message.take(30) + "…" else message
                sessionDao.updateTitle(sessionId, title)
            }
            sessionDao.updateMessageCount(sessionId, messageDao.getCount(sessionId))
        }

        QlhLogger.i(TAG, "user message saved: session=$sessionId skipUserSave=$skipUserSave")

        // 3. 路由到推理引擎
        val client = routeClient
        if (client != null) {
            // ============================================================
            // 全无模式：HTTP → PC 主节点
            // ============================================================
            return sendViaApi(
                client,
                sessionId,
                message,
                maxTokens,
                temperature,
                topP,
                showThinking,
                imageDataUrls,
            )
        }

        // ================================================================
        // 全有模式：本地 llama.cpp 推理
        // ================================================================
        val service = localService ?: inferenceService()
        if (service == null) {
            val errorMsg = MessageEntity(
                sessionId = sessionId,
                role = "assistant",
                content = "本地推理引擎未启动。请检查应用权限和模型文件是否就绪。"
            )
            messageDao.insert(errorMsg)
            sessionDao.updateMessageCount(sessionId, messageDao.getCount(sessionId))
            return Result.failure(IllegalStateException("推理引擎未启动"))
        }

        return try {
            QlhLogger.i(TAG, "full mode ensureModelLoaded start: session=$sessionId")
            // 确保模型已加载（使用用户在设置中选择的上下文长度）
            val loadResult = service.ensureModelLoaded(contextSize = service.modelContextSize)
            if (loadResult.isFailure) {
                val causeMessage = loadResult.exceptionOrNull()?.message.orEmpty()
                val hint = if (
                    causeMessage.contains("本地推理运行时") ||
                    causeMessage.contains("libqlh_llama_jni") ||
                    causeMessage.contains("UnsatisfiedLinkError")
                ) {
                    "请安装包含 llama.cpp 本地运行时的 full-local APK，" +
                        "或切换到全无模式连接 PC 主节点。"
                } else {
                    "请确认：\n" +
                        "1. 已在「设置 → 模型管理」选择 SAF 模型目录\n" +
                        "2. 已扫描并选中一个 .gguf 模型文件\n" +
                        "3. 设备剩余内存和存储空间充足"
                }
                val errorMsg = MessageEntity(
                    sessionId = sessionId,
                    role = "assistant",
                    content = "模型加载失败: $causeMessage\n\n$hint"
                )
                messageDao.insert(errorMsg)
                sessionDao.updateMessageCount(sessionId, messageDao.getCount(sessionId))
                return Result.failure(loadResult.exceptionOrNull()!!)
            }

            // 构建 Qwen chat prompt
            val prompt = buildQwenPrompt(message)

            QlhLogger.i(TAG, "full mode generate start: prompt=${prompt.length} maxTokens=$maxTokens")
            // 执行本地推理（非流式 — 流式版本由 ChatScreen 通过 ViewModel 调用）
            val genResult = if (imageDataUrls.isEmpty()) {
                service.generate(
                    prompt = prompt,
                    maxTokens = maxTokens,
                    temperature = temperature,
                    topP = topP,
                )
            } else {
                val decodedResult = decodeImageDataUrls(imageDataUrls)
                if (decodedResult.isFailure) {
                    Result.failure(decodedResult.exceptionOrNull()!!)
                } else {
                    service.generateMultimodal(
                        prompt = prompt,
                        imageBytes = decodedResult.getOrThrow(),
                        maxTokens = maxTokens,
                        temperature = temperature,
                        topP = topP,
                    )
                }
            }

            genResult.map { response ->
                val assistantMsg = MessageEntity(
                    sessionId = sessionId,
                    role = "assistant",
                    content = response,
                    metrics = """{"engine":"llama.cpp","mode":"full_local"}"""
                )
                messageDao.insert(assistantMsg)
                sessionDao.updateMessageCount(sessionId, messageDao.getCount(sessionId))
                Log.i(TAG, "本地推理完成: ${response.length} chars")
                response
            }
        } catch (e: Exception) {
            QlhLogger.e(TAG, "本地推理失败", e)
            val errorMsg = MessageEntity(
                sessionId = sessionId,
                role = "assistant",
                content = "本地推理失败: ${e.message}"
            )
            messageDao.insert(errorMsg)
            sessionDao.updateMessageCount(sessionId, messageDao.getCount(sessionId))
            Result.failure(e)
        }
    }

    /**
     * 流式发送消息（逐 token 推送）。
     *
     * 全有模式：通过 [InferenceService.generateStream] 获取 Flow。
     * 全无模式：通过 [ApiClient.chatStream]（阶段 3 实现）。
     */
    fun sendMessageStream(
        sessionId: Long,
        message: String,
        maxTokens: Int = 1024,
        temperature: Float = 0.7f,
        topP: Float = 0.9f
    ): Flow<String> {
        val service = inferenceService()
        if (service != null) {
            // 全有模式 — 本地流式推理
            return service.generateStream(
                prompt = buildQwenPrompt(message),
                maxTokens = maxTokens,
                temperature = temperature,
                topP = topP
            )
        }

        // 全无模式 — 返回错误 Flow（SSE 流式暂未接入）
        return kotlinx.coroutines.flow.flow {
            val client = apiClient()
            if (client != null) {
                throw UnsupportedOperationException(
                    "全无模式 SSE 流式暂未实现，请使用非流式 sendMessage()"
                )
            } else {
                throw IllegalStateException("推理引擎未启动且未配置主节点")
            }
        }
    }

    // ================================================================
    // 内部
    // ================================================================

    /** 全无模式：HTTP → PC 主节点 */
    private suspend fun sendViaApi(
        client: ApiClient,
        sessionId: Long,
        message: String,
        maxTokens: Int,
        temperature: Float,
        topP: Float,
        showThinking: Boolean,
        imageDataUrls: List<String>,
    ): Result<String> {
        QlhLogger.i(TAG, "sendViaApi: session=$sessionId message=${message.length} chars")
        thinPresenceHook?.invoke()
        val clientNodeId = clientNodeIdProvider?.invoke()
        val result = client.chat(
            ChatRequest(
                message = message,
                imageDataUrls = imageDataUrls,
                maxNewTokens = maxTokens,
                temperature = temperature,
                topP = topP,
                showThinking = showThinking,
                sessionId = sessionId.toString(),
                clientNodeId = clientNodeId,
                clientNodeType = "android",
                clientMode = "thin",
                routingPreference = "distributed_preferred",
                clientAppVariant = if (BuildConfig.IS_LITE) "lite" else "full",
                allowExternal = imageDataUrls.takeIf { it.isNotEmpty() }?.let { true },
                preferExternal = imageDataUrls.takeIf { it.isNotEmpty() }?.let { true },
            )
        )

        return result.map { response ->
            QlhLogger.i(TAG, "sendViaApi OK: content=${response.content.length} chars")
            thinPresenceHook?.invoke()
            val assistantMsg = MessageEntity(
                sessionId = sessionId,
                role = "assistant",
                content = response.content,
                metrics = response.metrics?.let { gson.toJson(it) }
            )
            messageDao.insert(assistantMsg)
            sessionDao.updateMessageCount(sessionId, messageDao.getCount(sessionId))
            response.content
        }.onFailure { e ->
            QlhLogger.e(TAG, "sendViaApi failed", e)
        }
    }

    /**
     * 构建 Qwen ChatML 格式 prompt（与 PC 端 _build_qwen_prompt 一致）。
     *
     * Android 全有模式下无多轮历史（每次独立推理），
     * 仅构建单轮 user→assistant prompt。
     */
    private fun buildQwenPrompt(userMessage: String): String {
        return buildString {
            append("<|im_start|>user\n")
            append(userMessage)
            append("<|im_end|>\n")
            append("<|im_start|>assistant\n")
        }
    }

    private fun decodeImageDataUrls(imageDataUrls: List<String>): Result<List<ByteArray>> =
        runCatching {
            imageDataUrls.map { dataUrl ->
                val comma = dataUrl.indexOf(',')
                if (!dataUrl.startsWith("data:image/", ignoreCase = true) || comma <= 0) {
                    throw IllegalArgumentException("图像必须是 data URL")
                }
                Base64.getDecoder().decode(dataUrl.substring(comma + 1))
            }
        }

    // ==================== 服务端同步 ====================

    /**
     * 从服务端同步会话列表到本地 Room。
     * 用于首次连接或手动刷新。
     */
    suspend fun syncSessionsFromServer() {
        val client = apiClient() ?: return
        client.getSessions().onSuccess { remoteSessions ->
            for (remote in remoteSessions) {
                val remoteId = remote.id.toLongOrNull() ?: continue
                val local = sessionDao.getById(remoteId)
                if (local == null) {
                    // 新会话 — 创建本地记录
                    sessionDao.insert(
                        SessionEntity(
                            id = remoteId,
                            title = remote.title,
                            messageCount = remote.messageCount
                        )
                    )
                    // 同步消息
                    syncMessagesFromServer(remoteId, client)
                }
            }
        }
    }

    private suspend fun syncMessagesFromServer(sessionId: Long, client: ApiClient) {
        client.getSessionMessages(sessionId.toString()).onSuccess { remoteMessages ->
            for (msg in remoteMessages) {
                messageDao.insert(
                    MessageEntity(
                        sessionId = sessionId,
                        role = msg.role,
                        content = msg.content,
                        metrics = msg.metrics?.let { gson.toJson(it) }
                    )
                )
            }
        }
    }
}
