package com.qlh.inference.data

import androidx.room.Dao
import androidx.room.Insert
import androidx.room.OnConflictStrategy
import androidx.room.Query
import kotlinx.coroutines.flow.Flow

@Dao
interface MessageDao {

    /** 获取某个会话的所有消息（按时间升序） */
    @Query("SELECT * FROM messages WHERE session_id = :sessionId ORDER BY timestamp ASC")
    fun getMessagesBySession(sessionId: Long): Flow<List<MessageEntity>>

    /** 获取某个会话最近 N 条消息（用于 Context 窗口） */
    @Query("SELECT * FROM messages WHERE session_id = :sessionId ORDER BY timestamp DESC LIMIT :limit")
    suspend fun getRecentMessages(sessionId: Long, limit: Int = 50): List<MessageEntity>

    /** 插入一条消息 */
    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insert(message: MessageEntity): Long

    /** 删除某会话所有消息 */
    @Query("DELETE FROM messages WHERE session_id = :sessionId")
    suspend fun deleteBySession(sessionId: Long)

    /** 删除单条消息 */
    @Query("DELETE FROM messages WHERE id = :messageId")
    suspend fun deleteById(messageId: Long)

    /** 获取会话消息数 */
    @Query("SELECT COUNT(*) FROM messages WHERE session_id = :sessionId")
    suspend fun getCount(sessionId: Long): Int
}
