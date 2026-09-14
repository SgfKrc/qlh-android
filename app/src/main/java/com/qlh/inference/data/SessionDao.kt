package com.qlh.inference.data

import androidx.room.Dao
import androidx.room.Insert
import androidx.room.OnConflictStrategy
import androidx.room.Query
import androidx.room.Update
import kotlinx.coroutines.flow.Flow

@Dao
interface SessionDao {

    /** 获取所有会话（按更新时间降序） */
    @Query("SELECT * FROM sessions ORDER BY updated_at DESC")
    fun getAllSessions(): Flow<List<SessionEntity>>

    /** 根据 ID 获取会话 */
    @Query("SELECT * FROM sessions WHERE id = :sessionId")
    suspend fun getById(sessionId: Long): SessionEntity?

    /** 创建会话，返回 ID */
    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insert(session: SessionEntity): Long

    /** 更新会话 */
    @Update
    suspend fun update(session: SessionEntity)

    /** 更新会话标题 */
    @Query("UPDATE sessions SET title = :title, updated_at = :updatedAt WHERE id = :sessionId")
    suspend fun updateTitle(sessionId: Long, title: String, updatedAt: Long = System.currentTimeMillis())

    /** 更新会话消息数 */
    @Query("UPDATE sessions SET message_count = :count, updated_at = :updatedAt WHERE id = :sessionId")
    suspend fun updateMessageCount(sessionId: Long, count: Int, updatedAt: Long = System.currentTimeMillis())

    /** 删除会话 */
    @Query("DELETE FROM sessions WHERE id = :sessionId")
    suspend fun deleteById(sessionId: Long)
}
