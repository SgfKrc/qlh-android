package com.qlh.inference.data

import androidx.room.ColumnInfo
import androidx.room.Entity
import androidx.room.PrimaryKey

@Entity(tableName = "sessions")
data class SessionEntity(
    @PrimaryKey(autoGenerate = true)
    val id: Long = 0,

    /** 会话标题（首条消息自动截取） */
    val title: String = "新对话",

    /** 创建时间 (epoch millis) */
    @ColumnInfo(name = "created_at")
    val createdAt: Long = System.currentTimeMillis(),

    /** 最后活跃时间 */
    @ColumnInfo(name = "updated_at")
    val updatedAt: Long = System.currentTimeMillis(),

    /** 消息数缓存（避免 JOIN COUNT） */
    @ColumnInfo(name = "message_count")
    val messageCount: Int = 0
)
