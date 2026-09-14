package com.qlh.inference.data

import androidx.room.ColumnInfo
import androidx.room.Entity
import androidx.room.ForeignKey
import androidx.room.Index
import androidx.room.PrimaryKey

@Entity(
    tableName = "messages",
    foreignKeys = [
        ForeignKey(
            entity = SessionEntity::class,
            parentColumns = ["id"],
            childColumns = ["session_id"],
            onDelete = ForeignKey.CASCADE
        )
    ],
    indices = [Index("session_id")]
)
data class MessageEntity(
    @PrimaryKey(autoGenerate = true)
    val id: Long = 0,

    @ColumnInfo(name = "session_id")
    val sessionId: Long,

    /** "user" | "assistant" | "system" */
    val role: String,

    val content: String,

    /** 时间戳 (epoch millis) */
    val timestamp: Long = System.currentTimeMillis(),

    /** JSON 格式的附加指标 (tok/s, engine 等) */
    val metrics: String? = null
)
