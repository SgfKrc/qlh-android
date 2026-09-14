package com.qlh.inference.ui

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.Chat
import androidx.compose.material.icons.filled.Add
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material.icons.filled.Forum
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.ExtendedFloatingActionButton
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.semantics.selected
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import com.qlh.inference.data.SessionEntity
import com.qlh.inference.ui.components.EmptyState
import com.qlh.inference.ui.components.QlhTopBar
import com.qlh.inference.ui.components.StatusChip
import com.qlh.inference.ui.theme.QlhShapeTokens
import com.qlh.inference.ui.theme.qlhBrandGold
import com.qlh.inference.ui.theme.qlhBrandGoldContainer
import com.qlh.inference.ui.theme.qlhOnBrandGoldContainer
import com.qlh.inference.ui.theme.qlhNeonGreen
import com.qlh.inference.ui.theme.qlhOnNeonGreen
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

// ================================================================
// 会话列表界面
// ================================================================

@Composable
fun SessionListScreen(
    sessions: List<SessionEntity>,
    currentSessionId: Long,
    onSessionClick: (Long) -> Unit,
    onCreateSession: () -> Unit,
    onDeleteSession: (Long) -> Unit,
    modifier: Modifier = Modifier
) {
    var deleteTarget by remember { mutableStateOf<SessionEntity?>(null) }

    Scaffold(
        floatingActionButton = {
            ExtendedFloatingActionButton(
                onClick = onCreateSession,
                modifier = Modifier.testTag("session_create"),
                containerColor = MaterialTheme.colorScheme.primaryContainer,
                contentColor = MaterialTheme.colorScheme.onPrimaryContainer,
                icon = {
                    Icon(
                        Icons.Default.Add,
                        contentDescription = null
                    )
                },
                text = { Text("新建会话") }
            )
        },
        containerColor = MaterialTheme.colorScheme.background,
        modifier = modifier.testTag("session_list_screen")
    ) { padding ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(padding)
        ) {
            QlhTopBar(
                title = "会话",
                subtitle = if (sessions.isEmpty()) "暂无会话" else "共 ${sessions.size} 个会话"
            )
            HorizontalDivider(color = MaterialTheme.colorScheme.outlineVariant.copy(alpha = 0.5f))

            if (sessions.isEmpty()) {
                // 空状态
                Box(
                    modifier = Modifier
                        .fillMaxSize()
                        .padding(horizontal = 32.dp),
                    contentAlignment = Alignment.Center
                ) {
                    EmptyState(
                        icon = Icons.Default.Forum,
                        title = "暂无会话",
                        subtitle = "点击右下角「新建会话」开始第一段对话"
                    )
                }
            } else {
                LazyColumn(
                    modifier = Modifier.fillMaxSize(),
                    contentPadding = PaddingValues(
                        start = 16.dp,
                        end = 16.dp,
                        top = 12.dp,
                        bottom = 96.dp
                    ),
                    verticalArrangement = Arrangement.spacedBy(10.dp)
                ) {
                    items(sessions, key = { it.id }) { session ->
                        SessionCard(
                            session = session,
                            isCurrent = session.id == currentSessionId,
                            onClick = { onSessionClick(session.id) },
                            onDelete = { deleteTarget = session }
                        )
                    }
                }
            }
        }
    }

    // ---- 删除确认对话框 ----
    deleteTarget?.let { session ->
        AlertDialog(
            onDismissRequest = { deleteTarget = null },
            title = { Text("删除会话") },
            text = {
                Text(
                    "确定删除「${session.title}」及其所有消息？\n此操作不可撤销。"
                )
            },
            confirmButton = {
                TextButton(
                    onClick = {
                        onDeleteSession(session.id)
                        deleteTarget = null
                    }
                ) {
                    Text(
                        "删除",
                        color = MaterialTheme.colorScheme.error
                    )
                }
            },
            dismissButton = {
                TextButton(onClick = { deleteTarget = null }) {
                    Text("取消")
                }
            }
        )
    }
}

// ================================================================
// 会话卡片
// ================================================================

@Composable
private fun SessionCard(
    session: SessionEntity,
    isCurrent: Boolean,
    onClick: () -> Unit,
    onDelete: () -> Unit
) {
    Card(
        onClick = onClick,
        modifier = Modifier
            .fillMaxWidth()
            .testTag("session_card_${session.id}")
            .semantics { selected = isCurrent },
        shape = MaterialTheme.shapes.medium,
        colors = CardDefaults.cardColors(
            containerColor = if (isCurrent) {
                MaterialTheme.colorScheme.tertiaryContainer
            } else {
                MaterialTheme.colorScheme.surfaceContainer
            }
        ),
        border = BorderStroke(
            1.dp,
            if (isCurrent) qlhBrandGold().copy(alpha = 0.78f)
            else MaterialTheme.colorScheme.outlineVariant.copy(alpha = 0.64f),
        ),
        elevation = CardDefaults.cardElevation(defaultElevation = 0.dp)
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 14.dp, vertical = 14.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            // 会话图标（圆角底色）
            Surface(
                modifier = Modifier.size(40.dp),
                shape = RoundedCornerShape(QlhShapeTokens.control),
                color = if (isCurrent) {
                    qlhBrandGoldContainer()
                } else {
                    MaterialTheme.colorScheme.secondaryContainer
                }
            ) {
                Box(contentAlignment = Alignment.Center) {
                    Icon(
                        Icons.AutoMirrored.Filled.Chat,
                        contentDescription = null,
                        modifier = Modifier.size(20.dp),
                        tint = if (isCurrent) {
                            qlhOnBrandGoldContainer()
                        } else {
                            MaterialTheme.colorScheme.onSecondaryContainer
                        }
                    )
                }
            }
            Spacer(modifier = Modifier.width(12.dp))

            // 标题和信息
            Column(modifier = Modifier.weight(1f)) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Text(
                        text = session.title,
                        style = MaterialTheme.typography.titleMedium,
                        fontWeight = if (isCurrent) FontWeight.SemiBold else FontWeight.Medium,
                        color = if (isCurrent) {
                            MaterialTheme.colorScheme.onTertiaryContainer
                        } else {
                            MaterialTheme.colorScheme.onSurface
                        },
                        maxLines = 2,
                        overflow = TextOverflow.Ellipsis,
                        modifier = Modifier.weight(1f)
                    )
                    if (isCurrent) {
                        Spacer(modifier = Modifier.width(8.dp))
                        StatusChip(
                            text = "当前",
                            containerColor = qlhNeonGreen(),
                            contentColor = qlhOnNeonGreen(),
                            showDot = false
                        )
                    }
                }
                Spacer(modifier = Modifier.height(3.dp))
                Text(
                    text = buildInfoText(session),
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    maxLines = 1,
                    overflow = TextOverflow.Ellipsis
                )
            }

            // 删除按钮
            IconButton(
                onClick = onDelete,
                modifier = Modifier.testTag("session_delete_${session.id}"),
            ) {
                Icon(
                    Icons.Default.Delete,
                    contentDescription = "删除会话",
                    modifier = Modifier.size(20.dp),
                    tint = MaterialTheme.colorScheme.outline
                )
            }
        }
    }
}

private fun buildInfoText(session: SessionEntity): String {
    val dateStr = formatDate(session.updatedAt)
    val countStr = "${session.messageCount} 条消息"
    return "$dateStr · $countStr"
}

private fun formatDate(timestamp: Long): String {
    val now = System.currentTimeMillis()
    val diff = now - timestamp
    return when {
        diff < 60_000 -> "刚刚"
        diff < 3600_000 -> "${diff / 60_000} 分钟前"
        diff < 86400_000 -> "${diff / 3600_000} 小时前"
        else -> {
            val sdf = SimpleDateFormat("MM-dd", Locale.getDefault())
            sdf.format(Date(timestamp))
        }
    }
}
