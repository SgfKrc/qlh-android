package com.qlh.inference

import com.google.gson.Gson
import com.google.gson.JsonElement
import com.google.gson.JsonNull
import com.google.gson.JsonObject
import com.google.gson.JsonParser
import com.qlh.inference.network.ModelIntentBinding
import com.qlh.inference.network.ModelIntentCatalog
import com.qlh.inference.network.ModelIntentCapacityDiagnostic
import com.qlh.inference.network.ModelIntentCapacityNode
import com.qlh.inference.network.ModelIntentOperation
import com.qlh.inference.network.validate
import org.junit.Assert.*
import org.junit.Test
import java.util.Locale

class ModelIntentCapacityTest {
    private val gson = Gson()
    private val capacityCode = "pipeline_distributed_capacity_insufficient"
    private val hostCode = "SUPPLY_RUNTIME_HOST_CAPACITY_INSUFFICIENT"
    private val unknownCode = "pipeline_distributed_workers_unavailable"
    private val generation = 1792056278284219200L

    private fun diagnostic(code: String = capacityCode): JsonObject = JsonParser.parseString("""{
        "reason_code":"$code", "reason":"候选层段超出节点预算", "scope":"candidate_plan",
        "suggested_action":"释放节点内存或增加可用节点后重新规划",
        "nodes":[{"node_id":"rog", "engine":"llama_cpp", "execution_device":"cpu", "resource":"ram",
            "available_bytes":1073741824, "required_bytes":1342177280, "deficit_bytes":268435456,
            "reserve_bytes":134217728, "layer_range":[20,24]}]
    }""").asJsonObject

    private fun operation(
        code: String = capacityCode,
        status: String = "waiting_nodes",
        loadedReady: Boolean = false,
        diagnostic: JsonElement? = diagnostic(code),
        reason: String = "当前在线节点无法容纳连续层段",
    ): ModelIntentOperation {
        val json = JsonParser.parseString("""{
            "operation_id":"supply_capacity", "config_id":"config_capacity", "generation":$generation,
            "intent":{"model_id":"qwen35-9b", "engine":"llama_cpp", "quant_type":"", "strategy":"auto",
                "timeout_seconds":3600},
            "nodes":{"master":{"status":"online"},"rog":{"status":"online"},"y700":{"status":"online"}}
        }""").asJsonObject.apply {
            addProperty("status", status)
            addProperty("loaded_ready", loadedReady)
            addProperty("reason_code", code)
            addProperty("reason", reason)
            if (diagnostic != null) add("capacity_diagnostic", diagnostic)
        }
        return gson.fromJson(json, ModelIntentOperation::class.java).validate()
    }

    private fun JsonObject.firstNode(): JsonObject = getAsJsonArray("nodes")[0].asJsonObject
    private fun text(operation: ModelIntentOperation) = formatModelIntentCapacity(operation).joinToString("\n")
    private fun assertNoBudgetNumbers(operation: ModelIntentOperation) {
        val displayed = text(operation)
        for (label in listOf("可用 ", "所需 ", "缺口 ", "预留 ")) assertFalse(displayed, displayed.contains(label))
    }

    @Test fun `all online but insufficient capacity uses owner budgets and keeps lifecycle`() {
        val op = operation()
        val before = gson.toJson(op)
        val displayed = text(op)
        assertEquals("等待容量", modelIntentStatusLabel(op))
        assertEquals("waiting_nodes", op.status)
        assertEquals(setOf("master", "rog", "y700"), op.nodes.keys)
        assertTrue(op.nodes.values.all { it.status == "online" })
        for (part in listOf(
            "容量不足（候选方案", "释放所示缺口不保证可执行", "候选层段超出节点预算",
            "rog · llama.cpp · CPU · RAM", "层段 [20, 24)",
            "可用 1.00 GiB (1073741824 bytes)", "所需 1.25 GiB (1342177280 bytes)",
            "缺口 256.00 MiB (268435456 bytes)", "含边界的预留 128.00 MiB (134217728 bytes)",
            "建议：释放节点内存或增加可用节点后重新规划",
        )) assertTrue(part, displayed.contains(part))
        assertFalse(op.readyForChat)
        assertEquals(before, gson.toJson(op))
    }

    @Test fun `actual worker joining never inherits a matching stale capacity diagnostic`() {
        val d = diagnostic("SUPPLY_WAITING_NODES").apply {
            add("pending_node_ids", JsonParser.parseString("""["stale-worker"]"""))
        }
        val op = operation(code = "SUPPLY_WAITING_NODES", diagnostic = d)
        assertEquals("等待节点", modelIntentStatusLabel(op))
        assertTrue(formatModelIntentCapacity(op).isEmpty())
        assertEquals("SUPPLY_WAITING_NODES", op.reasonCode)
    }

    @Test fun `failed host staging distinguishes host RAM from CUDA VRAM and fixed reserve`() {
        val d = diagnostic(hostCode).apply {
            addProperty("scope", "minimum_segment")
            firstNode().apply {
                addProperty("node_id", "master")
                addProperty("engine", "pytorch")
                addProperty("execution_device", "cuda")
                addProperty("resource", "host_ram")
                addProperty("available_bytes", 33554432)
                addProperty("required_bytes", 67154432)
                addProperty("deficit_bytes", 33600000)
                add("layer_range", JsonParser.parseString("[0,4]"))
            }
        }
        val op = operation(code = hostCode, status = "failed", diagnostic = d)
        val displayed = text(op)
        assertEquals("失败", modelIntentStatusLabel(op))
        assertTrue(displayed.contains("容量不足（最小层段"))
        assertTrue(displayed.contains("master · PyTorch · CUDA · 主机暂存 RAM"))
        assertTrue(displayed.contains("层段 [0, 4)"))
        assertTrue(displayed.contains("可用 32.00 MiB (33554432 bytes)"))
        assertTrue(displayed.contains("所需 64.04 MiB (67154432 bytes)"))
        assertTrue(displayed.contains("缺口 32.04 MiB (33600000 bytes)"))
        assertTrue(displayed.contains("固定预留 128.00 MiB (134217728 bytes)"))
        assertFalse(displayed.contains("含边界的预留"))
        assertFalse(displayed.contains("显存"))
        assertTrue(op.finished)
        assertFalse(op.readyForChat)
    }

    @Test fun `host capacity waiting uses capacity title without changing readiness`() {
        val op = operation(code = hostCode)
        assertEquals("等待容量", modelIntentStatusLabel(op))
        assertFalse(op.readyForChat)
        assertFalse(op.finished)
    }

    @Test fun `VRAM uses projected reserve and preserves explicitly zero owner deficit`() {
        val d = diagnostic().apply {
            firstNode().apply {
                addProperty("engine", "pytorch")
                addProperty("execution_device", "cuda")
                addProperty("resource", "vram")
                addProperty("deficit_bytes", 0)
            }
        }
        val displayed = text(operation(diagnostic = d))
        assertTrue(displayed.contains("PyTorch · CUDA · 显存（VRAM）"))
        assertTrue(displayed.contains("缺口 0 bytes"))
        assertTrue(displayed.contains("含边界的预留 128.00 MiB"))
        assertFalse(displayed.contains("主机暂存"))
    }

    @Test fun `failed capacity preserves matching budget and failure lifecycle`() {
        val op = operation(status = "failed")
        assertEquals("失败", modelIntentStatusLabel(op))
        assertTrue(text(op).contains("所需 1.25 GiB"))
        assertTrue(text(op).contains("缺口 256.00 MiB"))
        assertTrue(op.finished)
        assertFalse(op.readyForChat)
    }

    @Test fun `only waiting and failed may show capacity diagnosis`() {
        for (status in listOf("queued", "inspecting", "inventory", "planning", "building", "supplying",
            "verifying", "preparing", "cancelled", "superseded")) {
            val op = operation(status = status)
            assertEquals(status, modelIntentStatusLabel(status), modelIntentStatusLabel(op))
            assertTrue(status, formatModelIntentCapacity(op).isEmpty())
            assertFalse(op.readyForChat)
        }
    }

    @Test fun `ready retains its own loaded ready gate despite a stale capacity diagnosis`() {
        for (loaded in listOf(false, true)) {
            val op = operation(status = "ready", loadedReady = loaded)
            assertEquals("准备完成", modelIntentStatusLabel(op))
            assertTrue(formatModelIntentCapacity(op).isEmpty())
            assertEquals(loaded, op.readyForChat)
            val state = ModelIntentUiState(
                catalog = ModelIntentCatalog(binding = ModelIntentBinding(status = "connected")), operation = op,
            )
            assertEquals(loaded, state.readyForChat)
        }
    }

    @Test fun `loaded ready alone never turns waiting capacity into chat ready`() {
        val op = operation(loadedReady = true)
        assertFalse(op.readyForChat)
        assertEquals("等待容量", modelIntentStatusLabel(op))
        assertTrue(text(op).contains("缺口 "))
    }

    @Test fun `standard cause classifies legacy waiting without matching diagnostic details`() {
        for (status in listOf("waiting_nodes", "failed")) {
            val op = operation(code = hostCode, status = status, diagnostic = diagnostic(capacityCode))
            assertEquals(if (status == "waiting_nodes") "等待容量" else "失败", modelIntentStatusLabel(op))
            assertTrue(formatModelIntentCapacity(op).isEmpty())
            assertEquals(hostCode, op.reasonCode)
        }
    }

    @Test fun `unrelated failure never inherits previous diagnosis suggestions or pending nodes`() {
        val d = diagnostic().apply {
            add("pending_node_ids", JsonParser.parseString("""["joining-worker"]"""))
        }
        for (code in listOf("SUPPLY_DEADLINE_EXCEEDED", "SUPPLY_WORKER_FAILED")) {
            val op = operation(code = code, status = "failed", diagnostic = d)
            assertEquals("失败", modelIntentStatusLabel(op))
            assertTrue(formatModelIntentCapacity(op).isEmpty())
        }
    }

    @Test fun `known capacity reasons classify waiting even on old services`() {
        for (code in listOf(capacityCode, "pipeline_cluster_capacity_insufficient",
            "pipeline_capacity_single_node_insufficient", "pipeline_capacity_manual_insufficient",
            "pipeline_reshard_capacity_insufficient", "node_capacity_unavailable",
            "SUPPLY_CAPACITY_INSUFFICIENT", "SUPPLY_MASTER_CAPACITY_INSUFFICIENT", hostCode)) {
            val op = operation(code = code, diagnostic = null)
            assertEquals(code, "等待容量", modelIntentStatusLabel(op))
            assertTrue(formatModelIntentCapacity(op).isEmpty())
        }
    }

    @Test fun `unknown matching cause needs an explicit valid positive deficit`() {
        val op = operation(code = unknownCode)
        assertEquals("等待容量", modelIntentStatusLabel(op))
        assertTrue(text(op).contains("缺口 256.00 MiB"))
        assertEquals("失败", modelIntentStatusLabel(op.copy(status = "failed")))
        assertFalse(formatModelIntentCapacity(op.copy(status = "failed")).isEmpty())
    }

    @Test fun `unknown cause never uses another reasons positive deficit`() {
        val op = operation(code = unknownCode, diagnostic = diagnostic(capacityCode))
        assertEquals("等待节点", modelIntentStatusLabel(op))
        assertTrue(formatModelIntentCapacity(op).isEmpty())
    }

    @Test fun `unknown cause with zero missing or invalid deficit keeps old waiting title`() {
        for (raw in listOf("null", "0", "-1", "true", "1.1", "1e2", "\"256\"", "9223372036854775808")) {
            val d = diagnostic(unknownCode).apply { firstNode().add("deficit_bytes", JsonParser.parseString(raw)) }
            val op = operation(code = unknownCode, diagnostic = d)
            assertEquals(raw, "等待节点", modelIntentStatusLabel(op))
            assertTrue(raw, formatModelIntentCapacity(op).isEmpty())
        }
        val d = diagnostic(unknownCode).apply { firstNode().remove("deficit_bytes") }
        assertEquals("等待节点", modelIntentStatusLabel(operation(code = unknownCode, diagnostic = d)))
    }

    @Test fun `range contract rejection with no candidate does not claim a numeric shortage`() {
        val d = diagnostic(unknownCode).apply {
            addProperty("reason", "现有层段范围无法形成连续接力，未形成合法候选")
            firstNode().apply {
                remove("required_bytes")
                remove("deficit_bytes")
                remove("layer_range")
                addProperty("raw_model_deficit", 987654321L)
            }
        }
        val op = operation(code = unknownCode, diagnostic = d, reason = "允许层段约束不匹配")
        assertEquals("等待节点", modelIntentStatusLabel(op))
        assertTrue(formatModelIntentCapacity(op).isEmpty())
        assertEquals("允许层段约束不匹配", op.reason)
    }

    @Test fun `known rejection without candidate only displays known budgets`() {
        val d = diagnostic().apply {
            firstNode().apply {
                remove("required_bytes")
                remove("deficit_bytes")
                remove("layer_range")
                addProperty("raw_model_deficit", 987654321L)
            }
            addProperty("suggested_action", "未得到合法候选，无法给出可靠的逐节点需求或缺口")
        }
        val op = operation(diagnostic = d)
        val displayed = text(op)
        assertEquals("等待容量", modelIntentStatusLabel(op))
        assertTrue(displayed.contains("可用 1.00 GiB"))
        assertTrue(displayed.contains("含边界的预留 128.00 MiB"))
        for (absent in listOf("所需 ", "缺口 ", "层段 [", "987654321", "释放所示缺口")) {
            assertFalse(absent, displayed.contains(absent))
        }
        assertTrue(displayed.contains("无法给出可靠的逐节点需求或缺口"))
    }

    @Test fun `missing deficit is never recomputed from required minus available`() {
        val d = diagnostic().apply { firstNode().remove("deficit_bytes") }
        val displayed = text(operation(diagnostic = d))
        assertTrue(displayed.contains("可用 1.00 GiB"))
        assertTrue(displayed.contains("所需 1.25 GiB"))
        assertFalse(displayed.contains("缺口 "))
    }

    @Test fun `old response without diagnostic keeps owner reasons nodes and exact generation`() {
        val op = operation(code = "SUPPLY_WAITING_NODES", diagnostic = null, reason = "等待节点加入")
        assertNull(op.capacityDiagnostic)
        assertEquals("等待节点", modelIntentStatusLabel(op))
        assertEquals("等待节点加入", op.reason)
        assertEquals(generation, op.generation)
        assertTrue(op.nodes.containsKey("y700"))
        assertTrue(formatModelIntentCapacity(op).isEmpty())
    }

    @Test fun `malformed optional diagnostic does not discard the operation`() {
        for (raw in listOf("null", "[]", "\"legacy\"", "true", "42", "{}")) {
            val op = operation(diagnostic = JsonParser.parseString(raw))
            assertEquals(raw, "等待容量", modelIntentStatusLabel(op))
            assertEquals(generation, op.generation)
            assertTrue(raw, formatModelIntentCapacity(op).isEmpty())
        }
        assertNull(operation(diagnostic = JsonNull.INSTANCE).capacityDiagnostic)
    }

    @Test fun `partial node never invents missing fields or raw deficit`() {
        val d = diagnostic().apply {
            add("nodes", JsonParser.parseString("""[null,{}, {"node_id":"y700","resource":"ram",
                "available_bytes":5368709120,"raw_model_deficit":1122334455}]"""))
        }
        val displayed = text(operation(diagnostic = d))
        assertTrue(displayed.contains("y700 · RAM · 可用 5.00 GiB (5368709120 bytes)"))
        assertTrue(displayed.contains("未标识节点"))
        for (absent in listOf("所需 ", "缺口 ", "预留 ", "层段 [", "1122334455")) {
            assertFalse(absent, displayed.contains(absent))
        }
    }

    @Test fun `byte budgets reject coercion truncation overflow and negative values`() {
        for (raw in listOf("null", "true", "false", "-1", "1.5", "268435456.0", "1e3", "\"268435456\"",
            "[]", "{}", "9223372036854775808")) {
            val d = diagnostic().apply {
                firstNode().apply {
                    for (field in listOf("available_bytes", "required_bytes", "deficit_bytes", "reserve_bytes")) {
                        add(field, JsonParser.parseString(raw))
                    }
                }
            }
            val op = operation(diagnostic = d)
            val node = op.capacityDiagnostic!!.nodes.single()
            assertNull(raw, node.availableBytes)
            assertNull(raw, node.requiredBytes)
            assertNull(raw, node.deficitBytes)
            assertNull(raw, node.reserveBytes)
            assertNoBudgetNumbers(op)
            assertTrue(text(op).contains("层段 [20, 24)"))
        }
    }

    @Test fun `long budgets and generation preserve exact adjacent integers through Gson roundtrip`() {
        val d = diagnostic().apply {
            firstNode().apply {
                addProperty("available_bytes", generation)
                addProperty("required_bytes", generation + 1)
                addProperty("deficit_bytes", 1L)
                addProperty("reserve_bytes", Long.MAX_VALUE)
            }
        }
        val op = operation(diagnostic = d)
        val roundtrip = gson.fromJson(gson.toJson(op), ModelIntentOperation::class.java).validate()
        assertEquals(generation, roundtrip.generation)
        val node = roundtrip.capacityDiagnostic!!.nodes.single()
        assertEquals(generation, node.availableBytes!!)
        assertEquals(generation + 1, node.requiredBytes!!)
        assertEquals(Long.MAX_VALUE, node.reserveBytes!!)
        assertNotEquals(node.availableBytes, node.requiredBytes)
        val displayed = text(roundtrip)
        for (exact in listOf("($generation bytes)", "(${generation + 1} bytes)", "(${Long.MAX_VALUE} bytes)")) {
            assertTrue(exact, displayed.contains(exact))
        }
        assertTrue(displayed.contains("缺口 1 bytes"))
    }

    @Test fun `invalid or missing layer bounds never become a displayed segment`() {
        for (raw in listOf("null", "[]", "[2]", "[0,4,8]", "[4,2]", "[2,2]", "[-1,2]",
            "[0,true]", "[\"0\",4]", "[0,4.0]", "[0,4e0]", "[0,9223372036854775808]", "{}")) {
            val d = diagnostic().apply { firstNode().add("layer_range", JsonParser.parseString(raw)) }
            val displayed = text(operation(diagnostic = d))
            assertFalse(raw, displayed.contains("层段 ["))
            assertTrue(raw, displayed.contains("可用 1.00 GiB"))
        }
    }

    @Test fun `unrecognized scope never claims a candidate or minimum segment`() {
        for (raw in listOf("null", "\"future\"", "[]", "42")) {
            val d = diagnostic().apply { add("scope", JsonParser.parseString(raw)) }
            val displayed = text(operation(diagnostic = d))
            assertTrue(raw, displayed.contains("容量不足（范围未提供"))
            assertFalse(displayed.contains("容量不足（候选方案"))
            assertFalse(displayed.contains("容量不足（最小层段"))
        }
    }

    @Test fun `identical owner and diagnostic reasons are not repeated`() {
        val op = operation(reason = "候选层段超出节点预算")
        assertFalse(formatModelIntentCapacity(op).contains(op.reason))
        assertTrue(text(op).contains("可用 1.00 GiB"))
        assertEquals("候选层段超出节点预算", op.reason)
    }

    @Test fun `distinct diagnostic reason is retained once`() {
        assertEquals(1, formatModelIntentCapacity(operation()).count { it == "候选层段超出节点预算" })
    }

    @Test fun `pending nodes and suggestions supplement the capacity cause in waiting and failure`() {
        val d = diagnostic().apply {
            add("pending_node_ids", JsonParser.parseString("""["surface-new",null,"", "   ", true,5,"worker-new"]"""))
        }
        for (status in listOf("waiting_nodes", "failed")) {
            val op = operation(status = status, diagnostic = d)
            val displayed = text(op)
            assertEquals(listOf("surface-new", "worker-new"), op.capacityDiagnostic!!.pendingNodeIds)
            assertTrue(displayed.contains("仍在加入的节点：surface-new, worker-new"))
            assertTrue(displayed.contains("缺口 256.00 MiB"))
            assertTrue(displayed.contains("建议：释放节点内存或增加可用节点后重新规划"))
            assertEquals(capacityCode, op.reasonCode)
        }
    }

    @Test fun `invalid pending list never claims node joining`() {
        for (raw in listOf("null", "[]", "\"worker-new\"", "[null,1,true,\"\",\"   \"]", "{}")) {
            val d = diagnostic().apply { add("pending_node_ids", JsonParser.parseString(raw)) }
            assertFalse(raw, text(operation(diagnostic = d)).contains("仍在加入的节点"))
        }
    }

    @Test fun `malformed metadata never coerces into displayed strings or budgets`() {
        val d = diagnostic().apply {
            add("reason", JsonParser.parseString("42"))
            add("suggested_action", JsonParser.parseString("true"))
            firstNode().apply {
                add("node_id", JsonParser.parseString("{}"))
                add("engine", JsonParser.parseString("17"))
                add("execution_device", JsonParser.parseString("false"))
                add("resource", JsonParser.parseString("[]"))
            }
        }
        val displayed = text(operation(diagnostic = d))
        assertTrue(displayed.contains("未标识节点"))
        assertFalse(displayed.contains("建议："))
        assertFalse(displayed.contains("true"))
        assertFalse(displayed.contains("false"))
    }

    @Test fun `invalid reason code does not match or change legacy operation`() {
        for (raw in listOf("null", "true", "42", "{}", "[]", "\"\"")) {
            val d = diagnostic().apply { add("reason_code", JsonParser.parseString(raw)) }
            val op = operation(diagnostic = d)
            assertEquals("等待容量", modelIntentStatusLabel(op))
            assertTrue(raw, formatModelIntentCapacity(op).isEmpty())
        }
    }

    @Test fun `missing operation cause cannot be supplied by diagnostic alone`() {
        val op = operation().copy(reasonCode = null)
        assertEquals("等待节点", modelIntentStatusLabel(op))
        assertTrue(formatModelIntentCapacity(op).isEmpty())
    }

    @Test fun `invalid nodes container never turns into a guessed budget`() {
        for (raw in listOf("null", "{}", "\"legacy\"", "[null,1,true,[],\"node\"]")) {
            val d = diagnostic().apply { add("nodes", JsonParser.parseString(raw)) }
            val op = operation(diagnostic = d)
            assertTrue(raw, op.capacityDiagnostic!!.nodes.isEmpty())
            assertNoBudgetNumbers(op)
        }
    }

    @Test fun `readable units remain stable regardless of device locale`() {
        val original = Locale.getDefault()
        try {
            Locale.setDefault(Locale.GERMANY)
            val displayed = text(operation())
            assertTrue(displayed.contains("1.25 GiB (1342177280 bytes)"))
            assertFalse(displayed.contains("1,25 GiB"))
        } finally {
            Locale.setDefault(original)
        }
    }

    @Test fun `small byte counts show exact values without rounding to zero`() {
        val d = diagnostic().apply {
            firstNode().apply {
                addProperty("available_bytes", 0L)
                addProperty("required_bytes", 1024L)
                addProperty("deficit_bytes", 1023L)
                addProperty("reserve_bytes", 1L)
            }
        }
        val displayed = text(operation(diagnostic = d))
        for (part in listOf("可用 0 bytes", "所需 1.00 KiB (1024 bytes)", "缺口 1023 bytes", "预留 1 bytes")) {
            assertTrue(part, displayed.contains(part))
        }
    }

    @Test fun `typed partial legacy node also omits invalid budgets and ranges`() {
        val op = operation().copy(capacityDiagnostic = ModelIntentCapacityDiagnostic(
            reasonCode = capacityCode,
            nodes = listOf(ModelIntentCapacityNode(nodeId = "old", availableBytes = -1, layerRange = listOf(4, 2))),
        ))
        val displayed = text(op)
        assertTrue(displayed.contains("范围未提供"))
        assertTrue(displayed.contains("old"))
        assertFalse(displayed.contains("可用 "))
        assertFalse(displayed.contains("层段 ["))
    }
}
