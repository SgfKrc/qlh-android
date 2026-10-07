package com.qlh.inference.worker

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Test

class TaskWorkerProtocolTest {
    private val model = mapOf(
        "model_id" to "qwen_1_8b",
        "engine" to "llama_cpp",
        "format" to "gguf",
        "revision" to "local_v1",
        "sha256" to "a".repeat(64),
    )

    private val identity = TaskWorkerAttemptIdentity(
        workflowId = "wf_android_001",
        stageId = "candidate_a",
        attemptId = "att_android_001",
        leaseId = "lease_android_001",
        leaseEpoch = 1,
    )

    @Test
    fun `hello round trips canonically with android full worker capabilities`() {
        val hello = TaskWorkerProtocol.buildHello(
            nodeId = "android_worker_01",
            capabilities = mapOf(
                "stage_types" to listOf("full_inference"),
                "engines" to listOf("llama_cpp"),
                "models" to listOf(model),
                "max_concurrency" to 1,
            ),
            messageId = "msg_hello_android_01",
            sentAtMs = 1_700_000_000_000,
        )

        val bytes = TaskWorkerProtocol.encode(hello)
        val decoded = TaskWorkerProtocol.decode(bytes)

        assertEquals(hello.protocol, decoded.protocol)
        assertEquals(hello.version, decoded.version)
        assertEquals(hello.messageType, decoded.messageType)
        assertEquals(hello.messageId, decoded.messageId)
        assertEquals(hello.sentAtMs, decoded.sentAtMs)
        assertEquals(TaskWorkerProtocol.ANDROID_WORKER_KIND, decoded.payload["worker_kind"])
        assertEquals(
            bytes.toString(Charsets.UTF_8),
            TaskWorkerProtocol.encode(decoded).toString(Charsets.UTF_8),
        )
        assertTrue(bytes.size < TaskWorkerProtocol.MAX_MESSAGE_BYTES)
    }

    @Test
    fun `stage result binds output digest and attempt idempotency`() {
        val result = TaskWorkerProtocol.buildStageResult(
            identity = identity,
            providerId = "remote_android_worker_01",
            output = mapOf("content" to "answer", "usage" to mapOf("total_tokens" to 3)),
            metadata = mapOf("model" to "qwen_1_8b", "usage_estimated" to false),
            messageId = "msg_result_android_01",
            sentAtMs = 1_700_000_000_100,
        )
        val decoded = TaskWorkerProtocol.decode(TaskWorkerProtocol.encode(result))
        val digest = decoded.payload["output_sha256"] as String

        assertEquals(TaskWorkerProtocol.stageOutputSha256(decoded.payload["output"] as Map<String, Any?>), digest)
        assertEquals(
            "att_android_001:1:$digest",
            TaskWorkerProtocol.attemptIdempotencyKey(identity, digest),
        )
    }

    @Test
    fun `input digest and lease deadline are validated`() {
        val offer = TaskWorkerProtocol.buildStageOffer(
            identity = identity,
            requestId = "request_android_01",
            stageType = "full_inference",
            providerId = "remote_android_worker_01",
            leaseExpiresAtMs = 1_700_000_000_200,
            rootInput = mapOf("message" to "summarize"),
            dependencies = emptyMap(),
            modelIdentity = model,
            messageId = "msg_offer_android_01",
            sentAtMs = 1_700_000_000_100,
        )
        assertEquals(
            TaskWorkerProtocol.stageInputSha256(
                offer.payload["root_input"] as Map<String, Any?>,
                offer.payload["dependencies"] as Map<String, Any?>,
            ),
            offer.payload["input_sha256"],
        )

        val invalid = offer.copy(
            payload = offer.payload + ("lease_expires_at_ms" to offer.sentAtMs),
        )
        expectProtocolError("invalid_lease_deadline") {
            TaskWorkerProtocol.validate(invalid)
        }
    }

    @Test
    fun `strict validation rejects unknown fields and wrong worker kind`() {
        val hello = TaskWorkerProtocol.buildHello(
            nodeId = "android_worker_01",
            capabilities = mapOf(
                "stage_types" to listOf("full_inference"),
                "engines" to listOf("llama_cpp"),
                "models" to listOf(model),
                "max_concurrency" to 1,
            ),
            messageId = "msg_hello_android_02",
            sentAtMs = 1_700_000_000_000,
        )
        expectProtocolError("invalid_fields") {
            TaskWorkerProtocol.validate(hello.copy(payload = hello.payload + ("unexpected" to true)))
        }
        expectProtocolError("unsupported_worker_kind") {
            TaskWorkerProtocol.validate(
                hello.copy(payload = hello.payload + ("worker_kind" to "pc_full_worker"))
            )
        }
    }

    @Test
    fun `replay cache returns response and rejects message id conflict`() {
        val cache = TaskWorkerReplayCache(maxEntries = 1)
        val hello = TaskWorkerProtocol.buildHello(
            nodeId = "android_worker_01",
            capabilities = mapOf(
                "stage_types" to listOf("full_inference"),
                "engines" to listOf("llama_cpp"),
                "models" to listOf(model),
                "max_concurrency" to 1,
            ),
            messageId = "msg_hello_android_03",
            sentAtMs = 1_700_000_000_000,
        )
        val ack = TaskWorkerProtocol.buildHelloAck(
            coordinatorNodeId = "pc_master_01",
            accepted = true,
            selectedVersion = TaskWorkerProtocol.VERSION,
            reasonCode = "",
            messageId = "msg_ack_android_03",
            sentAtMs = 1_700_000_000_001,
        )

        cache.remember(hello, ack)
        assertEquals(ack, cache.replay(hello))
        assertEquals(1, cache.size())

        val conflict = hello.copy(sentAtMs = hello.sentAtMs + 1)
        expectProtocolError("message_id_conflict") { cache.replay(conflict) }
        assertNull(cache.replay(conflict.copy(messageId = "msg_other_android_03")))
    }

    @Test
    fun `decoder rejects malformed utf8 before json parsing`() {
        expectProtocolError("invalid_encoding") {
            TaskWorkerProtocol.decode(byteArrayOf(0x7b, 0x22, 0x80.toByte(), 0x22, 0x3a, 0x7d))
        }
    }

    private fun expectProtocolError(expectedCode: String, block: () -> Unit) {
        try {
            block()
            fail("expected $expectedCode")
        } catch (error: TaskWorkerProtocolException) {
            assertEquals(expectedCode, error.code)
            assertNotNull(error.field)
        }
    }

    // ---- T8：validate 分支补齐（测试修复票排期） ----

    private fun expectRejected(block: () -> Unit) {
        try {
            block()
            fail("expected TaskWorkerProtocolException")
        } catch (error: TaskWorkerProtocolException) {
            // 拒绝即通过：build 或 validate 任一步骤拒绝非法输入都是回归保护
        }
    }

    @Test
    fun `T8 safeId regex rejects illegal id characters`() {
        expectRejected {
            TaskWorkerProtocol.buildStageOffer(
                identity = identity,
                requestId = "bad id with space",
                stageType = "full_inference",
                providerId = "remote_android_worker_01",
                leaseExpiresAtMs = 1_700_000_000_200,
                rootInput = emptyMap(),
                dependencies = emptyMap(),
                modelIdentity = model,
                messageId = "msg_offer_badid_01",
                sentAtMs = 1_700_000_000_100,
            )
        }
    }

    @Test
    fun `T8 hello_ack negotiation contradiction is rejected`() {
        expectRejected {
            TaskWorkerProtocol.buildHelloAck(
                coordinatorNodeId = "coordinator_01",
                accepted = true,
                selectedVersion = 2,
                reasonCode = "busy",
                messageId = "msg_ack_contradiction_01",
                sentAtMs = 1_700_000_000_000,
            )
        }
    }

    @Test
    fun `T8 capability mismatch is rejected for android worker`() {
        expectRejected {
            TaskWorkerProtocol.buildHello(
                nodeId = "android_worker_01",
                capabilities = mapOf(
                    "stage_types" to listOf("text_generation"),
                    "engines" to listOf("llama_cpp"),
                    "max_concurrency" to 1,
                ),
                messageId = "msg_hello_badcap_01",
                sentAtMs = 1_700_000_000_000,
            )
        }
    }

    @Test
    fun `T8 numeric bounds are enforced`() {
        expectRejected {
            TaskWorkerProtocol.buildHello(
                nodeId = "android_worker_01",
                capabilities = mapOf(
                    "stage_types" to listOf("full_inference"),
                    "engines" to listOf("llama_cpp"),
                    "max_concurrency" to 1,
                ),
                messageId = "msg_hello_neg_01",
                sentAtMs = -1,
            )
        }
    }

    @Test
    fun `T8 stage acceptance disagreement is rejected`() {
        // accepted=true 且 reason 非空：accepted==reason.isNotEmpty 矛盾
        expectRejected {
            TaskWorkerProtocol.buildStageAccept(
                identity = identity,
                providerId = "remote_android_worker_01",
                accepted = true,
                reasonCode = "busy",
                retryable = false,
                messageId = "msg_accept_bad_01",
                sentAtMs = 1_700_000_000_100,
            )
        }
    }

    @Test
    fun `v3 layer_forward accepts a known middle_channel and rejects unknown ones`() {
        // ★ 2026-09-23：`middle_channel` 是 v3 层段的**可选**字段（值域与主仓同集合）。
        TaskWorkerProtocol.validate(layerForwardOffer(middleChannel = "keep_head_layer_out"))
        TaskWorkerProtocol.validate(layerForwardOffer(middleChannel = "extract_hidden"))
        // 缺省（不发该字段）= 旧行为，必须仍然通过（向后兼容）。
        TaskWorkerProtocol.validate(layerForwardOffer(middleChannel = null))
        expectProtocolError("unsupported_middle_channel") {
            TaskWorkerProtocol.validate(layerForwardOffer(middleChannel = "bogus_channel"))
        }
    }

    @Test
    fun `hello capabilities may advertise middle_channel and n_pos_per_embd`() {
        TaskWorkerProtocol.validate(
            TaskWorkerProtocol.buildHello(
                nodeId = "android_worker_01",
                capabilities = mapOf(
                    "stage_types" to listOf("full_inference", "layer_forward"),
                    "engines" to listOf("llama_cpp"),
                    "models" to emptyList<Map<String, Any?>>(),
                    "max_concurrency" to 1,
                    "middle_channel" to "keep_head_layer_out",
                    "n_pos_per_embd" to 4,
                ),
                messageId = "msg_capabilities_channel_01",
                sentAtMs = 1_700_000_000_000,
            ),
        )
        // 值域外的声明必须 fail-closed（不能靠调度侧猜）。
        expectProtocolError("invalid_capabilities") {
            TaskWorkerProtocol.validate(
                TaskWorkerProtocol.buildHello(
                    nodeId = "android_worker_01",
                    capabilities = mapOf(
                        "stage_types" to listOf("full_inference"),
                        "engines" to listOf("llama_cpp"),
                        "models" to emptyList<Map<String, Any?>>(),
                        "max_concurrency" to 1,
                        "n_pos_per_embd" to 3,
                    ),
                    messageId = "msg_capabilities_channel_02",
                    sentAtMs = 1_700_000_000_000,
                ),
            )
        }
    }

    @Test
    fun `hello rejects invalid global segment mode`() {
        expectProtocolError("invalid_capabilities") {
            TaskWorkerProtocol.validate(
                TaskWorkerProtocol.buildHello(
                    nodeId = "android_worker_01",
                    capabilities = artifactCapabilities() + ("segment_mode" to "whole"),
                    messageId = "msg_bad_segment_mode_01",
                    sentAtMs = 1_700_000_000_000,
                ),
            )
        }
    }

    @Test
    fun `hello validates per artifact range mode digest and exact fields`() {
        TaskWorkerProtocol.validate(
            TaskWorkerProtocol.buildHello(
                nodeId = "android_worker_01",
                capabilities = artifactCapabilities(),
                messageId = "msg_layer_artifacts_ok_01",
                sentAtMs = 1_700_000_000_000,
            ),
        )

        val invalidItems = listOf(
            validArtifact() + ("segment_mode" to "whole"),
            validArtifact() + ("layer_range" to listOf(4.5, 16)),
            validArtifact() + ("artifact_sha256" to "not-a-digest"),
            validArtifact() + ("unexpected" to true),
        )
        invalidItems.forEachIndexed { index, item ->
            expectProtocolError(if (index == 3) "invalid_fields" else "invalid_capabilities") {
                TaskWorkerProtocol.validate(
                    TaskWorkerProtocol.buildHello(
                        nodeId = "android_worker_01",
                        capabilities = artifactCapabilities(item),
                        messageId = "msg_bad_artifact_item_0$index",
                        sentAtMs = 1_700_000_000_000,
                    ),
                )
            }
        }
    }

    @Test
    fun `hello rejects artifact without matching advertised model identity`() {
        val capabilities = artifactCapabilities().toMutableMap()
        capabilities["models"] = emptyList<Map<String, Any?>>()
        expectProtocolError("invalid_capabilities") {
            TaskWorkerProtocol.validate(
                TaskWorkerProtocol.buildHello(
                    nodeId = "android_worker_01",
                    capabilities = capabilities,
                    messageId = "msg_artifact_model_missing_01",
                    sentAtMs = 1_700_000_000_000,
                ),
            )
        }
    }

    @Test
    fun `hello rejects duplicate model ids ranges and artifact ranges`() {
        val duplicateModels = artifactCapabilities().toMutableMap()
        duplicateModels["models"] = listOf(model, model)
        expectProtocolError("invalid_capabilities") {
            TaskWorkerProtocol.validate(
                TaskWorkerProtocol.buildHello(
                    nodeId = "android_worker_01",
                    capabilities = duplicateModels,
                    messageId = "msg_duplicate_models_01",
                    sentAtMs = 1_700_000_000_000,
                ),
            )
        }

        val duplicateRanges = artifactCapabilities().toMutableMap()
        duplicateRanges["layer_ranges"] = listOf(listOf(4, 16), listOf(4, 16))
        expectProtocolError("invalid_capabilities") {
            TaskWorkerProtocol.validate(
                TaskWorkerProtocol.buildHello(
                    nodeId = "android_worker_01",
                    capabilities = duplicateRanges,
                    messageId = "msg_duplicate_ranges_01",
                    sentAtMs = 1_700_000_000_000,
                ),
            )
        }

        val duplicateArtifacts = artifactCapabilities().toMutableMap()
        duplicateArtifacts["layer_artifacts"] = listOf(validArtifact(), validArtifact())
        expectProtocolError("invalid_capabilities") {
            TaskWorkerProtocol.validate(
                TaskWorkerProtocol.buildHello(
                    nodeId = "android_worker_01",
                    capabilities = duplicateArtifacts,
                    messageId = "msg_duplicate_artifacts_01",
                    sentAtMs = 1_700_000_000_000,
                ),
            )
        }
    }

    private fun validArtifact(): Map<String, Any?> = mapOf(
        "layer_range" to listOf(4, 16),
        "segment_mode" to "middle",
        "model_id" to "mid4-16.gguf",
        "artifact_sha256" to "b".repeat(64),
        "source_model_sha256" to "a".repeat(64),
    )

    private fun artifactCapabilities(
        artifact: Map<String, Any?> = validArtifact(),
    ): Map<String, Any?> = mapOf(
        "stage_types" to listOf("layer_forward"),
        "engines" to listOf("llama_cpp"),
        "models" to listOf(
            mapOf(
                "model_id" to "mid4-16.gguf",
                "engine" to "llama_cpp",
                "format" to "gguf",
                "revision" to "local",
                "sha256" to "b".repeat(64),
            ),
        ),
        "max_concurrency" to 1,
        "layer_ranges" to listOf(listOf(4, 16)),
        "layer_artifacts" to listOf(artifact),
    )

    /**
     * 构造一个最小合法 v3 层段 offer。
     *
     * `middleChannel` / `extraFields` 缺省 ⇒ 不发对应可选字段（旧行为）；
     * `hiddenDtype="float16"` 时 `root_input` 放 `hidden_f16`（1×4 ⇒ 8 字节）。
     */
    private fun layerForwardOffer(
        middleChannel: String? = null,
        hiddenDtype: String = "float32",
        extraFields: Map<String, Any?> = emptyMap(),
    ): TaskWorkerEnvelope {
        val hiddenField = if (hiddenDtype == "float16") "hidden_f16" else "hidden_f32"
        val hiddenBytes = if (hiddenDtype == "float16") 8 else 16
        return TaskWorkerProtocol.buildStageOffer(
            identity = identity,
            requestId = "request_channel_01",
            stageType = "layer_forward",
            providerId = "remote_android_worker_01",
            leaseExpiresAtMs = 2_000,
            rootInput = mapOf(
                hiddenField to java.util.Base64.getEncoder().encodeToString(ByteArray(hiddenBytes)),
                "context_size" to 2048,
                "want_hidden" to true,
            ),
            dependencies = emptyMap(),
            modelIdentity = model,
            messageId = "msg_channel_01",
            sentAtMs = 1_000,
            stageFields = mapOf(
                "layer_range" to listOf(4, 8),
                "handoff_at" to 4,
                "hidden_sha256" to "c".repeat(64),
                "hidden_spec" to mapOf("n_tokens" to 1, "n_embd" to 4, "dtype" to hiddenDtype),
            ) + (middleChannel?.let { mapOf("middle_channel" to it) } ?: emptyMap()) + extraFields,
        )
    }

    @Test
    fun `v3 layer_forward accepts float16 hidden and explicit multi-sequence positions`() {
        // ★ A13：`dtype` 与主仓对齐（`float32` / `float16` 都接受）
        TaskWorkerProtocol.validate(layerForwardOffer(hiddenDtype = "float16"))

        // ★ A12：`seq_ids` / `positions` 可选；长度必须等于 `hidden_spec.n_tokens`（这里 1）
        TaskWorkerProtocol.validate(
            layerForwardOffer(extraFields = mapOf(
                "seq_ids" to listOf(0),
                "positions" to listOf(0),
            )),
        )
        expectProtocolError("invalid_seq_ids") {
            TaskWorkerProtocol.validate(
                layerForwardOffer(extraFields = mapOf("seq_ids" to listOf(0, 1))),
            )
        }
        expectProtocolError("invalid_positions") {
            TaskWorkerProtocol.validate(
                layerForwardOffer(extraFields = mapOf("positions" to listOf(-1))),
            )
        }
    }

    @Test
    fun `DIST-NEXT-1 stage_cancelled carries a bounded execution state`() {
        val inFlight = TaskWorkerProtocol.buildStageCancelled(
            identity = identity,
            providerId = "remote_android_worker_01",
            reasonCode = "coordinator_cancelled",
            messageId = "msg_cancelled_state01",
            sentAtMs = 1_700_000_000_000L,
            executionState = TaskWorkerProtocol.EXECUTION_IN_FLIGHT,
        )
        assertEquals(
            TaskWorkerProtocol.EXECUTION_IN_FLIGHT,
            inFlight.payload["execution_state"],
        )
        // 编解码往返必须保持该字段（对端据此区分「已取消」与「已停止」）。
        //   ⚠️ 不比较整个 envelope：Gson 往返会把整数解析成 Double，Map 相等性会因此失败。
        val roundTripped = TaskWorkerProtocol.decode(TaskWorkerProtocol.encode(inFlight))
        assertEquals(TaskWorkerProtocol.EXECUTION_IN_FLIGHT, roundTripped.payload["execution_state"])
        assertEquals("coordinator_cancelled", roundTripped.payload["reason_code"])

        // 值域封闭：未知状态一律拒收，绝不照抄。
        expectProtocolError("unsupported_execution_state") {
            TaskWorkerProtocol.buildStageCancelled(
                identity = identity,
                providerId = "remote_android_worker_01",
                reasonCode = "coordinator_cancelled",
                messageId = "msg_cancelled_state02",
                sentAtMs = 1_700_000_000_000L,
                executionState = "stopped",
            )
        }

        // 旧对端（不带该字段）仍然合法 —— 可选字段不得变成必填。
        val legacy = TaskWorkerProtocol.build(
            messageType = TaskWorkerProtocol.STAGE_CANCELLED,
            payload = identity.asPayload() + mapOf(
                "provider_id" to "remote_android_worker_01",
                "reason_code" to "coordinator_cancelled",
            ),
            messageId = "msg_cancelled_state03",
            sentAtMs = 1_700_000_000_000L,
        )
        assertNull(legacy.payload["execution_state"])
        val legacyRoundTrip = TaskWorkerProtocol.decode(TaskWorkerProtocol.encode(legacy))
        assertNull(legacyRoundTrip.payload["execution_state"])
        assertEquals("remote_android_worker_01", legacyRoundTrip.payload["provider_id"])

        // 该字段只属于回程 ACK；coordinator 的取消请求不得携带它。
        expectProtocolError("invalid_fields") {
            TaskWorkerProtocol.build(
                messageType = TaskWorkerProtocol.STAGE_CANCEL,
                payload = identity.asPayload() + mapOf(
                    "reason_code" to "coordinator_cancelled",
                    "execution_state" to TaskWorkerProtocol.EXECUTION_STOPPED,
                ),
                messageId = "msg_cancel_state01",
                sentAtMs = 1_700_000_000_000L,
            )
        }
    }

    @Test
    fun `DIST-NEXT-6 hello advertises unusable artifact reasons without paths`() {
        val hello = TaskWorkerProtocol.buildHello(
            nodeId = "android_worker_01",
            capabilities = mapOf(
                "stage_types" to listOf("layer_forward"),
                "engines" to listOf("llama_cpp"),
                "models" to listOf(model),
                "max_concurrency" to 1,
                "layer_artifact_diagnostics" to listOf(
                    artifactDiagnostic(
                        errorCode = "artifact_missing",
                        manifest = "mid.manifest.json",
                        mode = "middle",
                        layerRange = listOf(4, 16),
                    ),
                    artifactDiagnostic(
                        errorCode = "manifest_unreadable",
                        manifest = "gone.manifest.json",
                        mode = "",
                        layerRange = null,
                    ),
                ),
            ),
            messageId = "msg_hello_artifactdiag1",
            sentAtMs = 1_700_000_000_000L,
        )

        val decoded = TaskWorkerProtocol.decode(TaskWorkerProtocol.encode(hello))
        val capabilities = decoded.payload["capabilities"] as Map<*, *>
        val diagnostics = capabilities["layer_artifact_diagnostics"] as List<*>
        assertEquals(2, diagnostics.size)
        assertEquals(
            "artifact_missing",
            (diagnostics[0] as Map<*, *>)["error_code"],
        )
    }

    @Test
    fun `DIST-NEXT-6 artifact diagnostics are validated fail closed`() {
        fun helloWith(value: Any?): TaskWorkerEnvelope = TaskWorkerProtocol.buildHello(
            nodeId = "android_worker_01",
            capabilities = mapOf(
                "stage_types" to listOf("layer_forward"),
                "engines" to listOf("llama_cpp"),
                "models" to listOf(model),
                "max_concurrency" to 1,
                "layer_artifact_diagnostics" to value,
            ),
            messageId = "msg_hello_artifactdiag2",
            sentAtMs = 1_700_000_000_000L,
        )

        // 空列表没有信息量：要么不给该键，要么给非空列表
        expectProtocolError("invalid_capabilities") { helloWith(emptyList<Any?>()) }

        // error code 值域封闭
        expectProtocolError("invalid_capabilities") {
            helloWith(listOf(artifactDiagnostic(errorCode = "whatever")))
        }

        // 键集固定：携带本地路径的额外键会被拒
        expectRejected {
            helloWith(listOf(artifactDiagnostic(extra = mapOf("path" to "/sdcard/models/x.gguf"))))
        }

        // 合法项通过（含 layer_range = null 的「读不到 manifest」）
        TaskWorkerProtocol.validate(helloWith(listOf(
            artifactDiagnostic(errorCode = "artifact_digest_mismatch", manifest = "x.manifest.json"),
        )))
    }

    private fun artifactDiagnostic(
        errorCode: String = "artifact_missing",
        manifest: String = "mid.manifest.json",
        architecture: String = "qwen35",
        mode: String = "middle",
        layerRange: List<Int>? = listOf(4, 16),
        artifactPresent: Boolean = false,
        extra: Map<String, Any?> = emptyMap(),
    ): Map<String, Any?> = mapOf(
        "error_code" to errorCode,
        "manifest" to manifest,
        "architecture" to architecture,
        "mode" to mode,
        "layer_range" to layerRange,
        "artifact_present" to artifactPresent,
    ) + extra

    @Test
    fun `DIST-NEXT-2 hidden wire budget matches the master frame limit`() {
        // 与主仓 `task_worker_protocol.hidden_wire_bytes` 同公式、同常量
        assertEquals(5_462L, TaskWorkerProtocol.hiddenWireBytes(1_024L))
        assertEquals(2_731L, TaskWorkerProtocol.hiddenWireBytes(1_024L, 2))
        assertEquals(
            TaskWorkerProtocol.MAX_MESSAGE_BYTES.toLong() -
                TaskWorkerProtocol.STAGE_FRAME_RESERVE_BYTES,
            TaskWorkerProtocol.stagePayloadBudgetBytes(),
        )
        val budget = TaskWorkerProtocol.stagePayloadBudgetBytes()
        // 2048 宽的 1 token ≈ 10.9 KB：预算内；2_000_000 元素 ≈ 10.7 MB：超预算
        assertTrue(TaskWorkerProtocol.hiddenWireBytes(2_048L) < budget)
        assertTrue(TaskWorkerProtocol.hiddenWireBytes(2_000_000L) > budget)
    }
}
