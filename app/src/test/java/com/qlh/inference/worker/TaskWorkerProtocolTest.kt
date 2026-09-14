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
}
