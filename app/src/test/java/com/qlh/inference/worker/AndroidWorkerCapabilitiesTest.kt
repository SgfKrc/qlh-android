package com.qlh.inference.worker

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Assert.assertThrows
import org.junit.Test

class AndroidWorkerCapabilitiesTest {
    @Test
    fun `resource gate defaults closed and withholds model identity`() {
        val capabilities = AndroidWorkerCapabilities.build(modelId = "qwen_1_8b")
        assertEquals(1, capabilities["max_concurrency"])
        assertEquals(AndroidWorkerCapabilities.DEFAULT_RUNTIME_PROFILE, capabilities["runtime_profile"])
        assertTrue((capabilities["models"] as List<*>).isEmpty())
        assertEquals(
            mapOf("admitted" to false, "reason_code" to "resource_gate_not_confirmed"),
            capabilities["resource_gate"],
        )
        val hello = TaskWorkerProtocol.buildHello(
            nodeId = "android_worker_01",
            capabilities = capabilities,
            messageId = "msg_capabilities_android01",
            sentAtMs = 1_700_000_000_000,
        )
        TaskWorkerProtocol.validate(hello)
    }

    @Test
    fun `layer budget derives from available memory and artifact size`() {
        val budget = AndroidWorkerCapabilities.computeLayerBudget(
            availableBytes = 8_000_000_000L,
            modelFileBytes = 1_000_000_000L,
            coveredLayers = 20,
            safetyFactor = 0.6,
        )

        assertNotNull(budget)
        assertEquals(50_000_000L, budget!!.perLayerBytes)
        assertEquals(96, budget.maxLayers)
        assertFalse(budget.localCut)
    }

    @Test
    fun `layer budget is withheld without evidence`() {
        assertNull(AndroidWorkerCapabilities.computeLayerBudget(0L, 1_000L, 10))
        assertNull(AndroidWorkerCapabilities.computeLayerBudget(1_000L, 0L, 10))
        assertNull(AndroidWorkerCapabilities.computeLayerBudget(1_000L, 1_000L, 0))
        assertNull(
            AndroidWorkerCapabilities.computeLayerBudget(
                1_000L, 1_000L, 10, safetyFactor = 0.0,
            ),
        )
    }

    @Test
    fun `build advertises layer budget only when it is known`() {
        val withBudget = AndroidWorkerCapabilities.build(
            modelId = "qwen3_5_2b",
            modelSha256 = "a".repeat(64),
            resourceAdmitted = true,
            layerRanges = listOf(listOf(4, 16)),
            layerBudget = AndroidWorkerCapabilities.LayerBudget(
                availableBytes = 100L,
                perLayerBytes = 10L,
                maxLayers = 8,
                localCut = true,
            ),
        )
        assertEquals(
            mapOf(
                "available_bytes" to 100L,
                "per_layer_bytes" to 10L,
                "max_layers" to 8,
                "local_cut" to true,
            ),
            withBudget["layer_budget"],
        )
        // 本端协议校验必须接受该可选键，否则真机 hello 会被自己拦下。
        TaskWorkerProtocol.validate(
            TaskWorkerProtocol.buildHello(
                nodeId = "android_worker_01",
                capabilities = withBudget,
                messageId = "msg_capabilities_budget",
                sentAtMs = 1_700_000_000_000,
            ),
        )

        val withoutBudget = AndroidWorkerCapabilities.build(
            modelId = "qwen3_5_2b",
            modelSha256 = "a".repeat(64),
            resourceAdmitted = true,
        )
        assertFalse(withoutBudget.containsKey("layer_budget"))
    }

    @Test
    fun `admitted gate advertises one exact llama model`() {
        val capabilities = AndroidWorkerCapabilities.build(
            modelId = "qwen_1_8b",
            modelFormat = "gguf",
            modelRevision = "local-v1",
            modelSha256 = "A".repeat(64),
            resourceAdmitted = true,
        )
        val models = capabilities["models"] as List<*>
        assertEquals(1, models.size)
        assertEquals(
            mapOf(
                "model_id" to "qwen_1_8b",
                "engine" to "llama_cpp",
                "format" to "gguf",
                "revision" to "local-v1",
                "sha256" to "a".repeat(64),
            ),
            models.single(),
        )
        val hello = TaskWorkerProtocol.buildHello(
            nodeId = "android_worker_01",
            capabilities = capabilities,
            messageId = "msg_capabilities_android02",
            sentAtMs = 1_700_000_000_000,
        )
        TaskWorkerProtocol.validate(hello)
    }

    @Test
    fun `middle_channel and n_pos_per_embd are advertised when native reports them`() {
        // ★ 2026-09-23：native `layerForwardInfo()` 上报 ⇒ worker capabilities 里要能看到
        //   中间段通道与 M-RoPE 位置分量数；缺省不写这两个键（向后兼容）。
        val advertised = AndroidWorkerCapabilities.build(
            modelId = "qwen3_5_9b",
            resourceAdmitted = true,
            middleChannel = "keep_head_layer_out",
            nPosPerEmbd = 4,
        )
        assertEquals("keep_head_layer_out", advertised["middle_channel"])
        assertEquals(4, (advertised["n_pos_per_embd"] as? Number)?.toInt())
        TaskWorkerProtocol.validate(
            TaskWorkerProtocol.buildHello(
                nodeId = "android_worker_01",
                capabilities = advertised,
                messageId = "msg_capabilities_channel_01",
                sentAtMs = 1_700_000_000_000,
            ),
        )

        // 不传 ⇒ 两个键都不出现（旧行为不变）。
        val plain = AndroidWorkerCapabilities.build(modelId = "qwen3_5_9b")
        assertTrue(!plain.containsKey("middle_channel"))
        assertTrue(!plain.containsKey("n_pos_per_embd"))
    }

    @Test
    fun `layer-only worker advertises ranges without full inference`() {
        val capabilities = AndroidWorkerCapabilities.build(
            modelId = "qwen3-5-2b",
            modelFormat = "gguf",
            modelRevision = "local-a86078042c3e",
            modelSha256 = "a".repeat(64),
            resourceAdmitted = true,
            layerRanges = listOf(listOf(4, 16)),
            fullInferenceAvailable = false,
        )
        assertEquals(listOf("layer_forward"), capabilities["stage_types"])
        assertEquals(listOf(listOf(4, 16)), capabilities["layer_ranges"])
        TaskWorkerProtocol.validate(
            TaskWorkerProtocol.buildHello(
                nodeId = "android_layer_only_01",
                capabilities = capabilities,
                messageId = "msg_capabilities_layer_only_01",
                sentAtMs = 1_700_000_000_000,
            ),
        )
    }

    @Test
    fun `multiple artifacts retain range mode and identity without unsafe global mode`() {
        val artifacts = listOf(
            AndroidWorkerCapabilities.LayerArtifactCapability(
                startLayer = 0,
                endLayerExclusive = 8,
                segmentMode = "head",
                modelId = "head0-8.gguf",
                artifactSha256 = "b".repeat(64),
                sourceModelSha256 = "a".repeat(64),
                sourceModelId = "qwen3-5-2b",
                hiddenSize = 2048,
                tokenizerSha256 = "d".repeat(64),
            ),
            AndroidWorkerCapabilities.LayerArtifactCapability(
                startLayer = 8,
                endLayerExclusive = 16,
                segmentMode = "middle",
                modelId = "mid8-16.gguf",
                artifactSha256 = "c".repeat(64),
                sourceModelSha256 = "a".repeat(64),
            ),
        )
        val capabilities = AndroidWorkerCapabilities.build(
            resourceAdmitted = true,
            fullInferenceAvailable = false,
            layerWorker = true,
            layerArtifacts = artifacts,
        )

        assertEquals(listOf(listOf(0, 8), listOf(8, 16)), capabilities["layer_ranges"])
        assertFalse(capabilities.containsKey("segment_mode"))
        assertEquals(artifacts.map { it.toMap() }, capabilities["layer_artifacts"])
        val models = capabilities["models"] as List<*>
        assertEquals(2, models.size)
        assertEquals("b".repeat(64), (models[0] as Map<*, *>)["sha256"])
        assertEquals("c".repeat(64), (models[1] as Map<*, *>)["sha256"])
        val advertisedArtifacts = capabilities["layer_artifacts"] as List<*>
        val headArtifact = advertisedArtifacts[0] as Map<*, *>
        assertEquals("qwen3-5-2b", headArtifact["source_model_id"])
        assertEquals(2048, headArtifact["hidden_size"])
        assertEquals("d".repeat(64), headArtifact["tokenizer_sha256"])
        TaskWorkerProtocol.validate(
            TaskWorkerProtocol.buildHello(
                nodeId = "android_multi_artifact_01",
                capabilities = capabilities,
                messageId = "msg_capabilities_artifacts_01",
                sentAtMs = 1_700_000_000_000,
            ),
        )
    }

    @Test
    fun `homogeneous artifacts emit legacy global mode`() {
        val capabilities = AndroidWorkerCapabilities.build(
            resourceAdmitted = true,
            layerArtifacts = listOf(
                AndroidWorkerCapabilities.LayerArtifactCapability(
                    4, 8, "middle", "mid4-8.gguf", "b".repeat(64),
                ),
                AndroidWorkerCapabilities.LayerArtifactCapability(
                    8, 12, "middle", "mid8-12.gguf", "c".repeat(64),
                ),
            ),
        )

        assertEquals("middle", capabilities["segment_mode"])
    }

    @Test
    fun `artifact identity replaces duplicate base model id`() {
        val capabilities = AndroidWorkerCapabilities.build(
            modelId = "tail.gguf",
            modelSha256 = "a".repeat(64),
            resourceAdmitted = true,
            layerArtifacts = listOf(
                AndroidWorkerCapabilities.LayerArtifactCapability(
                    12, 16, "tail", "tail.gguf", "b".repeat(64),
                ),
            ),
        )

        val models = capabilities["models"] as List<*>
        assertEquals(1, models.size)
        assertEquals("b".repeat(64), (models.single() as Map<*, *>)["sha256"])
        TaskWorkerProtocol.validate(
            TaskWorkerProtocol.buildHello(
                nodeId = "android_duplicate_model_01",
                capabilities = capabilities,
                messageId = "msg_capabilities_duplicate_01",
                sentAtMs = 1_700_000_000_000,
            ),
        )
    }

    @Test
    fun `builder rejects duplicate artifact ranges and model ids`() {
        val first = AndroidWorkerCapabilities.LayerArtifactCapability(
            4, 8, "middle", "mid4-8.gguf", "b".repeat(64),
        )
        assertThrows(IllegalArgumentException::class.java) {
            AndroidWorkerCapabilities.build(
                resourceAdmitted = true,
                layerArtifacts = listOf(
                    first,
                    AndroidWorkerCapabilities.LayerArtifactCapability(
                        4, 8, "middle", "mid4-8-v2.gguf", "c".repeat(64),
                    ),
                ),
            )
        }
        assertThrows(IllegalArgumentException::class.java) {
            AndroidWorkerCapabilities.build(
                resourceAdmitted = true,
                layerArtifacts = listOf(
                    first,
                    AndroidWorkerCapabilities.LayerArtifactCapability(
                        8, 12, "middle", "mid4-8.gguf", "c".repeat(64),
                    ),
                ),
            )
        }
    }

    @Test
    fun `artifact preflight contract is validated as an all or none tuple`() {
        assertThrows(IllegalArgumentException::class.java) {
            AndroidWorkerCapabilities.LayerArtifactCapability(
                4, 8, "middle", "mid4-8.gguf", "b".repeat(64),
                sourceModelId = "qwen3-5-2b",
            )
        }
        assertThrows(IllegalArgumentException::class.java) {
            AndroidWorkerCapabilities.LayerArtifactCapability(
                4, 8, "middle", "mid4-8.gguf", "b".repeat(64),
                sourceModelId = "bad id",
                hiddenSize = 2048,
                tokenizerSha256 = "c".repeat(64),
            )
        }
        assertThrows(IllegalArgumentException::class.java) {
            AndroidWorkerCapabilities.LayerArtifactCapability(
                4, 8, "middle", "mid4-8.gguf", "b".repeat(64),
                sourceModelId = "qwen3-5-2b",
                hiddenSize = 0,
                tokenizerSha256 = "c".repeat(64),
            )
        }
        assertThrows(IllegalArgumentException::class.java) {
            AndroidWorkerCapabilities.LayerArtifactCapability(
                4, 8, "middle", "mid4-8.gguf", "b".repeat(64),
                sourceModelId = "qwen3-5-2b",
                hiddenSize = 2048,
                tokenizerSha256 = "not-a-digest",
            )
        }
    }

    @Test
    fun `artifact filename is normalized to protocol safe model id`() {
        assertEquals(
            "mid_segment_.gguf",
            AndroidWorkerCapabilities.artifactModelId(
                "models/mid segment?.gguf",
                "b".repeat(64),
            ),
        )
    }

    @Test
    fun `layer-only worker marks alias identity explicitly`() {
        val capabilities = AndroidWorkerCapabilities.build(
            modelId = "layer-aaaaaaaaaaaaaaaa",
            modelFormat = "gguf",
            modelRevision = "local-aaaaaaaaaaaa",
            modelSha256 = "a".repeat(64),
            resourceAdmitted = true,
            layerRanges = listOf(listOf(4, 16)),
            fullInferenceAvailable = false,
            layerWorker = true,
        )
        assertEquals(true, capabilities["layer_worker"])
        TaskWorkerProtocol.validate(
            TaskWorkerProtocol.buildHello(
                nodeId = "android_layer_alias_01",
                capabilities = capabilities,
                messageId = "msg_capabilities_layer_alias_01",
                sentAtMs = 1_700_000_000_000,
            ),
        )
    }

    @Test
    fun `runtime profile is carried by hello capabilities`() {
        val capabilities = AndroidWorkerCapabilities.build(
            runtimeProfile = "llama_cpp_only",
        )
        assertEquals("llama_cpp_only", capabilities["runtime_profile"])
        TaskWorkerProtocol.validate(
            TaskWorkerProtocol.buildHello(
                nodeId = "android_profile_01",
                capabilities = capabilities,
                messageId = "msg_capabilities_profile_01",
                sentAtMs = 1_700_000_000_000,
            ),
        )
    }

    @Test
    fun `unknown runtime profile is fail-closed to unspecified`() {
        val capabilities = AndroidWorkerCapabilities.build(runtimeProfile = "torch_edge")
        assertEquals(AndroidWorkerCapabilities.UNSPECIFIED_RUNTIME_PROFILE, capabilities["runtime_profile"])
        TaskWorkerProtocol.validate(
            TaskWorkerProtocol.buildHello(
                nodeId = "android_profile_02",
                capabilities = capabilities,
                messageId = "msg_capabilities_profile_02",
                sentAtMs = 1_700_000_000_000,
            ),
        )
    }
}
