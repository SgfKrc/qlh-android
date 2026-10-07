package com.qlh.inference.worker

/** Builds the conservative capability snapshot sent by an Android Full Worker. */
object AndroidWorkerCapabilities {
    /** Release/runtime profile names shared with the Python worker protocol. */
    const val DEFAULT_RUNTIME_PROFILE: String = "llama_cpp_only"
    const val UNSPECIFIED_RUNTIME_PROFILE: String = "unspecified"
    private val SUPPORTED_RUNTIME_PROFILES = setOf(
        DEFAULT_RUNTIME_PROFILE,
        "torch_cpu",
        "torch_cuda",
        UNSPECIFIED_RUNTIME_PROFILE,
    )
    private val SEGMENT_MODES = setOf("head", "middle", "tail")
    private val SHA256 = Regex("[0-9a-fA-F]{64}")
    private val SAFE_ID = Regex("^[A-Za-z0-9_.:-]{1,128}$")

    /** One runnable artifact, bound to its own range, mode, and executable identity. */
    data class LayerArtifactCapability(
        val startLayer: Int,
        val endLayerExclusive: Int,
        val segmentMode: String,
        val modelId: String,
        val artifactSha256: String,
        val sourceModelSha256: String? = null,
    ) {
        init {
            require(startLayer >= 0 && endLayerExclusive > startLayer) {
                "layer artifact range must be non-empty"
            }
            require(segmentMode in SEGMENT_MODES) {
                "layer artifact segment mode must be head, middle, or tail"
            }
            require(SAFE_ID.matches(modelId)) { "layer artifact model id is invalid" }
            require(SHA256.matches(artifactSha256)) {
                "layer artifact digest must be a 64-char hex digest"
            }
            require(sourceModelSha256 == null || SHA256.matches(sourceModelSha256)) {
                "layer artifact source model digest must be a 64-char hex digest"
            }
        }

        fun toMap(): Map<String, Any> = linkedMapOf<String, Any>(
            "layer_range" to listOf(startLayer, endLayerExclusive),
            "segment_mode" to segmentMode,
            "model_id" to modelId,
            "artifact_sha256" to artifactSha256.lowercase(),
        ).also { output ->
            sourceModelSha256?.let { output["source_model_sha256"] = it.lowercase() }
        }

        fun modelIdentity(): Map<String, Any?> = mapOf(
            "model_id" to modelId,
            "engine" to DEFAULT_ENGINE,
            "format" to "gguf",
            "revision" to "local",
            "sha256" to artifactSha256.lowercase(),
        )
    }

    fun isValidSegmentMode(value: String?): Boolean = value in SEGMENT_MODES

    /** Normalize a manifest filename into the protocol's safe model-id alphabet. */
    fun artifactModelId(artifactName: String, artifactSha256: String): String {
        val normalized = artifactName
            .substringAfterLast('/').substringAfterLast('\\')
            .replace(Regex("[^A-Za-z0-9_.:-]"), "_")
            .take(128)
        return normalized.ifBlank { "layer-${artifactSha256.lowercase().take(16)}" }
    }

    fun normalizeRuntimeProfile(value: String?): String {
        val normalized = value?.trim()?.lowercase().orEmpty()
        return normalized.takeIf { it in SUPPORTED_RUNTIME_PROFILES }
            ?: UNSPECIFIED_RUNTIME_PROFILE
    }
    /**
     * 本 Android worker 支持的推理引擎集合 —— **能力探测的单一来源**。
     *
     * 新增/切换引擎时只改这里，不再向协议校验处散落字面量：`TaskWorkerProtocol`
     * 通过 [areAllEnginesSupported] / [isSupportedEngine] 判定，而不是拿
     * `listOf("llama_cpp")` 做相等比较。与主仓 `CORE-KOAKUMA-ENGINE-ABC-01` 的
     * 「能力差异以能力探测表达、而非 `if engine_type`」判据对齐。
     */
    val SUPPORTED_ENGINES: List<String> = listOf("llama_cpp")

    /**
     * ★ 2026-10-07（DIST-NEXT-6）：能力广告里携带的工件诊断条数上限。
     * 广告是控制面消息，不承载病态目录的完整清单。
     */
    const val MAX_ARTIFACT_DIAGNOSTICS: Int = 32

    /** 默认引擎（用于未显式指定引擎的模型身份）。 */
    val DEFAULT_ENGINE: String = SUPPORTED_ENGINES.first()

    /** 该引擎是否受支持。 */
    fun isSupportedEngine(engine: String?): Boolean =
        engine != null && SUPPORTED_ENGINES.contains(engine)

    /** 上报的引擎列表是否全部受支持（空列表视为不合法）。 */
    fun areAllEnginesSupported(engines: List<String>): Boolean =
        engines.isNotEmpty() && engines.all { SUPPORTED_ENGINES.contains(it) }

    /** 供错误信息展示的支持清单。 */
    fun describeSupportedEngines(): String = SUPPORTED_ENGINES.joinToString(", ")

    /**
     * 本 Android worker 支持的 stage 类型 —— 与 [SUPPORTED_ENGINES] 同一约定：
     * 同样作为能力探测的单一来源，协议校验不得对字面量做相等比较。
     *
     * ✅ **2026-09-20：层段（`layer_forward`）现已声明。** 三个加入条件已全部满足：
     * 1. `qlh_llama_jni.cpp` 提供 `nativeLayerForwardToken` /
     *    `nativeLayerForwardHidden` / `nativeLayerForwardInfo`（收 hidden、
     *    `llama_batch.embd` 注入、只算本节点层区间）；
     * 2. `AndroidFullWorkerStageExecutor.executeLayerForward` 能真正执行该 stage，
     *    且 `TaskWorkerService` 已把 `layerForward` 接到 `LocalInferenceEngine`；
     * 3. JVM 单测断言「本清单里每个 stage 类型都能被执行器处理」
     *    （`AndroidWorkerCapabilitiesStageParityTest`）—— 防的是**能力撒谎**。
     *
     * ⚠️ 契约边界（不因声明而放松）：
     * * 本节点是**中间段**（要产出 hidden）时，模型**必须**以
     *   `extractHidden = true` 加载；未开启则 native 返回 `-2`，
     *   上层如实失败（`extract_hidden_not_enabled`），**不返回空 hidden**。
     * * **验证阶梯当前到 D/E（真机）**：
     *   - ✅ **B**（Gradle/APK 构建通过，含 `buildCMakeDebug[arm64-v8a]`）
     *   - ✅ **C+**（**x86_64 数值正确性**）：用**同一份 llama.cpp
     *     （b9902 / 47e1de77a）源码**以 x86_64 目标重编译，做端到端对照 ——
     *     整模型取 `layer_inp(K)` 当上游 hidden，裁层模型用 `llama_batch.embd`
     *     注入后续算取 argmax，与整模型 token 路径的 argmax 逐 token 比对。
     *     实测**两个 prompt、共 11 步全部 MATCH**（exit code 0）。
     *   - ✅ **D/E（真机 ARM64 数值）**：同一份源码经 NDK 交叉编译为 aarch64 可执行文件，
     *     在 **Lenovo Y700（TB321FU / **SM8650 = Snapdragon 8 Gen 3** / Android 15 /
     *     `arm64-v8a`）**上重跑同一对照 ⇒ **5 步逐 token 全部 MATCH**（exit 0），
     *     且 token 序列与 x86_64 的 C+ **完全相同**（`576/9396/11/14019/9396`）。
     *     辅助判据：反汇编 `libggml-cpu.so` 统计到 `sdot`×1063、`smmla`×244
     *     ⇒ 确认跑的是 **dotprod + i8mm 快速 kernel**，**不是 baseline**。
     *   - ✅ **D0（真机控制面）**：APK 经无线调试装到真机并启动，官方
     *     `scripts/android_validation.py --install --launch` 得
     *     `status = "native-worker-candidate"`（`jvm_and_build` /
     *     `apk_device_control_plane` / `arm64_native_worker` **均 true**）。
     *   - ⚠️ **仍未验证：APK 内部的层段能力**。
     *     * D/E 数值证据来自 **Termux 里的原生可执行文件**，它**绕过了**
     *       `qlh_llama_jni.cpp` 的 JNI 封装、Kotlin 执行器与协议 v3 编解码；
     *     * D0 只证明「装得上 / 起得来 / ABI 对」，**没有跑过一次 `layer_forward`**。
     *     ⇒ 要断言「APK 里的层段真的能用」，需 App 内入口或 instrumentation test。
     *     详见 `docs/安卓验证阶梯D-E层-真机Termux方案-2026-09-20.md` §2.1 / §5.1 / §9.9。
     */
    val SUPPORTED_STAGE_TYPES: List<String> = listOf("full_inference", "layer_forward")

    /** 默认 stage 类型。 */
    val DEFAULT_STAGE_TYPE: String = SUPPORTED_STAGE_TYPES.first()

    /** 该 stage 类型是否受支持。 */
    fun isSupportedStageType(stageType: String?): Boolean =
        stageType != null && SUPPORTED_STAGE_TYPES.contains(stageType)

    /** 上报的 stage 类型列表是否全部受支持（空列表视为不合法）。 */
    fun areAllStageTypesSupported(stageTypes: List<String>): Boolean =
        stageTypes.isNotEmpty() && stageTypes.all { SUPPORTED_STAGE_TYPES.contains(it) }

    /** 供错误信息展示的 stage 支持清单。 */
    fun describeSupportedStageTypes(): String = SUPPORTED_STAGE_TYPES.joinToString(", ")

    fun modelIdentity(
        modelId: String,
        modelFormat: String,
        modelRevision: String,
        modelSha256: String,
        resourceAdmitted: Boolean,
    ): Map<String, Any?>? = if (
        resourceAdmitted && modelId.isNotBlank() && modelSha256.matches(Regex("[a-fA-F0-9]{64}"))
    ) {
        mapOf(
            "model_id" to modelId,
            "engine" to DEFAULT_ENGINE,
            "format" to modelFormat,
            "revision" to modelRevision,
            "sha256" to modelSha256.lowercase(),
        )
    } else {
        null
    }

    /** 设备自荐的层容量；见 [computeLayerBudget]。 */
    data class LayerBudget(
        val availableBytes: Long,
        val perLayerBytes: Long,
        val maxLayers: Int,
        val localCut: Boolean,
    ) {
        fun toMap(): Map<String, Any> = mapOf(
            "available_bytes" to availableBytes,
            "per_layer_bytes" to perLayerBytes,
            "max_layers" to maxLayers,
            "local_cut" to localCut,
        )
    }

    /**
     * 按可用内存与单层字节推算本节点可承载的层数上限。
     *
     * 与 `layer_ranges` 分工：`layer_ranges` 是"当前已就绪、马上能跑的区间"，
     * `layer_budget` 是"本地裁层之后能承载的层数上限"。主仓调度据此可以把任意
     * 连续区间分给本节点，而不是被预置工件的那一段钉死。
     *
     * 任一项缺少依据就返回 `null`（**不猜**）：此时不上报 `layer_budget`，
     * 调度退回只按 `layer_ranges` 与容量账分配。
     *
     * @param availableBytes 可用于承载层段权重的可用内存
     * @param modelFileBytes 已选模型工件的字节数
     * @param coveredLayers 该工件覆盖的层数（整模＝模型总层数）
     * @param safetyFactor 给 KV 缓存与运行时留的余量系数（0 < f <= 1）
     * @param localCut 是否具备本地裁层条件（整模与裁层工具都就位）
     */
    fun computeLayerBudget(
        availableBytes: Long,
        modelFileBytes: Long,
        coveredLayers: Int,
        safetyFactor: Double = 0.6,
        localCut: Boolean = false,
    ): LayerBudget? {
        if (availableBytes <= 0L || modelFileBytes <= 0L || coveredLayers <= 0) return null
        if (safetyFactor <= 0.0 || safetyFactor > 1.0) return null
        val perLayerBytes = modelFileBytes / coveredLayers
        if (perLayerBytes <= 0L) return null
        val budgetBytes = (availableBytes * safetyFactor).toLong()
        val maxLayers = (budgetBytes / perLayerBytes).toInt()
        if (maxLayers <= 0) return null
        return LayerBudget(
            availableBytes = budgetBytes,
            perLayerBytes = perLayerBytes,
            maxLayers = maxLayers,
            localCut = localCut,
        )
    }

    fun build(
        modelId: String = "",
        modelFormat: String = "gguf",
        modelRevision: String = "local",
        modelSha256: String = "",
        resourceAdmitted: Boolean = false,
        resourceReason: String = "resource_gate_not_confirmed",
        layerRanges: List<List<Int>> = emptyList(),
        /** False for a layer-only worker that must never accept full-model offers. */
        fullInferenceAvailable: Boolean = true,
        /** True when the advertised identity is a source-SHA layer alias. */
        layerWorker: Boolean = false,
        /** Runtime profile selected by the Android release variant. */
        runtimeProfile: String = DEFAULT_RUNTIME_PROFILE,
        // ★ 2026-09-23：中间段通道能力与 M-RoPE 位置分量数（来自 native 的 `layerForwardInfo()`）。
        //   `null` = 未声明 ⇒ 不写这两个键（协议侧它们都是可选的）。
        middleChannel: String? = null,
        nPosPerEmbd: Int? = null,
        /** 设备自荐的层容量（本地裁层后可承载的层数上限）；null = 缺少依据，不上报。 */
        layerBudget: LayerBudget? = null,
        /** Per-artifact records; unlike the legacy global mode these cannot conflate ranges. */
        layerArtifacts: List<LayerArtifactCapability> = emptyList(),
        /**
         * ★ 2026-10-07（DIST-NEXT-6）：**不可用**层段工件的结构化原因（无本地路径）。
         *
         * 广告可用工件时同时说明「哪些 manifest 被判不可用、因为什么」—— 否则主节点
         * 只能看到 `layer_range_not_advertised`，无法区分「文件不存在」「摘要不符」
         * 「架构/模式非法」与「没有该区间」。
         */
        layerArtifactDiagnostics: List<Map<String, Any?>> = emptyList(),
    ): Map<String, Any?> {
        require(
            layerArtifacts.map { it.startLayer to it.endLayerExclusive }.distinct().size ==
                layerArtifacts.size
        ) { "layer artifact ranges must be unique" }
        require(layerArtifacts.map { it.modelId }.distinct().size == layerArtifacts.size) {
            "layer artifact model ids must be unique"
        }
        val normalizedReason = if (resourceAdmitted) "" else resourceReason.ifBlank {
            "resource_gate_not_confirmed"
        }
        val model = modelIdentity(
            modelId, modelFormat, modelRevision, modelSha256, resourceAdmitted,
        )
        val advertisedRanges = if (layerArtifacts.isNotEmpty()) {
            layerArtifacts.map { listOf(it.startLayer, it.endLayerExclusive) }
        } else {
            layerRanges
        }
        val normalizedRanges = advertisedRanges
            .filter { it.size == 2 && it[0] >= 0 && it[1] > it[0] }
            .distinct()
        val artifactModels = layerArtifacts.map { it.modelIdentity() }
        val advertisedModels = buildList {
            val artifactModelIds = artifactModels.mapNotNull { it["model_id"] }.toSet()
            if (model?.get("model_id") !in artifactModelIds) model?.let(::add)
            addAll(artifactModels)
        }.distinctBy { it["model_id"] }
        val capabilities = linkedMapOf<String, Any?>(
            "stage_types" to if (normalizedRanges.isEmpty()) {
                listOf("full_inference")
            } else if (!fullInferenceAvailable) {
                listOf("layer_forward")
            } else {
                SUPPORTED_STAGE_TYPES
            },
            "engines" to SUPPORTED_ENGINES,
            "models" to advertisedModels,
            "max_concurrency" to 1,
            "runtime_profile" to normalizeRuntimeProfile(runtimeProfile),
            "resource_gate" to mapOf(
                "admitted" to resourceAdmitted,
                "reason_code" to normalizedReason,
            ),
        )
        if (normalizedRanges.isNotEmpty()) capabilities["layer_ranges"] = normalizedRanges
        if (layerWorker) capabilities["layer_worker"] = true
        // ★ 2026-09-23：把 native 上报的「中间段通道 / M-RoPE 位置分量数」并入 capabilities，
        //   让调度侧能**知道**本节点中间段实际可走哪条通道（值域与协议侧同集合），
        //   而不是靠 `extract_hidden` 去猜。
        if (middleChannel != null) capabilities["middle_channel"] = middleChannel
        if (nPosPerEmbd != null && nPosPerEmbd > 0) capabilities["n_pos_per_embd"] = nPosPerEmbd
        // ★ 2026-10-03：设备自荐的层容量（本地裁层后可承载的层数上限）。
        //   与 `layer_ranges` 区别：后者是手上工件现成能跑的区间，前者是能自裁并
        //   承载的上限 ⇒ 有了它，主仓调度才能分配任意连续区间。
        if (layerBudget != null) capabilities["layer_budget"] = layerBudget.toMap()
        if (layerArtifacts.isNotEmpty()) {
            capabilities["layer_artifacts"] = layerArtifacts.map { it.toMap() }
            // Compatibility for older coordinators is safe only when every artifact agrees.
            layerArtifacts.map { it.segmentMode }.distinct().singleOrNull()?.let {
                capabilities["segment_mode"] = it
            }
        }
        // ★ 2026-10-07（DIST-NEXT-6）：把**不可用**工件的原因一并广告出去（不含本地路径）。
        //   上限 32 条：广告是控制面消息，不承载病态目录的完整清单。
        if (layerArtifactDiagnostics.isNotEmpty()) {
            capabilities["layer_artifact_diagnostics"] =
                layerArtifactDiagnostics.take(MAX_ARTIFACT_DIAGNOSTICS)
        }
        return capabilities
    }
}
