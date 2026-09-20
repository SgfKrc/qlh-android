package com.qlh.inference.worker

/** Builds the conservative capability snapshot sent by an Android Full Worker. */
object AndroidWorkerCapabilities {
    /**
     * 本 Android worker 支持的推理引擎集合 —— **能力探测的单一来源**。
     *
     * 新增/切换引擎时只改这里，不再向协议校验处散落字面量：`TaskWorkerProtocol`
     * 通过 [areAllEnginesSupported] / [isSupportedEngine] 判定，而不是拿
     * `listOf("llama_cpp")` 做相等比较。与主仓 `CORE-KOAKUMA-ENGINE-ABC-01` 的
     * 「能力差异以能力探测表达、而非 `if engine_type`」判据对齐。
     */
    val SUPPORTED_ENGINES: List<String> = listOf("llama_cpp")

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
     *   - ✅ **D/E（真机 ARM64）**：同一份源码经 NDK 交叉编译为 aarch64 可执行文件，
     *     在 **Lenovo Y700（TB321FU / **SM8650 = Snapdragon 8 Gen 3** / Android 15 /
     *     `arm64-v8a`）**上重跑同一对照 ⇒ **5 步逐 token 全部 MATCH**（exit 0），
     *     且 token 序列与 x86_64 的 C+ **完全相同**（`576/9396/11/14019/9396`）。
     *     辅助判据：反汇编 `libggml-cpu.so` 统计到 `sdot`×1063、`smmla`×244
     *     ⇒ 确认跑的是 **dotprod + i8mm 快速 kernel**，**不是 baseline**。
     *   - ⚠️ **仍未验证：APK 应用链路**。上述 D/E 证据来自 **Termux 里的原生可执行文件**，
     *     它**绕过了** `qlh_llama_jni.cpp` 的 JNI 封装、Kotlin 执行器、协议 v3 编解码、
     *     APK 打包签名与安装 —— **从未在真机上装过 APK**。
     *     详见 `docs/安卓验证阶梯D-E层-真机Termux方案-2026-09-20.md` §2.1 与 §5.1。
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

    fun build(
        modelId: String = "",
        modelFormat: String = "gguf",
        modelRevision: String = "local",
        modelSha256: String = "",
        resourceAdmitted: Boolean = false,
        resourceReason: String = "resource_gate_not_confirmed",
    ): Map<String, Any?> {
        val normalizedReason = if (resourceAdmitted) "" else resourceReason.ifBlank {
            "resource_gate_not_confirmed"
        }
        val model = modelIdentity(
            modelId, modelFormat, modelRevision, modelSha256, resourceAdmitted,
        )
        return mapOf(
            "stage_types" to SUPPORTED_STAGE_TYPES,
            "engines" to SUPPORTED_ENGINES,
            "models" to (model?.let { listOf(it) } ?: emptyList<Map<String, Any?>>()),
            "max_concurrency" to 1,
            "resource_gate" to mapOf(
                "admitted" to resourceAdmitted,
                "reason_code" to normalizedReason,
            ),
        )
    }
}
