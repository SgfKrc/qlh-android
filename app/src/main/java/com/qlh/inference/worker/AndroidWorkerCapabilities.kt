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
     * ⚠️ **2026-09-20：层段（`layer_forward`）暂不在此声明。**
     * 主仓任务协议已升到 v3 并定义了 `layer_forward`（层段），但 Android 侧的
     * **JNI 尚未实现** `llama_batch.embd` 注入与层段前向 ⇒ 若现在声明，
     * 就会出现「声明支持、执行时 unsupported_stage_type」的**能力撒谎**，
     * 违反 fail-closed 纪律（`AndroidFullWorkerStageExecutor` 只认已实现的类型）。
     *
     * **加入条件（完成 A3 后）**：`qlh_llama_jni.cpp` 提供
     * `nativeLayerForward*`（收 hidden、注入、只算本节点层区间），
     * 且 `AndroidFullWorkerStageExecutor` 能真正执行该 stage ⇒ 再把
     * `"layer_forward"` 加进本列表，并在 JVM 单测里断言两者一致。
     */
    val SUPPORTED_STAGE_TYPES: List<String> = listOf("full_inference")

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
