package com.qlh.inference.status

data class AndroidRuntimeStatus(
    val nativeRuntimeAvailable: Boolean = false,
    val nativeRuntimeError: String? = null,
    val serviceRunning: Boolean = false,
    val inferenceMode: String = "thin",
    val isLite: Boolean = false,
    val system: SystemStatus = SystemStatus(),
    val memory: MemoryStatus = MemoryStatus(),
    val storage: StorageStatus = StorageStatus(),
    val gpu: GpuStatus = GpuStatus(),
    val backend: BackendStatus = BackendStatus(),
    val multimodal: MultimodalStatus = MultimodalStatus(),
    val model: ModelRuntimeStatus = ModelRuntimeStatus(),
    val context: ContextRuntimeStatus = ContextRuntimeStatus(),
)

data class SystemStatus(
    val manufacturer: String = "",
    val brand: String = "",
    val model: String = "",
    val device: String = "",
    val hardware: String = "",
    val socManufacturer: String = "",
    val socModel: String = "",
    val sdkInt: Int = 0,
    val androidRelease: String = "",
    val abis: List<String> = emptyList(),
    val cpuCores: Int = 0,
    val powerSaveMode: Boolean = false,
    val thermalStatus: String = "unknown",
)

data class MemoryStatus(
    val availableBytes: Long = 0L,
    val totalBytes: Long = 0L,
    val thresholdBytes: Long = 0L,
    val lowMemory: Boolean = false,
    val heapMaxBytes: Long = 0L,
    val heapTotalBytes: Long = 0L,
    val heapFreeBytes: Long = 0L,
    val lowRamDevice: Boolean = false,
)

data class StorageStatus(
    val filesAvailableBytes: Long = 0L,
    val filesTotalBytes: Long = 0L,
    val cacheAvailableBytes: Long = 0L,
    val cacheTotalBytes: Long = 0L,
)

data class GpuStatus(
    val vendor: String = "",
    val renderer: String = "",
    val version: String = "",
    val probeError: String? = null,
    val supportsGpuOffload: Boolean = false,
    val backendDevices: String = "",
    val note: String = "当前 Android Full/Lite 版本仅探测 GPU 作为设备画像；本地推理仍使用 CPU llama.cpp。Android GPU 版将作为单独版本规划。",
)

data class BackendStatus(
    val engine: String = "llama.cpp Android CPU",
    val systemInfo: String = "",
    val supportsMmap: Boolean = false,
    val supportsMlock: Boolean = false,
    val supportsGpuOffload: Boolean = false,
    val supportsRpc: Boolean = false,
)

data class MultimodalStatus(
    val visionSupported: Boolean = false,
    val assetsPresent: Boolean = false,
    val assetSizesVerified: Boolean = false,
    val ready: Boolean = false,
    val reason: String = "",
)

data class ModelRuntimeStatus(
    val selectedName: String = "",
    val selectedSizeBytes: Long = 0L,
    val selectedSource: String = "",
    val selectedUri: String = "",
    val loaded: Boolean = false,
    val loadedPath: String = "",
    val loadedSourceUri: String = "",
    val name: String = "",
    val backend: String = "",
    val params: String = "",
    val layers: String = "",
    val sizeBytes: Long = 0L,
    val vocabTokens: String = "",
    val embedding: String = "",
    val heads: String = "",
)

data class ContextRuntimeStatus(
    val configuredContextSize: Int = 0,
    val modelContextSize: String = "",
    val trainContextSize: String = "",
    val batchSize: String = "",
    val microBatchSize: String = "",
    val lastPromptTokens: Int = 0,
    val lastGeneratedTokens: Int = 0,
    val lastTotalTokens: Int = 0,
    val lastElapsedSeconds: Double = 0.0,
    val lastTokensPerSecond: Double = 0.0,
    val stopReason: String = "",
    val estimatedKvMemoryMb: Double = 0.0,
    val persistentKvReuseEnabled: Boolean = false,
    val note: String = "Android 当前每次生成前清空 llama memory；这里展示最后一次生成上下文与 KV 估算。",
)
