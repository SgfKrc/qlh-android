import java.util.Properties
import java.io.File
import javax.inject.Inject
import org.gradle.process.ExecOperations

plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.android)
    alias(libs.plugins.kotlin.compose)
    alias(libs.plugins.ksp)
}

// Build identity comes from the containing core checkout and its fixed Git graph.
// Use one executable path/name, with no shell command or revision environment override.
abstract class GenerateQlhBuildInfo : DefaultTask() {
    @get:Input
    abstract val pythonExecutable: Property<String>

    @get:InputFile
    @get:PathSensitive(PathSensitivity.RELATIVE)
    abstract val generatorScript: RegularFileProperty

    @get:Internal
    abstract val coreCheckout: DirectoryProperty

    @get:OutputDirectory
    abstract val assetsDirectory: DirectoryProperty

    @get:Inject
    abstract val execOperations: ExecOperations

    @TaskAction
    fun generate() {
        execOperations.exec {
            commandLine(
                pythonExecutable.get(), "-B", generatorScript.get().asFile.absolutePath,
                "--root", coreCheckout.get().asFile.absolutePath,
                "--output", assetsDirectory.file("build-info.json").get().asFile.absolutePath,
            )
        }
    }
}

// A package action, rather than a sibling task: all compile/native/merge
// dependencies must have completed before comparing the original source snapshot.
open class VerifyQlhBuildInfoAction @Inject constructor(
    private val execOperations: ExecOperations,
) : Action<Task> {
    lateinit var pythonExecutable: String
    lateinit var generatorScript: File
    lateinit var coreCheckout: File
    lateinit var snapshotFile: Provider<RegularFile>

    override fun execute(task: Task) {
        task.logger.lifecycle("Verifying QLH build-info snapshot before ${task.path}")
        execOperations.exec {
            commandLine(
                pythonExecutable, "-B", generatorScript.absolutePath,
                "--root", coreCheckout.absolutePath,
                "--verify", snapshotFile.get().asFile.absolutePath,
            )
        }
    }
}

val provenanceCoreRoot = rootProject.projectDir.parentFile
val provenancePython = providers.environmentVariable("QLH_BUILD_PYTHON").orElse("python")
val provenanceScript = provenanceCoreRoot.resolve("scripts/package_contract/build_provenance.py")
// Observe actual HEADs, source bytes and dirty state even when timestamps/HEAD did
// not change. This input also validates the selected Python and exact Git roots.
val provenanceSnapshot = providers.exec {
    commandLine(
        provenancePython.get(), "-B", provenanceScript.absolutePath,
        "--root", provenanceCoreRoot.absolutePath, "--print",
    )
}.standardOutput.asText

val generateQlhBuildInfo = tasks.register<GenerateQlhBuildInfo>("generateQlhBuildInfo") {
    pythonExecutable.set(provenancePython)
    generatorScript.set(provenanceScript)
    coreCheckout.set(provenanceCoreRoot)
    assetsDirectory.set(layout.buildDirectory.dir("generated/qlhBuildInfo/assets"))
    inputs.property("sourceProvenance", provenanceSnapshot)
    // A copied generated asset or cached configuration must never reuse old identity.
    outputs.upToDateWhen { false }
    outputs.doNotCacheIf("Build source provenance must be freshly observed") { true }
}

androidComponents {
    onVariants(selector().all()) { variant ->
        variant.sources.assets?.addGeneratedSourceDirectory(
            generateQlhBuildInfo, GenerateQlhBuildInfo::assetsDirectory,
        )
    }
}
tasks.named("preBuild") {
    dependsOn(generateQlhBuildInfo)
}

// Use the actual APK task provider for every application variant (debug/release).
// doFirst runs after its dependencies and before AGP writes/signs the APK. JVM
// compile/test tasks do not invoke this action. Never refresh metadata here.
android.applicationVariants.all {
    packageApplicationProvider.configure {
        val originalSnapshot = generateQlhBuildInfo.flatMap {
            it.assetsDirectory.file("build-info.json")
        }
        inputs.file(originalSnapshot).withPropertyName("qlhBuildInfoSnapshot")
            .withPathSensitivity(PathSensitivity.RELATIVE)
        outputs.upToDateWhen { false }
        outputs.doNotCacheIf("APK packaging must verify the original source snapshot") { true }
        val verification = objects.newInstance(VerifyQlhBuildInfoAction::class.java).apply {
            pythonExecutable = provenancePython.get()
            generatorScript = provenanceScript
            coreCheckout = provenanceCoreRoot
            snapshotFile = originalSnapshot
        }
        doFirst("verifyQlhBuildInfo", verification)
    }
}

val keystorePropertiesFile = rootProject.file("keystore.properties")
val keystoreProperties = Properties()
if (keystorePropertiesFile.exists()) {
    keystorePropertiesFile.inputStream().use { keystoreProperties.load(it) }
}

val releaseVersionFile = rootProject.file("version.properties")
check(releaseVersionFile.isFile) {
    "Missing generated version.properties; run python scripts/version_contract.py --write from the core checkout."
}
val releaseVersion = Properties().apply {
    releaseVersionFile.inputStream().use { load(it) }
}

// ★ 2026-09-30：`keystore-lite.properties` / lite 专属签名随 full/lite 合并一并移除。

android {
    namespace = "com.qlh.inference"
    compileSdk = 36
    ndkVersion = "27.2.12479018"

    defaultConfig {
        // ★ 2026-09-30 产品基线整改：**放弃 full/lite 区分** —— 原先 `full`/`lite` 两个
        //   productFlavor 合并为单一默认变体（能力面 = 原 full：内置 llama 运行时）。
        //   理由：产品基线以 TUI 为主，Android 侧不再需要"瘦客户端"变体；
        //   `IS_LITE` 仍是既有 Kotlin 分支读取的符号，统一置 `false`（代码分支保留，零行为回归）。
        applicationId = "com.qlh.inference"
        buildConfigField("boolean", "IS_LITE", "false")
        minSdk = 26
        targetSdk = 34
        versionCode = releaseVersion.getProperty("versionCode").toInt()
        versionName = releaseVersion.getProperty("productVersion")

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"

        ndk {
            abiFilters += listOf("arm64-v8a")
        }

        externalNativeBuild {
            cmake {
                cppFlags += listOf("-std=c++17")
                arguments += listOf(
                    "-DANDROID_STL=c++_shared",
                    // Android 15+ 的 16 KB page 设备要求 ELF LOAD 段 16 KB 对齐；
                    // NDK r27 通过该开关带上 -Wl,-z,max-page-size=16384（无需改源码）。
                    "-DANDROID_SUPPORT_FLEXIBLE_PAGE_SIZES=ON",
                    // ★ 2026-10-08（真机性能根因）：此前**没有传任何 arch 参数** ⇒
                    //   `GGML_CPU_ARM_ARCH` 为空 ⇒ ggml-cpu 退到 baseline NEON，设备上
                    //   `ggml_cpu_has_dotprod()/ggml_cpu_has_matmul_int8()` 恒为 0（编译期宏）。
                    //   真机实测：Y700（骁龙 8 Gen 3）上 1062 tokens × 4 层 >43 秒，远超硬件
                    //   应有水平。带 dotprod/fp16/i8mm 的同类构建在 Termux 手工验证里已存在
                    //   （`build-aarch64/CMakeCache.txt` ⇒ `armv8.6-a+dotprod+fp16+i8mm`）。
                    "-DGGML_CPU_ARM_ARCH=armv8.6-a+dotprod+fp16+i8mm",
                )
            }
        }
    }

    signingConfigs {
        if (keystorePropertiesFile.exists()) {
            create("release") {
                storeFile = rootProject.file(keystoreProperties["storeFile"] as String)
                storePassword = keystoreProperties["storePassword"] as String
                keyAlias = keystoreProperties["keyAlias"] as String
                keyPassword = keystoreProperties["keyPassword"] as String
            }
        }
    }

    testOptions {
        // ★ 2026-10-07（真机 P0 修复的测试前提）：JVM 单测里 `android.util.Log.*` 默认抛
        //   `RuntimeException: Method ... not mocked`。`QlhLogger` 每个方法都会调 `Log.*`，
        //   于是"取消收敛"这条**带日志的关键路径**在测试里会中断 —— 真机缺陷因此被掩盖。
        //   打开官方的 return-default 语义后，日志调用返回默认值，不参与失败判定。
        unitTests.isReturnDefaultValues = true
    }

    buildTypes {
        release {
            // ★ 2026-09-30：full/lite 合并后只有一套签名（`keystore.properties`）。
            signingConfig = signingConfigs.findByName("release")
            isMinifyEnabled = true
            isShrinkResources = true
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
        debug {
            isMinifyEnabled = false
            applicationIdSuffix = ".debug"
            versionNameSuffix = "-debug"
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    kotlinOptions {
        jvmTarget = "17"
    }

    packaging {
        jniLibs {
            useLegacyPackaging = false
        }
    }

    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/CMakeLists.txt")
            version = "3.22.1"
        }
    }

    buildFeatures {
        compose = true
        buildConfig = true
    }
}

dependencies {
    // AndroidX Core
    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.lifecycle.runtime.ktx)
    implementation(libs.androidx.lifecycle.viewmodel.compose)
    implementation(libs.androidx.lifecycle.runtime.compose)
    implementation(libs.androidx.activity.compose)
    implementation(libs.androidx.splash.screen)

    // Compose BOM
    implementation(platform(libs.compose.bom))
    implementation(libs.compose.ui)
    implementation(libs.compose.ui.graphics)
    implementation(libs.compose.ui.tooling.preview)
    implementation(libs.compose.material3)
    implementation(libs.compose.material.icons.extended)
    debugImplementation(libs.compose.ui.tooling)
    debugImplementation(libs.compose.ui.test.manifest)
    androidTestImplementation(platform(libs.compose.bom))
    androidTestImplementation(libs.compose.ui.test.junit4)
    androidTestImplementation(libs.androidx.test.ext.junit)
    // 显式对齐 runner：否则解析到 1.5.0，与 core 1.6.1 / monitor 1.7.1 混搭会让
    // ActivityScenario 在 teardown 找不到 InstrumentationActivityInvoker$EmptyActivity。
    androidTestImplementation(libs.androidx.test.runner)
    androidTestImplementation(libs.junit)

    // JVM 单元测试（纯逻辑，无需设备）
    testImplementation(libs.junit)

    // Navigation
    implementation(libs.androidx.navigation.compose)

    // Room
    implementation(libs.androidx.room.runtime)
    implementation(libs.androidx.room.ktx)
    ksp(libs.androidx.room.compiler)

    // OkHttp
    implementation(libs.okhttp)
    implementation(libs.okhttp.logging)

    // Gson
    implementation(libs.gson)

    // DataStore
    implementation(libs.androidx.datastore.preferences)

    // Coroutines
    implementation(libs.kotlinx.coroutines.android)

    // AndroidKeyStore does not expose an exportable Ed25519 seed on API 35.
    implementation(libs.bouncycastle.crypto)
}

// Room schema 导出目录 — 必须在顶层，不能在 defaultConfig 内
ksp {
    arg("room.schemaLocation", "$projectDir/schemas")
}
