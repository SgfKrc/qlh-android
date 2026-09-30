import java.util.Properties

plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.android)
    alias(libs.plugins.kotlin.compose)
    alias(libs.plugins.ksp)
}

val keystorePropertiesFile = rootProject.file("keystore.properties")
val keystoreProperties = Properties()
if (keystorePropertiesFile.exists()) {
    keystorePropertiesFile.inputStream().use { keystoreProperties.load(it) }
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
        versionCode = 6
        versionName = "0.1.8.3"

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
}

// Room schema 导出目录 — 必须在顶层，不能在 defaultConfig 内
ksp {
    arg("room.schemaLocation", "$projectDir/schemas")
}
