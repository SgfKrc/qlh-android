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

val keystoreLitePropertiesFile = rootProject.file("keystore-lite.properties")
val keystoreLiteProperties = Properties()
if (keystoreLitePropertiesFile.exists()) {
    keystoreLitePropertiesFile.inputStream().use { keystoreLiteProperties.load(it) }
}

android {
    namespace = "com.qlh.inference"
    compileSdk = 36
    ndkVersion = "27.2.12479018"

    defaultConfig {
        minSdk = 26
        targetSdk = 34
        versionCode = 6
        versionName = "0.1.8.3"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
    }

    signingConfigs {
        if (keystorePropertiesFile.exists()) {
            create("releaseFull") {
                storeFile = rootProject.file(keystoreProperties["storeFile"] as String)
                storePassword = keystoreProperties["storePassword"] as String
                keyAlias = keystoreProperties["keyAlias"] as String
                keyPassword = keystoreProperties["keyPassword"] as String
            }
        }
        if (keystoreLitePropertiesFile.exists()) {
            create("releaseLite") {
                storeFile = rootProject.file(keystoreLiteProperties["storeFile"] as String)
                storePassword = keystoreLiteProperties["storePassword"] as String
                keyAlias = keystoreLiteProperties["keyAlias"] as String
                keyPassword = keystoreLiteProperties["keyPassword"] as String
            }
        }
        // 向后兼容的回退签名（无 flavor 构建时使用）
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
            // 签名由 productFlavors 分别指定，避免 lite 被 buildType 回退到 full 签名
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

    flavorDimensions += "version"
    productFlavors {
        create("full") {
            dimension = "version"
            applicationId = "com.qlh.inference"
            versionNameSuffix = ""
            buildConfigField("boolean", "IS_LITE", "false")
            signingConfig = signingConfigs.findByName("releaseFull")

            ndk {
                abiFilters += listOf("arm64-v8a")
            }

            externalNativeBuild {
                cmake {
                    cppFlags += listOf("-std=c++17")
                    arguments += listOf("-DANDROID_STL=c++_shared")
                }
            }
        }
        create("lite") {
            dimension = "version"
            applicationId = "com.qlh.inference.lite"
            versionNameSuffix = "-lite"
            buildConfigField("boolean", "IS_LITE", "true")
            signingConfig = signingConfigs.findByName("releaseLite")

            ndk {
                abiFilters += listOf("arm64-v8a")
            }

            externalNativeBuild {
                cmake {
                    arguments += listOf("-DQLH_LITE=ON")
                }
            }
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
