# QLH Edge Inference — ProGuard Rules

# Keep Gson serialization classes
-keepattributes Signature
-keepattributes *Annotation*
-dontwarn sun.misc.**
-keep class com.google.gson.** { *; }
-keep class com.qlh.inference.network.** { *; }

# Keep Room entities
-keep class com.qlh.inference.data.** { *; }

# OkHttp
-dontwarn okhttp3.**
-dontwarn okio.**

# JNI native methods for llama.cpp Android bridge
-keepclasseswithmembernames class com.qlh.inference.service.LocalInferenceEngine {
    native <methods>;
}
-keep class com.qlh.inference.service.LocalInferenceEngine { *; }
