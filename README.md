# qlh-android

Standalone Android client for QLH distributed inference.

This repository owns the Android JVM/UI, instrumentation tests, JNI bridge,
Gradle build, resources, and the Android-native llama.cpp integration. The
QLH core service and model fleet remain in the sibling `qlh` repository.

## Build

```text
gradlew.bat :app:assembleFullDebug
gradlew.bat :app:testFullDebugUnitTest
```

The `app/src/main/cpp/llama.cpp` submodule is pinned to
`47e1de77aa0f06bf73cfd8c5281d95979f89fcbe`. Do not silently upgrade it; update
the pin and run the native/JVM gates together.

The Android client does not own PC PyTorch execution, Web UI code, image
generation, or model weights. Model assets are user-managed through the QLH
model contract.

The canonical remote is `https://github.com/SgfKrc/qlh-android.git`. This
checkout is the local migration baseline.
