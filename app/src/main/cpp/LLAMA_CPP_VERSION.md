# llama.cpp 版本说明（Android Full 原生构建）

- 上游仓库：`https://github.com/ggml-org/llama.cpp`
- 固定 commit：`47e1de77aa0f06bf73cfd8c5281d95979f89fcbe`
- 接入方式：Git submodule（2026-08-08 由 vendored 源码树迁移；本仓库是 Android 独立交付边界）
- 用途：Android full-local GGUF 推理运行时（JNI：`qlh_llama_jni.cpp`，CMake `add_subdirectory(llama.cpp build-llama)`）
- Android 验证：迁移后 Full Release 构建通过（2026-08-08，assembleFullRelease BUILD SUCCESSFUL）；Lite 不依赖本目录。真机回归待做（结果后补）。
- 项目补丁：有，`conversion/base.py` 增加旧 Qwen `layer_norm_epsilon` 候选键。补丁源文件保留在主仓 `qlh/tools/model_tools/patches/llama-cpp-converter-qwen-eps.patch`，Android 仓只维护本地工作树状态，不向上游 submodule 推送。
