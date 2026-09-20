# llama.cpp 版本说明（Android Full 原生构建）

- 上游仓库：`https://github.com/ggml-org/llama.cpp`
- 固定 commit：`47e1de77aa0f06bf73cfd8c5281d95979f89fcbe`（与 QLH 主仓 `scripts/model_tools/llama_quantize.lock.json` 对齐）
- 接入方式：Git submodule（2026-08-08 由 vendored 源码树迁移；本仓库是 Android 独立交付边界）
- 用途：Android full-local GGUF 推理运行时（JNI：`qlh_llama_jni.cpp`，CMake `add_subdirectory(llama.cpp build-llama)`）
- Android 验证：迁移后 Full Release 构建通过（2026-08-08，assembleFullRelease BUILD SUCCESSFUL）；Lite 不依赖本目录。真机回归待做（结果后补）。
- 项目补丁：有，统一由主仓 `scripts/model_tools/patches/` 管理，摘要登记在 `scripts/model_tools/llama_quantize.lock.json`。当前挂一个补丁：`llama-cpp-layer-forward-api.patch` —— 把子模块内**早已实现但未导出**的 `set/get_embeddings_layer_inp`、`set_embeddings_nextn` 等 API 补进 `include/llama.h`，并让 `src/models/qwen2.cpp` 登记 `t_layer_inp`（层段接力 `layer_forward` 所需的上游能力；不改计算语义）。原 legacy Qwen ε 补丁（`conversion/base.py` 的 `layer_norm_epsilon` 候选键）已于 **2026-09-20 随 Qwen-1.8B 退役移除**。Android 仓只维护本地工作树状态，不向上游 submodule 推送。可在主仓执行 `python scripts/model_tools/sync_llama_cpp.py --dry-run` 检查，或显式加 `--apply` 应用。
