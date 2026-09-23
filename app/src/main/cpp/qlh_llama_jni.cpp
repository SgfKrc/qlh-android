#include <android/log.h>
#include <jni.h>
#include <sys/sysinfo.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

#include "ggml-backend.h"
#include "llama.h"
#include "mtmd-helper.h"
#include "mtmd.h"

#define QLH_LOG_TAG "QlhLlamaJni"
#define QLH_LOGI(...) __android_log_print(ANDROID_LOG_INFO, QLH_LOG_TAG, __VA_ARGS__)
#define QLH_LOGW(...) __android_log_print(ANDROID_LOG_WARN, QLH_LOG_TAG, __VA_ARGS__)
#define QLH_LOGE(...) __android_log_print(ANDROID_LOG_ERROR, QLH_LOG_TAG, __VA_ARGS__)

struct QlhGenerationStats {
    int prompt_tokens = 0;
    int generated_tokens = 0;
    int total_tokens = 0;
    double elapsed_seconds = 0.0;
    double tokens_per_second = 0.0;
    std::string stop_reason;
};

struct QlhLlamaContext {
    llama_model * model = nullptr;
    llama_context * ctx = nullptr;
    llama_sampler * sampler = nullptr;
    mtmd_context * vision = nullptr;
    const llama_vocab * vocab = nullptr;
    int n_ctx = 0;
    int n_threads = 0;
    int n_threads_batch = 0;
    int n_gpu_layers = 0;
    //: ★ 2026-09-20（层段）：context 是否开启了隐藏态导出。只有为 true 时
    //: `llama_get_embeddings_ith` 才有值 ⇒ 中间层段节点必须以此为前提，
    //: `nativeLayerForwardHidden` 据此**如实报错**而不是回退成空 hidden。
    bool extract_hidden = false;
    QlhGenerationStats last_stats;
};

static void free_vision(QlhLlamaContext * qctx) {
    if (qctx != nullptr && qctx->vision != nullptr) {
        mtmd_free(qctx->vision);
        qctx->vision = nullptr;
    }
}

static std::once_flag g_backend_once;

static void ensure_backend_initialized() {
    std::call_once(g_backend_once, []() {
        ggml_backend_load_all();
        llama_backend_init();
        QLH_LOGI("llama backend initialized: %s", llama_print_system_info());
    });
}

static void throw_java(JNIEnv * env, const char * message) {
    jclass cls = env->FindClass("java/lang/IllegalStateException");
    if (cls != nullptr) {
        env->ThrowNew(cls, message);
    }
}

static void map_put(JNIEnv * env, jobject map, jmethodID put_method, const char * key, const std::string & value) {
    jstring j_key = env->NewStringUTF(key);
    jstring j_value = env->NewStringUTF(value.c_str());
    env->CallObjectMethod(map, put_method, j_key, j_value);
    env->DeleteLocalRef(j_key);
    env->DeleteLocalRef(j_value);
}

static jobject new_string_map(JNIEnv * env, jmethodID * put_method_out) {
    jclass map_class = env->FindClass("java/util/HashMap");
    jmethodID map_init = env->GetMethodID(map_class, "<init>", "()V");
    *put_method_out = env->GetMethodID(
        map_class,
        "put",
        "(Ljava/lang/Object;Ljava/lang/Object;)Ljava/lang/Object;"
    );
    return env->NewObject(map_class, map_init);
}

static int available_threads() {
    const int cores = std::max(1, get_nprocs());
    return std::max(2, std::min(4, cores - 1));
}

static bool valid_utf8(const std::string & s) {
    const auto * bytes = reinterpret_cast<const unsigned char *>(s.c_str());
    size_t i = 0;
    while (i < s.size()) {
        unsigned char c = bytes[i];
        size_t n = 0;
        if ((c & 0x80) == 0) {
            n = 1;
        } else if ((c & 0xE0) == 0xC0) {
            n = 2;
        } else if ((c & 0xF0) == 0xE0) {
            n = 3;
        } else if ((c & 0xF8) == 0xF0) {
            n = 4;
        } else {
            return false;
        }
        if (i + n > s.size()) {
            return false;
        }
        for (size_t j = 1; j < n; ++j) {
            if ((bytes[i + j] & 0xC0) != 0x80) {
                return false;
            }
        }
        i += n;
    }
    return true;
}

static std::vector<llama_token> tokenize(JNIEnv * env, QlhLlamaContext * qctx, const std::string & text) {
    int n_tokens = -llama_tokenize(
        qctx->vocab,
        text.c_str(),
        static_cast<int32_t>(text.size()),
        nullptr,
        0,
        true,
        true
    );
    if (n_tokens <= 0) {
        throw_java(env, "Prompt tokenize failed");
        return {};
    }

    std::vector<llama_token> tokens(static_cast<size_t>(n_tokens));
    int actual = llama_tokenize(
        qctx->vocab,
        text.c_str(),
        static_cast<int32_t>(text.size()),
        tokens.data(),
        static_cast<int32_t>(tokens.size()),
        true,
        true
    );
    if (actual < 0) {
        throw_java(env, "Prompt tokenize failed");
        return {};
    }
    tokens.resize(static_cast<size_t>(actual));
    return tokens;
}

static std::string token_to_piece(QlhLlamaContext * qctx, llama_token token) {
    char buffer[256];
    int n = llama_token_to_piece(qctx->vocab, token, buffer, sizeof(buffer), 0, true);
    if (n < 0) {
        std::vector<char> large(static_cast<size_t>(-n));
        n = llama_token_to_piece(qctx->vocab, token, large.data(), static_cast<int32_t>(large.size()), 0, true);
        if (n < 0) {
            return {};
        }
        return std::string(large.data(), static_cast<size_t>(n));
    }
    return std::string(buffer, static_cast<size_t>(n));
}

static const char * backend_device_type_name(enum ggml_backend_dev_type type) {
    switch (type) {
        case GGML_BACKEND_DEVICE_TYPE_CPU: return "cpu";
        case GGML_BACKEND_DEVICE_TYPE_GPU: return "gpu";
        case GGML_BACKEND_DEVICE_TYPE_IGPU: return "igpu";
        case GGML_BACKEND_DEVICE_TYPE_ACCEL: return "accel";
        case GGML_BACKEND_DEVICE_TYPE_META: return "meta";
        default: return "unknown";
    }
}

static std::string backend_devices_summary() {
    std::ostringstream oss;
    const size_t count = ggml_backend_dev_count();
    for (size_t i = 0; i < count; ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (dev == nullptr) continue;
        if (oss.tellp() > 0) oss << "; ";
        size_t free_mem = 0;
        size_t total_mem = 0;
        ggml_backend_dev_memory(dev, &free_mem, &total_mem);
        oss << ggml_backend_dev_name(dev)
            << "(" << backend_device_type_name(ggml_backend_dev_type(dev)) << ")";
        if (total_mem > 0) {
            oss << " mem=" << total_mem;
        }
    }
    return oss.str();
}

static double estimate_kv_memory_mb(QlhLlamaContext * qctx, int tokens) {
    if (qctx == nullptr || qctx->model == nullptr || tokens <= 0) return 0.0;
    const double layers = std::max(1, llama_model_n_layer(qctx->model));
    const double embd = std::max(1, llama_model_n_embd(qctx->model));
    const double bytes = static_cast<double>(tokens) * layers * embd * 2.0 * 2.0;
    return bytes / 1024.0 / 1024.0;
}

extern "C" JNIEXPORT jlong JNICALL
Java_com_qlh_inference_service_LocalInferenceEngine_nativeLoadModel(
    JNIEnv * env,
    jobject /* thiz */,
    jstring j_path,
    jint j_n_ctx,
    jboolean j_extract_hidden,
    jint j_gpu_layers
) {
    ensure_backend_initialized();

    const char * path_chars = env->GetStringUTFChars(j_path, nullptr);
    if (path_chars == nullptr) {
        return 0;
    }
    std::string model_path(path_chars);
    env->ReleaseStringUTFChars(j_path, path_chars);

    QLH_LOGI("loading model: %s", model_path.c_str());

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = std::max(-1, static_cast<int>(j_gpu_layers));
    if (!llama_supports_gpu_offload() && model_params.n_gpu_layers != 0) {
        QLH_LOGW("GPU offload requested but no llama backend is available; using CPU");
        model_params.n_gpu_layers = 0;
    }

    llama_model * model = llama_model_load_from_file(model_path.c_str(), model_params);
    if (model == nullptr) {
        QLH_LOGE("llama_model_load_from_file failed: %s", model_path.c_str());
        return 0;
    }

    const int n_threads = available_threads();
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = static_cast<uint32_t>(std::max(512, static_cast<int>(j_n_ctx)));
    // ★ 2026-09-20 修 BUG：原先 `n_batch = min(512, n_ctx)`、`n_ubatch = min(256, n_batch)`。
    //   层段（layer_forward）每步注入的是**整段序列**的 hidden，长 prefill 会超过 512
    //   ⇒ `llama_decode` 越界（实验侧实测 rc=0xC0000409）。批量必须能吃下最长序列，
    //   故与 n_ctx 联动（上游 llama.cpp 的默认 n_batch 也是 2048 量级）。
    ctx_params.n_batch = ctx_params.n_ctx;
    ctx_params.n_ubatch = ctx_params.n_ctx;
    ctx_params.n_threads = n_threads;
    ctx_params.n_threads_batch = n_threads;
    ctx_params.no_perf = true;
    // ★ 2026-09-20（层段）：中间层段节点需要把「本段末位置的输出 hidden」交给下一段，
    //   而 `llama_get_embeddings_ith` 仅在 context 开启隐藏态导出时才有值。
    //   ⚠️ 字段名随版本而异：**android 子模块的 llama.cpp 为 `b9902`，用的是
    //      `embeddings`**；桌面实验 fork（较新）叫 `extract_hidden_states`。
    //      这正是「两侧必须同构建」的一个具体体现 —— 不要盲目照搬桌面侧写法。
    //   ⚠️ 代价：开启后会保留隐藏态张量，有额外内存/时间成本 ⇒ **默认关闭**，
    //      只有要当**中间层段**时才置 true（末段只需 logits）。
    ctx_params.embeddings = (j_extract_hidden == JNI_TRUE);

    llama_context * ctx = llama_init_from_model(model, ctx_params);
    if (ctx == nullptr) {
        QLH_LOGE("llama_init_from_model failed");
        llama_model_free(model);
        return 0;
    }

    auto sampler_params = llama_sampler_chain_default_params();
    sampler_params.no_perf = true;
    llama_sampler * sampler = llama_sampler_chain_init(sampler_params);
    llama_sampler_chain_add(sampler, llama_sampler_init_top_k(40));
    llama_sampler_chain_add(sampler, llama_sampler_init_top_p(0.9f, 1));
    llama_sampler_chain_add(sampler, llama_sampler_init_temp(0.7f));
    llama_sampler_chain_add(sampler, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

    auto * qctx = new QlhLlamaContext();
    qctx->model = model;
    qctx->ctx = ctx;
    qctx->sampler = sampler;
    qctx->vocab = llama_model_get_vocab(model);
    qctx->n_ctx = static_cast<int>(llama_n_ctx(ctx));
    qctx->n_threads = llama_n_threads(ctx);
    qctx->n_threads_batch = llama_n_threads_batch(ctx);
    qctx->n_gpu_layers = model_params.n_gpu_layers;
    qctx->extract_hidden = (j_extract_hidden == JNI_TRUE);

    QLH_LOGI("model loaded: ctx=%d n_batch=%u extract_hidden=%d gpu_layers=%d threads=%d",
             qctx->n_ctx, ctx_params.n_batch,
             ctx_params.embeddings ? 1 : 0, qctx->n_gpu_layers, n_threads);
    return reinterpret_cast<jlong>(qctx);
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_qlh_inference_service_LocalInferenceEngine_nativeLoadMultimodalProjector(
    JNIEnv * env,
    jobject /* thiz */,
    jlong ptr,
    jstring j_mmproj_path
) {
    auto * qctx = reinterpret_cast<QlhLlamaContext *>(ptr);
    if (qctx == nullptr || qctx->model == nullptr || j_mmproj_path == nullptr) {
        throw_java(env, "Model is not loaded");
        return JNI_FALSE;
    }

    const char * path_chars = env->GetStringUTFChars(j_mmproj_path, nullptr);
    if (path_chars == nullptr) {
        return JNI_FALSE;
    }
    std::string mmproj_path(path_chars);
    env->ReleaseStringUTFChars(j_mmproj_path, path_chars);

    mtmd_context_params params = mtmd_context_params_default();
    params.use_gpu = false;
    params.n_threads = qctx->n_threads;
    params.batch_max_tokens = std::min(1024, qctx->n_ctx);
    params.warmup = false;
    mtmd_context * vision = mtmd_init_from_file(
        mmproj_path.c_str(), qctx->model, params
    );
    if (vision == nullptr || !mtmd_support_vision(vision)) {
        if (vision != nullptr) mtmd_free(vision);
        QLH_LOGW("mmproj does not provide vision capability: %s", mmproj_path.c_str());
        return JNI_FALSE;
    }
    free_vision(qctx);
    qctx->vision = vision;
    QLH_LOGI("mmproj loaded with vision capability");
    return JNI_TRUE;
}

extern "C" JNIEXPORT void JNICALL
Java_com_qlh_inference_service_LocalInferenceEngine_nativeFreeModel(
    JNIEnv * /* env */,
    jobject /* thiz */,
    jlong ptr
) {
    auto * qctx = reinterpret_cast<QlhLlamaContext *>(ptr);
    if (qctx == nullptr) {
        return;
    }
    free_vision(qctx);
    if (qctx->sampler != nullptr) {
        llama_sampler_free(qctx->sampler);
    }
    if (qctx->ctx != nullptr) {
        llama_free(qctx->ctx);
    }
    if (qctx->model != nullptr) {
        llama_model_free(qctx->model);
    }
    delete qctx;
    QLH_LOGI("model freed");
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_qlh_inference_service_LocalInferenceEngine_nativeGenerate(
    JNIEnv * env,
    jobject /* thiz */,
    jlong ptr,
    jstring j_prompt,
    jint j_max_tokens,
    jfloat j_temperature,
    jfloat j_top_p,
    jobject on_token
) {
    auto * qctx = reinterpret_cast<QlhLlamaContext *>(ptr);
    if (qctx == nullptr || qctx->model == nullptr || qctx->ctx == nullptr || qctx->sampler == nullptr) {
        throw_java(env, "Model is not loaded");
        return env->NewStringUTF("");
    }

    const auto t_start = std::chrono::steady_clock::now();
    QlhGenerationStats stats;
    stats.stop_reason = "unknown";

    const char * prompt_chars = env->GetStringUTFChars(j_prompt, nullptr);
    if (prompt_chars == nullptr) {
        return env->NewStringUTF("");
    }
    std::string prompt(prompt_chars);
    env->ReleaseStringUTFChars(j_prompt, prompt_chars);

    llama_memory_clear(llama_get_memory(qctx->ctx), true);

    if (qctx->sampler != nullptr) {
        llama_sampler_free(qctx->sampler);
    }
    auto sampler_params = llama_sampler_chain_default_params();
    sampler_params.no_perf = true;
    qctx->sampler = llama_sampler_chain_init(sampler_params);
    llama_sampler_chain_add(qctx->sampler, llama_sampler_init_top_k(40));
    llama_sampler_chain_add(qctx->sampler, llama_sampler_init_top_p(std::max(0.01f, std::min(1.0f, static_cast<float>(j_top_p))), 1));
    llama_sampler_chain_add(qctx->sampler, llama_sampler_init_temp(std::max(0.0f, static_cast<float>(j_temperature))));
    llama_sampler_chain_add(qctx->sampler, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

    std::vector<llama_token> prompt_tokens = tokenize(env, qctx, prompt);
    stats.prompt_tokens = static_cast<int>(prompt_tokens.size());
    if (env->ExceptionCheck() || prompt_tokens.empty()) {
        stats.stop_reason = "tokenize_error";
        qctx->last_stats = stats;
        return env->NewStringUTF("");
    }

    if (static_cast<int>(prompt_tokens.size()) >= qctx->n_ctx) {
        stats.stop_reason = "context_overflow";
        qctx->last_stats = stats;
        throw_java(env, "Prompt exceeds context window");
        return env->NewStringUTF("");
    }

    llama_batch batch = llama_batch_get_one(prompt_tokens.data(), static_cast<int32_t>(prompt_tokens.size()));
    if (llama_decode(qctx->ctx, batch) != 0) {
        stats.stop_reason = "prompt_decode_error";
        qctx->last_stats = stats;
        throw_java(env, "llama_decode failed while processing prompt");
        return env->NewStringUTF("");
    }

    jclass callback_class = env->GetObjectClass(on_token);
    jmethodID invoke_method = callback_class == nullptr
        ? nullptr
        : env->GetMethodID(callback_class, "invoke", "(Ljava/lang/Object;)Ljava/lang/Object;");

    std::string output;
    std::string pending_utf8;
    llama_token next_token = LLAMA_TOKEN_NULL;
    const int max_tokens = std::max(1, static_cast<int>(j_max_tokens));
    stats.stop_reason = "max_tokens";

    for (int i = 0; i < max_tokens; ++i) {
        next_token = llama_sampler_sample(qctx->sampler, qctx->ctx, -1);
        llama_sampler_accept(qctx->sampler, next_token);

        if (llama_vocab_is_eog(qctx->vocab, next_token)) {
            stats.stop_reason = "eog";
            break;
        }
        stats.generated_tokens += 1;

        std::string piece = token_to_piece(qctx, next_token);
        pending_utf8 += piece;
        if (valid_utf8(pending_utf8)) {
            output += pending_utf8;
            if (invoke_method != nullptr && !pending_utf8.empty()) {
                jstring j_piece = env->NewStringUTF(pending_utf8.c_str());
                env->CallObjectMethod(on_token, invoke_method, j_piece);
                env->DeleteLocalRef(j_piece);
                if (env->ExceptionCheck()) {
                    stats.stop_reason = "callback_exception";
                    break;
                }
            }
            pending_utf8.clear();
        }

        llama_batch next_batch = llama_batch_get_one(&next_token, 1);
        if (llama_decode(qctx->ctx, next_batch) != 0) {
            stats.stop_reason = "decode_error";
            qctx->last_stats = stats;
            throw_java(env, "llama_decode failed during generation");
            return env->NewStringUTF(output.c_str());
        }
    }

    if (!pending_utf8.empty() && valid_utf8(pending_utf8)) {
        output += pending_utf8;
    }

    const auto t_end = std::chrono::steady_clock::now();
    stats.total_tokens = stats.prompt_tokens + stats.generated_tokens;
    stats.elapsed_seconds = std::chrono::duration<double>(t_end - t_start).count();
    if (stats.elapsed_seconds > 0.0) {
        stats.tokens_per_second = static_cast<double>(stats.generated_tokens) / stats.elapsed_seconds;
    }
    qctx->last_stats = stats;
    return env->NewStringUTF(output.c_str());
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_qlh_inference_service_LocalInferenceEngine_nativeGenerateMultimodal(
    JNIEnv * env,
    jobject /* thiz */,
    jlong ptr,
    jstring j_prompt,
    jobjectArray j_images,
    jint j_max_tokens,
    jfloat j_temperature,
    jfloat j_top_p,
    jobject on_token
) {
    auto * qctx = reinterpret_cast<QlhLlamaContext *>(ptr);
    if (qctx == nullptr || qctx->model == nullptr || qctx->ctx == nullptr ||
        qctx->sampler == nullptr || qctx->vision == nullptr) {
        throw_java(env, "Multimodal model is not ready");
        return env->NewStringUTF("");
    }
    if (j_images == nullptr || env->GetArrayLength(j_images) <= 0) {
        throw_java(env, "At least one image is required");
        return env->NewStringUTF("");
    }

    const char * prompt_chars = env->GetStringUTFChars(j_prompt, nullptr);
    if (prompt_chars == nullptr) return env->NewStringUTF("");
    std::string prompt(prompt_chars);
    env->ReleaseStringUTFChars(j_prompt, prompt_chars);
    const char * marker = mtmd_default_marker();
    if (prompt.find(marker) == std::string::npos) {
        prompt = std::string(marker) + "\n" + prompt;
    }

    llama_memory_clear(llama_get_memory(qctx->ctx), true);
    if (qctx->sampler != nullptr) llama_sampler_free(qctx->sampler);
    auto sampler_params = llama_sampler_chain_default_params();
    sampler_params.no_perf = true;
    qctx->sampler = llama_sampler_chain_init(sampler_params);
    llama_sampler_chain_add(qctx->sampler, llama_sampler_init_top_k(40));
    llama_sampler_chain_add(qctx->sampler, llama_sampler_init_top_p(
        std::max(0.01f, std::min(1.0f, static_cast<float>(j_top_p))), 1));
    llama_sampler_chain_add(qctx->sampler, llama_sampler_init_temp(
        std::max(0.0f, static_cast<float>(j_temperature))));
    llama_sampler_chain_add(qctx->sampler, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

    std::vector<mtmd_bitmap *> bitmaps;
    std::vector<const mtmd_bitmap *> bitmap_refs;
    const jsize image_count = env->GetArrayLength(j_images);
    bitmaps.reserve(static_cast<size_t>(image_count));
    bitmap_refs.reserve(static_cast<size_t>(image_count));
    for (jsize i = 0; i < image_count; ++i) {
        auto image = static_cast<jbyteArray>(env->GetObjectArrayElement(j_images, i));
        if (image == nullptr) {
            throw_java(env, "Image payload is null");
            for (auto * bitmap : bitmaps) mtmd_bitmap_free(bitmap);
            return env->NewStringUTF("");
        }
        const jsize length = env->GetArrayLength(image);
        if (length <= 0 || length > 8 * 1024 * 1024) {
            env->DeleteLocalRef(image);
            throw_java(env, "Image payload exceeds the Android limit");
            for (auto * bitmap : bitmaps) mtmd_bitmap_free(bitmap);
            return env->NewStringUTF("");
        }
        jboolean is_copy = JNI_FALSE;
        auto * bytes = env->GetByteArrayElements(image, &is_copy);
        if (bytes == nullptr) {
            env->DeleteLocalRef(image);
            throw_java(env, "Unable to access image payload");
            for (auto * bitmap : bitmaps) mtmd_bitmap_free(bitmap);
            return env->NewStringUTF("");
        }
        auto wrapper = mtmd_helper_bitmap_init_from_buf(
            qctx->vision,
            reinterpret_cast<const unsigned char *>(bytes),
            static_cast<size_t>(length),
            false
        );
        if (bytes != nullptr) env->ReleaseByteArrayElements(image, bytes, JNI_ABORT);
        env->DeleteLocalRef(image);
        if (wrapper.video_ctx != nullptr) mtmd_helper_video_free(wrapper.video_ctx);
        if (wrapper.bitmap == nullptr) {
            throw_java(env, "MTMD image decode failed");
            for (auto * bitmap : bitmaps) mtmd_bitmap_free(bitmap);
            return env->NewStringUTF("");
        }
        bitmaps.push_back(wrapper.bitmap);
        bitmap_refs.push_back(wrapper.bitmap);
    }

    mtmd_input_chunks * chunks = mtmd_input_chunks_init();
    mtmd_input_text input_text{
        prompt.c_str(),
        true,
        true
    };
    const int32_t tokenize_result = mtmd_tokenize(
        qctx->vision,
        chunks,
        &input_text,
        bitmap_refs.data(),
        bitmap_refs.size()
    );
    for (auto * bitmap : bitmaps) mtmd_bitmap_free(bitmap);
    if (tokenize_result != 0) {
        mtmd_input_chunks_free(chunks);
        throw_java(env, "MTMD prompt/image tokenize failed");
        return env->NewStringUTF("");
    }

    QlhGenerationStats stats;
    stats.stop_reason = "max_tokens";
    const size_t chunk_count = mtmd_input_chunks_size(chunks);
    llama_pos n_past = 0;
    for (size_t i = 0; i < chunk_count; ++i) {
        const mtmd_input_chunk * chunk = mtmd_input_chunks_get(chunks, i);
        const int32_t eval_result = mtmd_helper_eval_chunk_single(
            qctx->vision,
            qctx->ctx,
            chunk,
            n_past,
            0,
            llama_n_batch(qctx->ctx),
            i + 1 == chunk_count,
            &n_past
        );
        if (eval_result != 0) {
            mtmd_input_chunks_free(chunks);
            stats.stop_reason = "mtmd_decode_error";
            qctx->last_stats = stats;
            throw_java(env, "MTMD decode failed");
            return env->NewStringUTF("");
        }
        stats.prompt_tokens += static_cast<int>(mtmd_input_chunk_get_n_tokens(chunk));
    }
    mtmd_input_chunks_free(chunks);
    if (stats.prompt_tokens >= qctx->n_ctx) {
        stats.stop_reason = "context_overflow";
        qctx->last_stats = stats;
        throw_java(env, "Multimodal prompt exceeds context window");
        return env->NewStringUTF("");
    }

    jclass callback_class = env->GetObjectClass(on_token);
    jmethodID invoke_method = callback_class == nullptr
        ? nullptr
        : env->GetMethodID(callback_class, "invoke", "(Ljava/lang/Object;)Ljava/lang/Object;");
    std::string output;
    std::string pending_utf8;
    const int max_tokens = std::max(1, static_cast<int>(j_max_tokens));
    const auto t_start = std::chrono::steady_clock::now();
    for (int i = 0; i < max_tokens; ++i) {
        llama_token next_token = llama_sampler_sample(qctx->sampler, qctx->ctx, -1);
        llama_sampler_accept(qctx->sampler, next_token);
        if (llama_vocab_is_eog(qctx->vocab, next_token)) {
            stats.stop_reason = "eog";
            break;
        }
        stats.generated_tokens += 1;
        pending_utf8 += token_to_piece(qctx, next_token);
        if (valid_utf8(pending_utf8)) {
            output += pending_utf8;
            if (invoke_method != nullptr && !pending_utf8.empty()) {
                jstring piece = env->NewStringUTF(pending_utf8.c_str());
                env->CallObjectMethod(on_token, invoke_method, piece);
                env->DeleteLocalRef(piece);
                if (env->ExceptionCheck()) {
                    stats.stop_reason = "callback_exception";
                    break;
                }
            }
            pending_utf8.clear();
        }
        llama_batch next_batch = llama_batch_get_one(&next_token, 1);
        if (llama_decode(qctx->ctx, next_batch) != 0) {
            stats.stop_reason = "decode_error";
            qctx->last_stats = stats;
            throw_java(env, "llama_decode failed during multimodal generation");
            return env->NewStringUTF(output.c_str());
        }
    }
    if (!pending_utf8.empty() && valid_utf8(pending_utf8)) output += pending_utf8;
    const auto t_end = std::chrono::steady_clock::now();
    stats.total_tokens = stats.prompt_tokens + stats.generated_tokens;
    stats.elapsed_seconds = std::chrono::duration<double>(t_end - t_start).count();
    if (stats.elapsed_seconds > 0.0) {
        stats.tokens_per_second = static_cast<double>(stats.generated_tokens) / stats.elapsed_seconds;
    }
    qctx->last_stats = stats;
    return env->NewStringUTF(output.c_str());
}

extern "C" JNIEXPORT jobject JNICALL
Java_com_qlh_inference_service_LocalInferenceEngine_nativeGetModelInfo(
    JNIEnv * env,
    jobject /* thiz */,
    jlong ptr
) {
    auto * qctx = reinterpret_cast<QlhLlamaContext *>(ptr);
    if (qctx == nullptr || qctx->model == nullptr) {
        throw_java(env, "Model is not loaded");
        return nullptr;
    }

    jmethodID put_method = nullptr;
    jobject map = new_string_map(env, &put_method);

    auto put = [&](const char * key, const std::string & value) {
        map_put(env, map, put_method, key, value);
    };

    char desc[256] = {0};
    llama_model_desc(qctx->model, desc, sizeof(desc));

    put("name", desc);
    put("n_ctx", std::to_string(llama_n_ctx(qctx->ctx)));
    put("n_ctx_train", std::to_string(llama_model_n_ctx_train(qctx->model)));
    put("n_batch", std::to_string(llama_n_batch(qctx->ctx)));
    put("n_ubatch", std::to_string(llama_n_ubatch(qctx->ctx)));
    put("n_layer", std::to_string(llama_model_n_layer(qctx->model)));
    put("n_params", std::to_string(llama_model_n_params(qctx->model)));
    put("size_bytes", std::to_string(llama_model_size(qctx->model)));
    put("n_embd", std::to_string(llama_model_n_embd(qctx->model)));
    put("n_head", std::to_string(llama_model_n_head(qctx->model)));
    put("n_head_kv", std::to_string(llama_model_n_head_kv(qctx->model)));
    put("vocab_tokens", std::to_string(llama_vocab_n_tokens(qctx->vocab)));
    put("n_threads", std::to_string(qctx->n_threads));
    put("n_threads_batch", std::to_string(qctx->n_threads_batch));
    put("ftype", std::to_string(static_cast<int>(llama_model_ftype(qctx->model))));
    put("backend", qctx->n_gpu_layers == 0
        ? "llama.cpp Android CPU"
        : "llama.cpp Android GPU offload");
    put("n_gpu_layers", std::to_string(qctx->n_gpu_layers));
    put("supports_gpu_offload", llama_supports_gpu_offload() ? "true" : "false");
    put("estimated_kv_memory_mb", std::to_string(estimate_kv_memory_mb(qctx, qctx->n_ctx)));

    return map;
}

extern "C" JNIEXPORT jobject JNICALL
Java_com_qlh_inference_service_LocalInferenceEngine_nativeGetBackendInfo(
    JNIEnv * env,
    jobject /* thiz */
) {
    ensure_backend_initialized();

    jmethodID put_method = nullptr;
    jobject map = new_string_map(env, &put_method);
    map_put(env, map, put_method, "system_info", llama_print_system_info());
    map_put(env, map, put_method, "supports_mmap", llama_supports_mmap() ? "true" : "false");
    map_put(env, map, put_method, "supports_mlock", llama_supports_mlock() ? "true" : "false");
    map_put(env, map, put_method, "supports_gpu_offload", llama_supports_gpu_offload() ? "true" : "false");
    map_put(env, map, put_method, "supports_rpc", llama_supports_rpc() ? "true" : "false");
    map_put(env, map, put_method, "backend_device_count", std::to_string(ggml_backend_dev_count()));
    map_put(env, map, put_method, "backend_devices", backend_devices_summary());
    return map;
}

extern "C" JNIEXPORT jobject JNICALL
Java_com_qlh_inference_service_LocalInferenceEngine_nativeGetMultimodalCapability(
    JNIEnv * env,
    jobject /* thiz */,
    jlong ptr
) {
    jmethodID put_method = nullptr;
    jobject map = new_string_map(env, &put_method);
    auto * qctx = reinterpret_cast<QlhLlamaContext *>(ptr);
    const bool vision = qctx != nullptr && qctx->vision != nullptr &&
        mtmd_support_vision(qctx->vision);
    map_put(env, map, put_method, "vision_supported", vision ? "true" : "false");
    map_put(env, map, put_method, "audio_supported", "false");
    map_put(env, map, put_method, "mtmd_compiled", "true");
    map_put(env, map, put_method, "reason", vision ? "ready" : "mmproj not loaded");
    return map;
}

extern "C" JNIEXPORT jobject JNICALL
Java_com_qlh_inference_service_LocalInferenceEngine_nativeGetLastGenerationStats(
    JNIEnv * env,
    jobject /* thiz */,
    jlong ptr
) {
    auto * qctx = reinterpret_cast<QlhLlamaContext *>(ptr);
    if (qctx == nullptr) {
        throw_java(env, "Model is not loaded");
        return nullptr;
    }

    const QlhGenerationStats & stats = qctx->last_stats;
    jmethodID put_method = nullptr;
    jobject map = new_string_map(env, &put_method);
    map_put(env, map, put_method, "prompt_tokens", std::to_string(stats.prompt_tokens));
    map_put(env, map, put_method, "generated_tokens", std::to_string(stats.generated_tokens));
    map_put(env, map, put_method, "total_tokens", std::to_string(stats.total_tokens));
    map_put(env, map, put_method, "elapsed_seconds", std::to_string(stats.elapsed_seconds));
    map_put(env, map, put_method, "tokens_per_second", std::to_string(stats.tokens_per_second));
    map_put(env, map, put_method, "stop_reason", stats.stop_reason);
    map_put(env, map, put_method, "max_tokens", std::to_string(qctx->n_ctx));
    map_put(env, map, put_method, "utilization", qctx->n_ctx > 0
        ? std::to_string(static_cast<double>(stats.total_tokens) / static_cast<double>(qctx->n_ctx))
        : "0");
    map_put(env, map, put_method, "estimated_memory_mb", std::to_string(estimate_kv_memory_mb(qctx, stats.total_tokens)));
    return map;
}

// ============================================================================
// Koakuma Android RPC worker（票 11 / P0-3）
//
// 目的：让 Android App 在**自己进程内**跑 ggml RPC server，把本机算力暴露给
// 集群（对应 PC 侧 scripts/llama_rpc_sim.py 的 `ggml-rpc-server` 角色），
// 而不需要 fork 子进程 —— Android 上无法像桌面那样起独立可执行文件。
//
// 上游约束（llama.cpp 47e1de77a）：`ggml_backend_rpc_start_server` 是**阻塞**
// 函数且**没有对应的优雅停止 API**（`tools/rpc/rpc-server.cpp` 靠进程退出结束）。
// 因此这里：
//   * Start 只负责在独立线程里进入 server 循环；
//   * Stop 只能做「逻辑停止」（清运行标志），无法中断已阻塞的 accept 循环 ——
//     这一点通过 Status 的 `stop_supported=false` 显式告知上层，不做虚假承诺。
// ============================================================================

#include "ggml-rpc.h"

#include <atomic>
#include <thread>

struct QlhRpcWorker {
    std::thread th;
    std::atomic<bool> running{false};
    std::string endpoint;
    std::string cache_dir;
    int n_threads = 0;
    int n_devices = 0;
    std::string last_error;
    std::string devices_desc;
    long long started_at_ms = 0;
    long long stopped_at_ms = 0;
};

static QlhRpcWorker g_rpc_worker;
static std::mutex g_rpc_mutex;

static long long qlh_now_ms() {
    using namespace std::chrono;
    return duration_cast<milliseconds>(steady_clock::now().time_since_epoch()).count();
}

static std::string qlh_jstring_to_utf8(JNIEnv * env, jstring value) {
    if (value == nullptr) {
        return std::string();
    }
    const char * chars = env->GetStringUTFChars(value, nullptr);
    if (chars == nullptr) {
        return std::string();
    }
    std::string out(chars);
    env->ReleaseStringUTFChars(value, chars);
    return out;
}

// 候选设备：本机所有非 RPC 后端设备（避免把 RPC 设备自身再暴露出去形成自环）
static std::vector<ggml_backend_dev_t> qlh_rpc_candidate_devices(std::string * desc_out) {
    std::vector<ggml_backend_dev_t> out;
    std::ostringstream desc;
    const size_t n = ggml_backend_dev_count();
    for (size_t i = 0; i < n; ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (dev == nullptr) {
            continue;
        }
        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
        const char * reg_name = (reg != nullptr) ? ggml_backend_reg_name(reg) : nullptr;
        if (reg_name != nullptr && std::string(reg_name) == "RPC") {
            continue;
        }
        out.push_back(dev);
        if (desc.tellp() > 0) {
            desc << ", ";
        }
        desc << ggml_backend_dev_name(dev);
    }
    if (desc_out != nullptr) {
        *desc_out = desc.str();
    }
    return out;
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_qlh_inference_service_LocalInferenceEngine_nativeRpcWorkerStart(
    JNIEnv * env, jobject /* thiz */, jstring j_endpoint, jint n_threads, jstring j_cache_dir
) {
    if (j_endpoint == nullptr) {
        throw_java(env, "endpoint must not be null");
        return JNI_FALSE;
    }
    const std::string endpoint = qlh_jstring_to_utf8(env, j_endpoint);
    const std::string cache_dir = qlh_jstring_to_utf8(env, j_cache_dir);

    std::lock_guard<std::mutex> lock(g_rpc_mutex);
    if (g_rpc_worker.running.load()) {
        throw_java(env, "RPC worker is already running");
        return JNI_FALSE;
    }
    if (g_rpc_worker.th.joinable()) {
        g_rpc_worker.th.join();       // 回收上一次已退出的线程
    }

    ensure_backend_initialized();

    std::string devices_desc;
    std::vector<ggml_backend_dev_t> devices = qlh_rpc_candidate_devices(&devices_desc);
    if (devices.empty()) {
        throw_java(env, "no local device available for the RPC worker");
        return JNI_FALSE;
    }

    g_rpc_worker.endpoint = endpoint;
    g_rpc_worker.cache_dir = cache_dir;
    g_rpc_worker.n_threads = (n_threads > 0) ? static_cast<int>(n_threads) : available_threads();
    g_rpc_worker.n_devices = static_cast<int>(devices.size());
    g_rpc_worker.devices_desc = devices_desc;
    g_rpc_worker.last_error.clear();
    g_rpc_worker.started_at_ms = qlh_now_ms();
    g_rpc_worker.stopped_at_ms = 0;
    g_rpc_worker.running.store(true);

    const int worker_threads = g_rpc_worker.n_threads;
    const char * cache_ptr = g_rpc_worker.cache_dir.empty() ? nullptr : g_rpc_worker.cache_dir.c_str();

    QLH_LOGI("RPC worker starting: endpoint=%s threads=%d devices=%s",
             endpoint.c_str(), worker_threads, devices_desc.c_str());

    // mutable：devices.data() 需要非 const 指针（lambda 默认按 const 捕获副本）
    g_rpc_worker.th = std::thread([endpoint, cache_dir, devices, worker_threads]() mutable {
        const char * cache = cache_dir.empty() ? nullptr : cache_dir.c_str();
        ggml_backend_rpc_start_server(
            endpoint.c_str(), cache, static_cast<size_t>(worker_threads),
            devices.size(), devices.data());
        // start_server 返回即表示 server 循环结束
        {
            std::lock_guard<std::mutex> guard(g_rpc_mutex);
            g_rpc_worker.running.store(false);
            g_rpc_worker.stopped_at_ms = qlh_now_ms();
        }
        QLH_LOGI("RPC worker stopped: endpoint=%s", endpoint.c_str());
    });
    (void) cache_ptr;

    return JNI_TRUE;
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_qlh_inference_service_LocalInferenceEngine_nativeRpcWorkerStop(
    JNIEnv * env, jobject /* thiz */
) {
    (void) env;
    std::lock_guard<std::mutex> lock(g_rpc_mutex);
    if (!g_rpc_worker.running.load() && !g_rpc_worker.th.joinable()) {
        return JNI_FALSE;
    }
    // 逻辑停止：仅清标志。上游没有优雅停止 API，阻塞中的 accept 循环无法从此处中断，
    // 真正的结束依赖 server 自身返回或进程退出（见 Status 的 stop_supported=false）。
    g_rpc_worker.running.store(false);
    g_rpc_worker.stopped_at_ms = qlh_now_ms();
    QLH_LOGW("RPC worker logical stop requested; upstream has no graceful shutdown API");
    return JNI_TRUE;
}

extern "C" JNIEXPORT jobject JNICALL
Java_com_qlh_inference_service_LocalInferenceEngine_nativeRpcWorkerStatus(
    JNIEnv * env, jobject /* thiz */
) {
    std::lock_guard<std::mutex> lock(g_rpc_mutex);
    jmethodID put_method = nullptr;
    jobject map = new_string_map(env, &put_method);
    map_put(env, map, put_method, "running", g_rpc_worker.running.load() ? "true" : "false");
    map_put(env, map, put_method, "endpoint", g_rpc_worker.endpoint);
    map_put(env, map, put_method, "cache_dir", g_rpc_worker.cache_dir);
    map_put(env, map, put_method, "n_threads", std::to_string(g_rpc_worker.n_threads));
    map_put(env, map, put_method, "n_devices", std::to_string(g_rpc_worker.n_devices));
    map_put(env, map, put_method, "devices", g_rpc_worker.devices_desc);
    map_put(env, map, put_method, "started_at_ms", std::to_string(g_rpc_worker.started_at_ms));
    map_put(env, map, put_method, "stopped_at_ms", std::to_string(g_rpc_worker.stopped_at_ms));
    map_put(env, map, put_method, "last_error", g_rpc_worker.last_error);
    map_put(env, map, put_method, "proto_version", std::to_string(RPC_PROTO_MAJOR_VERSION) + "." +
            std::to_string(RPC_PROTO_MINOR_VERSION) + "." + std::to_string(RPC_PROTO_PATCH_VERSION));
    // 如实上报能力边界：上游无优雅停止，Stop 只能做逻辑标记
    map_put(env, map, put_method, "stop_supported", "false");
    return map;
}

// 探活：把 endpoint 当远端 RPC 设备访问，读取其显存/内存信息以确认可达
extern "C" JNIEXPORT jobject JNICALL
Java_com_qlh_inference_service_LocalInferenceEngine_nativeRpcProbe(
    JNIEnv * env, jobject /* thiz */, jstring j_endpoint
) {
    const std::string endpoint = qlh_jstring_to_utf8(env, j_endpoint);
    jmethodID put_method = nullptr;
    jobject map = new_string_map(env, &put_method);
    map_put(env, map, put_method, "endpoint", endpoint);
    if (endpoint.empty()) {
        map_put(env, map, put_method, "reachable", "false");
        map_put(env, map, put_method, "error", "endpoint is empty");
        return map;
    }

    ensure_backend_initialized();

    size_t free_bytes = 0;
    size_t total_bytes = 0;
    bool ok = false;
    try {
        ggml_backend_rpc_get_device_memory(endpoint.c_str(), 0, &free_bytes, &total_bytes);
        ok = (total_bytes > 0);
    } catch (...) {
        ok = false;
    }
    map_put(env, map, put_method, "reachable", ok ? "true" : "false");
    map_put(env, map, put_method, "free_bytes", std::to_string(free_bytes));
    map_put(env, map, put_method, "total_bytes", std::to_string(total_bytes));
    if (!ok) {
        map_put(env, map, put_method, "error", "no response from RPC endpoint");
    }
    return map;
}

// ============================================================================
// 层段（layer_forward）—— 用 llama.cpp 承「一段层」，参与主仓层流水线
// ============================================================================
//
// ## 为什么需要
// Android 要用 llama.cpp 参与**层流水线**：加载**裁层 GGUF**（只含尾段层），
// 接收上游的 hidden 并注入（`llama_batch.embd`），只算自己负责的层区间。
// —— 这条路径就是主仓 `CORE-RELAY-01` / `L→L Relay` 的 L 侧一边。
// 主仓任务协议 v3 已定义 `layer_forward` stage 与 `layer_range`/`handoff_at`/
// `hidden_sha256`/`hidden_spec` 字段（见 `src/task_worker_protocol.py`）。
//
// ## 实现来源
// 移植自已跑通的实验实现
//   `build/cross-framework-layer-poc/llama.cpp/tools/hs-extract-batch/relay-gen-dl.cpp`
// （D→L 接力，逐 token 一致）。**三个已踩过的坑必须保留**：
//   ① `n_batch`/`n_ubatch` 必须 ≥ 最长序列，否则 `llama_decode` 越界崩溃
//      （实测 rc=0xC0000409）；本文件在 nativeLoadModel 侧已按 n_ctx 设置。
//   ② **pos 数组必须按 `n_pos_per_embd * n_tokens` 提供** —— llama.cpp 的
//      `ubatch_add` 在 M-RoPE + `embd` 批次会按该长度读 `pos`，而
//      `llama_batch_init` 只分配 `n_tokens` 个。这是上游 issue #28963 / #28902
//      的**调用方修法**（上游 #28910 只修 pos==NULL 路径，不覆盖我们的显式 pos）。
//   ③ `embd` 注入路径**进程内 bitwise 非确定** ⇒ 验收**只认 per-token argmax**，
//      不得要求 bitwise 一致（主仓 `RELAY_ACCEPTANCE = "per_token_argmax"`）。

//: M-RoPE 的位置段数（**不要硬编码**）：由模型自身的 rope 类型决定 —— MROPE / IMROPE ⇒ 4，
//: 其余（NORM / NEOX…）⇒ 1。Qwen3.5 是 IMROPE（`rope.dimension_sections` 4 段）；
//: 对 `n_pos_per_embd == 1` 的模型，planar 重建没有意义（`llama_batch_init` 分配的
//: `n_tokens` 个 pos 就是正确布局）。与主仓 `qlh_keep_head.c` / `llama_engine.py` 同一判定。
static int qlh_n_pos_per_embd(const llama_model * model) {
    const enum llama_rope_type rope = (model != nullptr) ? llama_model_rope_type(model)
                                                         : LLAMA_ROPE_TYPE_NONE;
    return (rope == LLAMA_ROPE_TYPE_MROPE || rope == LLAMA_ROPE_TYPE_IMROPE) ? 4 : 1;
}

// ★ M-RoPE 的 **embd 注入通道**位置布局（与主仓 shim 同一修法）：
//   llama.cpp 在 `batch.token == nullptr`（embd 注入）时按 planar 读 `batch.pos[j*n_tokens + i]`
//   （`llama-batch.cpp::ubatch_add`：`src_off = batch.token ? 0 : j*batch.n_tokens`），
//   而 `llama_batch_init()` 只分配 `n_tokens` 个 pos ⇒ 必须重建为 `n_pos * n_tokens`。
//   ⚠️ 源值取**已经填好的 `batch.pos[i]`**（来自显式 `positions` 或 `pos_base + i`）——
//   不要改用 `pos_base + i` 重填，否则 `...KeepHeadSeq` 传来的显式多序列位置会被丢弃
//   （旧实现的实错）。`n_pos <= 1` 时无需重建。
static bool qlh_rebuild_pos_planar(llama_batch & batch, int n_tokens, int n_pos) {
    if (n_pos <= 1) {
        return true;
    }
    const size_t pos_len = (size_t) n_pos * (size_t) n_tokens;
    auto * pos_ext = static_cast<llama_pos *>(malloc(pos_len * sizeof(llama_pos)));
    if (pos_ext == nullptr) {
        return false;
    }
    for (int j = 0; j < n_pos; ++j) {
        for (int i = 0; i < n_tokens; ++i) {
            pos_ext[(size_t) j * (size_t) n_tokens + (size_t) i] = batch.pos[i];
        }
    }
    free(batch.pos);
    batch.pos = pos_ext;
    return true;
}

// 把 f32 hidden 注入 embd 批次并从本节点层段继续前向；返回末位置 argmax token。
//
// out_hidden 非空时，额外把「末位置的输出 hidden」拷回它（供中间层段节点继续接力）。
static jint qlh_layer_forward_impl(
    JNIEnv * env,
    jlong model_ptr,
    jfloatArray j_hidden,
    jint n_tokens,
    jint pos_base,
    jfloatArray out_hidden
) {
    if (model_ptr == 0 || j_hidden == nullptr || n_tokens <= 0) {
        return -1;
    }
    auto * qctx = reinterpret_cast<QlhLlamaContext *>(model_ptr);
    if (qctx->ctx == nullptr || qctx->model == nullptr) {
        return -1;
    }

    const int n_embd_inp = llama_model_n_embd_inp(qctx->model);
    const jsize arr_len = env->GetArrayLength(j_hidden);
    if (arr_len != (jsize) n_tokens * n_embd_inp) {
        // 形状不符：明确失败，不做静默截断/补齐
        return -1;
    }

    std::vector<float> hidden((size_t) arr_len);
    env->GetFloatArrayRegion(j_hidden, 0, arr_len, hidden.data());
    if (env->ExceptionCheck()) {
        return -1;
    }

    // 每步从零重算（与 relay-gen-dl 默认模式一致）：避免跨步 KV 复用带来的
    // 序列号/位置耦合，先把正确性钉死；增量模式（keep-kv）是后续优化位。
    llama_memory_t mem = llama_get_memory(qctx->ctx);
    if (mem) {
        llama_memory_clear(mem, true);
    }

    llama_batch batch = llama_batch_init(n_tokens, n_embd_inp, /*n_seq_max=*/1);
    if (batch.embd == nullptr) {
        return -1;
    }
    for (int i = 0; i < n_tokens; ++i) {
        batch.pos[i] = pos_base + i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = (i + 1 == n_tokens) ? 1 : 0;
    }
    batch.n_tokens = n_tokens;
    std::memcpy(batch.embd, hidden.data(), hidden.size() * sizeof(float));

    // ★ 坑②：pos 按 n_pos_per_embd * n_tokens 重建（上游 #28963 调用方修法）。
    if (!qlh_rebuild_pos_planar(batch, n_tokens, qlh_n_pos_per_embd(qctx->model))) {
        llama_batch_free(batch);
        return -1;
    }

    jint result = -1;
    const int rc = llama_decode(qctx->ctx, batch);
    if (rc == 0) {
        const float * logits = llama_get_logits_ith(qctx->ctx, n_tokens - 1);
        if (logits != nullptr) {
            const int n_vocab = llama_vocab_n_tokens(qctx->vocab);
            int best = 0;
            for (int i = 1; i < n_vocab; ++i) {
                if (logits[i] > logits[best]) {
                    best = i;
                }
            }
            result = (jint) best;
        }
        if (out_hidden != nullptr) {
            // ★ 2026-09-20：中间层段必须由 load 时开启 `extract_hidden_states`，
            //   否则 `llama_get_embeddings_ith` 无值。此处**如实处理**：
            //   未开启时把结果标记为「不能承中间段」，由调用方 fail-closed，
            //   而不是悄悄回传一个全零 hidden 让下游算出错误结果。
            if (!qctx->extract_hidden) {
                llama_batch_free(batch);
                return -2;  // -2 = 需要 extract_hidden_states（load 时开启）
            }
            const float * hidden_out = llama_get_embeddings_ith(qctx->ctx, n_tokens - 1);
            if (hidden_out == nullptr) {
                llama_batch_free(batch);
                return -1;
            }
            if (env->GetArrayLength(out_hidden) != (jsize) n_embd_inp) {
                llama_batch_free(batch);
                return -1;
            }
            env->SetFloatArrayRegion(out_hidden, 0, (jsize) n_embd_inp, hidden_out);
            if (env->ExceptionCheck()) {
                llama_batch_free(batch);
                return -1;
            }
        }
    }

    // batch.pos 已被我们替换为 malloc 的 pos_ext；llama_batch_free 会 free() 它，
    // 与 llama_batch_init 的分配方式一致（同为 malloc），故这里统一走 free 路径。
    llama_batch_free(batch);
    return result;
}

// 层段前向：返回末位置 argmax token（本节点是**末段**时用）。
extern "C" JNIEXPORT jint JNICALL
Java_com_qlh_inference_service_LocalInferenceEngine_nativeLayerForwardToken(
    JNIEnv * env, jobject /* thiz */, jlong model_ptr,
    jfloatArray j_hidden, jint n_tokens, jint pos_base
) {
    return qlh_layer_forward_impl(env, model_ptr, j_hidden, n_tokens, pos_base, nullptr);
}

// 层段前向（中间段）：除 argmax 外，把末位置输出 hidden 拷回 out_hidden，
// 供上层继续交给下一段。形状不符时返回 -1。
extern "C" JNIEXPORT jint JNICALL
Java_com_qlh_inference_service_LocalInferenceEngine_nativeLayerForwardHidden(
    JNIEnv * env, jobject /* thiz */, jlong model_ptr,
    jfloatArray j_hidden, jint n_tokens, jint pos_base, jfloatArray out_hidden
) {
    return qlh_layer_forward_impl(env, model_ptr, j_hidden, n_tokens, pos_base, out_hidden);
}

// ★ 2026-09-23：中间段的「层输出（`output_norm` **之前**）」改用 `layer_inp` 的
//   `lid == n_layer` 槽位（与主仓 `scripts/model_tools/keep_head_shim/qlh_keep_head.c` 同一通道）。
//
//   为什么不再用 nextn：各架构把 `t_h_nextn` 挂在**不同位置** —— qwen2 在 `output_norm` 之前
//   （QLH 2026-09-20 补丁），而 **qwen35 在之后**（那里的消费方是 MTP head）⇒ 同一个 nextn 通道
//   在 Qwen3.5 上会多一次 RMSNorm，接力首步即分叉（主仓 9B 实测 1/32 → 修后 32/32）。
//   `layer_inp` 的 `n_layer` 槽位在语义上**只会**是"末层输出"，与架构无关：槽位由
//   `src/llama-context.cpp` 多分配一个、由各架构 graph 在 `output_norm` 之前登记
//   （llama / qwen2 / qwen3 / qwen35 / qwen3moe / qwen35moe / qwen3next / gemma4 / openai-moe）。
//   ⚠️ **未登记该槽位的架构会在 decode 时 GGML_ABORT**（不是可回退的错误码）⇒ 接入新架构前
//   必须先确认它登记了该槽位。
static void qlh_layer_out_set(llama_context * ctx, const llama_model * model, bool enable) {
    if (ctx == nullptr || model == nullptr) {
        return;
    }
    llama_set_embeddings_layer_inp(ctx, (uint32_t) llama_model_n_layer(model), enable);
}

// 取「末位 token 的层输出（`output_norm` 之前）」——`layer_inp` 通道给的是**稠密**
// `[n_tokens, n_embd]`，故按行偏移取末位。返回 nullptr ⇒ 通道不可用（调用方 fail-closed）。
static const float * qlh_layer_out_get(llama_context * ctx, const llama_model * model,
                                       int n_tokens) {
    if (ctx == nullptr || model == nullptr || n_tokens <= 0) {
        return nullptr;
    }
    const float * base = llama_get_embeddings_layer_inp(ctx,
            (uint32_t) llama_model_n_layer(model));
    if (base == nullptr) {
        return nullptr;
    }
    return base + (size_t) (n_tokens - 1) * (size_t) llama_model_n_embd_inp(model);
}

// ---------------------------------------------------------------------------
// ★ 2026-09-21：keep-head 中间段 —— 与主仓 `scripts/model_tools/keep_head_shim/qlh_keep_head.c`
//   **同语义**（任何一侧改动必须同步，判据同为 per-token argmax）。
//
//   问题：上面的 `nativeLayerForwardHidden` 用 `llama_get_embeddings_ith` 取中间段输出，
//   而该通道返回的是 **`output_norm(H)`** —— 比层接力上游/中间段所需的 hidden 多一次
//   归一化（主仓实测：与 `RMSNorm(H)*model.norm.weight` 的 rel_err=0.0018 / cos=0.999998，
//   同通道的端到端对照 first_mismatch=2）。拿它当中间段输出，下游会收到被多归一化一次
//   的激活 ⇒ 三段链路必然分叉。
//
//   解：走补丁导出的 **`layer_inp` 的 `lid == n_layer` 槽位**（= 第 n_layer 层的输入 =
//   末层输出，`output_norm` **之前**）—— 见上方 `qlh_layer_out_set/get`。**不再用 nextn 通道**：
//   各架构 `t_h_nextn` 的挂点不同（qwen2 在 norm 之前，qwen35 在**之后** —— 后者是 MTP head 的
//   消费方），同一个 nextn 通道在 Qwen3.5 上会多一次 RMSNorm，接力首步即分叉。
//
//   ⚠️ 前提：批次的**每个 token** 都标记输出（`batch.logits[i] = 1`）—— 旧 nextn 通道只对
//      「有输出的行」计算末层 hidden，只标末位却按 n_tokens 读会撞 GGML_ASSERT
//      "tensor read out of bounds"（主仓 shim 实测踩过）。走 layer_inp 后这条不再是硬约束，
//      保持全开可以不必依赖具体的 sched 行为。
//
//   返回：末位置 argmax（≥0）；-1 形状/参数错；-3 = **层输出通道不可用**（该架构没登记
//   `t_layer_inp[n_layer]` 槽位）—— 调用方必须 fail-closed，不得退回 embeddings 通道。
static jint qlh_layer_forward_keep_head_impl(
    JNIEnv * env,
    jlong model_ptr,
    jfloatArray j_hidden,
    jint n_tokens,
    jint pos_base,
    jfloatArray out_hidden,
    // ★ P3 多序列：显式 `n_seq_id / seq_ids / positions`（可为 nullptr ⇒ 单序列旧行为）
    const jint * n_seq_id,
    const jint * seq_ids,
    const jint * positions
) {
    if (model_ptr == 0 || j_hidden == nullptr || out_hidden == nullptr || n_tokens <= 0) {
        return -1;
    }
    auto * qctx = reinterpret_cast<QlhLlamaContext *>(model_ptr);
    if (qctx->ctx == nullptr || qctx->model == nullptr) {
        return -1;
    }

    const int n_embd_inp = llama_model_n_embd_inp(qctx->model);
    const jsize arr_len = env->GetArrayLength(j_hidden);
    if (arr_len != (jsize) n_tokens * n_embd_inp) {
        return -1;
    }
    if (env->GetArrayLength(out_hidden) != (jsize) n_embd_inp) {
        return -1;
    }

    std::vector<float> hidden((size_t) arr_len);
    env->GetFloatArrayRegion(j_hidden, 0, arr_len, hidden.data());
    if (env->ExceptionCheck()) {
        return -1;
    }

    // 与 Hidden 版一致：每步从零重算，先把正确性钉死（增量 KV 是后续优化位）。
    llama_memory_t mem = llama_get_memory(qctx->ctx);
    if (mem) {
        llama_memory_clear(mem, true);
    }

    // ★ 开启层输出导出（`layer_inp` 的 `lid == n_layer` 槽位；稠密 [n_tokens, n_embd]）
    qlh_layer_out_set(qctx->ctx, qctx->model, true);

    llama_batch batch = llama_batch_init(n_tokens, n_embd_inp, /*n_seq_max=*/1);
    if (batch.embd == nullptr) {
        qlh_layer_out_set(qctx->ctx, qctx->model, false);
        return -1;
    }
    for (int i = 0; i < n_tokens; ++i) {
        // ★ P3 多序列：给了显式数组就逐 token 绑定（多序列交错推进时不依赖隐式位置递增）
        batch.pos[i]       = (positions != nullptr) ? (llama_pos) positions[i]
                                                    : (llama_pos) (pos_base + i);
        batch.n_seq_id[i]  = (n_seq_id != nullptr && n_seq_id[i] > 0) ? n_seq_id[i] : 1;
        batch.seq_id[i][0] = (seq_ids != nullptr) ? seq_ids[i] : 0;
        batch.logits[i]    = 1;   // ★ 每个 token 都标（见上文断言说明）
    }
    batch.n_tokens = n_tokens;
    std::memcpy(batch.embd, hidden.data(), hidden.size() * sizeof(float));

    // pos 按 n_pos_per_embd * n_tokens 重建（与 Hidden 版同一修法）；源值取已填好的
    // `batch.pos`（保留显式 `positions`，不要用 `pos_base + i` 覆盖）。
    if (!qlh_rebuild_pos_planar(batch, n_tokens, qlh_n_pos_per_embd(qctx->model))) {
        llama_batch_free(batch);
        qlh_layer_out_set(qctx->ctx, qctx->model, false);
        return -1;
    }

    jint result = -1;
    const int rc = llama_decode(qctx->ctx, batch);
    if (rc == 0) {
        const float * logits = llama_get_logits_ith(qctx->ctx, n_tokens - 1);
        if (logits != nullptr) {
            const int n_vocab = llama_vocab_n_tokens(qctx->vocab);
            int best = 0;
            for (int i = 1; i < n_vocab; ++i) {
                if (logits[i] > logits[best]) {
                    best = i;
                }
            }
            result = (jint) best;
        }
        const float * hidden_out = qlh_layer_out_get(qctx->ctx, qctx->model, n_tokens);
        if (hidden_out == nullptr) {
            llama_batch_free(batch);
            qlh_layer_out_set(qctx->ctx, qctx->model, false);
            return -3;  // 层输出通道不可用 ⇒ 调用方 fail-closed
        }
        env->SetFloatArrayRegion(out_hidden, 0, (jsize) n_embd_inp, hidden_out);
        if (env->ExceptionCheck()) {
            llama_batch_free(batch);
            qlh_layer_out_set(qctx->ctx, qctx->model, false);
            return -1;
        }
    }

    llama_batch_free(batch);
    // 复位：层输出导出只服务本次中间段调用，不污染后续普通推理
    //（该槽位会占 n_embd * n_batch 的缓冲，见 llama-context.cpp 的 output_reserve）。
    qlh_layer_out_set(qctx->ctx, qctx->model, false);
    return result;
}

// 层段前向（中间段，keep-head 语义）：吐**末层输出（output_norm 之前）**。
extern "C" JNIEXPORT jint JNICALL
Java_com_qlh_inference_service_LocalInferenceEngine_nativeLayerForwardHiddenKeepHead(
    JNIEnv * env, jobject /* thiz */, jlong model_ptr,
    jfloatArray j_hidden, jint n_tokens, jint pos_base, jfloatArray out_hidden
) {
    return qlh_layer_forward_keep_head_impl(env, model_ptr, j_hidden, n_tokens, pos_base,
                                            out_hidden, nullptr, nullptr, nullptr);
}

// ★ P3：层段前向（中间段，keep-head 语义，**多序列**）—— 显式 seq_ids / positions。
//
// 与 `...HiddenKeepHead` 的唯一区别是序列绑定方式：这里逐 token 给出
// `n_seq_id` / `seq_ids` / `positions`（三者长度都必须等于 nTokens，可为 null 表示不指定），
// 与主仓 `LlamaCppEngine.forward_layers_from_hidden(seq_ids=..., positions=...)` 同一契约。
// 多序列交错推进时必须用这个入口 —— 否则 llama.cpp 会按隐式位置递增报
// "tokens ... have inconsistent sequence positions"。
//
// 返回同 `...HiddenKeepHead`：末位 argmax（>=0）/ -1 形状错 / -3 层输出通道不可用。
extern "C" JNIEXPORT jint JNICALL
Java_com_qlh_inference_service_LocalInferenceEngine_nativeLayerForwardHiddenKeepHeadSeq(
    JNIEnv * env, jobject /* thiz */, jlong model_ptr,
    jfloatArray j_hidden, jint n_tokens, jint pos_base,
    jintArray j_n_seq_id, jintArray j_seq_ids, jintArray j_positions,
    jfloatArray out_hidden
) {
    if (n_tokens <= 0) {
        return -1;
    }
    std::vector<jint> n_seq_id;
    std::vector<jint> seq_ids;
    std::vector<jint> positions;
    auto read_optional = [&](jintArray source, std::vector<jint> & target) -> bool {
        if (source == nullptr) {
            return true;
        }
        if (env->GetArrayLength(source) != (jsize) n_tokens) {
            return false;   // 形状不符：明确失败，不做静默截断/补齐
        }
        target.resize((size_t) n_tokens);
        env->GetIntArrayRegion(source, 0, (jsize) n_tokens, target.data());
        return env->ExceptionCheck() == JNI_FALSE;
    };
    if (!read_optional(j_n_seq_id, n_seq_id)
        || !read_optional(j_seq_ids, seq_ids)
        || !read_optional(j_positions, positions)) {
        return -1;
    }
    return qlh_layer_forward_keep_head_impl(
        env, model_ptr, j_hidden, n_tokens, pos_base, out_hidden,
        n_seq_id.empty() ? nullptr : n_seq_id.data(),
        seq_ids.empty() ? nullptr : seq_ids.data(),
        positions.empty() ? nullptr : positions.data());
}

// 层段能力探测：上报本节点能否承层段、hidden 宽度、以及「层段运行必需的批量下限」。
// 与 `AndroidWorkerCapabilities` 的「能力探测是单一来源」约定一致 —— 上层据此决定
// 是否把本节点纳入层流水线，而不是靠 `if engine_type` 猜。
extern "C" JNIEXPORT jobject JNICALL
Java_com_qlh_inference_service_LocalInferenceEngine_nativeLayerForwardInfo(
    JNIEnv * env, jobject /* thiz */, jlong model_ptr
) {
    jmethodID put_method = nullptr;
    jobject map = new_string_map(env, &put_method);
    const bool usable = (model_ptr != 0);
    map_put(env, map, put_method, "layer_forward_supported", usable ? "true" : "false");
    if (!usable) {
        map_put(env, map, put_method, "reason", "model not loaded");
        return map;
    }
    auto * qctx = reinterpret_cast<QlhLlamaContext *>(model_ptr);
    if (qctx->model == nullptr) {
        map_put(env, map, put_method, "layer_forward_supported", "false");
        map_put(env, map, put_method, "reason", "model handle is empty");
        return map;
    }
    const int n_embd_inp = llama_model_n_embd_inp(qctx->model);
    const int n_layer = (int) llama_model_n_layer(qctx->model);
    map_put(env, map, put_method, "n_embd", std::to_string(n_embd_inp));
    map_put(env, map, put_method, "n_layer", std::to_string(n_layer));
    // 裁层 GGUF 天然只含尾段层 ⇒ `n_layer` 即本节点实际负责的层数；
    // 层区间的**源模型**编号由主仓按 `layer_range` 下发，不在此臆测。
    map_put(env, map, put_method, "hidden_dtype", "float32");
    map_put(env, map, put_method, "n_pos_per_embd", std::to_string(qlh_n_pos_per_embd(qctx->model)));
    map_put(env, map, put_method, "acceptance", "per_token_argmax");
    // ★ 2026-09-20：能否承**中间段**取决于 load 时是否开了隐藏态导出。
    //   末段（只要 argmax）不受该开关影响 ⇒ 两个能力分别上报。
    map_put(env, map, put_method, "can_tail", "true");
    map_put(env, map, put_method, "extract_hidden", qctx->extract_hidden ? "true" : "false");
    map_put(env, map, put_method, "can_middle", qctx->extract_hidden ? "true" : "false");
    if (!qctx->extract_hidden) {
        map_put(env, map, put_method, "middle_reason",
                "extract_hidden_states not enabled at load time");
    }
    // ★ 2026-09-23：中间段的**正确**通道。`extract_hidden`（embeddings 通道）给的是
    //   `output_norm(H)`，比层接力所需的 hidden 多一次归一化，两者不可混用；
    //   keep-head 走 **`layer_inp` 的 `lid == n_layer` 槽位**（末层输出，`output_norm` 之前）
    //   ⇒ 与主仓 D→L / L→L / 三段语义一致，且不随各架构 `t_h_nextn` 的挂点变化。
    //   能力单独上报：调用方据此选择通道，而不是靠 extract_hidden 猜。
    map_put(env, map, put_method, "keep_head_middle", "true");
    map_put(env, map, put_method, "middle_channel", "keep_head_layer_out");
    return map;
}
