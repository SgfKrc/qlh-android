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
    jint j_n_ctx
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
    model_params.n_gpu_layers = 0;

    llama_model * model = llama_model_load_from_file(model_path.c_str(), model_params);
    if (model == nullptr) {
        QLH_LOGE("llama_model_load_from_file failed: %s", model_path.c_str());
        return 0;
    }

    const int n_threads = available_threads();
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = static_cast<uint32_t>(std::max(512, static_cast<int>(j_n_ctx)));
    ctx_params.n_batch = std::min<uint32_t>(512, ctx_params.n_ctx);
    ctx_params.n_ubatch = std::min<uint32_t>(256, ctx_params.n_batch);
    ctx_params.n_threads = n_threads;
    ctx_params.n_threads_batch = n_threads;
    ctx_params.no_perf = true;

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

    QLH_LOGI("model loaded: ctx=%d threads=%d", qctx->n_ctx, n_threads);
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
    put("backend", "llama.cpp Android CPU");
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
