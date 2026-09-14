/**
 * QLH Lite stub — 薄客户端不需要 llama.cpp 本地推理。
 * 提供了一个空实现的 JNI 库，避免 System.loadLibrary("qlh_llama_jni") 崩溃。
 */

#include <jni.h>
#include <android/log.h>

#define TAG "qlh_llama_jni(stub)"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)

JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM *vm, void *reserved) {
    LOGI("Lite stub loaded — 本地推理不可用，请使用 thin 模式");
    return JNI_VERSION_1_6;
}
