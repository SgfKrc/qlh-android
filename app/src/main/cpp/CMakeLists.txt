cmake_minimum_required(VERSION 3.22.1)

project(qlh_llama_jni LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Lite 版本跳过原生编译 — 仅需 HTTP thin 模式，不需要 llama.cpp
option(QLH_LITE "Skip llama.cpp build for lite variant" OFF)
if(QLH_LITE)
    message(STATUS "QLH Lite — 跳过原生库编译（thin-only 模式）")
    # 构建一个空桩 .so，防止 App 启动时 System.loadLibrary 崩溃
    add_library(qlh_llama_jni SHARED qlh_llama_jni_stub.cpp)
    target_link_libraries(qlh_llama_jni android log)
    return()
endif()

if(DEFINED ANDROID_ABI)
    message(STATUS "QLH Android ABI: ${ANDROID_ABI}")
    if(NOT ANDROID_ABI STREQUAL "arm64-v8a")
        message(FATAL_ERROR "Unsupported ABI for local GGUF inference: ${ANDROID_ABI}")
    endif()
endif()

# ---- llama.cpp submodule 前置检查（Full 构建必需；Lite 已在上面提前返回） ----
set(QLH_LLAMA_CPP_DIR "${CMAKE_CURRENT_LIST_DIR}/llama.cpp")
if(NOT EXISTS "${QLH_LLAMA_CPP_DIR}/CMakeLists.txt")
    message(FATAL_ERROR
        "llama.cpp submodule is not initialized. Run: "
        "git submodule update --init --recursive")
endif()

set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build llama.cpp as static libs" FORCE)
set(LLAMA_BUILD_COMMON OFF CACHE BOOL "Do not build llama common utils" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "Do not build llama tests" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "Do not build llama tools" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "Do not build llama examples" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "Do not build llama server" FORCE)
set(LLAMA_BUILD_APP OFF CACHE BOOL "Do not build llama app" FORCE)
set(LLAMA_BUILD_MTMD ON CACHE BOOL "Build the standalone multimodal bridge" FORCE)
set(MTMD_VIDEO OFF CACHE BOOL "Disable video support in Android bridge" FORCE)
set(GGML_NATIVE OFF CACHE BOOL "Disable host-native flags for Android cross compile" FORCE)
set(GGML_OPENMP OFF CACHE BOOL "Avoid libomp packaging for the first Android build" FORCE)
set(GGML_CPU_KLEIDIAI OFF CACHE BOOL "Avoid build-time KleidiAI download" FORCE)
set(GGML_BACKEND_DL OFF CACHE BOOL "Use statically linked backend" FORCE)

add_subdirectory(llama.cpp build-llama)

add_library(qlh_llama_jni SHARED qlh_llama_jni.cpp)

target_include_directories(qlh_llama_jni PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/llama.cpp/include
    ${CMAKE_CURRENT_LIST_DIR}/llama.cpp/ggml/include
    ${CMAKE_CURRENT_LIST_DIR}/llama.cpp/tools/mtmd
)

target_link_libraries(qlh_llama_jni
    llama
    mtmd
    android
    log
)
