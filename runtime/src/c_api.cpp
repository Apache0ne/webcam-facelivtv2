#include "facelivt/runtime.hpp"
#include <algorithm>
#include <cstring>
#include <filesystem>

#ifdef _WIN32
#define FLVT_API extern "C" __declspec(dllexport)
#else
#define FLVT_API extern "C" __attribute__((visibility("default")))
#endif

namespace {
void error_text(char* out, size_t capacity, const char* message) {
    if (!out || !capacity) return;
    const size_t n = std::min(capacity - 1, std::strlen(message));
    std::memcpy(out, message, n);
    out[n] = '\0';
}
}

// Opaque handles must be created, used, and destroyed on the same thread.
// Buffers belong to the caller; no allocation crosses the DLL boundary.
FLVT_API void* flvt_create(const char* model, const char* manifest, int max_batch,
                          char* error, size_t capacity) noexcept {
    error_text(error, capacity, "");
    try {
        if (!model || !manifest || max_batch < 1 || max_batch > 64)
            throw std::runtime_error("invalid model, manifest, or max_batch (1..64)");
        facelivt::RuntimeOptions options;
        options.max_batch = max_batch;
        return new facelivt::Runtime(std::filesystem::u8path(model),
                                    std::filesystem::u8path(manifest), options);
    } catch (const std::exception& e) { error_text(error, capacity, e.what()); }
      catch (...) { error_text(error, capacity, "unknown runtime creation error"); }
    return nullptr;
}

FLVT_API int flvt_infer(void* handle, const uint8_t* bgr, size_t input_bytes,
                       int batch, float* output, size_t output_count,
                       char* error, size_t capacity) noexcept {
    error_text(error, capacity, "");
    try {
        if (!handle || !bgr || !output || batch < 1 || batch > 64)
            throw std::runtime_error("invalid inference arguments");
        if (input_bytes != size_t(batch) * 112 * 112 * 3 || output_count != size_t(batch) * 512)
            throw std::runtime_error("expected Bx112x112x3 BGR bytes and Bx512 output floats");
        static_cast<facelivt::Runtime*>(handle)->infer_host(bgr, batch, output, true, true);
        return 0;
    } catch (const std::exception& e) { error_text(error, capacity, e.what()); }
      catch (...) { error_text(error, capacity, "unknown inference error"); }
    return 1;
}

FLVT_API void flvt_destroy(void* handle) noexcept {
    delete static_cast<facelivt::Runtime*>(handle);
}

FLVT_API int flvt_infer_device(void* handle, const uint8_t* bgr, size_t input_bytes,
                              int batch, float* output, size_t output_count,
                              char* error, size_t capacity) noexcept {
    error_text(error, capacity, "");
    try {
        if (!handle || !bgr || !output || batch < 1 || batch > 64)
            throw std::runtime_error("invalid device inference arguments");
        if (input_bytes != size_t(batch)*112*112*3 || output_count != size_t(batch)*512)
            throw std::runtime_error("invalid device inference buffer size");
        auto* runtime = static_cast<facelivt::Runtime*>(handle);
        runtime->infer_device(bgr, batch, output, true, true);
        facelivt::cuda_check(cudaStreamSynchronize(runtime->stream()), "finish device inference");
        return 0;
    } catch (const std::exception& e) { error_text(error, capacity, e.what()); }
      catch (...) { error_text(error, capacity, "unknown device inference error"); }
    return 1;
}
