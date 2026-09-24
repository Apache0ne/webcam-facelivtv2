#pragma once

#include <cuda_runtime_api.h>
#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <thread>
#include <vector>

namespace Ort {
class Env;
class Session;
class IoBinding;
class MemoryInfo;
class Value;
}

namespace facelivt {

struct ScrfdFace {
    std::array<float,4> box{};          // x1, y1, x2, y2 in source-frame pixels
    std::array<float,10> landmarks{};   // five (x, y) pairs in source-frame pixels
    float confidence=0;
};

struct alignas(16) ScrfdCandidate {
    float box[4];
    float confidence;
    float landmarks[10];
    float padding[1]{};
};

struct ScrfdOptions {
    int device=0;
    int input_size=640;
    int max_faces=8;
    float score_threshold=0.65f;
    float nms_threshold=0.4f;
    size_t gpu_memory_limit=1024ull*1024ull*1024ull;
};

// SCRFD inference and image preparation stay on CUDA. Only thresholded face
// candidates are copied back because tracking, enrollment and UI need boxes.
class ScrfdDetector {
public:
    explicit ScrfdDetector(const std::filesystem::path& model,
                           ScrfdOptions options={});
    ~ScrfdDetector();
    ScrfdDetector(const ScrfdDetector&)=delete;
    ScrfdDetector& operator=(const ScrfdDetector&)=delete;

    // frame_bgra is a packed uint8 BGRA image already resident on options.device.
    std::vector<ScrfdFace> detect(const uint8_t* frame_bgra,int width,int height);

    // Produce packed BGR uint8 crops in device memory, ready for Runtime::infer_device.
    void align(const uint8_t* frame_bgra,int width,int height,
               std::span<const ScrfdFace> faces,uint8_t* device_bgr_crops);

    cudaStream_t stream() const { return stream_; }
    int input_size() const { return options_.input_size; }

private:
    struct Output {
        std::string name;
        std::vector<int64_t> shape;
        float* device_data=nullptr;
        size_t elements=0;
    };
    struct Head {
        int stride=0;
        int count=0;
        float* scores=nullptr;
        float* boxes=nullptr;
        float* landmarks=nullptr;
    };

    void check_thread() const;
    void initialize_session(const std::filesystem::path& model);
    void allocate_outputs();
    void cleanup() noexcept;

    ScrfdOptions options_;
    std::thread::id owner_thread_{};
    cudaStream_t stream_=nullptr;
    float* device_input_=nullptr;
    int* device_candidate_count_=nullptr;
    ScrfdCandidate* device_candidates_=nullptr;
    float* device_affine_=nullptr;
    ScrfdCandidate* host_candidates_=nullptr;
    int host_candidate_count_=0;
    size_t candidate_capacity_=0;

    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::Session> session_;
    std::unique_ptr<Ort::IoBinding> binding_;
    std::unique_ptr<Ort::MemoryInfo> cuda_memory_;
    std::unique_ptr<Ort::Value> input_value_;
    std::vector<std::unique_ptr<Ort::Value>> output_values_;
    std::vector<Output> outputs_;
    std::array<Head,3> heads_{};
};

} // namespace facelivt
