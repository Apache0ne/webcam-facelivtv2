#pragma once
#include "model.hpp"
#include "kernel_registry.hpp"
#include <cstdint>
#include <filesystem>
#include <map>
#include <tuple>
#include <vector>

namespace facelivt {

struct RuntimeOptions {
    int device=0;
    int max_batch=64;
    bool bgr_input=true;
    bool normalize_output=true;
};

struct Match {
    int best_index=-1;
    float best_score=-1.0f;
    int second_index=-1;
    float second_score=-1.0f;
};

class Runtime {
public:
    Runtime(const std::filesystem::path& model,
            const std::filesystem::path& kernel_manifest,
            RuntimeOptions options={});
    ~Runtime();
    Runtime(const Runtime&)=delete;
    Runtime& operator=(const Runtime&)=delete;

    int max_batch() const { return opt_.max_batch; }
    cudaStream_t stream() const { return stream_; }

    // Host API: input = packed B*112*112*3 bytes; output = B*512 floats.
    void infer_host(const uint8_t* input, int batch, float* output,
                    bool bgr_input, bool normalize);
    void infer_host(const uint8_t* input, int batch, float* output) {
        infer_host(input,batch,output,opt_.bgr_input,opt_.normalize_output);
    }

    // Device API uses stable internal graph buffers and D2D copies.
    void infer_device(const uint8_t* d_input, int batch, float* d_output,
                      bool bgr_input, bool normalize,
                      cudaStream_t input_ready_stream=nullptr);

    // Gallery rows are normalized during upload. `infer_and_score_host` returns
    // batch x gallery_count cosine scores; top-k reduction is a later optimization.
    void set_gallery(const float* embeddings, int count);
    int gallery_count() const { return gallery_count_; }
    void infer_and_score_host(const uint8_t* input, int batch, float* scores,
                              bool bgr_input=true);
    void infer_and_match_host(const uint8_t* input, int batch, Match* matches,
                              bool bgr_input=true);

private:
    using GraphKey=std::tuple<int,bool,bool>;
    struct GraphExec { cudaGraph_t graph=nullptr; cudaGraphExec_t exec=nullptr; };

    void allocate_buffers();
    void free_buffers();
    void enqueue_network(int batch,bool bgr_input,bool normalize,cudaStream_t stream);
    cudaGraphExec_t graph_for(int batch,bool bgr_input,bool normalize);

    void launch_preprocess(int batch,bool bgr,void* out,cudaStream_t stream);
    void launch_stem(const char* alias,int batch,const void* x,const char* wname,const char* bname,void* y,cudaStream_t stream);
    void launch_dw3(int batch,int H,int W,int C,int stride,const void* x,const char* base,void* y,cudaStream_t stream);
    void launch_dw4(int batch,const void* x,const char* base,void* y,cudaStream_t stream);
    void launch_pw(int M,int K,int N,bool gelu,bool residual,
                   const void* x,const char* base,const void* residual_ptr,void* y,cudaStream_t stream);
    void launch_mhla(int batch,int C,int S,const void* x,const char* base,void* y,cudaStream_t stream);
    void launch_embed_out(int batch,bool normalize,const void* x,float* y,cudaStream_t stream);
    void launch_gallery_scores(const float* query,int batch,float* scores,cudaStream_t stream);
    void launch_top2(int batch,cudaStream_t stream);

    RuntimeOptions opt_;
    Model model_;
    KernelRegistry kernels_;
    cudaStream_t stream_=nullptr;

    uint8_t* d_input_u8_=nullptr;
    void* d_a_=nullptr;
    void* d_b_=nullptr;
    void* d_hidden_=nullptr;
    void* d_embed_half_=nullptr;
    float* d_output_=nullptr;
    size_t activation_bytes_=0;

    float* d_gallery_=nullptr;
    float* d_scores_=nullptr;
    int* d_best_idx_=nullptr;
    float* d_best_score_=nullptr;
    int* d_second_idx_=nullptr;
    float* d_second_score_=nullptr;
    int gallery_count_=0;

    uint8_t* h_input_pinned_=nullptr;
    float* h_output_pinned_=nullptr;

    std::map<GraphKey,GraphExec> graphs_;
};
}
