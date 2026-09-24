#include "facelivt/runtime.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace facelivt {
namespace {
constexpr int kInputH=112, kInputW=112, kInputC=3, kEmbedding=512;
constexpr size_t kMaxActivationElemsPerFace=100352; // max of 56*56*32 and 28*28*128
constexpr size_t kInputBytesPerFace=static_cast<size_t>(kInputH)*kInputW*kInputC;
std::string pw_alias(int K,int N,bool gelu,bool residual){
    std::ostringstream o; o<<"pw_k"<<K<<"_n"<<N<<"_g"<<(gelu?1:0)<<"_r"<<(residual?1:0); return o.str();
}
std::string dw_alias(int H,int C,int stride){
    std::ostringstream o; o<<"dw3_h"<<H<<"_c"<<C<<"_s"<<stride; return o.str();
}
}

Runtime::Runtime(const std::filesystem::path& model,
                 const std::filesystem::path& kernel_manifest,
                 RuntimeOptions options):opt_(options){
    if(opt_.max_batch<1) throw std::runtime_error("max_batch must be >= 1");
    cuda_check(cudaSetDevice(opt_.device),"cudaSetDevice");
    // Force Runtime API context creation before loading Driver API modules.
    cuda_check(cudaFree(nullptr),"initialize CUDA context");
    model_.load(model);
    kernels_.load(kernel_manifest,opt_.device);
    cuda_check(cudaStreamCreateWithFlags(&stream_,cudaStreamNonBlocking),"cudaStreamCreate");
    allocate_buffers();
}

Runtime::~Runtime(){
    for(auto& kv:graphs_){
        if(kv.second.exec) cudaGraphExecDestroy(kv.second.exec);
        if(kv.second.graph) cudaGraphDestroy(kv.second.graph);
    }
    if(stream_) cudaStreamDestroy(stream_);
    free_buffers();
}

void Runtime::allocate_buffers(){
    activation_bytes_=static_cast<size_t>(opt_.max_batch)*kMaxActivationElemsPerFace*sizeof(uint16_t);
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_input_u8_),static_cast<size_t>(opt_.max_batch)*kInputBytesPerFace),"malloc input");
    cuda_check(cudaMalloc(&d_a_,activation_bytes_),"malloc activation A");
    cuda_check(cudaMalloc(&d_b_,activation_bytes_),"malloc activation B");
    cuda_check(cudaMalloc(&d_hidden_,activation_bytes_),"malloc hidden");
    cuda_check(cudaMalloc(&d_embed_half_,static_cast<size_t>(opt_.max_batch)*kEmbedding*sizeof(uint16_t)),"malloc embedding half");
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_output_),static_cast<size_t>(opt_.max_batch)*kEmbedding*sizeof(float)),"malloc output");
    cuda_check(cudaHostAlloc(reinterpret_cast<void**>(&h_input_pinned_),static_cast<size_t>(opt_.max_batch)*kInputBytesPerFace,cudaHostAllocPortable),"alloc pinned input");
    cuda_check(cudaHostAlloc(reinterpret_cast<void**>(&h_output_pinned_),static_cast<size_t>(opt_.max_batch)*kEmbedding*sizeof(float),cudaHostAllocPortable),"alloc pinned output");
}

void Runtime::free_buffers(){
    if(d_second_score_) cudaFree(d_second_score_);
    if(d_second_idx_) cudaFree(d_second_idx_);
    if(d_best_score_) cudaFree(d_best_score_);
    if(d_best_idx_) cudaFree(d_best_idx_);
    if(d_scores_) cudaFree(d_scores_);
    if(d_gallery_) cudaFree(d_gallery_);
    if(d_output_) cudaFree(d_output_);
    if(d_embed_half_) cudaFree(d_embed_half_);
    if(d_hidden_) cudaFree(d_hidden_);
    if(d_b_) cudaFree(d_b_);
    if(d_a_) cudaFree(d_a_);
    if(d_input_u8_) cudaFree(d_input_u8_);
    if(h_output_pinned_) cudaFreeHost(h_output_pinned_);
    if(h_input_pinned_) cudaFreeHost(h_input_pinned_);
    d_second_score_=nullptr; d_second_idx_=nullptr; d_best_score_=nullptr; d_best_idx_=nullptr;
    d_scores_=nullptr; d_gallery_=nullptr; d_output_=nullptr; d_embed_half_=nullptr;
    d_hidden_=d_b_=d_a_=nullptr; d_input_u8_=nullptr; gallery_count_=0;
    h_output_pinned_=nullptr; h_input_pinned_=nullptr;
}

void Runtime::launch_preprocess(int batch,bool bgr,void* out,cudaStream_t stream){
    const int n=batch*kInputH*kInputW*kInputC;
    void* src=d_input_u8_; void* dst=out; int nn=n;
    void* args[]={&src,&dst,&nn};
    kernels_.launch(bgr?"preprocess_swap1":"preprocess_swap0",ceil_div(n,1024),1,1,args,stream);
}

void Runtime::launch_stem(const char* alias,int batch,const void* x,const char* wname,const char* bname,void* y,cudaStream_t stream){
    const bool first=std::string(alias)=="stem_3_32_gelu";
    const int oh=first?56:28;
    const int ow=oh;
    const int n=first?32:64;
    void* xa=const_cast<void*>(x); void* wa=model_.tensor(wname).device; void* ba=model_.tensor(bname).device; void* ya=y;
    void* args[]={&xa,&wa,&ba,&ya};
    kernels_.launch(alias,batch*oh*ow,ceil_div(n,32),1,args,stream);
}

void Runtime::launch_dw3(int batch,int H,int W,int C,int stride,const void* x,const char* base,void* y,cudaStream_t stream){
    int oh=(H+stride-1)/stride, ow=(W+stride-1)/stride;
    std::string ws=std::string(base)+".w", bs=std::string(base)+".b";
    void* xa=const_cast<void*>(x); void* wa=model_.tensor(ws).device; void* ba=model_.tensor(bs).device; void* ya=y;
    void* args[]={&xa,&wa,&ba,&ya};
    kernels_.launch(dw_alias(H,C,stride),batch*oh*ow,ceil_div(C,64),1,args,stream);
}

void Runtime::launch_dw4(int batch,const void* x,const char* base,void* y,cudaStream_t stream){
    std::string ws=std::string(base)+".w", bs=std::string(base)+".b";
    void* xa=const_cast<void*>(x); void* wa=model_.tensor(ws).device; void* ba=model_.tensor(bs).device; void* ya=y;
    void* args[]={&xa,&wa,&ba,&ya};
    kernels_.launch("dw4_global_c1284",batch,ceil_div(1284,64),1,args,stream);
}

void Runtime::launch_pw(int M,int K,int N,bool gelu,bool residual,
                        const void* x,const char* base,const void* residual_ptr,void* y,cudaStream_t stream){
    std::string ws=std::string(base)+".w", bs=std::string(base)+".b";
    void* xa=const_cast<void*>(x); void* wa=model_.tensor(ws).device; void* ba=model_.tensor(bs).device;
    void* ra=const_cast<void*>(residual_ptr?residual_ptr:x); void* ya=y; int mm=M;
    void* args[]={&xa,&wa,&ba,&ra,&ya,&mm};
    const int bm=(N<=512)?32:16;
    kernels_.launch(pw_alias(K,N,gelu,residual),ceil_div(M,bm),ceil_div(N,64),1,args,stream);
}

void Runtime::launch_mhla(int batch,int C,int S,const void* x,const char* base,void* y,cudaStream_t stream){
    auto get=[&](const char* suffix){ return model_.tensor(std::string(base)+suffix).device; };
    void* xa=const_cast<void*>(x); void* aa=get(".alpha"); void* be=get(".beta"); void* w=get(".w"); void* bi=get(".b"); void* ls=get(".ls"); void* ya=y;
    void* args[]={&xa,&aa,&be,&w,&bi,&ls,&ya};
    std::ostringstream alias; alias<<"mhla_c"<<C<<"_s"<<S;
    const int head_dim=C/4;
    kernels_.launch(alias.str(),batch,4,ceil_div(head_dim,32),args,stream);
}

void Runtime::launch_embed_out(int batch,bool normalize,const void* x,float* y,cudaStream_t stream){
    void* xa=const_cast<void*>(x); void* ya=y;
    void* args[]={&xa,&ya};
    kernels_.launch(normalize?"embed_out_norm1":"embed_out_norm0",batch,1,1,args,stream);
}

void Runtime::enqueue_network(int batch,bool bgr_input,bool normalize,cudaStream_t s){
    void* x=d_a_; void* y=d_b_;
    auto swap_xy=[&](){ std::swap(x,y); };

    launch_preprocess(batch,bgr_input,x,s);
    launch_stem("stem_3_32_gelu",batch,x,"stem.0.w","stem.0.b",y,s); swap_xy();
    launch_stem("stem_32_64",batch,x,"stem.1.w","stem.1.b",y,s); swap_xy();

    int H=28,W=28,C=64;
    auto ffn=[&](const std::string& base){
        const int M=batch*H*W;
        launch_pw(M,C,2*C,true,false,x,(base+".fc1").c_str(),nullptr,d_hidden_,s);
        launch_pw(M,2*C,C,false,true,d_hidden_,(base+".fc2").c_str(),x,y,s);
        swap_xy();
    };
    auto stage_repmix=[&](int stage,int depth){
        for(int b=0;b<depth;b++){
            std::string base="stage."+std::to_string(stage)+".block."+std::to_string(b);
            launch_dw3(batch,H,W,C,1,x,(base+".dw").c_str(),y,s); swap_xy();
            ffn(base+".ffn");
        }
    };
    auto stage_mhla=[&](int stage,int depth){
        for(int b=0;b<depth;b++){
            std::string base="stage."+std::to_string(stage)+".block."+std::to_string(b);
            launch_dw3(batch,H,W,C,1,x,(base+".dw").c_str(),y,s); swap_xy();
            launch_mhla(batch,C,H*W,x,(base+".mhla").c_str(),y,s); swap_xy();
            ffn(base+".ffn");
        }
    };
    auto merge=[&](int mi,int newC){
        std::string base="merge."+std::to_string(mi);
        launch_dw3(batch,H,W,C,2,x,(base+".dw").c_str(),y,s); swap_xy();
        H=(H+1)/2; W=(W+1)/2;
        int M=batch*H*W;
        launch_pw(M,C,newC,false,false,x,(base+".pw").c_str(),nullptr,y,s); swap_xy();
        C=newC;
        ffn(base+".ffn");
    };

    stage_repmix(0,3);
    merge(0,128);
    stage_repmix(1,3);
    merge(1,256);
    stage_mhla(2,9);
    merge(2,512);
    stage_mhla(3,3);

    // pre-head: 512 -> 1284 at 4x4, then 4x4 depthwise reduction to 1x1.
    launch_pw(batch*16,512,1284,false,false,x,"prehead.pw",nullptr,y,s); swap_xy();
    launch_dw4(batch,x,"prehead.dw4",y,s); swap_xy();
    launch_pw(batch,1284,512,false,false,x,"head.fc",nullptr,d_embed_half_,s);
    launch_embed_out(batch,normalize,d_embed_half_,d_output_,s);
}

cudaGraphExec_t Runtime::graph_for(int batch,bool bgr_input,bool normalize){
    if(batch<1 || batch>opt_.max_batch) throw std::runtime_error("batch outside configured max_batch");
    GraphKey key{batch,bgr_input,normalize};
    auto it=graphs_.find(key); if(it!=graphs_.end()) return it->second.exec;
    GraphExec ge;
    // First-use graph construction is cold-path; ensure capture begins from an idle stream.
    cuda_check(cudaStreamSynchronize(stream_),"sync before CUDA graph capture");
    cuda_check(cudaStreamBeginCapture(stream_,cudaStreamCaptureModeGlobal),"begin CUDA graph capture");
    try { enqueue_network(batch,bgr_input,normalize,stream_); }
    catch(...) { cudaStreamEndCapture(stream_,&ge.graph); if(ge.graph) cudaGraphDestroy(ge.graph); throw; }
    cuda_check(cudaStreamEndCapture(stream_,&ge.graph),"end CUDA graph capture");
    cuda_check(cudaGraphInstantiate(&ge.exec,ge.graph,nullptr,nullptr,0),"instantiate CUDA graph");
    graphs_[key]=ge;
    return ge.exec;
}

void Runtime::infer_host(const uint8_t* input,int batch,float* output,bool bgr_input,bool normalize){
    if(!input||!output) throw std::runtime_error("null inference pointer");
    if(batch<1||batch>opt_.max_batch) throw std::runtime_error("invalid batch");
    const size_t in_bytes=static_cast<size_t>(batch)*kInputBytesPerFace;
    const size_t out_bytes=static_cast<size_t>(batch)*kEmbedding*sizeof(float);
    std::memcpy(h_input_pinned_,input,in_bytes);
    cuda_check(cudaMemcpyAsync(d_input_u8_,h_input_pinned_,in_bytes,cudaMemcpyHostToDevice,stream_),"H2D input");
    cuda_check(cudaGraphLaunch(graph_for(batch,bgr_input,normalize),stream_),"cudaGraphLaunch");
    cuda_check(cudaMemcpyAsync(h_output_pinned_,d_output_,out_bytes,cudaMemcpyDeviceToHost,stream_),"D2H output");
    cuda_check(cudaStreamSynchronize(stream_),"sync inference");
    std::memcpy(output,h_output_pinned_,out_bytes);
}

void Runtime::infer_device(const uint8_t* d_input,int batch,float* d_output,bool bgr_input,bool normalize,
                           cudaStream_t input_ready_stream){
    if(!d_input||!d_output) throw std::runtime_error("null device inference pointer");
    if(batch<1||batch>opt_.max_batch) throw std::runtime_error("invalid batch");
    const size_t in_bytes=static_cast<size_t>(batch)*kInputBytesPerFace;
    const size_t out_bytes=static_cast<size_t>(batch)*kEmbedding*sizeof(float);
    if(input_ready_stream && input_ready_stream!=stream_){
        cudaEvent_t ready=nullptr;
        cuda_check(cudaEventCreateWithFlags(&ready,cudaEventDisableTiming),"create input-ready event");
        try {
            cuda_check(cudaEventRecord(ready,input_ready_stream),"record input-ready event");
            cuda_check(cudaStreamWaitEvent(stream_,ready,0),"wait for device input producer");
        } catch(...) { cudaEventDestroy(ready); throw; }
        cuda_check(cudaEventDestroy(ready),"release input-ready event");
    }
    cuda_check(cudaMemcpyAsync(d_input_u8_,d_input,in_bytes,cudaMemcpyDeviceToDevice,stream_),"D2D input");
    cuda_check(cudaGraphLaunch(graph_for(batch,bgr_input,normalize),stream_),"cudaGraphLaunch");
    cuda_check(cudaMemcpyAsync(d_output,d_output_,out_bytes,cudaMemcpyDeviceToDevice,stream_),"D2D output");
}

void Runtime::set_gallery(const float* embeddings,int count){
    if(!embeddings || count<=0) throw std::runtime_error("invalid gallery");
    if(count>4096) throw std::runtime_error("current GPU top-2 kernel supports up to 4096 gallery identities");
    std::vector<float> norm(static_cast<size_t>(count)*kEmbedding);
    for(int i=0;i<count;i++){
        const float* src=embeddings+static_cast<size_t>(i)*kEmbedding;
        float ss=0.f; for(int j=0;j<kEmbedding;j++) ss+=src[j]*src[j];
        float inv=1.0f/std::sqrt(std::max(ss,1.0e-20f));
        for(int j=0;j<kEmbedding;j++) norm[static_cast<size_t>(i)*kEmbedding+j]=src[j]*inv;
    }
    if(d_second_score_){ cudaFree(d_second_score_); d_second_score_=nullptr; }
    if(d_second_idx_){ cudaFree(d_second_idx_); d_second_idx_=nullptr; }
    if(d_best_score_){ cudaFree(d_best_score_); d_best_score_=nullptr; }
    if(d_best_idx_){ cudaFree(d_best_idx_); d_best_idx_=nullptr; }
    if(d_scores_){ cudaFree(d_scores_); d_scores_=nullptr; }
    if(d_gallery_){ cudaFree(d_gallery_); d_gallery_=nullptr; }
    gallery_count_=count;
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_gallery_),norm.size()*sizeof(float)),"malloc gallery");
    cuda_check(cudaMemcpy(d_gallery_,norm.data(),norm.size()*sizeof(float),cudaMemcpyHostToDevice),"upload gallery");
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_scores_),static_cast<size_t>(opt_.max_batch)*count*sizeof(float)),"malloc gallery scores");
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_best_idx_),static_cast<size_t>(opt_.max_batch)*sizeof(int)),"malloc best idx");
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_best_score_),static_cast<size_t>(opt_.max_batch)*sizeof(float)),"malloc best score");
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_second_idx_),static_cast<size_t>(opt_.max_batch)*sizeof(int)),"malloc second idx");
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_second_score_),static_cast<size_t>(opt_.max_batch)*sizeof(float)),"malloc second score");
}

void Runtime::launch_gallery_scores(const float* query,int batch,float* scores,cudaStream_t s){
    if(gallery_count_<=0) throw std::runtime_error("gallery not set");
    void* q=const_cast<float*>(query); void* g=d_gallery_; void* sc=scores; int N=gallery_count_;
    void* args[]={&q,&g,&sc,&N};
    kernels_.launch("gallery_scores",batch,ceil_div(gallery_count_,64),1,args,s);
}

void Runtime::infer_and_score_host(const uint8_t* input,int batch,float* scores,bool bgr_input){
    if(!input || !scores) throw std::runtime_error("null inference/score pointer");
    if(batch<1 || batch>opt_.max_batch) throw std::runtime_error("invalid batch");
    if(gallery_count_<=0) throw std::runtime_error("gallery not set");
    const size_t in_bytes=static_cast<size_t>(batch)*kInputBytesPerFace;
    std::memcpy(h_input_pinned_,input,in_bytes);
    cuda_check(cudaMemcpyAsync(d_input_u8_,h_input_pinned_,in_bytes,cudaMemcpyHostToDevice,stream_),"H2D input");
    cuda_check(cudaGraphLaunch(graph_for(batch,bgr_input,true),stream_),"cudaGraphLaunch");
    launch_gallery_scores(d_output_,batch,d_scores_,stream_);
    cuda_check(cudaMemcpyAsync(scores,d_scores_,static_cast<size_t>(batch)*gallery_count_*sizeof(float),cudaMemcpyDeviceToHost,stream_),"D2H scores");
    cuda_check(cudaStreamSynchronize(stream_),"sync inference+score");
}
void Runtime::launch_top2(int batch,cudaStream_t s){
    void* sc=d_scores_; void* bi=d_best_idx_; void* bs=d_best_score_; void* si=d_second_idx_; void* ss=d_second_score_; int N=gallery_count_;
    void* args[]={&sc,&bi,&bs,&si,&ss,&N};
    kernels_.launch("top2_scores_4096",batch,1,1,args,s);
}

void Runtime::infer_and_match_host(const uint8_t* input,int batch,Match* matches,bool bgr_input){
    if(!input || !matches) throw std::runtime_error("null match input/output");
    if(batch<1 || batch>opt_.max_batch) throw std::runtime_error("invalid batch");
    if(gallery_count_<=0) throw std::runtime_error("gallery not set");
    const size_t in_bytes=static_cast<size_t>(batch)*kInputBytesPerFace;
    std::memcpy(h_input_pinned_,input,in_bytes);
    cuda_check(cudaMemcpyAsync(d_input_u8_,h_input_pinned_,in_bytes,cudaMemcpyHostToDevice,stream_),"H2D input");
    cuda_check(cudaGraphLaunch(graph_for(batch,bgr_input,true),stream_),"cudaGraphLaunch");
    launch_gallery_scores(d_output_,batch,d_scores_,stream_);
    launch_top2(batch,stream_);
    std::vector<int> bidx(batch),sidx(batch);
    std::vector<float> bscore(batch),sscore(batch);
    cuda_check(cudaMemcpyAsync(bidx.data(),d_best_idx_,batch*sizeof(int),cudaMemcpyDeviceToHost,stream_),"D2H best idx");
    cuda_check(cudaMemcpyAsync(bscore.data(),d_best_score_,batch*sizeof(float),cudaMemcpyDeviceToHost,stream_),"D2H best score");
    cuda_check(cudaMemcpyAsync(sidx.data(),d_second_idx_,batch*sizeof(int),cudaMemcpyDeviceToHost,stream_),"D2H second idx");
    cuda_check(cudaMemcpyAsync(sscore.data(),d_second_score_,batch*sizeof(float),cudaMemcpyDeviceToHost,stream_),"D2H second score");
    cuda_check(cudaStreamSynchronize(stream_),"sync inference+match");
    for(int i=0;i<batch;i++) matches[i]=Match{bidx[i],bscore[i],sidx[i],sscore[i]};
}

}
