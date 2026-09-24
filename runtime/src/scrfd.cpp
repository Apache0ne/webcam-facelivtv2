#include "facelivt/scrfd.hpp"
#include "facelivt/common.hpp"
#include <onnxruntime_cxx_api.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace facelivt {

cudaError_t launch_scrfd_preprocess(const uint8_t*,int,int,int,int,int,float*,cudaStream_t);
cudaError_t launch_scrfd_decode(const float*,const float*,const float*,int,int,int,
                                float,float,float,ScrfdCandidate*,int*,int,cudaStream_t);
cudaError_t launch_scrfd_align(const uint8_t*,int,int,const float*,uint8_t*,int,cudaStream_t);

namespace {
constexpr std::array<int,3> kStrides{8,16,32};
constexpr std::array<std::array<float,2>,5> kAlignmentTemplate{{
    {38.2946f,51.6963f},{73.5318f,51.5014f},{56.0252f,71.7366f},
    {41.5493f,92.3655f},{70.7299f,92.2041f}
}};
constexpr int kMaximumNmsCandidates=256;

float intersection_over_union(const ScrfdCandidate& a,const ScrfdCandidate& b) {
    const float left=std::max(a.box[0],b.box[0]);
    const float top=std::max(a.box[1],b.box[1]);
    const float right=std::min(a.box[2],b.box[2]);
    const float bottom=std::min(a.box[3],b.box[3]);
    const float area_intersection=std::max(0.0f,right-left+1.0f)*std::max(0.0f,bottom-top+1.0f);
    const float area_a=std::max(0.0f,a.box[2]-a.box[0]+1.0f)*std::max(0.0f,a.box[3]-a.box[1]+1.0f);
    const float area_b=std::max(0.0f,b.box[2]-b.box[0]+1.0f)*std::max(0.0f,b.box[3]-b.box[1]+1.0f);
    return area_intersection/std::max(area_a+area_b-area_intersection,1.0e-8f);
}

std::array<float,6> inverse_similarity_transform(const ScrfdFace& face) {
    float source_mean_x=0,source_mean_y=0,target_mean_x=0,target_mean_y=0;
    for(int i=0;i<5;++i){
        const float x=face.landmarks[i*2],y=face.landmarks[i*2+1];
        if(!std::isfinite(x) || !std::isfinite(y)) throw std::runtime_error("SCRFD returned non-finite landmarks");
        source_mean_x+=x/5.0f;
        source_mean_y+=y/5.0f;
        target_mean_x+=kAlignmentTemplate[i][0]/5.0f;
        target_mean_y+=kAlignmentTemplate[i][1]/5.0f;
    }
    float variance=0,alpha=0,beta=0;
    for(int i=0;i<5;++i){
        const float sx=face.landmarks[i*2]-source_mean_x;
        const float sy=face.landmarks[i*2+1]-source_mean_y;
        const float tx=kAlignmentTemplate[i][0]-target_mean_x;
        const float ty=kAlignmentTemplate[i][1]-target_mean_y;
        variance+=sx*sx+sy*sy;
        alpha+=sx*tx+sy*ty;
        beta+=sx*ty-sy*tx;
    }
    const float magnitude=std::sqrt(alpha*alpha+beta*beta);
    if(variance<1.0e-8f || magnitude<1.0e-8f) throw std::runtime_error("SCRFD landmarks are degenerate");
    const float scale=magnitude/variance;
    const float cosine=alpha/magnitude;
    const float sine=beta/magnitude;
    const float a=cosine/scale,b=sine/scale,c=-sine/scale,d=cosine/scale;
    const float tx=target_mean_x-(scale*cosine*source_mean_x-scale*sine*source_mean_y);
    const float ty=target_mean_y-(scale*sine*source_mean_x+scale*cosine*source_mean_y);
    return {a,b,-(a*tx+b*ty),c,d,-(c*tx+d*ty)};
}

int stride_for_count(int input_size,int count) {
    for(int stride:kStrides){
        const int grid=input_size/stride;
        if(count==grid*grid*2) return stride;
    }
    return 0;
}
}

ScrfdDetector::ScrfdDetector(const std::filesystem::path& model,ScrfdOptions options)
    :options_(options),owner_thread_(std::this_thread::get_id()) {
    if(options_.device<0) throw std::runtime_error("CUDA device index must be nonnegative");
    if(options_.input_size<32 || options_.input_size%32!=0)
        throw std::runtime_error("SCRFD input size must be a positive multiple of 32");
    if(options_.max_faces<1 || options_.max_faces>64)
        throw std::runtime_error("SCRFD max_faces must be between 1 and 64");
    if(!(options_.score_threshold>=0.0f && options_.score_threshold<=1.0f) ||
       !(options_.nms_threshold>0.0f && options_.nms_threshold<1.0f))
        throw std::runtime_error("invalid SCRFD score or NMS threshold");
    try {
        cuda_check(cudaSetDevice(options_.device),"select SCRFD CUDA device");
        cuda_check(cudaFree(nullptr),"initialize SCRFD CUDA context");
        cuda_check(cudaStreamCreateWithFlags(&stream_,cudaStreamNonBlocking),"create SCRFD CUDA stream");
        const size_t input_elements=static_cast<size_t>(3)*options_.input_size*options_.input_size;
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&device_input_),input_elements*sizeof(float)),"allocate SCRFD input");
        for(int stride:kStrides){
            const size_t grid=options_.input_size/stride;
            candidate_capacity_+=grid*grid*2;
        }
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&device_candidate_count_),sizeof(int)),"allocate SCRFD candidate counter");
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&device_candidates_),candidate_capacity_*sizeof(ScrfdCandidate)),"allocate SCRFD candidates");
        cuda_check(cudaHostAlloc(reinterpret_cast<void**>(&host_candidates_),candidate_capacity_*sizeof(ScrfdCandidate),cudaHostAllocPortable),"allocate pinned SCRFD candidate staging");
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&device_affine_),static_cast<size_t>(options_.max_faces)*6*sizeof(float)),"allocate CUDA alignment matrices");
        initialize_session(model);
    } catch(...) {
        cleanup();
        throw;
    }
}

ScrfdDetector::~ScrfdDetector() { cleanup(); }

void ScrfdDetector::cleanup() noexcept {
    binding_.reset();
    input_value_.reset();
    output_values_.clear();
    session_.reset();
    cuda_memory_.reset();
    env_.reset();
    for(auto& output:outputs_){
        if(output.device_data) cudaFree(output.device_data);
        output.device_data=nullptr;
    }
    outputs_.clear();
    if(device_affine_) cudaFree(device_affine_);
    if(host_candidates_) cudaFreeHost(host_candidates_);
    if(device_candidates_) cudaFree(device_candidates_);
    if(device_candidate_count_) cudaFree(device_candidate_count_);
    if(device_input_) cudaFree(device_input_);
    if(stream_) cudaStreamDestroy(stream_);
    device_affine_=nullptr;
    host_candidates_=nullptr;
    device_candidates_=nullptr;
    device_candidate_count_=nullptr;
    device_input_=nullptr;
    stream_=nullptr;
}

void ScrfdDetector::initialize_session(const std::filesystem::path& model) {
    env_=std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING,"facelivt_scrfd");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_options.SetInterOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    session_options.AddConfigEntry("session.disable_cpu_ep_fallback","1");
    session_options.AddFreeDimensionOverrideByName("?",options_.input_size);

    Ort::CUDAProviderOptions cuda_options;
    std::unordered_map<std::string,std::string> provider_options{
        {"device_id",std::to_string(options_.device)},
        {"user_compute_stream",std::to_string(reinterpret_cast<uintptr_t>(stream_))},
        {"cudnn_conv_algo_search","HEURISTIC"},
        {"enable_cuda_graph","0"},
        {"do_copy_in_default_stream","1"},
        {"use_tf32","1"},
        {"gpu_mem_limit",std::to_string(options_.gpu_memory_limit)}
    };
    cuda_options.Update(provider_options);
    session_options.AppendExecutionProvider_CUDA_V2(*cuda_options);
    session_=std::make_unique<Ort::Session>(*env_,model.wstring().c_str(),session_options);
    if(session_->GetInputCount()!=1) throw std::runtime_error("SCRFD model must expose exactly one input tensor");
    auto input_name=session_->GetInputNameAllocated(0,Ort::AllocatorWithDefaultOptions{});
    cuda_memory_=std::make_unique<Ort::MemoryInfo>("Cuda",OrtAllocatorType::OrtDeviceAllocator,
                                                  options_.device,OrtMemTypeDefault);
    const std::array<int64_t,4> input_shape{1,3,options_.input_size,options_.input_size};
    const size_t input_elements=static_cast<size_t>(3)*options_.input_size*options_.input_size;
    input_value_=std::make_unique<Ort::Value>(Ort::Value::CreateTensor<float>(
        *cuda_memory_,device_input_,input_elements,input_shape.data(),input_shape.size()));
    binding_=std::make_unique<Ort::IoBinding>(*session_);
    binding_->BindInput(input_name.get(),*input_value_);
    allocate_outputs();
}

void ScrfdDetector::allocate_outputs() {
    const size_t count=session_->GetOutputCount();
    if(count!=9) throw std::runtime_error("SCRFD-10G model must expose nine score, box, and landmark outputs");
    outputs_.reserve(count);
    output_values_.reserve(count);
    for(size_t i=0;i<count;++i){
        auto name=session_->GetOutputNameAllocated(i,Ort::AllocatorWithDefaultOptions{});
        auto info=session_->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo();
        if(info.GetElementType()!=ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
            throw std::runtime_error("SCRFD outputs must be float32");
        auto shape=info.GetShape();
        if(shape.size()!=2 || shape[0]<=0 || shape[1]<=0)
            throw std::runtime_error("SCRFD outputs must have fixed [anchors, features] shapes");
        const int output_count=static_cast<int>(shape[0]);
        const int features=static_cast<int>(shape[1]);
        const int stride=stride_for_count(options_.input_size,output_count);
        if(!stride || (features!=1 && features!=4 && features!=10))
            throw std::runtime_error("SCRFD output shape is not a score, box, or five-landmark head");
        Head* head=&heads_[stride==8?0:(stride==16?1:2)];
        if(head->stride && head->count!=output_count)
            throw std::runtime_error("inconsistent SCRFD output anchor count");
        head->stride=stride;
        head->count=output_count;

        const size_t elements=static_cast<size_t>(shape[0])*shape[1];
        float* device_data=nullptr;
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&device_data),elements*sizeof(float)),"allocate SCRFD output");
        try {
            auto value=std::make_unique<Ort::Value>(Ort::Value::CreateTensor<float>(
                *cuda_memory_,device_data,elements,shape.data(),shape.size()));
            binding_->BindOutput(name.get(),*value);
            if(features==1){
                if(head->scores) throw std::runtime_error("duplicate SCRFD score output");
                head->scores=device_data;
            } else if(features==4){
                if(head->boxes) throw std::runtime_error("duplicate SCRFD box output");
                head->boxes=device_data;
            } else {
                if(head->landmarks) throw std::runtime_error("duplicate SCRFD landmark output");
                head->landmarks=device_data;
            }
            Output output;
            output.name=name.get();
            output.shape=std::move(shape);
            output.device_data=device_data;
            output.elements=elements;
            outputs_.push_back(std::move(output));
            output_values_.push_back(std::move(value));
        } catch(...) {
            cudaFree(device_data);
            throw;
        }
    }
    for(const auto& head:heads_){
        if(!head.stride || !head.scores || !head.boxes || !head.landmarks)
            throw std::runtime_error("SCRFD model is missing an output head");
    }
}

void ScrfdDetector::check_thread() const {
    if(std::this_thread::get_id()!=owner_thread_)
        throw std::runtime_error("Use and close SCRFD on the thread that created it");
    cuda_check(cudaSetDevice(options_.device),"select SCRFD CUDA device");
}

std::vector<ScrfdFace> ScrfdDetector::detect(const uint8_t* frame_bgra,int width,int height) {
    check_thread();
    if(!frame_bgra || width<=0 || height<=0) throw std::runtime_error("invalid GPU BGRA frame");
    const float ratio=std::min(static_cast<float>(options_.input_size)/width,
                               static_cast<float>(options_.input_size)/height);
    const int resized_width=std::max(1,static_cast<int>(width*ratio));
    const int resized_height=std::max(1,static_cast<int>(height*ratio));
    const float scale_x=static_cast<float>(resized_width)/width;
    const float scale_y=static_cast<float>(resized_height)/height;
    cuda_check(launch_scrfd_preprocess(frame_bgra,width,height,resized_width,resized_height,
                                       options_.input_size,device_input_,stream_),"launch SCRFD preprocessing");
    cuda_check(cudaMemsetAsync(device_candidate_count_,0,sizeof(int),stream_),"reset SCRFD candidate count");
    session_->Run(Ort::RunOptions{nullptr},*binding_);
    binding_->SynchronizeOutputs();
    for(const auto& head:heads_){
        cuda_check(launch_scrfd_decode(head.scores,head.boxes,head.landmarks,head.count,
                                       options_.input_size,head.stride,scale_x,scale_y,
                                       options_.score_threshold,device_candidates_,
                                       device_candidate_count_,static_cast<int>(candidate_capacity_),
                                       stream_),"launch SCRFD head decode");
    }
    cuda_check(cudaMemcpyAsync(&host_candidate_count_,device_candidate_count_,sizeof(int),
                               cudaMemcpyDeviceToHost,stream_),"copy SCRFD candidate count");
    cuda_check(cudaStreamSynchronize(stream_),"finish SCRFD candidate count");
    if(host_candidate_count_<0 || static_cast<size_t>(host_candidate_count_)>candidate_capacity_)
        throw std::runtime_error("SCRFD returned an invalid candidate count");
    if(host_candidate_count_==0) return {};
    cuda_check(cudaMemcpyAsync(host_candidates_,device_candidates_,
                               static_cast<size_t>(host_candidate_count_)*sizeof(ScrfdCandidate),
                               cudaMemcpyDeviceToHost,stream_),"copy thresholded SCRFD candidates");
    cuda_check(cudaStreamSynchronize(stream_),"finish SCRFD candidate transfer");

    std::vector<ScrfdCandidate> candidates(host_candidates_,host_candidates_+host_candidate_count_);
    std::stable_sort(candidates.begin(),candidates.end(),[](const auto& a,const auto& b){
        return a.confidence>b.confidence;
    });
    std::vector<ScrfdCandidate> selected;
    selected.reserve(std::min<size_t>(candidates.size(),kMaximumNmsCandidates));
    for(const auto& candidate:candidates){
        bool suppressed=false;
        for(const auto& previous:selected){
            if(intersection_over_union(candidate,previous)>options_.nms_threshold){
                suppressed=true;
                break;
            }
        }
        if(!suppressed){
            selected.push_back(candidate);
            if(selected.size()==kMaximumNmsCandidates) break;
        }
    }
    std::stable_sort(selected.begin(),selected.end(),[](const auto& a,const auto& b){
        const float area_a=std::max(0.0f,a.box[2]-a.box[0])*std::max(0.0f,a.box[3]-a.box[1]);
        const float area_b=std::max(0.0f,b.box[2]-b.box[0])*std::max(0.0f,b.box[3]-b.box[1]);
        return area_a>area_b;
    });
    if(selected.size()>static_cast<size_t>(options_.max_faces)) selected.resize(options_.max_faces);
    std::vector<ScrfdFace> faces;
    faces.reserve(selected.size());
    for(const auto& candidate:selected){
        ScrfdFace face;
        std::copy(std::begin(candidate.box),std::end(candidate.box),face.box.begin());
        std::copy(std::begin(candidate.landmarks),std::end(candidate.landmarks),face.landmarks.begin());
        face.confidence=candidate.confidence;
        faces.push_back(face);
    }
    return faces;
}

void ScrfdDetector::align(const uint8_t* frame_bgra,int width,int height,
                          std::span<const ScrfdFace> faces,uint8_t* device_bgr_crops) {
    check_thread();
    if(faces.empty()) return;
    if(!frame_bgra || !device_bgr_crops || width<=0 || height<=0)
        throw std::runtime_error("invalid CUDA alignment input");
    if(faces.size()>static_cast<size_t>(options_.max_faces))
        throw std::runtime_error("face batch exceeds the configured alignment capacity");
    std::vector<float> transforms;
    transforms.reserve(faces.size()*6);
    for(const auto& face:faces){
        const auto matrix=inverse_similarity_transform(face);
        transforms.insert(transforms.end(),matrix.begin(),matrix.end());
    }
    cuda_check(cudaMemcpyAsync(device_affine_,transforms.data(),transforms.size()*sizeof(float),
                               cudaMemcpyHostToDevice,stream_),"upload face alignment transforms");
    cuda_check(launch_scrfd_align(frame_bgra,width,height,device_affine_,device_bgr_crops,
                                  static_cast<int>(faces.size()),stream_),"launch CUDA face alignment");
}

} // namespace facelivt
