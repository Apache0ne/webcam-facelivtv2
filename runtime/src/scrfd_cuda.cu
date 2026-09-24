#include "facelivt/scrfd.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>

namespace facelivt {
namespace {
constexpr int kThreads=256;

__device__ float read_channel(const uint8_t* frame,int width,int height,int x,int y,int channel) {
    if(x<0 || x>=width || y<0 || y>=height) return 0.0f;
    return static_cast<float>(frame[(static_cast<size_t>(y)*width+x)*4+channel]);
}

__global__ void preprocess_bgra_kernel(const uint8_t* frame,int width,int height,
                                       int resized_width,int resized_height,int input_size,
                                       float* output) {
    const int plane=input_size*input_size;
    const int count=plane*3;
    const int index=blockIdx.x*blockDim.x+threadIdx.x;
    if(index>=count) return;
    const int channel=index/plane;
    const int pixel=index-channel*plane;
    const int x=pixel%input_size;
    const int y=pixel/input_size;
    if(x>=resized_width || y>=resized_height) {
        output[index]=-127.5f/128.0f;
        return;
    }
    const float sx=fmaxf(0.0f,fminf(width-1.0f,(x+0.5f)*width/resized_width-0.5f));
    const float sy=fmaxf(0.0f,fminf(height-1.0f,(y+0.5f)*height/resized_height-0.5f));
    const int x0=static_cast<int>(floorf(sx));
    const int y0=static_cast<int>(floorf(sy));
    const int x1=min(x0+1,width-1);
    const int y1=min(y0+1,height-1);
    const float fx=sx-x0, fy=sy-y0;
    const int source_channel=2-channel; // camera BGRA -> model RGB
    const float top=read_channel(frame,width,height,x0,y0,source_channel)*(1-fx)+
                    read_channel(frame,width,height,x1,y0,source_channel)*fx;
    const float bottom=read_channel(frame,width,height,x0,y1,source_channel)*(1-fx)+
                       read_channel(frame,width,height,x1,y1,source_channel)*fx;
    const float value=__uint2float_rn(__float2uint_rn(top*(1-fy)+bottom*fy));
    output[index]=(value-127.5f)/128.0f;
}

__global__ void decode_head_kernel(const float* scores,const float* distances,
                                   const float* offsets,int count,int input_size,
                                   int stride,float scale_x,float scale_y,float threshold,
                                   ScrfdCandidate* output,int* output_count,int capacity) {
    const int index=blockIdx.x*blockDim.x+threadIdx.x;
    if(index>=count) return;
    const float score=scores[index];
    if(!isfinite(score) || score<threshold) return;
    const int grid_width=input_size/stride;
    const int cell=index>>1; // SCRFD uses two anchors per cell at every level.
    const float center_x=static_cast<float>(cell%grid_width*stride);
    const float center_y=static_cast<float>(cell/grid_width*stride);
    const float* box=distances+static_cast<size_t>(index)*4;
    const float* points=offsets+static_cast<size_t>(index)*10;
    const int slot=atomicAdd(output_count,1);
    if(slot>=capacity) return;
    ScrfdCandidate& candidate=output[slot];
    candidate.box[0]=(center_x-box[0]*stride)/scale_x;
    candidate.box[1]=(center_y-box[1]*stride)/scale_y;
    candidate.box[2]=(center_x+box[2]*stride)/scale_x;
    candidate.box[3]=(center_y+box[3]*stride)/scale_y;
    candidate.confidence=score;
    for(int i=0;i<10;++i)
        candidate.landmarks[i]=(i&1 ? center_y : center_x)+points[i]*stride;
    for(int i=0;i<10;i+=2){
        candidate.landmarks[i]/=scale_x;
        candidate.landmarks[i+1]/=scale_y;
    }
    candidate.padding[0]=0;
}

__global__ void align_bgra_kernel(const uint8_t* frame,int width,int height,
                                 const float* inverse_affine,uint8_t* output,
                                 int face_count) {
    constexpr int crop_size=112;
    constexpr int pixels=crop_size*crop_size;
    const int index=blockIdx.x*blockDim.x+threadIdx.x;
    const int total=face_count*pixels;
    if(index>=total) return;
    const int face=index/pixels;
    const int pixel=index-face*pixels;
    const float x=static_cast<float>(pixel%crop_size);
    const float y=static_cast<float>(pixel/crop_size);
    const float* m=inverse_affine+face*6;
    const float sx=nearbyintf((m[0]*x+m[1]*y+m[2])*32.0f)*(1.0f/32.0f);
    const float sy=nearbyintf((m[3]*x+m[4]*y+m[5])*32.0f)*(1.0f/32.0f);
    const int x0=static_cast<int>(floorf(sx));
    const int y0=static_cast<int>(floorf(sy));
    const int x1=x0+1, y1=y0+1;
    const float fx=sx-x0, fy=sy-y0;
    const size_t destination=(static_cast<size_t>(face)*pixels+pixel)*3;
    for(int c=0;c<3;++c){
        const float top=read_channel(frame,width,height,x0,y0,c)*(1-fx)+
                        read_channel(frame,width,height,x1,y0,c)*fx;
        const float bottom=read_channel(frame,width,height,x0,y1,c)*(1-fx)+
                           read_channel(frame,width,height,x1,y1,c)*fx;
        const float value=top*(1-fy)+bottom*fy;
        output[destination+c]=static_cast<uint8_t>(max(0,min(255,__float2int_rn(value))));
    }
}
}

cudaError_t launch_scrfd_preprocess(const uint8_t* frame,int width,int height,
                                    int resized_width,int resized_height,int input_size,
                                    float* output,cudaStream_t stream) {
    const int count=input_size*input_size*3;
    preprocess_bgra_kernel<<<(count+kThreads-1)/kThreads,kThreads,0,stream>>>(
        frame,width,height,resized_width,resized_height,input_size,output);
    return cudaGetLastError();
}

cudaError_t launch_scrfd_decode(const float* scores,const float* distances,
                                const float* offsets,int count,int input_size,int stride,
                                float scale_x,float scale_y,float threshold,
                                ScrfdCandidate* output,int* output_count,int capacity,
                                cudaStream_t stream) {
    decode_head_kernel<<<(count+kThreads-1)/kThreads,kThreads,0,stream>>>(
        scores,distances,offsets,count,input_size,stride,scale_x,scale_y,threshold,
        output,output_count,capacity);
    return cudaGetLastError();
}

cudaError_t launch_scrfd_align(const uint8_t* frame,int width,int height,
                               const float* inverse_affine,uint8_t* output,
                               int face_count,cudaStream_t stream) {
    constexpr int pixels=112*112;
    const int count=face_count*pixels;
    align_bgra_kernel<<<(count+kThreads-1)/kThreads,kThreads,0,stream>>>(
        frame,width,height,inverse_affine,output,face_count);
    return cudaGetLastError();
}

} // namespace facelivt
