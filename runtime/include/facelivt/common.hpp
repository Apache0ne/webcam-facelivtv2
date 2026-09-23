#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace facelivt {
inline void cuda_check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(what)+": "+cudaGetErrorString(e));
}
inline void cu_check(CUresult e, const char* what) {
    if (e == CUDA_SUCCESS) return;
    const char* name=nullptr; const char* msg=nullptr;
    cuGetErrorName(e,&name); cuGetErrorString(e,&msg);
    throw std::runtime_error(std::string(what)+": "+(name?name:"CUerror")+" "+(msg?msg:""));
}
inline int ceil_div(int a, int b) { return (a+b-1)/b; }
}
