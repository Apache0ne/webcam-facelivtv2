#include "facelivt/runtime.hpp"
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

int main(int argc,char** argv){
    if(argc<3){ std::cerr<<"usage: facelivt_bench model.flvt kernel_manifest.tsv [batch=1] [iters=1000]\n"; return 2; }
    try {
    int batch=argc>3?std::stoi(argv[3]):1;
    int iters=argc>4?std::stoi(argv[4]):1000;
    if(batch<1 || iters<1) throw std::runtime_error("batch and iters must be positive");
    facelivt::RuntimeOptions opt; opt.max_batch=std::max(64,batch);
    std::cerr<<"Loading model and CUDA kernels...\n";
    facelivt::Runtime rt(argv[1],argv[2],opt);
    std::vector<uint8_t> input(static_cast<size_t>(batch)*112*112*3);
    std::mt19937 rng(1234); for(auto& v:input) v=static_cast<uint8_t>(rng()&255);
    std::vector<float> out(static_cast<size_t>(batch)*512);
    // first call includes CUDA graph construction; keep it outside timing
    std::cerr<<"Warming up batch "<<batch<<"...\n";
    rt.infer_host(input.data(),batch,out.data());
    auto t0=std::chrono::steady_clock::now();
    for(int i=0;i<iters;i++) rt.infer_host(input.data(),batch,out.data());
    auto t1=std::chrono::steady_clock::now();
    double sec=std::chrono::duration<double>(t1-t0).count();
    std::cout<<"batch="<<batch<<" iters="<<iters<<" seconds="<<sec
             <<" faces_per_sec="<<(double(batch)*iters/sec)
             <<" ms_per_batch="<<(sec*1000.0/iters)<<"\n";
    } catch(const std::exception& e) {
        std::cerr<<"error: "<<e.what()<<"\n";
        return 1;
    }
    return 0;
}
