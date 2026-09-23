#include "facelivt/kernel_registry.hpp"
#include <fstream>
#include <sstream>
#include <vector>

namespace facelivt {
KernelRegistry::~KernelRegistry(){ for(auto& kv:kernels_) if(kv.second.module) cuModuleUnload(kv.second.module); }

void KernelRegistry::load(const std::filesystem::path& manifest){
    if(!kernels_.empty()) throw std::runtime_error("kernel registry already loaded");
    cu_check(cuInit(0),"cuInit");
    std::ifstream f(manifest); if(!f) throw std::runtime_error("cannot open kernel manifest: "+manifest.string());
    std::string line; std::getline(f,line); // header
    const auto base=manifest.parent_path();
    while(std::getline(f,line)){
        if(line.empty()) continue;
        std::vector<std::string> cols; std::stringstream ss(line); std::string c;
        while(std::getline(ss,c,'\t')) cols.push_back(c);
        if(cols.size()<5) throw std::runtime_error("invalid manifest row: "+line);
        KernelInfo ki; ki.warps=std::stoi(cols[3]); ki.shared=std::stoi(cols[4]);
        if(cols.size()>5) ki.scratch_args=std::stoi(cols[5]);
        if(ki.scratch_args<0 || ki.scratch_args>2) throw std::runtime_error("unsupported Triton scratch argument count");
        auto cubin=(base/cols[1]).string();
        cu_check(cuModuleLoad(&ki.module,cubin.c_str()),("cuModuleLoad "+cubin).c_str());
        cu_check(cuModuleGetFunction(&ki.function,ki.module,cols[2].c_str()),("cuModuleGetFunction "+cols[0]).c_str());
        kernels_.emplace(cols[0],ki);
    }
}
const KernelInfo& KernelRegistry::get(const std::string& alias) const {
    auto it=kernels_.find(alias); if(it==kernels_.end()) throw std::runtime_error("missing Triton kernel: "+alias); return it->second;
}
void KernelRegistry::launch(const std::string& alias,unsigned gx,unsigned gy,unsigned gz,std::span<void*> args,cudaStream_t stream) const {
    const auto& k=get(alias);
    CUdeviceptr scratch=0;
    std::vector<void*> params(args.begin(),args.end());
    for(int i=0;i<k.scratch_args;++i) params.push_back(&scratch);
    cu_check(cuLaunchKernel(k.function,gx,gy,gz,static_cast<unsigned>(k.warps*32),1,1,
                            static_cast<unsigned>(k.shared),reinterpret_cast<CUstream>(stream),params.data(),nullptr),
             ("launch "+alias).c_str());
}
}
