#pragma once
#include "common.hpp"
#include <filesystem>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

namespace facelivt {
struct KernelInfo {
    CUmodule module=nullptr;
    CUfunction function=nullptr;
    int warps=4;
    int shared=0;
    int scratch_args=0;
};

class KernelRegistry {
public:
    KernelRegistry()=default;
    explicit KernelRegistry(const std::filesystem::path& manifest) { load(manifest); }
    ~KernelRegistry();
    KernelRegistry(const KernelRegistry&)=delete;
    KernelRegistry& operator=(const KernelRegistry&)=delete;

    void load(const std::filesystem::path& manifest);
    const KernelInfo& get(const std::string& alias) const;
    void launch(const std::string& alias,
                unsigned gx,unsigned gy,unsigned gz,
                std::span<void*> args,cudaStream_t stream) const;
private:
    std::unordered_map<std::string,KernelInfo> kernels_;
};
}
