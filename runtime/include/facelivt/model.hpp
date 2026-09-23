#pragma once
#include "common.hpp"
#include <array>
#include <cstdint>
#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>

namespace facelivt {
struct TensorView {
    std::array<uint32_t,4> dims{1,1,1,1};
    uint32_t ndim=0;
    uint32_t dtype=0;
    uint64_t bytes=0;
    uint64_t offset=0;
    void* device=nullptr;
};

class Model {
public:
    Model()=default;
    explicit Model(const std::filesystem::path& path) { load(path); }
    ~Model();
    Model(const Model&)=delete;
    Model& operator=(const Model&)=delete;
    Model(Model&&) noexcept;
    Model& operator=(Model&&) noexcept;

    void load(const std::filesystem::path& path);
    const TensorView& tensor(const std::string& name) const;
    bool has(const std::string& name) const;
    size_t tensor_count() const { return tensors_.size(); }
    uint64_t device_bytes() const { return data_bytes_; }

private:
    void reset();
    void* device_blob_=nullptr;
    uint64_t data_bytes_=0;
    std::unordered_map<std::string,TensorView> tensors_;
};
}
