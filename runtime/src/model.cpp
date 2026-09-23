#include "facelivt/model.hpp"
#include <cstring>
#include <fstream>
#include <utility>

namespace facelivt {
#pragma pack(push,1)
struct DiskHeader {
    char magic[8];
    uint32_t version;
    uint32_t tensor_count;
    uint64_t table_offset;
    uint64_t data_offset;
    uint64_t data_bytes;
    uint32_t dtype;
    char reserved[20];
};
struct DiskDesc {
    char name[96];
    uint32_t dtype;
    uint32_t ndim;
    uint32_t dims[4];
    uint64_t offset;
    uint64_t bytes;
    uint64_t reserved;
};
#pragma pack(pop)
static_assert(sizeof(DiskHeader)==64);
static_assert(sizeof(DiskDesc)==144);

Model::~Model(){ reset(); }
Model::Model(Model&& o) noexcept { *this=std::move(o); }
Model& Model::operator=(Model&& o) noexcept {
    if(this!=&o){ reset(); device_blob_=o.device_blob_; data_bytes_=o.data_bytes_; tensors_=std::move(o.tensors_); o.device_blob_=nullptr; o.data_bytes_=0; }
    return *this;
}
void Model::reset(){ if(device_blob_) cudaFree(device_blob_); device_blob_=nullptr; data_bytes_=0; tensors_.clear(); }

void Model::load(const std::filesystem::path& path){
    reset();
    std::ifstream f(path,std::ios::binary);
    if(!f) throw std::runtime_error("cannot open model: "+path.string());
    DiskHeader h{}; f.read(reinterpret_cast<char*>(&h),sizeof(h));
    if(!f || std::memcmp(h.magic,"FLV2BIN\0",8)!=0) throw std::runtime_error("invalid FLVT magic");
    if(h.version!=1 || h.dtype!=1) throw std::runtime_error("unsupported FLVT version/dtype");
    f.seekg(static_cast<std::streamoff>(h.table_offset));
    std::vector<DiskDesc> descs(h.tensor_count);
    f.read(reinterpret_cast<char*>(descs.data()),static_cast<std::streamsize>(descs.size()*sizeof(DiskDesc)));
    if(!f) throw std::runtime_error("truncated FLVT tensor table");
    std::vector<uint8_t> blob(static_cast<size_t>(h.data_bytes));
    f.seekg(static_cast<std::streamoff>(h.data_offset));
    f.read(reinterpret_cast<char*>(blob.data()),static_cast<std::streamsize>(blob.size()));
    if(!f) throw std::runtime_error("truncated FLVT weight blob");
    cuda_check(cudaMalloc(&device_blob_,blob.size()),"cudaMalloc model");
    cuda_check(cudaMemcpy(device_blob_,blob.data(),blob.size(),cudaMemcpyHostToDevice),"copy model");
    data_bytes_=h.data_bytes;
    for(const auto& d:descs){
        if(d.offset+d.bytes>h.data_bytes) throw std::runtime_error("FLVT tensor out of bounds");
        size_t len=0; while(len<sizeof(d.name)&&d.name[len]) ++len;
        std::string name(d.name,len);
        TensorView t; t.dtype=d.dtype; t.ndim=d.ndim; t.bytes=d.bytes; t.offset=d.offset;
        for(int i=0;i<4;i++) t.dims[i]=d.dims[i];
        t.device=static_cast<uint8_t*>(device_blob_)+d.offset;
        tensors_.emplace(std::move(name),t);
    }
}
const TensorView& Model::tensor(const std::string& name) const {
    auto it=tensors_.find(name); if(it==tensors_.end()) throw std::runtime_error("missing model tensor: "+name); return it->second;
}
bool Model::has(const std::string& name) const { return tensors_.find(name)!=tensors_.end(); }
}
