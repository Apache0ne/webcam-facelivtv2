#include "facelivt/runtime.hpp"
#include <fstream>
#include <iostream>
#include <vector>

int main(int argc,char** argv){
    if(argc!=5){
        std::cerr<<"usage: facelivt_embed model.flvt kernel_manifest.tsv face112.bgr output.f32\n";
        return 2;
    }
    try {
        constexpr size_t N=112*112*3;
        std::vector<uint8_t> face(N);
        std::ifstream in(argv[3],std::ios::binary);
        in.read(reinterpret_cast<char*>(face.data()),static_cast<std::streamsize>(face.size()));
        if(static_cast<size_t>(in.gcount())!=face.size()) throw std::runtime_error("face file must be exactly 112*112*3 bytes");
        facelivt::Runtime rt(argv[1],argv[2]);
        std::vector<float> emb(512);
        rt.infer_host(face.data(),1,emb.data(),true,true);
        std::ofstream out(argv[4],std::ios::binary);
        out.write(reinterpret_cast<const char*>(emb.data()),static_cast<std::streamsize>(emb.size()*sizeof(float)));
        if(!out) throw std::runtime_error("failed writing output embedding");
        std::cout<<"wrote 512 normalized float32 values to "<<argv[4]<<"\n";
    } catch(const std::exception& e){ std::cerr<<"error: "<<e.what()<<"\n"; return 1; }
    return 0;
}
