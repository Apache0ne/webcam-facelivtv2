#include "facelivt/runtime.hpp"
#include <fstream>
#include <iostream>
#include <vector>

// Minimal matcher example. Thresholds must be calibrated on your own enrollment
// and walk-by data; this example deliberately does not hard-code one.
int main(int argc,char** argv){
    if(argc!=5){
        std::cerr<<"usage: attendance_match model.flvt manifest.tsv gallery.f32 face112.bgr\n";
        return 2;
    }
    try {
        std::ifstream gf(argv[3],std::ios::binary|std::ios::ate);
        auto bytes=gf.tellg(); gf.seekg(0);
        if(bytes<=0 || (static_cast<size_t>(bytes)%(512*sizeof(float)))!=0) throw std::runtime_error("gallery.f32 must contain N*512 float32 values");
        int n=static_cast<int>(static_cast<size_t>(bytes)/(512*sizeof(float)));
        std::vector<float> gallery(static_cast<size_t>(n)*512); gf.read(reinterpret_cast<char*>(gallery.data()),bytes);
        std::vector<uint8_t> face(112*112*3); std::ifstream ff(argv[4],std::ios::binary); ff.read(reinterpret_cast<char*>(face.data()),face.size());
        if(static_cast<size_t>(ff.gcount())!=face.size()) throw std::runtime_error("face must be raw 112x112x3 BGR");
        facelivt::Runtime rt(argv[1],argv[2]);
        rt.set_gallery(gallery.data(),n);
        facelivt::Match m;
        rt.infer_and_match_host(face.data(),1,&m,true);
        std::cout<<"best_index="<<m.best_index<<" best_score="<<m.best_score
                 <<" second_index="<<m.second_index<<" second_score="<<m.second_score
                 <<" margin="<<(m.best_score-m.second_score)<<"\n";
    } catch(const std::exception& e){ std::cerr<<"error: "<<e.what()<<"\n"; return 1; }
}
