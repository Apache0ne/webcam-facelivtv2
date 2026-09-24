#include "facelivt/scrfd.hpp"
#include "facelivt/common.hpp"
#include <fstream>
#include <iostream>
#include <vector>

int main(int argc,char** argv){
    if(argc<3){
        std::cerr<<"usage: facelivt_scrfd_smoke detector.onnx frame.bgra [width=640] [height=480]\n";
        return 2;
    }
    const int width=argc>3?std::stoi(argv[3]):640;
    const int height=argc>4?std::stoi(argv[4]):480;
    if(width<=0 || height<=0) throw std::runtime_error("frame dimensions must be positive");
    const size_t bytes=static_cast<size_t>(width)*height*4;
    std::vector<uint8_t> host_frame(bytes);
    std::ifstream input(argv[2],std::ios::binary);
    if(!input || !input.read(reinterpret_cast<char*>(host_frame.data()),static_cast<std::streamsize>(bytes)))
        throw std::runtime_error("could not read the complete packed BGRA test frame");
    if(input.peek()!=std::ifstream::traits_type::eof())
        throw std::runtime_error("test frame has trailing bytes; expected width*height*4 bytes");

    uint8_t* device_frame=nullptr;
    facelivt::cuda_check(cudaMalloc(reinterpret_cast<void**>(&device_frame),bytes),"allocate test frame");
    try {
        facelivt::cuda_check(cudaMemcpy(device_frame,host_frame.data(),bytes,cudaMemcpyHostToDevice),"upload test frame");
        facelivt::ScrfdDetector detector(argv[1]);
        const auto faces=detector.detect(device_frame,width,height);
        std::cout<<"SCRFD CUDA: "<<faces.size()<<" face(s)\n";
        for(size_t i=0;i<faces.size();++i){
            const auto& face=faces[i];
            std::cout<<i<<" score="<<face.confidence<<" box="
                     <<face.box[0]<<","<<face.box[1]<<","<<face.box[2]<<","<<face.box[3]<<"\n";
        }
        if(!faces.empty()){
            uint8_t* crops=nullptr;
            facelivt::cuda_check(cudaMalloc(reinterpret_cast<void**>(&crops),faces.size()*112*112*3),"allocate aligned crop test output");
            try {
                detector.align(device_frame,width,height,faces,crops);
                facelivt::cuda_check(cudaStreamSynchronize(detector.stream()),"finish CUDA alignment");
            } catch(...) { cudaFree(crops); throw; }
            cudaFree(crops);
            std::cout<<"CUDA five-point alignment completed.\n";
        }
    } catch(...) {
        cudaFree(device_frame);
        throw;
    }
    cudaFree(device_frame);
    return 0;
}
