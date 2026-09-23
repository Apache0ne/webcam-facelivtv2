#include "facelivt/microbatcher.hpp"
#include <cstring>
#include <algorithm>
#include <exception>
#include <stdexcept>

namespace facelivt {
MicroBatcher::MicroBatcher(Runtime& runtime,int max_batch,std::chrono::microseconds window,bool bgr)
    :runtime_(runtime),max_batch_(max_batch),window_(window),bgr_(bgr){
    if(max_batch_<1 || max_batch_>runtime_.max_batch()) throw std::runtime_error("invalid microbatch max_batch");
    thread_=std::thread(&MicroBatcher::worker,this);
}
MicroBatcher::~MicroBatcher(){ stop(); }
void MicroBatcher::stop(){
    {
        std::lock_guard<std::mutex> g(mu_);
        if(stopping_) return;
        stopping_=true;
    }
    cv_.notify_all();
    if(thread_.joinable()) thread_.join();
}
std::future<MicroBatcher::Embedding> MicroBatcher::submit(const uint8_t* face){
    if(!face) throw std::runtime_error("null face crop");
    auto r=std::make_unique<Request>();
    std::memcpy(r->face.data(),face,r->face.size());
    auto fut=r->promise.get_future();
    {
        std::lock_guard<std::mutex> g(mu_);
        if(stopping_) throw std::runtime_error("microbatcher stopped");
        queue_.push_back(std::move(r));
    }
    cv_.notify_one();
    return fut;
}
void MicroBatcher::worker(){
    while(true){
        std::vector<std::unique_ptr<Request>> batch;
        {
            std::unique_lock<std::mutex> lk(mu_);
            cv_.wait(lk,[&]{return stopping_ || !queue_.empty();});
            if(stopping_ && queue_.empty()) return;
            auto deadline=std::chrono::steady_clock::now()+window_;
            while(!stopping_ && static_cast<int>(queue_.size())<max_batch_){
                if(cv_.wait_until(lk,deadline)==std::cv_status::timeout) break;
            }
            int n=std::min<int>(max_batch_,static_cast<int>(queue_.size()));
            batch.reserve(n);
            for(int i=0;i<n;i++){ batch.push_back(std::move(queue_.front())); queue_.pop_front(); }
        }
        std::vector<uint8_t> packed(batch.size()*112*112*3);
        for(size_t i=0;i<batch.size();i++)
            std::memcpy(packed.data()+i*112*112*3,batch[i]->face.data(),112*112*3);
        std::vector<float> out(batch.size()*512);
        try {
            runtime_.infer_host(packed.data(),static_cast<int>(batch.size()),out.data(),bgr_,true);
            for(size_t i=0;i<batch.size();i++){
                Embedding e{};
                std::memcpy(e.data(),out.data()+i*512,512*sizeof(float));
                batch[i]->promise.set_value(std::move(e));
            }
        } catch(...) {
            auto ep=std::current_exception();
            for(auto& r:batch) r->promise.set_exception(ep);
        }
    }
}
}
