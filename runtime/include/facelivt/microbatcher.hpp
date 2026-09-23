#pragma once
#include "runtime.hpp"
#include <array>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <future>
#include <mutex>
#include <memory>
#include <thread>
#include <vector>

namespace facelivt {
class MicroBatcher {
public:
    using Embedding=std::array<float,512>;
    MicroBatcher(Runtime& runtime,int max_batch=16,std::chrono::microseconds window=std::chrono::microseconds(1500),bool bgr=true);
    ~MicroBatcher();
    MicroBatcher(const MicroBatcher&)=delete;
    MicroBatcher& operator=(const MicroBatcher&)=delete;

    std::future<Embedding> submit(const uint8_t* face112x112x3);
    void stop();
private:
    struct Request {
        std::array<uint8_t,112*112*3> face{};
        std::promise<Embedding> promise;
    };
    void worker();
    Runtime& runtime_;
    int max_batch_;
    std::chrono::microseconds window_;
    bool bgr_;
    std::mutex mu_;
    std::condition_variable cv_;
    std::deque<std::unique_ptr<Request>> queue_;
    bool stopping_=false;
    std::thread thread_;
};
}
