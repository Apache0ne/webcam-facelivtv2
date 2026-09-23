cmake_minimum_required(VERSION 3.24)
project(facelivt_triton_runtime LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# The Windows CUDA driver library links LIBCMT. Use the same CRT in our targets
# unless the caller explicitly selects a different runtime.
if(MSVC AND NOT DEFINED CMAKE_MSVC_RUNTIME_LIBRARY)
  set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()
find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)

add_library(facelivt_runtime STATIC
    runtime/src/model.cpp
    runtime/src/kernel_registry.cpp
    runtime/src/runtime.cpp
    runtime/src/gallery.cpp
    runtime/src/microbatcher.cpp
)
target_include_directories(facelivt_runtime PUBLIC runtime/include)
target_link_libraries(facelivt_runtime PUBLIC CUDA::cudart CUDA::cuda_driver Threads::Threads)
if (MSVC)
  target_compile_options(facelivt_runtime PRIVATE /O2 /EHsc)
else()
  target_compile_options(facelivt_runtime PRIVATE -O3)
endif()

add_executable(facelivt_embed examples/embed.cpp)
target_link_libraries(facelivt_embed PRIVATE facelivt_runtime)

add_executable(facelivt_bench bench/bench.cpp)
target_link_libraries(facelivt_bench PRIVATE facelivt_runtime)

add_executable(facelivt_match examples/attendance_match.cpp)
target_link_libraries(facelivt_match PRIVATE facelivt_runtime)

# Persistent native runtime for the optional Python webcam interface.
set_target_properties(facelivt_runtime PROPERTIES POSITION_INDEPENDENT_CODE ON)
add_library(facelivt_native SHARED runtime/src/c_api.cpp)
target_link_libraries(facelivt_native PRIVATE facelivt_runtime)

if(WIN32)
  add_library(facelivt_video SHARED runtime/src/video_gpu.cpp)
  target_compile_definitions(facelivt_video PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
  target_link_libraries(facelivt_video PRIVATE CUDA::cudart d3d11 dxgi d3dcompiler
                        d2d1 dwrite mfplat mfreadwrite mfuuid mf ole32)
endif()
