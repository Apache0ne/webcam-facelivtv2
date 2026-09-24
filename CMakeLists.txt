cmake_minimum_required(VERSION 3.24)
project(FaceLiVTv2Native LANGUAGES CXX)

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

if(WIN32)
  add_library(facelivt_video SHARED runtime/src/video_gpu.cpp)
  target_compile_definitions(facelivt_video PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
  target_link_libraries(facelivt_video PRIVATE CUDA::cudart d3d11 dxgi d3dcompiler
                        d2d1 dwrite mfplat mfreadwrite mfuuid mf ole32)

  option(FACELIVT_BUILD_NATIVE_DETECTOR "Build the native CUDA SCRFD attendance application" ON)
  if(FACELIVT_BUILD_NATIVE_DETECTOR)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
      if(CUDAToolkit_VERSION VERSION_LESS "13.0")
        set(_default_detector_arches "75;80;86;87;89;90;100;120")
      else()
        # CUDA 13 targets Turing and newer; older GPUs use the CUDA 12.x lane.
        set(_default_detector_arches "75;80;86;87;89;90;100;120")
      endif()
      set(CMAKE_CUDA_ARCHITECTURES "${_default_detector_arches}" CACHE STRING
          "CUDA architectures compiled into native detector kernels")
    endif()
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)

    set(FACELIVT_ONNXRUNTIME_VERSION "1.27.0" CACHE STRING
        "Microsoft ONNX Runtime CUDA package version (1.27.x for CUDA 13, 1.26.x for CUDA 12)")
    set(FACELIVT_ONNXRUNTIME_ROOT
        "${CMAKE_SOURCE_DIR}/build/deps/onnxruntime-gpu-windows-${FACELIVT_ONNXRUNTIME_VERSION}" CACHE PATH
        "Extracted Microsoft ONNX Runtime CUDA C++ package")
    find_path(FACELIVT_ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h
      HINTS "${FACELIVT_ONNXRUNTIME_ROOT}/buildTransitive/native/include" NO_DEFAULT_PATH)
    find_library(FACELIVT_ONNXRUNTIME_LIBRARY onnxruntime
      HINTS "${FACELIVT_ONNXRUNTIME_ROOT}/runtimes/win-x64/native" NO_DEFAULT_PATH)
    find_file(FACELIVT_ORT_RUNTIME_DLL onnxruntime.dll
      HINTS "${FACELIVT_ONNXRUNTIME_ROOT}/runtimes/win-x64/native" NO_DEFAULT_PATH)
    find_file(FACELIVT_ORT_CUDA_DLL onnxruntime_providers_cuda.dll
      HINTS "${FACELIVT_ONNXRUNTIME_ROOT}/runtimes/win-x64/native" NO_DEFAULT_PATH)
    find_file(FACELIVT_ORT_PROVIDERS_DLL onnxruntime_providers_shared.dll
      HINTS "${FACELIVT_ONNXRUNTIME_ROOT}/runtimes/win-x64/native" NO_DEFAULT_PATH)
    if(NOT FACELIVT_ONNXRUNTIME_INCLUDE_DIR OR NOT FACELIVT_ONNXRUNTIME_LIBRARY OR
       NOT FACELIVT_ORT_RUNTIME_DLL OR NOT FACELIVT_ORT_CUDA_DLL OR NOT FACELIVT_ORT_PROVIDERS_DLL)
      message(FATAL_ERROR
        "ONNX Runtime CUDA C++ files are missing. Run .\\tools\\fetch_onnxruntime.ps1 first.")
    endif()

    add_library(facelivt_scrfd STATIC runtime/src/scrfd.cpp runtime/src/scrfd_cuda.cu)
    target_include_directories(facelivt_scrfd PUBLIC runtime/include PRIVATE ${FACELIVT_ONNXRUNTIME_INCLUDE_DIR})
    target_compile_definitions(facelivt_scrfd PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
    target_link_libraries(facelivt_scrfd PUBLIC CUDA::cudart ${FACELIVT_ONNXRUNTIME_LIBRARY})
    target_compile_options(facelivt_scrfd PRIVATE
      $<$<COMPILE_LANGUAGE:CXX>:/O2> $<$<COMPILE_LANGUAGE:CXX>:/EHsc>)
    set_target_properties(facelivt_scrfd PROPERTIES CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
    add_custom_command(TARGET facelivt_scrfd POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different "${FACELIVT_ORT_RUNTIME_DLL}" "$<TARGET_FILE_DIR:facelivt_scrfd>"
      COMMAND ${CMAKE_COMMAND} -E copy_if_different "${FACELIVT_ORT_CUDA_DLL}" "$<TARGET_FILE_DIR:facelivt_scrfd>"
      COMMAND ${CMAKE_COMMAND} -E copy_if_different "${FACELIVT_ORT_PROVIDERS_DLL}" "$<TARGET_FILE_DIR:facelivt_scrfd>")

    add_executable(facelivt_scrfd_smoke examples/scrfd_smoke.cpp)
    target_link_libraries(facelivt_scrfd_smoke PRIVATE facelivt_scrfd)

    add_executable(facelivt_attendance WIN32 examples/native_attendance.cpp)
    target_compile_definitions(facelivt_attendance PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN UNICODE _UNICODE)
    target_link_libraries(facelivt_attendance PRIVATE facelivt_runtime facelivt_scrfd facelivt_video
                          bcrypt comctl32 shell32 user32)
  endif()
endif()
