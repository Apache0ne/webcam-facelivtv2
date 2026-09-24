# Windows release build

The published runtime is the native `facelivt_attendance.exe`. `run_webcam.ps1` is a thin launcher; it does not start Python. The package includes FaceLiVT `.flvt` weights, SCRFD ONNX weights, architecture-specific Triton cubins, ONNX Runtime, CUDA/cuDNN runtime DLLs, and the Media Foundation/D3D11 video DLL.

## Release lanes

- CUDA 13.x pairs with ONNX Runtime 1.27 and cuDNN 9.16; current native kernels target SM 7.5, 8.0, 8.6, 8.7, 8.9, 9.0, 10.0, and 12.0.
- The planned CUDA 12.x lane (12.8 or newer) pairs with ONNX Runtime 1.26 and cuDNN 9.16; native kernels additionally target SM 6.1 and 7.0. Its package builder is ready, but no CUDA 12 ZIP is published until the lane has been built and tested with CUDA Toolkit 12.8+.

ONNX Runtime documents CUDA 12.8-built packages as compatible with CUDA 12.x and CUDA 13-built packages with CUDA 13.x; cuDNN major versions must also match. Each archive is assembled from one CUDA lane and must not mix provider DLLs or cuDNN builds.

CUDA 13 no longer compiles device code for pre-Turing GPUs, so the CUDA 12 lane is the path for Pascal/Volta. The manifests allow the FaceLiVT runtime to select matching Triton cubins, while SCRFD custom CUDA code is built by CMake for the same architectures. Cross-compilation of cubins is not a substitute for testing on physical GPUs. Only the RTX 5060 Laptop / SM 12.0 live configuration has been exercised end-to-end to date.

## Developer tools

Builds require Windows, MSVC and a Windows SDK, CMake 3.24+, a matching CUDA Toolkit, Python 3.12, CUDA-enabled PyTorch, and `triton-windows`. Python runs only during checkpoint conversion or cubin generation. The release package contains none of the Python environment.

```powershell
.\setup_windows.ps1
.\download_scrfd.ps1
.\build_windows.ps1 -CudaMajor 13 -Python 'C:\path\to\python.exe'
.\package_release.ps1 -CudaMajor 13 -ReleaseVersion 0.1.0
```

The package script refuses mismatched toolkit builds, missing architecture entries, unverified cuDNN wheels, and missing runtime files. It stages an explicit allowlist, so `data/`, build logs, Python files, and source checkpoints are not included. Inspect every proposed artifact and license notice before a public release.

## Current validation boundary

Native SCRFD/CUDA inference, custom FaceLiVT CUDA inference, and webcam capture/display have run on the RTX 5060 Laptop. Repeat clean extraction and camera tests on additional Windows NVIDIA systems before describing other GPU families as verified. Similarity thresholds are trial values; model-output parity and recognition accuracy need separate validation.
