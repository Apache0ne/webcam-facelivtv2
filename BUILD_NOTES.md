# Windows release build

Each CUDA lane produces a small app ZIP and a large runtime ZIP. The app ZIP contains the FaceLiVT native executables and video DLL, both model files, architecture cubins and manifest, notices, and app-local x64 Microsoft C++ runtime DLLs. The runtime ZIP contains the lane's CUDA runtime DLLs, cuDNN, ONNX Runtime CUDA provider, and their notices. Runtime bundle revisions are independent from app versions (currently `r1.0.0`), and should be bumped only when those bundled dependency files change. Extract both archives into the same directory for a first install; an app-only update does not need a new runtime ZIP when its revision is unchanged.

## Release lanes

- CUDA 13.x uses ONNX Runtime 1.27 and cuDNN 9.16; CUDA 12.x uses ONNX Runtime 1.26 and cuDNN 9.16. Both lanes compile native kernels for SM 7.5, 8.0, 8.6, 8.7, 8.9, 9.0, 10.0, and 12.0.
- cuDNN 9.16 requires compute capability 7.5 or newer for both lanes; Pascal (6.1) and Volta (7.0) GPUs cannot run the bundled SCRFD detector runtime.

The app and runtime ZIPs must use the same CUDA lane and release version. Do not mix CUDA 12 and CUDA 13 files. The CUDA 13 lane was validated end-to-end on an RTX 5060 Laptop, but that does not verify every listed architecture. Cross-compiling kernels on a CPU runner confirms build coverage, not physical GPU inference.

## Developer and CI builds

Developer builds use Windows, Visual Studio 2022 C++ Build Tools and a Windows SDK, CMake 3.24+, the matching CUDA Toolkit, Python 3.12, CUDA-enabled PyTorch, and `triton-windows`. Python runs only to convert a checkpoint or compile cubins; no Python tools ship in the app/runtime ZIPs.

```powershell
.\setup_windows.ps1
.\download_scrfd.ps1
.\build_windows.ps1 -CudaMajor 13 -Python 'C:\path\to\python.exe'
.\package_release.ps1 -CudaMajor 13 -ReleaseVersion 0.1.2
.\tests\verify_split_release.ps1 -CudaMajor 13 -ReleaseVersion 0.1.2
```

`.github/workflows/windows-release.yml` builds both CUDA lanes on GitHub-hosted Windows CPU runners when a `v*` tag is pushed. The workflow installs the matching CUDA compiler libraries, uses `setup_windows.ps1 -AllowCpuBuild` to cross-compile cubins without a GPU, and runs a package layout and SHA-256 check. It skips inference tests because those require an NVIDIA GPU. Webcam and GPU smoke tests still need a physical GPU system.

`.github/workflows/linux-build.yml` checks the portable C++ FaceLiVT core on Ubuntu for CUDA 12 and 13 on pushes and pull requests. `build_linux.sh` builds the embedding, matching, and benchmark examples and the architecture cubins. The Linux target does not include the Windows-only webcam attendance window or SCRFD Media Foundation/D3D11 path.

`package_release.ps1` rejects mismatched toolkits, incomplete architecture manifests, unverified cuDNN wheels, and missing dependency DLLs. It stages an explicit allowlist, so `data/`, build logs, Python files, and source checkpoints are not included.

## Current validation boundary

Native SCRFD/CUDA inference, custom FaceLiVT CUDA inference, and webcam capture/display have run on the RTX 5060 Laptop with CUDA 13. New CI release builds establish that both code lanes compile and are packaged consistently; they do not execute GPU inference. Similarity thresholds remain trial values, and model-output parity and recognition accuracy need separate validation.
