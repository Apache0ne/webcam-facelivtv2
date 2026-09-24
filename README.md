# FaceLiVTv2 Native CUDA Attendance

Windows webcam attendance built around one native C++20/CUDA runtime. SCRFD-10G face detection and five-point alignment run through ONNX Runtime's CUDA provider; FaceLiVTv2 embeddings and attendance matching use the project's custom CUDA kernels. Webcam capture and preview use Media Foundation, D3D11, and CUDA GPU surfaces.

## Run a release package

For a first install, download a matching CUDA 12 or CUDA 13 pair from Releases:

1. Extract the `facelivt-runtime-windows-cudaXX` ZIP into a new folder.
2. Extract the matching `facelivt-app-windows-cudaXX` ZIP into that same folder.
3. Double-click `facelivt_attendance.exe`.

The smaller app ZIP contains the FaceLiVT and SCRFD-10G models, native app, architecture kernels, and app-local Microsoft C++ runtime files. The larger runtime ZIP contains the CUDA, cuDNN, and ONNX Runtime DLLs for that lane. Runtime bundles have their own revision, currently `r1.0.0`, separate from the app version; app-only updates do not require downloading that large ZIP again while the required runtime revision stays the same. Do not mix CUDA 12 and CUDA 13 ZIPs. A compatible NVIDIA display driver and webcam are required; Python, PyTorch, Triton, a CUDA Toolkit, WSL, and a separate Visual C++ Redistributable are not needed to run the app.

Both CUDA lanes contain kernels for SM 7.5, 8.0, 8.6, 8.7, 8.9, 9.0, 10.0, and 12.0. This includes GeForce RTX 20, 30, 40, and 50 series models with one of those compute capabilities, plus selected workstation and data-center GPUs. The bundled cuDNN detector runtime requires compute capability 7.5 or newer, so Pascal (6.1) and Volta (7.0) GPUs are not supported. The listed kernels have not all been individually tested on physical GPUs; the CUDA 13 lane has been validated on an RTX 5060 Laptop GPU. Check [NVIDIA's compute capability list](https://developer.nvidia.com/cuda/gpus) for a specific card. Camera availability and GPU surface support depend on the Windows driver and camera.

The webcam image is processed locally. The app creates `data/` for enrollment embeddings and daily attendance records, and `logs/native-attendance.log` for diagnostics. Those files are personal data: keep them private and do not commit or share them.

Read [WEBCAM.md](WEBCAM.md) for use and troubleshooting, and [THIRD_PARTY.md](THIRD_PARTY.md) for model and bundled-library terms. SCRFD pretrained weights are provided under non-commercial research terms.

## Build from source (developers)

Ordinary users should download the matching app/runtime ZIP pair. Developers who need to change native code or generate a different model can use Windows, Visual Studio 2022 C++ Build Tools, CMake 3.24+, the matching CUDA Toolkit, and Python 3.12 with CUDA-enabled PyTorch and Triton. Python is only used to convert a supplied checkpoint and compile architecture-specific cubins; the packaged app remains native.

```powershell
# Optional: prepare a project-local developer environment.
.\setup_windows.ps1

# Download the separately licensed SCRFD detector weights.
.\download_scrfd.ps1

# The Python environment is build-only. CPU-only CI can pass -AllowCpuBuild.
.\build_windows.ps1 -CudaMajor 13

# Assemble the smaller app ZIP and its matching runtime ZIP.
.\package_release.ps1 -CudaMajor 13 -ReleaseVersion 0.1.2
```

Use an existing Python environment with `-Python 'C:\path\to\python.exe'`. To build a CUDA 12 lane, install CUDA Toolkit 12.8 or newer and pass `-CudaMajor 12`; each lane selects a matching ONNX Runtime and cuDNN package. Build scripts never install WSL or copy the project venv into a release.

`-KernelArch` can select architectures explicitly, for example `-KernelArch 75,80,86,89,90,100,120`. Package builds validate the kernel manifest and native CUDA architecture list before writing the ZIP. Run the app from the extracted package to test the webcam; its **Benchmark GPU** button measures inference throughput on a replayed frame independently of camera FPS.

## Project layout

- `runtime/`, `examples/native_attendance.cpp`: native FaceLiVT runtime, CUDA SCRFD and attendance application.
- `kernels/`, `tools/build_kernels.py`: Triton kernel source and developer-time cubin compilation only.
- `converter/`: developer-time source-checkpoint conversion to the `.flvt` runtime format.
- `package_release.ps1`: creates separate app and large runtime ZIPs for the selected CUDA lane; never copies personal data.
- `.github/workflows/windows-release.yml`: builds both CUDA lanes on GitHub-hosted Windows CPU runners and publishes four ZIP assets when a version tag is pushed. CI compiles GPU code but cannot run GPU inference or webcam tests.
- `third_party/`: upstream code-license reference.
