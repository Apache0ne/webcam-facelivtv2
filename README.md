# FaceLiVTv2 Native CUDA Attendance

Windows webcam attendance built around one native C++20/CUDA runtime. SCRFD-10G face detection and five-point alignment run through ONNX Runtime's CUDA provider; FaceLiVTv2 embeddings and attendance matching use the project's custom CUDA kernels. Webcam capture and preview use Media Foundation, D3D11, and CUDA GPU surfaces.

## Run a release package

Download the currently published CUDA 13 Windows x64 ZIP from the repository's Releases page, extract the complete ZIP, then run `run_webcam.ps1` or double-click `facelivt_attendance.exe`. The app includes both model files, architecture-specific kernels, ONNX Runtime, cuDNN, and the CUDA runtime DLLs. It does not need Python, PyTorch, Triton, a CUDA Toolkit, or WSL at runtime. A CUDA 12 package is prepared by the same build scripts but still needs a matching CUDA 12 native build before it can be released.

You need a compatible NVIDIA GPU/driver and the Microsoft Visual C++ 2015–2022 x64 Redistributable. The CUDA 13 package targets Turing and newer (SM 7.5+). Camera availability and supported GPU surfaces depend on the Windows driver and camera.

The webcam image is processed locally. The app creates `data/` for enrollment embeddings and daily attendance records, and `logs/native-attendance.log` for diagnostics. Those files are personal data: keep them private and do not commit or share them.

Read [WEBCAM.md](WEBCAM.md) for use and troubleshooting, and [THIRD_PARTY.md](THIRD_PARTY.md) for model and bundled-library terms. SCRFD pretrained weights are provided under non-commercial research terms.

## Build from source (developers)

Ordinary users should use a release ZIP. Developers who need to change native code or generate a different model can use Windows, Visual Studio C++ Build Tools, CMake 3.24+, the matching CUDA Toolkit, and Python 3.12 with CUDA-enabled PyTorch and Triton. The Python environment is only for converting a supplied checkpoint and compiling the project's architecture-specific Triton cubins; the packaged runtime remains native.

```powershell
# Optional: prepare a project-local developer environment.
.\setup_windows.ps1

# Download the separately licensed SCRFD detector weights.
.\download_scrfd.ps1

# Build the CUDA 13 lane (or use -CudaMajor 12 with CUDA Toolkit 12.8+).
.\build_windows.ps1 -CudaMajor 13

# Assemble a self-contained package.
.\package_release.ps1 -CudaMajor 13 -ReleaseVersion 0.1.0
```

Use an existing Python environment with `-Python 'C:\path\to\python.exe'`. To build a CUDA 12 lane, install CUDA Toolkit 12.8 or newer and pass `-CudaMajor 12`; each CUDA lane uses its matching ONNX Runtime and cuDNN. Build scripts never install WSL or copy the project venv into the release.

`-KernelArch` can select architectures explicitly, for example `-KernelArch 75,80,86,89,90,100,120`. Package builds validate the kernel manifest and native CUDA architecture list before writing the ZIP. Run the app from the extracted package to test the webcam; its **Benchmark GPU** button measures inference throughput on a replayed frame independently of camera FPS.

## Project layout

- `runtime/`, `examples/native_attendance.cpp`: native FaceLiVT runtime, CUDA SCRFD and attendance application.
- `kernels/`, `tools/build_kernels.py`: Triton kernel source and developer-time cubin compilation only.
- `converter/`: developer-time source-checkpoint conversion to the `.flvt` runtime format.
- `package_release.ps1`: stages only runtime binaries, required DLLs, models, kernels, and notices; never copies personal data.
- `third_party/`: upstream code-license reference.
