# Native Windows webcam attendance

The release app is a standalone C++/CUDA Windows x64 program. Detection, five-point alignment, FaceLiVT inference, matching, and the video preview use CUDA/GPU paths. The release ZIP contains the model files and matching CUDA, cuDNN, and ONNX Runtime DLLs. No Python environment, WSL, CUDA Toolkit, OpenCV package, or compiler is needed to run it.

## Start the app

1. Download the CUDA 13 Windows x64 ZIP from the Releases page.
2. Unzip the complete archive.
3. Double-click `facelivt_attendance.exe`.

From PowerShell, you can also run:

```powershell
.\run_webcam.ps1
```

Choose another camera with `-Camera 1`. Add `-Benchmark` to benchmark on the first live frame after startup:

```powershell
.\run_webcam.ps1 -Camera 0 -Benchmark
```

The window also has a **Benchmark GPU** button. It replays a captured frame through SCRFD and FaceLiVT without waiting for another camera frame, so the result separates inference throughput from the camera's own FPS. The live preview cannot exceed the source camera's frame rate.

The app asks for one name, then collects eight clear enrollment samples while only that person is in view. Matching and attendance are local. The trial threshold is not an accuracy guarantee or a liveness check; calibrate it for the intended environment.

## Requirements and supported GPUs

- Windows 10/11 x64 and a compatible NVIDIA driver.
- NVIDIA GPU architecture supported by the selected release asset.
- The package bundles its x64 Visual C++ runtime DLLs; no separate redistributable installation is needed.
- A webcam exposed by Media Foundation. The current preview requests a 640x480 GPU surface; unsupported camera/driver formats produce an error instead of switching inference to CPU.

The CUDA 13 package contains kernels for SM 7.5, 8.0, 8.6, 8.7, 8.9, 9.0, 10.0, and 12.0. These cover GeForce RTX 20/30/40/50 series and selected workstation/data-center GPUs; GTX 10-series and older cards are not covered by this ZIP. The only end-to-end webcam validation so far is on an RTX 5060 Laptop GPU. SM 8.7 cubins are included, but this Windows x64 package is not a Jetson release. The source build supports a future CUDA 12 lane for Pascal and newer (SM 6.1+), but that ZIP is not published yet because a matching CUDA 12 build and test are still needed. ONNX Runtime 1.27 is the CUDA 13 lane; ONNX Runtime 1.26 is the CUDA 12 lane. A toolkit install is needed only to build from source, never to run a published ZIP.

## Data and diagnostics

The app stores `data/enrollments.json`, daily attendance JSON, and a lock file beside the executable. Enrollment files contain names and face embeddings. Keep `data/` private; it is excluded from Git and release packages. Logs are written to `logs/native-attendance.log` beside the app.

If startup fails, check the message in the app and inspect the log. Confirm the ZIP was completely extracted, the matching CUDA lane was downloaded, the NVIDIA driver is current enough for that lane, and the Visual C++ redistributable is installed. For detector inference problems, see the model and runtime notices in [THIRD_PARTY.md](THIRD_PARTY.md).

## Build and validate from source

Python is only a developer tool for source checkpoint conversion and Triton cubin generation. `setup_windows.ps1` prepares those tools; `build_windows.ps1` compiles the C++/CUDA application and stages no Python into the package. After building and downloading SCRFD weights, create a ZIP with `package_release.ps1`.

```powershell
.\setup_windows.ps1
.\download_scrfd.ps1
.\build_windows.ps1 -CudaMajor 13
.\package_release.ps1 -CudaMajor 13 -ReleaseVersion 0.1.1
```

To use an existing Python environment, pass its `python.exe` to both scripts. CUDA 12 builds require a CUDA 12.8+ Toolkit; CUDA 13 builds require a CUDA 13.x Toolkit. Run `facelivt_scrfd_smoke.exe` and `facelivt_embed.exe` from the built or staged folder to verify both native GPU inference components without launching the camera.
