# Native Windows webcam attendance

The release app is a standalone C++/CUDA Windows x64 program. Detection, five-point alignment, FaceLiVT inference, matching, and the video preview use CUDA/GPU paths. The app ZIP and its matching CUDA runtime ZIP together contain the models and all required runtime DLLs. No Python environment, WSL, CUDA Toolkit, OpenCV package, or compiler is needed to run it.

## Start the app

1. Download the matching CUDA 12 or CUDA 13 app ZIP and runtime ZIP from Releases.
2. Extract both ZIPs into the same folder.
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

Both CUDA lanes contain kernels for SM 7.5, 8.0, 8.6, 8.7, 8.9, 9.0, 10.0, and 12.0. The bundled cuDNN detector runtime requires compute capability 7.5 or newer, so Pascal (6.1) and Volta (7.0) GPUs are not supported by these packages. The listed architecture kernels are not all individually tested on physical GPUs. The only end-to-end webcam validation so far is on an RTX 5060 Laptop GPU using CUDA 13. SM 8.7 cubins are included, but this Windows x64 package is not a Jetson release. ONNX Runtime 1.27 is used by the CUDA 13 lane; ONNX Runtime 1.26 is used by CUDA 12. A toolkit install is needed only to build from source, never to run a published ZIP.

## Data and diagnostics

The app stores `data/enrollments.json`, daily attendance JSON, and a lock file beside the executable. Enrollment files contain names and face embeddings. Keep `data/` private; it is excluded from Git and release packages. Logs are written to `logs/native-attendance.log` beside the app.

If startup fails, check the message in the app and inspect the log. Confirm both matching ZIPs were extracted into the same folder, the NVIDIA driver is current enough for that lane, and Windows has access to the webcam. For detector inference problems, see the model and runtime notices in [THIRD_PARTY.md](THIRD_PARTY.md).

## Build and validate from source

Python is only a developer tool for source checkpoint conversion and Triton cubin generation. `setup_windows.ps1` prepares those tools; `build_windows.ps1` compiles the C++/CUDA application and stages no Python into the package. After building and downloading SCRFD weights, create a ZIP with `package_release.ps1`.

```powershell
.\setup_windows.ps1
.\download_scrfd.ps1
.\build_windows.ps1 -CudaMajor 13
.\package_release.ps1 -CudaMajor 13 -ReleaseVersion 0.1.2
```

To use an existing Python environment, pass its `python.exe` to both scripts. CUDA 12 builds require a CUDA 12.8+ Toolkit; CUDA 13 builds require a CUDA 13.x Toolkit. Run `facelivt_scrfd_smoke.exe` and `facelivt_embed.exe` from the built or staged folder to verify both native GPU inference components without launching the camera.
