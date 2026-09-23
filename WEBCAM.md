# Live webcam attendance

The webcam app is currently supported on Windows with an NVIDIA CUDA GPU. It
uses Media Foundation and D3D11 for camera capture and display, ONNX Runtime's
CUDA provider for SCRFD-10G, and the native FaceLiVT CUDA runtime for recognition.
There is no CPU inference or CPU video fallback. The native C++ inference runtime
also has a Linux build script, but the webcam capture/display frontend is Windows
specific.

## Requirements

- Windows x64 with a working NVIDIA driver and CUDA-capable GPU.
- Visual Studio C++ Build Tools, CMake 3.24 or newer, and CUDA Toolkit for the
  native DLLs.
- Python 3.12 recommended for the tested native Windows Triton build, plus a
  CUDA-enabled PyTorch install, `triton-windows`, NumPy, Pillow, OpenCV,
  `onnxruntime-gpu`, and Tkinter.
- The x64 Microsoft Visual C++ 2015-2022 Redistributable, required by native
  Python/Triton libraries ([download](https://aka.ms/vs/17/release/vc_redist.x64.exe)).
- A camera that Media Foundation can expose as a D3D11 GPU surface in the
  app's 640x480 BGRA/RGB32 mode. Cameras that only provide CPU buffers will fail
  with an explicit error.

The verified development machine used Windows, an RTX 5060 Laptop GPU, CUDA
Toolkit 13.3, Visual Studio 2026, Python 3.12, PyTorch 2.11 with CUDA 12.8,
triton-windows 3.6.0.post26, and ONNX Runtime GPU 1.26.0. Other driver, GPU,
Python, and library combinations need local verification. PyTorch and ONNX Runtime
must load compatible CUDA and cuDNN libraries; the app imports PyTorch before
creating the ONNX Runtime session so its CUDA libraries are available on Windows.

The simplest route creates a project-local Python 3.12 environment and installs
the tested CUDA 12.8 PyTorch, Triton, and webcam packages:

```powershell
.\setup_windows.ps1
```

The installer needs the Python 3.12 launcher (`py`) and internet access. To use
an existing environment instead, pass its Python executable and environment
directory. `-SkipTorchInstall` preserves its already-installed CUDA PyTorch:

```powershell
.\setup_windows.ps1 -Python 'C:\path\to\python.exe' -EnvironmentPath 'C:\path\to\existing-venv' -SkipTorchInstall
```

Pass the same executable to the build and run scripts when reusing that
environment:

```powershell
.\build_windows.ps1 -Python 'C:\path\to\python.exe'
.\run_webcam.ps1 -Python 'C:\path\to\python.exe'
```

Tkinter is included with most Windows Python installers. `triton-windows` is
needed to compile GPU kernels; it is not needed for inference after building.
Direct dependency versions are pinned in `requirements-build.txt` and
`requirements-webcam.txt` to match the verified Windows setup.

## Model files and build

The native runtime loads the converted `facelivtv2-l.fp16.flvt` weight
container. The repository includes this file so users can skip converting the
original `.pt` checkpoint. It is not tied to a GPU architecture. If omitted,
the build script can create it from a local `facelivtv2-l.pt`. Users still
compile Triton kernels for their own GPU and build the native DLLs locally.
The ignore rules keep raw `.pt` files out to avoid duplicating the weights.
Confirm the applicable redistribution terms for the converted model before
publishing it. SCRFD is
downloaded separately because its public pretrained weights have different,
non-commercial research terms described in [THIRD_PARTY.md](THIRD_PARTY.md).
Keep personal attendance data out of the repository.

To force regenerating the converted weights from the `.pt` instead of using an
existing `.flvt` file, run `.\build_windows.ps1 -Model .\facelivtv2-l.pt`.

The FaceLiVT model is already in the project root. Download the separately
licensed SCRFD detector with the helper, which verifies its SHA-256 before
keeping the file:

```powershell
.\download_scrfd.ps1
```

Read [THIRD_PARTY.md](THIRD_PARTY.md) for the SCRFD weight terms before use.

Then build the runtime from the project directory. The script defaults to the
new `.venv` and bundled checkpoint; it checks CMake, MSVC, CUDA Toolkit, Python,
and CUDA GPU visibility before compiling:

```powershell
.\build_windows.ps1
```

The script uses the shared `.flvt` weights if present; otherwise it converts
the `.pt` checkpoint. It then compiles Triton kernels for the NVIDIA GPU
available on that build machine and builds the native targets. Build the
kernels on the GPU architecture you intend to run for best compatibility and
performance. Kernel files are local build outputs; the `.flvt` weight container
can be shared with the source code.

## Run

First check CUDA-to-D3D11 display without opening the camera:

```powershell
.\.venv\Scripts\python.exe .\webcam_app.py --video-self-test
```

Close that window, then launch the live app:

```powershell
.\run_webcam.ps1 -Camera 0
```

Use `-Camera 1` for another device. The app fails rather than silently switching
to CPU if the camera does not provide the required GPU surface. The camera's
enumeration order may differ across systems.

Enter a name and click Enroll with only that person in view. The app collects
eight samples. It recognizes up to eight faces per processed frame. Matching
uses a trial cosine threshold of 0.55, a runner-up margin of 0.08, and four
consecutive observations over at least 0.4 seconds; these settings are not
calibrated accuracy guarantees or liveness checks.

## Local data and validation

`data/` contains `enrollments.json`, daily attendance JSON files, a lock file,
and `webcam.log`. Enrollment records contain names and face embeddings, which
are biometric data even though the app does not save face images. Keep this
directory private. Writes are atomic and only one app can use a data directory
at a time. Invalid JSON or a model fingerprint mismatch stops the app without
replacing the existing records.

Run the webcam and native integration checks with:

```powershell
& .\.venv\Scripts\python.exe -m unittest discover -s tests -p test_webcam.py -v
```

The native integration checks require the built DLLs, generated kernels, model
file, and a CUDA GPU. The webcam test itself requires a usable camera and should
be performed interactively.
