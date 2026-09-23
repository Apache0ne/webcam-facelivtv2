# Build notes

The C++ inference runtime requires C++20, CUDA Toolkit libraries, and an NVIDIA
driver that supports the generated kernels. CMake 3.24 or newer is required.
Windows uses MSVC; Linux uses the supplied `build_linux.sh` script. The webcam
capture and D3D11 preview DLL are Windows-only.

Kernel generation has been smoke-tested on Windows with Python 3.12,
triton-windows 3.6.0.post26, CUDA Toolkit 13.3, Visual Studio 2026, and an RTX
5060 Laptop GPU. Triton emits cubins and a manifest under `generated/kernels`;
these are generated locally and are intentionally excluded from source control.
Regenerate the cubins and manifest together whenever the builder or Triton
version changes. Build on the target NVIDIA architecture when possible.

The converted `.flvt` file contains FP16 model tensors and is independent of
the GPU architecture. It can be distributed to avoid repeating checkpoint
conversion. The generated Triton cubins remain specific to the build target and
must be compiled locally for each supported GPU architecture.

`build_windows.ps1` and `build_linux.sh` both use an existing
`facelivtv2-l.fp16.flvt` when present, or convert `facelivtv2-l.pt` if the
converted file is absent. Passing a `.pt` path to either build script forces a
fresh conversion.

The webcam frontend also needs a CUDA-enabled PyTorch installation,
`onnxruntime-gpu`, NumPy, Pillow, and OpenCV in its Python environment. It
requires CUDA/cuDNN libraries compatible with the ONNX Runtime CUDA provider.
The app imports PyTorch before initializing ONNX Runtime to expose its CUDA
libraries on Windows. See `WEBCAM.md` for setup and camera requirements.

Windows targets default to the static MSVC runtime because the CUDA driver
library links against LIBCMT. Python is not needed at inference time for the
native C++ runtime; it is used to convert checkpoints, compile Triton kernels,
and run the optional webcam UI.
