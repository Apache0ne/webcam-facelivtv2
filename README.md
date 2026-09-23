# FaceLiVTv2-L C++ / Triton CUDA runtime

Purpose-built inference runtime for the official `facelivtv2_l` 112x112 FaceLiVTv2 checkpoint.

The native inference runtime builds on Windows and Linux with an NVIDIA CUDA
GPU. The optional live attendance application currently supports Windows only
because camera capture and preview use Media Foundation and D3D11. It requires
locally generated GPU kernels. The repository shares the smaller converted
`facelivtv2-l.fp16.flvt` weights so users can skip checkpoint conversion. The
build script can still create that file from a local `facelivtv2-l.pt` if the
converted file is absent; raw `.pt` files stay ignored to avoid duplicating the
weights. Confirm the model weights' redistribution rights before publishing.
The SCRFD detector weights are downloaded separately under their own terms. See
[WEBCAM.md](WEBCAM.md) for the Windows setup and run steps, and
[THIRD_PARTY.md](THIRD_PARTY.md) for model licensing notes.

## Windows quick start

On Windows x64 with an NVIDIA GPU, C++ Build Tools, CMake, and CUDA Toolkit, run
these from PowerShell in the project folder. The detector download is separate
because its pretrained weights have different licensing terms. Review
`THIRD_PARTY.md` before running the downloader:

```powershell
.\setup_windows.ps1
.\download_scrfd.ps1
.\build_windows.ps1
.\run_webcam.ps1
```

The setup script creates `.venv` and installs the tested Python/CUDA packages.
The build script uses shared converted weights when available, otherwise
converts the `.pt` checkpoint, then compiles kernels for the GPU in that
machine. `WEBCAM.md` includes instructions for using an existing Python
environment and obtaining the detector file.

## Design

- final application is C++20 + CUDA Driver/Runtime only
- Triton is used at build time to compile custom CUDA kernels into cubins
- no PyTorch, Python, ONNX Runtime, TensorRT, timm, or LibTorch in deployment
- NHWC FP16 activations
- FP32 accumulation in Tensor-Core `tl.dot` pointwise kernels
- official structural reparameterization is performed by the converter
- stage RepConv residual identities are folded into depthwise weights
- BatchNorm is folded into Conv/Linear weights
- one fused MHLA kernel handles affine + 4 spatial linears + layer-scale + residual
- fixed-shape CUDA graphs are captured per batch size
- optional GPU gallery cosine scoring

## Build order

1. Optionally convert the original checkpoint if you do not already have the
   converted weight file:

```bash
python converter/convert_facelivtv2_l.py \
  --checkpoint facelivtv2-l.pt \
  --output facelivtv2-l.fp16.flvt
```

2. Compile Triton kernels for the GPU present on the build machine:

```bash
python tools/build_kernels.py --out generated/kernels
```

This writes `generated/kernels/kernel_manifest.tsv` and cubins.

3. Build C++:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j
```

4. Runtime usage is demonstrated in `examples/embed.cpp`.

## Input contract

The C++ API accepts a packed batch of already-aligned 112x112 8-bit 3-channel face crops. The preprocessing kernel converts RGB/BGR bytes to FP16 NHWC and applies the official normalization:

`x = x / 127.5 - 1.0`

The output is 512-dimensional. Optional L2 normalization is performed on GPU.

## Status

Native Windows smoke tests and live webcam attendance have run on an RTX 5060
Laptop GPU with CUDA Toolkit 13.3, Visual Studio 2026, and triton-windows
3.6.0.post26. All 31 kernels compiled, and the webcam showed GPU detection,
recognition, camera capture, and D3D11 preview. This is not yet a clean-machine
portability guarantee; generated kernels are built for the local GPU, and model
parity/recognition accuracy still need broader validation. See `TODO.md` and
`WEBCAM.md` for details.
