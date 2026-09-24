# Third-party code, models, and runtime libraries

Read these terms before redistributing the release or using it commercially. A code license does not automatically apply to model weights.

## FaceLiVTv2

The C++ runtime and converter implement the FaceLiVTv2-L architecture supplied for this project. The upstream BSD-3-Clause code license is reproduced in [`third_party/FaceLiVT_LICENSE_BSD3.txt`](third_party/FaceLiVT_LICENSE_BSD3.txt).

`facelivtv2-l.fp16.flvt` is a converted checkpoint artifact supplied for this project. The upstream code license does not by itself establish rights for the checkpoint or converted weights. Confirm the checkpoint's terms before redistribution or commercial use.

## SCRFD-10G detector

The detector ONNX file is downloaded separately as `models/scrfd_10g_bnkps.onnx`; the download script verifies its SHA-256. SCRFD source and pretrained-model terms are published by [InsightFace](https://github.com/deepinsight/insightface). InsightFace states that its public pretrained models are available for non-commercial research; commercial use requires separate permission from the provider. The release includes a copy of this notice.

## Bundled GPU/runtime dependencies

- NVIDIA CUDA runtime DLLs are redistributed under NVIDIA's [CUDA EULA](https://docs.nvidia.com/cuda/eula/). The package script takes only the CUDA runtime/provider DLLs needed by the application from the selected CUDA Toolkit; it does not bundle the toolkit or compiler.
- NVIDIA cuDNN 9.16 DLLs and their license are taken from the official NVIDIA `nvidia-cudnn-cu12` or `nvidia-cudnn-cu13` Windows wheel. The release archive includes the wheel's license text.
- Microsoft ONNX Runtime CUDA provider DLLs are from the official [`Microsoft.ML.OnnxRuntime.Gpu.Windows`](https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime.Gpu.Windows) package. The package's MIT license and third-party notices are included when present in the extracted SDK.
- The package copies the x64 Microsoft Visual C++ runtime DLLs from the installed Visual Studio redistributable folder so end users do not need to install the separate Visual C++ Redistributable. Microsoft redistribution terms are documented in the [Visual Studio license terms](https://visualstudio.microsoft.com/license-terms/).
- The native project links Windows system libraries (Media Foundation, D3D11, and related APIs). Those remain part of Windows and are not copied into the package.

The release builder pins the cuDNN wheel SHA-256 and verifies the model download. Each app and runtime ZIP contains its own `SHA256SUMS.txt` covering every file in that ZIP.
