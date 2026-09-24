#!/usr/bin/env bash
set -euo pipefail

usage() {
  printf '%s\n' \
    'Build the Linux FaceLiVT CUDA core and command-line examples.' \
    '' \
    'Usage: ./build_linux.sh [12|13] [--model CHECKPOINT]' \
    '       ./build_linux.sh --cuda-major 12|13 [--model CHECKPOINT]' \
    '' \
    'Requirements: CMake, a C++20 compiler, a matching CUDA Toolkit, Python 3.12,' \
    'and Triton 3.6.0. This builds the native embedding/matching/benchmark core;' \
    'the webcam attendance UI currently uses Windows Media Foundation and D3D11.' \
    '' \
    'CUDA_ROOT or CUDA_PATH may point to the selected toolkit installation.'
}

cuda_major=13
model=""
if [[ $# -gt 0 && "$1" != -* && "$1" != 12 && "$1" != 13 ]]; then
  # Preserve the original build_linux.sh CHECKPOINT invocation.
  model="$1"
  shift
fi
while (($#)); do
  case "$1" in
    12|13) cuda_major="$1" ;;
    --cuda-major)
      (($# >= 2)) || { echo "--cuda-major needs 12 or 13." >&2; exit 2; }
      cuda_major="$2"
      shift
      ;;
    --model)
      (($# >= 2)) || { echo "--model needs a checkpoint path." >&2; exit 2; }
      model="$2"
      shift
      ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done
if [[ "$cuda_major" != 12 && "$cuda_major" != 13 ]]; then
  echo "CUDA major version must be 12 or 13." >&2
  exit 2
fi

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$project_root"
python="${PYTHON:-python3}"
weights="$project_root/facelivtv2-l.fp16.flvt"
if [[ -n "$model" ]]; then
  "$python" "$project_root/converter/convert_facelivtv2_l.py" --checkpoint "$model" --output "$weights"
elif [[ ! -f "$weights" ]]; then
  if [[ -f "$project_root/facelivtv2-l.pt" ]]; then
    "$python" "$project_root/converter/convert_facelivtv2_l.py" --checkpoint "$project_root/facelivtv2-l.pt" --output "$weights"
  else
    echo "No FaceLiVT weights found. Add facelivtv2-l.fp16.flvt or pass a checkpoint with --model." >&2
    exit 1
  fi
fi

cuda_root="${CUDA_ROOT:-${CUDA_PATH:-}}"
if [[ -z "$cuda_root" ]]; then
  nvcc_path="$(command -v nvcc || true)"
  if [[ -n "$nvcc_path" ]]; then
    cuda_root="$(dirname -- "$(dirname -- "$nvcc_path")")"
  fi
fi
if [[ -z "$cuda_root" || ! -x "$cuda_root/bin/nvcc" ]]; then
  echo "CUDA Toolkit $cuda_major was not found. Set CUDA_ROOT or CUDA_PATH to its install directory." >&2
  exit 1
fi

nvcc_version="$("$cuda_root/bin/nvcc" --version)"
if ! grep -Eq "release[[:space:]]+$cuda_major\." <<<"$nvcc_version"; then
  echo "The toolkit at '$cuda_root' is not CUDA $cuda_major." >&2
  exit 1
fi

"$python" - <<'PY'
import sys
import triton

if sys.version_info[:2] != (3, 12):
    raise SystemExit(f"Python 3.12 is required (found {sys.version.split()[0]}).")
if triton.__version__ != "3.6.0":
    raise SystemExit(f"Triton 3.6.0 is required (found {triton.__version__}).")
print(f"Python {sys.version.split()[0]} | Triton {triton.__version__}")
PY

build_dir="${BUILD_DIR:-$project_root/build/cuda$cuda_major-linux}"
kernel_dir="${KERNEL_DIR:-$project_root/generated/cuda$cuda_major-linux/kernels}"
mkdir -p -- "$build_dir"
if [[ -z "${TRITON_CACHE_DIR:-}" ]]; then
  export TRITON_CACHE_DIR="$build_dir/triton_cache"
fi

arches=(75 80 86 87 89 90 100 120)
kernel_args=("--out" "$kernel_dir")
for arch in "${arches[@]}"; do
  kernel_args+=("--arch" "$arch")
done
"$python" "$project_root/tools/build_kernels.py" "${kernel_args[@]}"

cmake -S "$project_root" -B "$build_dir" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCUDAToolkit_ROOT="$cuda_root"
cmake --build "$build_dir" --parallel

for target in facelivt_embed facelivt_match facelivt_bench; do
  if [[ ! -x "$build_dir/$target" ]]; then
    echo "Expected Linux build output is missing: $build_dir/$target" >&2
    exit 1
  fi
done
if [[ ! -s "$kernel_dir/kernel_manifest.tsv" ]]; then
  echo "Triton did not produce the kernel manifest: $kernel_dir/kernel_manifest.tsv" >&2
  exit 1
fi

echo "Linux CUDA $cuda_major build complete: $build_dir"
echo "Built the FaceLiVT native core and CLI examples. A GPU is needed to run inference."
