#!/usr/bin/env bash
set -euo pipefail
MODEL="${1:-}"
if [[ -n "$MODEL" ]]; then
  python converter/convert_facelivtv2_l.py --checkpoint "$MODEL" --output facelivtv2-l.fp16.flvt
elif [[ ! -f facelivtv2-l.fp16.flvt ]]; then
  if [[ -f facelivtv2-l.pt ]]; then
    python converter/convert_facelivtv2_l.py --checkpoint facelivtv2-l.pt --output facelivtv2-l.fp16.flvt
  else
    echo "No FaceLiVT weights found. Add facelivtv2-l.fp16.flvt or facelivtv2-l.pt, or pass a .pt path." >&2
    exit 1
  fi
fi
python tools/build_kernels.py --out generated/kernels
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j"$(nproc)"
