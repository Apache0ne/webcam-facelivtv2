"""Benchmark the CUDA webcam processing path without camera/display pacing."""
import argparse
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from webcam.detector import SCRFD, align_faces_cuda
from webcam.native import NativeRuntime


def measure(function, frames):
    times = []
    for _ in range(frames):
        start = time.perf_counter()
        function()
        times.append((time.perf_counter() - start) * 1000)
    return {"mean_ms": float(np.mean(times)), "median_ms": float(np.median(times)),
            "p95_ms": float(np.percentile(times, 95)), "passes_per_second": 1000 / float(np.mean(times))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--image", type=Path, help="Optional input image; no image data is saved")
    parser.add_argument("--output", type=Path, default=ROOT / "build/gpu_webcam_benchmark.json")
    args = parser.parse_args()
    if args.frames < 1:
        parser.error("frames must be positive")
    cv2.setNumThreads(4)
    import torch
    if args.image:
        frame = cv2.imread(str(args.image))
        if frame is None:
            parser.error("Cannot read the input image")
        frame = cv2.resize(frame, (640, 480))
    else:
        frame = np.random.default_rng(77).integers(0, 256, (480, 640, 3), dtype=np.uint8)
    detector = SCRFD(ROOT / "models/scrfd_10g_bnkps.onnx")
    frame = torch.as_tensor(frame,device="cuda:0")
    faces = detector.detect(frame)[:8]
    crop = align_faces_cuda(frame,faces[:1]) if faces else torch.zeros((1,112,112,3),dtype=torch.uint8,device="cuda:0")
    with NativeRuntime(ROOT) as runtime:
        def pipeline():
            detected = detector.detect(frame)[:8]
            if detected:
                runtime.infer(align_faces_cuda(frame,detected))

        for _ in range(10):
            detector.detect(frame)
            runtime.infer(crop)
        report = {"frames": args.frames, "detected_faces": len(faces),
                  "detector": measure(lambda: detector.detect(frame), args.frames),
                  "recognition_one_crop": measure(lambda: runtime.infer(crop), args.frames),
                  "combined": measure(pipeline, args.frames),
                  "note": "GPU-resident processing capacity, excluding camera, preview, and database. Combined includes recognition only for detected faces."}
    # Profiling events can be captured inside CUDA graphs. Use a separate
    # session to audit providers, so the timed graph has no profiling overhead.
    audit = SCRFD(ROOT / "models/scrfd_10g_bnkps.onnx", profile_prefix=ROOT / "build/scrfd_cuda_profile")
    audit.detect(frame)
    events = json.loads(Path(audit.session.end_profiling()).read_text())
    providers = sorted({event.get("args", {}).get("provider") for event in events
                        if event.get("args", {}).get("provider")})
    if providers != ["CUDAExecutionProvider"]:
        raise RuntimeError(f"Expected only CUDA nodes in trace, found {providers}")
    report["detector_node_providers"] = providers
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
