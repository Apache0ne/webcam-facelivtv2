#!/usr/bin/env python3
"""Compare C++ output.f32 to golden NPZ. NOT executed when package was created."""
import argparse
import numpy as np
ap=argparse.ArgumentParser(); ap.add_argument("--golden",required=True); ap.add_argument("--cpp",required=True); a=ap.parse_args()
g=np.load(a.golden)["fp32_norm"].reshape(-1).astype(np.float32)
c=np.fromfile(a.cpp,dtype=np.float32)
if c.size!=512: raise SystemExit(f"expected 512 cpp floats, got {c.size}")
cos=float(np.dot(g,c)/(max(np.linalg.norm(g)*np.linalg.norm(c),1e-20)))
print("max_abs",float(np.max(np.abs(g-c))))
print("mean_abs",float(np.mean(np.abs(g-c))))
print("cosine",cos)
