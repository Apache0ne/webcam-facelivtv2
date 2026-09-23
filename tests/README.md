# Validation phase (not run yet)

1. Create a real aligned 112x112 BGR crop as raw bytes.
2. Generate official FP32/reparameterized golden values with `generate_golden.py`.
3. Convert the checkpoint and compile the custom runtime.
4. Run `facelivt_embed` to produce `output.f32`.
5. Run `compare_cpp.py`.
6. Only after end-to-end parity, start layer-by-layer and performance profiling.

This directory is intentionally scaffolding only at this stage.
