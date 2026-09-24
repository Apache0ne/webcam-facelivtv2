#!/usr/bin/env python3
"""Compile FaceLiVT kernels into architecture-specific cubins.

Python/Triton are build-time tools only. The deployed C++ runtime selects the
best compatible cubin from the manifest and does not embed Python.
"""
import argparse
import importlib.util
import re
from pathlib import Path

import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

ROOT = Path(__file__).resolve().parents[1]
KSRC = ROOT / "kernels" / "facelivt_kernels.py"
spec = importlib.util.spec_from_file_location("facelivt_kernels", KSRC)
K = importlib.util.module_from_spec(spec)
spec.loader.exec_module(K)


def make_source(fn, signature, constants):
    """Compatibility helper across recent Triton 3.x ASTSource signatures."""
    try:
        return ASTSource(fn=fn, signature=signature, constexprs=constants)
    except TypeError:
        ordered = [signature[name] for name in fn.arg_names if name not in constants]
        return ASTSource(fn=fn, signature=",".join(ordered), constexprs=constants)


def parse_arch(value):
    value = value.lower().removeprefix("sm_").removeprefix("sm")
    if not value.isdigit():
        raise argparse.ArgumentTypeError("architecture must look like 120, sm120, or sm_120")
    arch = int(value)
    if arch < 50 or arch > 200:
        raise argparse.ArgumentTypeError("compute capability must be between sm_50 and sm_200")
    return arch


def compile_one(alias, fn, signature, constants, out_dir, target,
                num_warps=4, num_stages=2):
    compiled = triton.compile(
        make_source(fn, signature, constants), target=target,
        options={"num_warps": num_warps, "num_stages": num_stages},
    )
    asm = compiled.asm
    cubin = asm.get("cubin")
    if cubin is None:
        raise RuntimeError(f"{alias}: cubin not emitted; available asm={list(asm.keys())}")
    path = out_dir / f"{alias}.cubin"
    path.write_bytes(cubin)
    md = compiled.metadata
    entry = getattr(md, "name", alias)
    warps = int(getattr(md, "num_warps", num_warps))
    shared = int(getattr(md, "shared", 0) or 0)
    if any(getattr(md, name, 0) for name in ("global_scratch_size", "profile_scratch_size")):
        raise RuntimeError(f"{alias}: runtime does not support nonzero Triton scratch buffers")
    params = re.search(r"\.entry\s+" + re.escape(entry) + r"\s*\((.*?)\)", asm["ptx"], re.S)
    if params is None:
        raise RuntimeError(f"{alias}: cannot read kernel parameter ABI")
    scratch_args = len(re.findall(r"\.param\b", params.group(1))) - len(signature)
    if scratch_args not in (0, 1, 2):
        raise RuntimeError(f"{alias}: unsupported extra kernel parameters: {scratch_args}")
    return alias, path.name, entry, warps, shared, scratch_args


def kernel_specs():
    """Return the fixed FaceLiVT model's kernel signatures and specializations."""
    rows = []

    for swap in (0, 1):
        alias = f"preprocess_swap{swap}"
        sig = {"src": "*u8", "dst": "*fp16", "n_elem": "i32"}
        rows.append((alias, K.preprocess_u8_kernel, sig,
                     {"SWAP_RB": swap, "BLOCK": 1024}, 4, 1))

    for alias, H, W, C, N, gelu in [
        ("stem_3_32_gelu", 112, 112, 3, 32, 1),
        ("stem_32_64", 56, 56, 32, 64, 0),
    ]:
        sig = {"x": "*fp16", "w": "*fp16", "bias": "*fp16", "y": "*fp16"}
        rows.append((alias, K.conv3s2_nhwc_kernel, sig,
                     {"H": H, "W": W, "C": C, "N": N, "GELU": gelu,
                      "BLOCK_N": 32, "BLOCK_C": 32}, 4, 2))

    for H, W, C, stride in [
        (28, 28, 64, 1), (28, 28, 64, 2),
        (14, 14, 128, 1), (14, 14, 128, 2),
        (7, 7, 256, 1), (7, 7, 256, 2),
        (4, 4, 512, 1),
    ]:
        alias = f"dw3_h{H}_c{C}_s{stride}"
        sig = {"x": "*fp16", "w": "*fp16", "bias": "*fp16", "y": "*fp16"}
        rows.append((alias, K.depthwise3_nhwc_kernel, sig,
                     {"H": H, "W": W, "C": C, "STRIDE": stride, "BLOCK_C": 64}, 4, 1))

    sig = {"x": "*fp16", "w": "*fp16", "bias": "*fp16", "y": "*fp16"}
    rows.append(("dw4_global_c1284", K.depthwise4_global_nhwc_kernel, sig,
                 {"C": 1284, "BLOCK_C": 64}, 4, 1))

    specs = set()
    for c in (64, 128, 256, 512):
        specs.add((c, 2 * c, 1, 0))
        specs.add((2 * c, c, 0, 1))
    specs.update({(64, 128, 0, 0), (128, 256, 0, 0), (256, 512, 0, 0),
                  (512, 1284, 0, 0), (1284, 512, 0, 0)})
    for kk, nn, gelu, residual in sorted(specs):
        alias = f"pw_k{kk}_n{nn}_g{gelu}_r{residual}"
        sig = {"a": "*fp16", "w": "*fp16", "bias": "*fp16",
               "residual": "*fp16", "out": "*fp16", "M": "i32"}
        constants = {"K": kk, "N": nn, "GELU": gelu, "HAS_RESIDUAL": residual,
                     "BLOCK_M": 32 if nn <= 512 else 16, "BLOCK_N": 64, "BLOCK_K": 32}
        rows.append((alias, K.pointwise_gemm_kernel, sig, constants, 4, 2))

    for C, S, head_dim, block_s in ((256, 49, 64, 64), (512, 16, 128, 16)):
        alias = f"mhla_c{C}_s{S}"
        sig = {"x": "*fp16", "alpha": "*fp16", "beta": "*fp16",
               "weight": "*fp16", "bias": "*fp16", "ls": "*fp16", "out": "*fp16"}
        constants = {"C": C, "S": S, "HEAD_DIM": head_dim, "BLOCK_C": 32, "BLOCK_S": block_s}
        rows.append((alias, K.mhla_fused_kernel, sig, constants, 4, 2))

    for normalize in (0, 1):
        sig = {"inp": "*fp16", "out": "*fp32"}
        rows.append((f"embed_out_norm{normalize}", K.embedding_out_kernel, sig,
                     {"NORMALIZE": normalize, "EMB": 512, "BLOCK": 512}, 4, 1))

    rows.append(("gallery_scores", K.gallery_scores_kernel,
                 {"query": "*fp32", "gallery": "*fp32", "scores": "*fp32", "N": "i32"},
                 {"EMB": 512, "BLOCK_N": 64, "BLOCK_K": 32}, 4, 1))
    rows.append(("top2_scores_4096", K.top2_scores_kernel,
                 {"scores": "*fp32", "best_idx": "*i32", "best_score": "*fp32",
                  "second_idx": "*i32", "second_score": "*fp32", "N": "i32"},
                 {"BLOCK_N": 4096}, 4, 1))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "generated" / "kernels"))
    ap.add_argument("--arch", action="append", type=parse_arch,
                    help="Compute capability to compile (repeatable; e.g. --arch sm120 --arch sm89). Defaults to the current GPU.")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    architectures = args.arch or [int(triton.runtime.driver.active.get_current_target().arch)]
    if len(set(architectures)) != len(architectures):
        ap.error("each --arch value must be unique")

    manifest = out / "kernel_manifest.tsv"
    with manifest.open("w", newline="\n") as file:
        file.write("arch\talias\tcubin\tentry\twarps\tshared\tscratch_args\n")
        for arch in architectures:
            target = GPUTarget("cuda", arch, 32)
            arch_dir = out / f"sm{arch}"
            arch_dir.mkdir(parents=True, exist_ok=True)
            for alias, fn, signature, constants, warps, stages in kernel_specs():
                result = compile_one(alias, fn, signature, constants, arch_dir,
                                     target, warps, stages)
                alias, filename, entry, nw, shared, scratch = result
                file.write("\t".join(map(str, (arch, alias, f"sm{arch}/{filename}",
                                                entry, nw, shared, scratch))) + "\n")
                file.flush()
            print(f"compiled {len(kernel_specs())} kernels for sm_{arch}")
    print(f"wrote {len(architectures)} architecture set(s) to {manifest}")


if __name__ == "__main__":
    main()
