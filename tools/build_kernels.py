#!/usr/bin/env python3
"""Compile FaceLiVT Triton kernels to cubins for the C++ CUDA-driver runtime.

Python/Triton are build-time only. The deployed process does not embed Python.
"""
import argparse
import importlib.util
import re
from pathlib import Path

import triton
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
        ordered = []
        for name in fn.arg_names:
            if name in constants:
                continue
            ordered.append(signature[name])
        return ASTSource(fn=fn, signature=",".join(ordered), constexprs=constants)


def compile_one(alias, fn, signature, constants, out_dir, num_warps=4, num_stages=2):
    target = triton.runtime.driver.active.get_current_target()
    src = make_source(fn, signature, constants)
    compiled = triton.compile(
        src, target=target,
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
    # Recent Triton versions append global/profile scratch pointers to the ABI,
    # even when neither scratch buffer is used by the kernel.
    if any(getattr(md, name, 0) for name in ("global_scratch_size", "profile_scratch_size")):
        raise RuntimeError(f"{alias}: runtime does not support nonzero Triton scratch buffers")
    params = re.search(r"\.entry\s+" + re.escape(entry) + r"\s*\((.*?)\)", asm["ptx"], re.S)
    if params is None:
        raise RuntimeError(f"{alias}: cannot read kernel parameter ABI")
    scratch_args = len(re.findall(r"\.param\b", params.group(1))) - len(signature)
    if scratch_args not in (0, 1, 2):
        raise RuntimeError(f"{alias}: unsupported extra kernel parameters: {scratch_args}")
    return alias, path.name, entry, warps, shared, scratch_args


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "generated" / "kernels"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []

    for swap in (0, 1):
        alias = f"preprocess_swap{swap}"
        sig = {"src":"*u8", "dst":"*fp16", "n_elem":"i32"}
        rows.append(compile_one(alias, K.preprocess_u8_kernel, sig,
                                {"SWAP_RB":swap, "BLOCK":1024}, out, 4, 1))

    for alias,H,W,C,N,gelu in [
        ("stem_3_32_gelu",112,112,3,32,1),
        ("stem_32_64",56,56,32,64,0),
    ]:
        sig={"x":"*fp16","w":"*fp16","bias":"*fp16","y":"*fp16"}
        rows.append(compile_one(alias,K.conv3s2_nhwc_kernel,sig,
            {"H":H,"W":W,"C":C,"N":N,"GELU":gelu,"BLOCK_N":32,"BLOCK_C":32},out,4,2))

    for H,W,C,stride in [
        (28,28,64,1),(28,28,64,2),
        (14,14,128,1),(14,14,128,2),
        (7,7,256,1),(7,7,256,2),
        (4,4,512,1),
    ]:
        alias=f"dw3_h{H}_c{C}_s{stride}"
        sig={"x":"*fp16","w":"*fp16","bias":"*fp16","y":"*fp16"}
        rows.append(compile_one(alias,K.depthwise3_nhwc_kernel,sig,
            {"H":H,"W":W,"C":C,"STRIDE":stride,"BLOCK_C":64},out,4,1))

    sig={"x":"*fp16","w":"*fp16","bias":"*fp16","y":"*fp16"}
    rows.append(compile_one("dw4_global_c1284",K.depthwise4_global_nhwc_kernel,sig,
        {"C":1284,"BLOCK_C":64},out,4,1))

    specs=set()
    for c in (64,128,256,512):
        specs.add((c,2*c,1,0))
        specs.add((2*c,c,0,1))
    specs.update({
        (64,128,0,0),(128,256,0,0),(256,512,0,0),
        (512,1284,0,0),(1284,512,0,0),
    })
    for kk,nn,gelu,res in sorted(specs):
        alias=f"pw_k{kk}_n{nn}_g{gelu}_r{res}"
        sig={"a":"*fp16","w":"*fp16","bias":"*fp16","residual":"*fp16","out":"*fp16","M":"i32"}
        bm=32 if nn <= 512 else 16
        bn=64
        rows.append(compile_one(alias,K.pointwise_gemm_kernel,sig,
            {"K":kk,"N":nn,"GELU":gelu,"HAS_RESIDUAL":res,
             "BLOCK_M":bm,"BLOCK_N":bn,"BLOCK_K":32},out,4,2))

    for C,S,hd,bs in ((256,49,64,64),(512,16,128,16)):
        alias=f"mhla_c{C}_s{S}"
        sig={"x":"*fp16","alpha":"*fp16","beta":"*fp16","weight":"*fp16","bias":"*fp16","ls":"*fp16","out":"*fp16"}
        rows.append(compile_one(alias,K.mhla_fused_kernel,sig,
            {"C":C,"S":S,"HEAD_DIM":hd,"BLOCK_C":32,"BLOCK_S":bs},out,4,2))

    for norm in (0,1):
        alias=f"embed_out_norm{norm}"
        sig={"inp":"*fp16","out":"*fp32"}
        rows.append(compile_one(alias,K.embedding_out_kernel,sig,
            {"NORMALIZE":norm,"EMB":512,"BLOCK":512},out,4,1))

    sig={"query":"*fp32","gallery":"*fp32","scores":"*fp32","N":"i32"}
    rows.append(compile_one("gallery_scores",K.gallery_scores_kernel,sig,
        {"EMB":512,"BLOCK_N":64,"BLOCK_K":32},out,4,1))

    sig={"scores":"*fp32","best_idx":"*i32","best_score":"*fp32","second_idx":"*i32","second_score":"*fp32","N":"i32"}
    rows.append(compile_one("top2_scores_4096",K.top2_scores_kernel,sig,
        {"BLOCK_N":4096},out,4,1))

    manifest=out/"kernel_manifest.tsv"
    with manifest.open("w", newline="\n") as f:
        f.write("alias\tcubin\tentry\twarps\tshared\tscratch_args\n")
        for row in rows:
            f.write("\t".join(map(str,row))+"\n")
    print(f"wrote {len(rows)} cubins")
    print(manifest)

if __name__ == "__main__":
    main()
