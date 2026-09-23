#!/usr/bin/env python3
import argparse
import hashlib
import math
import os
import struct
from collections import OrderedDict

import numpy as np
import torch

MAGIC = b"FLV2BIN\0"
VERSION = 1
DTYPE_F16 = 1
HEADER_FMT = "<8sIIQQQI20s"   # 64 bytes
DESC_FMT = "<96sII4IQQQ"      # 144 bytes
HEADER_SIZE = struct.calcsize(HEADER_FMT)
DESC_SIZE = struct.calcsize(DESC_FMT)
ALIGN = 256
BN_EPS = 1e-5


def align_up(x, a=ALIGN):
    return (x + a - 1) // a * a


def as_np(t):
    return t.detach().cpu().float().numpy()


def bn_fold_conv(w, b, gamma, beta, mean, var, eps=BN_EPS):
    w = as_np(w)
    if b is None:
        b = np.zeros((w.shape[0],), dtype=np.float32)
    else:
        b = as_np(b)
    gamma = as_np(gamma)
    beta = as_np(beta)
    mean = as_np(mean)
    var = as_np(var)
    scale = gamma / np.sqrt(var + eps)
    w = w * scale.reshape((-1, 1, 1, 1))
    b = beta + (b - mean) * scale
    return w.astype(np.float32), b.astype(np.float32)


def fuse_convbn(sd, prefix):
    # prefix points to Conv2d_BN, e.g. `patch_embedds.1.spatial.1`
    w = sd[prefix + ".c.weight"]
    return bn_fold_conv(
        w, None,
        sd[prefix + ".bn.weight"],
        sd[prefix + ".bn.bias"],
        sd[prefix + ".bn.running_mean"],
        sd[prefix + ".bn.running_var"],
    )


def fuse_repconv(sd, prefix, add_identity=False):
    # prefix points to RepConv
    w3 = as_np(sd[prefix + ".conv.weight"])
    b3 = as_np(sd[prefix + ".conv.bias"])
    w1 = as_np(sd[prefix + ".repconv.weight"])
    b1 = as_np(sd[prefix + ".repconv.bias"])
    kh, kw = w3.shape[2:]
    rh, rw = w1.shape[2:]
    ph = (kh - rh) // 2
    pw = (kw - rw) // 2
    w1p = np.pad(w1, ((0,0),(0,0),(ph, kh-rh-ph),(pw, kw-rw-pw)))
    w = w3 + w1p
    b = b3 + b1
    gamma = as_np(sd[prefix + ".bn.weight"])
    beta = as_np(sd[prefix + ".bn.bias"])
    mean = as_np(sd[prefix + ".bn.running_mean"])
    var = as_np(sd[prefix + ".bn.running_var"])
    scale = gamma / np.sqrt(var + BN_EPS)
    w = w * scale.reshape((-1,1,1,1))
    b = beta + (b - mean) * scale
    if add_identity:
        # Official deploy() adds identity after RepConv+BN deployment.
        assert w.shape[1] == 1, "identity folding is only valid for depthwise mixer"
        cy, cx = kh // 2, kw // 2
        w[:, 0, cy, cx] += 1.0
    return w.astype(np.float32), b.astype(np.float32)


def fuse_bn_linear(sd, prefix):
    # prefix points to BN_Linear (`head.classifier`)
    gamma = as_np(sd[prefix + ".bn.weight"])
    beta = as_np(sd[prefix + ".bn.bias"])
    mean = as_np(sd[prefix + ".bn.running_mean"])
    var = as_np(sd[prefix + ".bn.running_var"])
    W = as_np(sd[prefix + ".l.weight"])
    b_lin = as_np(sd[prefix + ".l.bias"])
    scale = gamma / np.sqrt(var + BN_EPS)
    bn_bias = beta - mean * scale
    Wf = W * scale.reshape((1, -1))
    bf = W @ bn_bias + b_lin
    return Wf.astype(np.float32), bf.astype(np.float32)


def conv_oihw_to_hwio(w):
    return np.transpose(w, (2,3,1,0)).copy()


def dw_o1hw_to_hwc(w):
    assert w.shape[1] == 1
    return np.transpose(w[:,0,:,:], (1,2,0)).copy()


def pw_oihw_to_io(w):
    assert w.shape[2:] == (1,1)
    return w[:, :, 0, 0].T.copy()


def linear_oi_to_io(w):
    return w.T.copy()


class Pack:
    def __init__(self):
        self.tensors = OrderedDict()

    def add(self, name, a):
        a = np.ascontiguousarray(a, dtype=np.float16)
        if name in self.tensors:
            raise KeyError(name)
        self.tensors[name] = a

    def add_dense_conv(self, name, w, b):
        self.add(name + ".w", conv_oihw_to_hwio(w))
        self.add(name + ".b", b)

    def add_dw(self, name, w, b):
        self.add(name + ".w", dw_o1hw_to_hwc(w))
        self.add(name + ".b", b)

    def add_pw(self, name, w, b):
        self.add(name + ".w", pw_oihw_to_io(w))
        self.add(name + ".b", b)

    def write(self, path):
        count = len(self.tensors)
        table_offset = HEADER_SIZE
        data_offset = align_up(table_offset + count * DESC_SIZE)
        descs = []
        cursor = 0
        blobs = []
        for name, arr in self.tensors.items():
            cursor = align_up(cursor)
            raw = arr.tobytes(order="C")
            dims = list(arr.shape[:4]) + [1] * max(0, 4-arr.ndim)
            dims = dims[:4]
            descs.append((name, arr.ndim, dims, cursor, len(raw)))
            if len(blobs) == 0:
                current = 0
            else:
                current = sum(len(x) for x in blobs)
            if current < cursor:
                blobs.append(bytes(cursor-current))
            blobs.append(raw)
            cursor += len(raw)
        data_bytes = cursor
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        with open(path, "wb") as f:
            f.write(struct.pack(
                HEADER_FMT, MAGIC, VERSION, count,
                table_offset, data_offset, data_bytes,
                DTYPE_F16, bytes(20)
            ))
            for name, ndim, dims, off, nbytes in descs:
                nb = name.encode("utf-8")
                if len(nb) >= 96:
                    raise ValueError(f"tensor name too long: {name}")
                nb = nb + bytes(96-len(nb))
                f.write(struct.pack(
                    DESC_FMT, nb, DTYPE_F16, ndim,
                    dims[0], dims[1], dims[2], dims[3],
                    off, nbytes, 0
                ))
            pos = f.tell()
            if pos < data_offset:
                f.write(bytes(data_offset-pos))
            for blob in blobs:
                f.write(blob)
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1<<20), b""):
                h.update(chunk)
        return h.hexdigest(), count, data_bytes


def add_ffn(pack, sd, out_prefix, src_prefix):
    w, b = fuse_convbn(sd, src_prefix + ".pw1")
    pack.add_pw(out_prefix + ".fc1", w, b)
    w, b = fuse_convbn(sd, src_prefix + ".pw2")
    pack.add_pw(out_prefix + ".fc2", w, b)


def build(sd):
    p = Pack()

    # Stem: StemLayer(3 -> 32 -> 64), both RepConv, only first followed by GELU.
    for i in range(2):
        w, b = fuse_repconv(sd, f"patch_embedds.0.stem.{i}.0", add_identity=False)
        p.add_dense_conv(f"stem.{i}", w, b)

    dims = [64, 128, 256, 512]
    depths = [3, 3, 9, 3]

    # Patch merges exist before stages 1,2,3 and are patch_embedds indices 1..3.
    for merge_i in range(1, 4):
        base = f"patch_embedds.{merge_i}"
        w, b = fuse_repconv(sd, base + ".spatial.0", add_identity=False)
        p.add_dw(f"merge.{merge_i-1}.dw", w, b)
        w, b = fuse_convbn(sd, base + ".spatial.1")
        p.add_pw(f"merge.{merge_i-1}.pw", w, b)
        add_ffn(p, sd, f"merge.{merge_i-1}.ffn", base + ".channel.m")

    # Stages.
    for s, depth in enumerate(depths):
        for bi in range(depth):
            block = f"stages.{s}.blocks.{bi}.block"
            # all v2-L token mixers begin with residual depthwise RepConv
            w, b = fuse_repconv(sd, block + ".0.m", add_identity=True)
            p.add_dw(f"stage.{s}.block.{bi}.dw", w, b)
            if s < 2:
                add_ffn(p, sd, f"stage.{s}.block.{bi}.ffn", block + ".1.m")
            else:
                mhla = block + ".1.m"
                alpha = as_np(sd[mhla + ".norm.alpha"]).reshape(dims[s])
                beta  = as_np(sd[mhla + ".norm.beta"]).reshape(dims[s])
                ls    = as_np(sd[mhla + ".ls"]).reshape(dims[s])
                S = 49 if s == 2 else 16
                weights = np.stack([as_np(sd[f"{mhla}.lin.{h}.weight"]) for h in range(4)], axis=0)
                biases  = np.stack([as_np(sd[f"{mhla}.lin.{h}.bias"]) for h in range(4)], axis=0)
                assert weights.shape == (4, S, S)
                p.add(f"stage.{s}.block.{bi}.mhla.alpha", alpha)
                p.add(f"stage.{s}.block.{bi}.mhla.beta", beta)
                p.add(f"stage.{s}.block.{bi}.mhla.ls", ls)
                p.add(f"stage.{s}.block.{bi}.mhla.w", weights)
                p.add(f"stage.{s}.block.{bi}.mhla.b", biases)
                add_ffn(p, sd, f"stage.{s}.block.{bi}.ffn", block + ".2.m")

    # Pre-head 512 -> 1284 pointwise, then per-channel 4x4 global conv.
    w, b = fuse_convbn(sd, "pre_head.0")
    p.add_pw("prehead.pw", w, b)
    w, b = fuse_convbn(sd, "pre_head.1")
    p.add_dw("prehead.dw4", w, b)

    # BN_Linear 1284 -> 512.
    w, b = fuse_bn_linear(sd, "head.classifier")
    p.add("head.fc.w", linear_oi_to_io(w))
    p.add("head.fc.b", b)
    return p


def validate_key_contract(sd):
    required = [
        "patch_embedds.0.stem.0.0.conv.weight",
        "patch_embedds.0.stem.1.0.conv.weight",
        "stages.2.blocks.8.block.1.m.lin.3.weight",
        "stages.3.blocks.2.block.1.m.lin.3.weight",
        "pre_head.0.c.weight",
        "pre_head.1.c.weight",
        "head.classifier.l.weight",
    ]
    missing = [k for k in required if k not in sd]
    if missing:
        raise RuntimeError("checkpoint does not match facelivtv2_l; missing: " + ", ".join(missing))
    if tuple(sd["head.classifier.l.weight"].shape) != (512, 1284):
        raise RuntimeError("expected FaceLiVTv2-L head [512,1284]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    obj = torch.load(args.checkpoint, map_location="cpu")
    if isinstance(obj, dict) and "state_dict" in obj:
        obj = obj["state_dict"]
    if not isinstance(obj, (dict, OrderedDict)):
        raise TypeError("checkpoint must contain a state_dict-like mapping")
    # tolerate DistributedDataParallel prefix
    if all(k.startswith("module.") for k in obj.keys()):
        obj = OrderedDict((k[7:], v) for k, v in obj.items())
    validate_key_contract(obj)
    pack = build(obj)
    sha, count, nbytes = pack.write(args.output)
    print(f"wrote: {args.output}")
    print(f"packed tensors: {count}")
    print(f"weight bytes: {nbytes}")
    print(f"sha256: {sha}")


if __name__ == "__main__":
    main()
