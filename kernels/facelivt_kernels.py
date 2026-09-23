import triton
import triton.language as tl


@triton.jit
def preprocess_u8_kernel(src, dst, n_elem, SWAP_RB: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elem
    c = offs % 3
    pix_base = offs - c
    if SWAP_RB:
        src_c = tl.where(c == 0, 2, tl.where(c == 2, 0, 1))
    else:
        src_c = c
    x = tl.load(src + pix_base + src_c, mask=mask, other=0).to(tl.float32)
    y = x * (1.0 / 127.5) - 1.0
    tl.store(dst + offs, y.to(tl.float16), mask=mask)


@triton.jit
def conv3s2_nhwc_kernel(
    x, w, bias, y,
    H: tl.constexpr, W: tl.constexpr,
    C: tl.constexpr, N: tl.constexpr,
    GELU: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_C: tl.constexpr,
):
    # Same-padding 3x3, stride 2. H/W/C/N are static; B controls launch grid.
    OH: tl.constexpr = (H + 1) // 2
    OW: tl.constexpr = (W + 1) // 2
    p = tl.program_id(0)
    pn = tl.program_id(1)
    b = p // (OH * OW)
    rem = p - b * (OH * OW)
    oy = rem // OW
    ox = rem - oy * OW
    offs_n = pn * BLOCK_N + tl.arange(0, BLOCK_N)
    nmask = offs_n < N
    acc = tl.load(bias + offs_n, mask=nmask, other=0.0).to(tl.float32)
    offs_c = tl.arange(0, BLOCK_C)
    cmask = offs_c < C
    for ky in range(3):
        iy = oy * 2 + ky - 1
        for kx in range(3):
            ix = ox * 2 + kx - 1
            spatial = (iy >= 0) & (iy < H) & (ix >= 0) & (ix < W)
            xp = ((b * H + iy) * W + ix) * C + offs_c
            a = tl.load(x + xp, mask=cmask & spatial, other=0.0).to(tl.float32)
            wp = (((ky * 3 + kx) * C + offs_c[:, None]) * N + offs_n[None, :])
            ww = tl.load(w + wp, mask=cmask[:, None] & nmask[None, :], other=0.0).to(tl.float32)
            acc += tl.sum(ww * a[:, None], axis=0)
    z = acc
    if GELU:
        z = 0.5 * z * (1.0 + tl.erf(z * 0.7071067811865476))
    outp = ((b * OH + oy) * OW + ox) * N + offs_n
    tl.store(y + outp, z.to(tl.float16), mask=nmask)


@triton.jit
def depthwise3_nhwc_kernel(
    x, w, bias, y,
    H: tl.constexpr, W: tl.constexpr, C: tl.constexpr,
    STRIDE: tl.constexpr, BLOCK_C: tl.constexpr,
):
    OH: tl.constexpr = (H + STRIDE - 1) // STRIDE
    OW: tl.constexpr = (W + STRIDE - 1) // STRIDE
    p = tl.program_id(0)
    pc = tl.program_id(1)
    b = p // (OH * OW)
    rem = p - b * (OH * OW)
    oy = rem // OW
    ox = rem - oy * OW
    c = pc * BLOCK_C + tl.arange(0, BLOCK_C)
    cmask = c < C
    acc = tl.load(bias + c, mask=cmask, other=0.0).to(tl.float32)
    for ky in range(3):
        iy = oy * STRIDE + ky - 1
        for kx in range(3):
            ix = ox * STRIDE + kx - 1
            spatial = (iy >= 0) & (iy < H) & (ix >= 0) & (ix < W)
            xp = ((b * H + iy) * W + ix) * C + c
            wp = (ky * 3 + kx) * C + c
            xv = tl.load(x + xp, mask=cmask & spatial, other=0.0)
            wv = tl.load(w + wp, mask=cmask, other=0.0)
            acc += xv.to(tl.float32) * wv.to(tl.float32)
    yp = ((b * OH + oy) * OW + ox) * C + c
    tl.store(y + yp, acc.to(tl.float16), mask=cmask)


@triton.jit
def depthwise4_global_nhwc_kernel(
    x, w, bias, y,
    C: tl.constexpr, BLOCK_C: tl.constexpr,
):
    b = tl.program_id(0)
    pc = tl.program_id(1)
    c = pc * BLOCK_C + tl.arange(0, BLOCK_C)
    cmask = c < C
    acc = tl.load(bias + c, mask=cmask, other=0.0).to(tl.float32)
    for s in range(16):
        xp = (b * 16 + s) * C + c
        wp = s * C + c
        xv = tl.load(x + xp, mask=cmask, other=0.0)
        wv = tl.load(w + wp, mask=cmask, other=0.0)
        acc += xv.to(tl.float32) * wv.to(tl.float32)
    tl.store(y + b * C + c, acc.to(tl.float16), mask=cmask)


@triton.jit
def pointwise_gemm_kernel(
    a, w, bias, residual, out,
    M,
    K: tl.constexpr, N: tl.constexpr,
    GELU: tl.constexpr, HAS_RESIDUAL: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        av = tl.load(
            a + offs_m[:, None] * K + offs_k[None, :],
            mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0,
        )
        bv = tl.load(
            w + offs_k[:, None] * N + offs_n[None, :],
            mask=(offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0,
        )
        acc += tl.dot(av, bv)
    acc += tl.load(bias + offs_n, mask=offs_n < N, other=0.0)[None, :]
    if GELU:
        acc = 0.5 * acc * (1.0 + tl.erf(acc * 0.7071067811865476))
    if HAS_RESIDUAL:
        r = tl.load(
            residual + offs_m[:, None] * N + offs_n[None, :],
            mask=(offs_m[:, None] < M) & (offs_n[None, :] < N), other=0.0,
        )
        acc += r
    tl.store(
        out + offs_m[:, None] * N + offs_n[None, :],
        acc.to(tl.float16),
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


@triton.jit
def mhla_fused_kernel(
    x, alpha, beta, weight, bias, ls, out,
    C: tl.constexpr, S: tl.constexpr,
    HEAD_DIM: tl.constexpr, BLOCK_C: tl.constexpr, BLOCK_S: tl.constexpr,
):
    b = tl.program_id(0)
    head = tl.program_id(1)
    cb = tl.program_id(2)
    lc = cb * BLOCK_C + tl.arange(0, BLOCK_C)
    cmask = lc < HEAD_DIM
    c = head * HEAD_DIM + lc
    sp = tl.arange(0, BLOCK_S)
    smask = sp < S
    xp = (b * S + sp[None, :]) * C + c[:, None]
    xv = tl.load(x + xp, mask=cmask[:, None] & smask[None, :], other=0.0).to(tl.float32)
    av = tl.load(alpha + c, mask=cmask, other=0.0).to(tl.float32)[:, None]
    bv = tl.load(beta + c, mask=cmask, other=0.0).to(tl.float32)[:, None]
    xv_aff = xv * av + bv

    # PyTorch Linear over spatial dimension: y[o] = sum_i x[i] * W[o,i] + b[o].
    i = tl.arange(0, BLOCK_S)
    o = tl.arange(0, BLOCK_S)
    imask = i < S
    omask = o < S
    wp = head * S * S + o[None, :] * S + i[:, None]  # [i,o] = W[o,i]
    wm = tl.load(weight + wp, mask=imask[:, None] & omask[None, :], other=0.0)
    mixed = tl.dot(xv_aff.to(tl.float16), wm.to(tl.float16))
    bb = tl.load(bias + head * S + o, mask=omask, other=0.0).to(tl.float32)[None, :]
    lsv = tl.load(ls + c, mask=cmask, other=0.0).to(tl.float32)[:, None]
    z = xv + (mixed + bb) * lsv
    tl.store(out + xp, z.to(tl.float16), mask=cmask[:, None] & smask[None, :])


@triton.jit
def embedding_out_kernel(inp, out, NORMALIZE: tl.constexpr, EMB: tl.constexpr, BLOCK: tl.constexpr):
    b = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < EMB
    x = tl.load(inp + b * EMB + offs, mask=mask, other=0.0).to(tl.float32)
    if NORMALIZE:
        ss = tl.sum(x * x, axis=0)
        inv = tl.rsqrt(tl.maximum(ss, 1.0e-20))
        x *= inv
    tl.store(out + b * EMB + offs, x, mask=mask)


@triton.jit
def gallery_scores_kernel(query, gallery, scores, N, EMB: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    b = tl.program_id(0)
    nb = tl.program_id(1)
    offs_n = nb * BLOCK_N + tl.arange(0, BLOCK_N)
    nmask = offs_n < N
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for k0 in range(0, EMB, BLOCK_K):
        k = k0 + tl.arange(0, BLOCK_K)
        kmask = k < EMB
        q = tl.load(query + b * EMB + k, mask=kmask, other=0.0).to(tl.float32)
        g = tl.load(
            gallery + offs_n[:, None] * EMB + k[None, :],
            mask=nmask[:, None] & kmask[None, :], other=0.0,
        ).to(tl.float32)
        acc += tl.sum(g * q[None, :], axis=1)
    tl.store(scores + b * N + offs_n, acc, mask=nmask)

@triton.jit
def top2_scores_kernel(scores, best_idx, best_score, second_idx, second_score,
                       N, BLOCK_N: tl.constexpr):
    b = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N
    x = tl.load(scores + b * N + offs, mask=mask, other=-1.0e30).to(tl.float32)
    v1 = tl.max(x, axis=0)
    i1 = tl.argmax(x, axis=0)
    x2 = tl.where(offs == i1, -1.0e30, x)
    v2 = tl.max(x2, axis=0)
    i2 = tl.argmax(x2, axis=0)
    tl.store(best_idx + b, i1)
    tl.store(best_score + b, v1)
    tl.store(second_idx + b, i2)
    tl.store(second_score + b, v2)
