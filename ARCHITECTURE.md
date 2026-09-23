# Exact FaceLiVTv2-L deployment graph

Internal activation layout: NHWC FP16.

```text
u8 BGR/RGB 112x112x3
  preprocess -> FP16 normalize
  3x3 s2 3->32 + GELU                       56x56x32
  3x3 s2 32->64                             28x28x64

  stage0 x3:
    residual-folded DW3x3
    PW 64->128 + GELU
    PW 128->64 + residual

  merge0:
    DW3x3 s2                                 14x14x64
    PW 64->128                              14x14x128
    PW 128->256 + GELU
    PW 256->128 + residual

  stage1 x3: DW + FFN                       14x14x128

  merge1:
    DW3x3 s2                                  7x7x128
    PW 128->256                               7x7x256
    FFN 256->512->256

  stage2 x9:
    residual-folded DW3x3                     7x7x256
    fused MHLA: affine + 4*(49x49) + ls + residual
    FFN 256->512->256

  merge2:
    DW3x3 s2                                  4x4x256
    PW 256->512                               4x4x512
    FFN 512->1024->512

  stage3 x3:
    residual-folded DW3x3                     4x4x512
    fused MHLA: affine + 4*(16x16) + ls + residual
    FFN 512->1024->512

  PW 512->1284                                4x4x1284
  DW4x4 per channel                           1x1x1284
  BN-folded Linear 1284->512                  512
  optional L2 normalize                       512 float32
```

The converter emits 202 semantic FP16 tensors rather than retaining the 628 training-state entries.
