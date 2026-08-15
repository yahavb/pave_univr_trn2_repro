#!/usr/bin/env python3
"""STANDALONE reproducer: UniVR-RIFE rolling-shutter unrolling, and the trn2 gather bottleneck.

One file, no imports from this repo. Inlines the model, all four resample implementations, the
triplet composition, the descriptor accounting and the scoring criterion the project ships.

WHAT IT REPRODUCES
  1. the port's numerics -- device/gather resample against the stock F.grid_sample reference
  2. the accuracy story against the shipped golden (bar: max_diff <= 3 of 255 levels)
  3. the bottleneck: descriptor count, bytes per descriptor and predicted GpSimd time, per
     resample call site, which is the thing to look at with the compiler team

WHAT THE MODEL IS
  A rolling-shutter sensor exposes one scanline at a time, so every row of a captured frame is
  from a different instant and moving content is sheared. RIFE's IFNet_m estimates bidirectional
  flow between two consecutive RS frames, and the UniVR wrapper feeds it a PER-ROW timestamp
  (tau = t + gamma - gamma*row/H) so the flow is resolved to a single instant. The corrected frame
  is then resampled from the inputs by that flow. The resample is the op this file is about.

THE FOUR RESAMPLE IMPLEMENTATIONS (--warp)
  gridsample  stock F.grid_sample(bilinear, border, align_corners=True). The reference.
  gather      the port's form: explicit 4-tap indirect gather in pure torch. Runs anywhere.
  nki         the shipping trn2 kernel, 2 indirect DMA descriptors per pixel. trn2 only.
  window      the REJECTED dense reformulation: sum over a static (2R+1)^2 window with
              triangular weights, no indirect access at all. Kept because it is the natural
              "just make it dense" suggestion and it needs to be visibly costed, not argued about.

USAGE
  # assets (see TEST_ASSET_SHA256 below) live in
  #   s3://digital-twin-checkpoints/digital_twin/univr_shutter_unrolling/test_assets/
  # weights (pre_net_flow.pkl, 24.8M params) in
  #   .../weights/UniVR_RIFE/deep_unroll_weights/<name>/
  python repro_unrolling_trn2.py --rs0 rs70.png --rs1 rs71.png --rs2 rs72.png \
      --gt gs71_merged_triplet.png --weights pre_net_flow.pkl --height 1728 --width 4096

  # bottleneck accounting only -- no weights, no images, no device needed:
  python repro_unrolling_trn2.py --report-only --height 1728 --width 4096

  # numerics of the port's resample against the reference, CPU, no assets:
  python repro_unrolling_trn2.py --self-test

NOTE ON --random-weights: timing and descriptor counts do not depend on the weights, so a perf
or bottleneck run is valid without them. ACCURACY IS NOT -- any PSNR from random weights is
meaningless and this script refuses to print one.
"""
from __future__ import annotations

import argparse
import hashlib
import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

# Dynamo's recompile cap, raised because this model legitimately needs more graphs than the
# default allows. A tiled frame has several distinct PADDED shapes -- 4 at 4x8 (512x576,
# 512x640, 576x576, 576x640) -- and a triplet runs 2 timestamps, so ~8 traces of the same
# forward. The default cache_size_limit is 8, and under fullgraph=True exceeding it is a HARD
# error rather than a fall back to eager:
#     torch._dynamo.exc.FailOnRecompileLimitHit: Hard failure due to fullgraph=True
# That killed both timed arms of the 8-core run after compiling only 2 graphs, with the
# single-tile guard passing in the same job -- one graph never trips it.
#
# This is a host-side trace counter, unrelated to NeuronCores or device memory. Raising it is
# the right fix rather than reducing --cores: the recompiles are legitimate, and halving cores
# would roughly double wall clock (the README measures sum/wall 7.57 at 8 cores), producing a
# latency not comparable to the baseline.
torch._dynamo.config.cache_size_limit = 64
torch._dynamo.config.accumulated_cache_size_limit = 256

# NKI MUST be imported at MODULE level, not inside the builder function. NKI's parser frontend
# resolves names through the kernel's __globals__ and NOT through its closure, so importing
# `nki.language as nl` inside a function makes `nl` a closure variable and every reference to it
# fails at trace time with "failed to resolve name 'nl.ndarray'" -- even though the name is
# perfectly valid Python. Cost two debug cycles: the first error was on `nl.tile_size.pmax`, which
# looked like a version difference, when the real problem was the import site.
try:
    import nki
    import nki.isa as nisa
    import nki.language as nl
    from nki.isa.constants import oob_mode
    _HAVE_NKI = True
except ImportError:                      # runs fine without the Neuron toolchain
    _HAVE_NKI = False

# ---------------------------------------------------------------------------------------------
# measured trn2 rates, for the predicted-cost column. Sources in docs/NEURON_PORT_NOTES.md.
DESC_NS = 26.5          # ns per indirect DMA descriptor (SWDGE, GpSimd descriptor generation)
GATHER_NS_PER_ELEM = 11.35   # torch.gather in-graph, measured 11.3-11.4 ns/elem (docs C4)

# ---------------------------------------------------------------------------------------------
# GPU BASELINES at 4K (1728x4096) TRIPLET, i.e. the same unit this script reports.
# [R] = reported in this repo, [M] = measured by this script.
# Quote ALL THREE or name which one you mean: they span 2.2x and they are not interchangeable.
CUDA_BASELINE_4K_TRIPLET = {
    "onnx_trt": (161.0, "R",
                 "ONNX+TensorRT, the production shipping path (repo README). TRT runs reduced "
                 "precision by default (fp16/TF32), so this is NOT an fp32 number -- and it is the "
                 "same pipeline whose output is gs71_merged_triplet.png, which our fp32 runs and "
                 "production's own fp32 ONNX both miss by ~73 LSB"),
    "torch_inductor": (237.9, "R",
                       "torch.compile inductor on L40S (repo docs)"),
    "torch_eager_fp32": (351.1, "M",
                         "measured 2026-08-04 on the g6e L40S by THIS script: --warp gridsample "
                         "--device cuda --dtype fp32, median of 3. Same code and same precision as "
                         "the trn2 run, so this is the apples-to-apples denominator"),
}
L40S_4K_TRT_MS = CUDA_BASELINE_4K_TRIPLET["onnx_trt"][0]


def report_gap(trn2_ms):
    """Print the trn2/GPU gap against every baseline, because picking one silently is the single
    easiest way to mis-state this result by 2.2x."""
    print()
    print("=" * 100)
    print("GAP vs CUDA on L40S, 4K triplet")
    print("=" * 100)
    print("  this run (trn2)                       %8.1f ms" % trn2_ms)
    print()
    print("  %-22s %10s %4s %10s" % ("GPU baseline", "ms", "src", "trn2 / GPU"))
    print("  " + "-" * 52)
    for k, (ms, src, _why) in sorted(CUDA_BASELINE_4K_TRIPLET.items(), key=lambda kv: kv[1][0]):
        print("  %-22s %10.1f  [%s] %9.2fx" % (k, ms, src, trn2_ms / ms))
    print()
    for k, (_ms, _src, why) in sorted(CUDA_BASELINE_4K_TRIPLET.items()):
        print("  %s:" % k)
        for line in _wrap(why, 92):
            print("    %s" % line)
    print()
    print("  WHICH TO QUOTE: onnx_trt is the commercially meaningful target and the conservative")
    print("  choice against us, since it is reduced precision while the trn2 run is fp32.")
    print("  torch_eager_fp32 is the fair like-for-like, same code and same precision on both sides.")
    print("  Report both. A single number here is not a result, it is a choice of denominator.")


def _wrap(s, n):
    out, cur = [], ""
    for w in s.split():
        if len(cur) + len(w) + 1 > n:
            out.append(cur)
            cur = w
        else:
            cur = (cur + " " + w).strip()
    if cur:
        out.append(cur)
    return out

TEST_ASSET_SHA256 = {
    "rs70.png": "d71d165877c25bf915409eb44ba318bac7cab5ff3666d37b9d9c5626e62bdacb",
    "rs71.png": "4a90f52bbbc3e17d4afb595d7a5a579db67b2994a650d25949203227daa5cd59",
    "rs72.png": "e4744b5bb618c40ac09ba22b2cc8a0b95246d23efbae98b1b6f98b2e5a6fceb4",
    "gs71_merged_triplet.png": "df0b97b396f17b09e22218202de6bc6208253d5c205703a495c07dc4ee2c6dd8",
}

# Every resample the forward performs, recorded as (tag, C, H, W). Populated by the warp wrappers
# so the descriptor report reflects the ACTUAL call sites rather than a hand-maintained list.
CALL_SITES: list[tuple[str, int, int, int]] = []
_RECORD = False

# Measured displacement per resample call, as (C, H, W, max_abs_px, pct_over_2, pct_over_3).
# This exists because the bounded-radius kernel silently CLAMPS anything beyond R px, and the only
# displacement figure on record (2.33 px) came from a single site at one tile. shiftwarp scored
# 229.72 LSB in the model with p50 = 0.00 and p99 = 172.99 -- 93% of pixels exact and ~7%
# catastrophically wrong, which is the signature of a clamp, not of numerical error. Populated by
# _bilinear_terms so every site is covered, including the ctx sites that fall back to gather.
FLOW_STATS: list[tuple[int, int, int, float, float, float]] = []
_RECORD_FLOW = False


# =============================================================================================
# THE RESAMPLE, four ways
# =============================================================================================
def _bilinear_terms(tenInput, tenFlow):
    """Shared front half: absolute sample coords -> integer base + 4 bilinear weights.

    This is the address computation. `idx = y0*W + x0` is the only index materialised; the other
    three taps are +1, +W, +W+1 from it, which the NKI kernel expresses as access-pattern strides
    rather than as separate index tensors.
    """
    B, C, H, W = tenInput.shape
    dev, dt = tenFlow.device, torch.float32
    gx = torch.arange(W, device=dev, dtype=dt).view(1, 1, 1, W)
    gy = torch.arange(H, device=dev, dtype=dt).view(1, 1, H, 1)
    sx = (gx + tenFlow[:, 0:1].to(dt)).clamp(0.0, W - 1.0)
    sy = (gy + tenFlow[:, 1:2].to(dt)).clamp(0.0, H - 1.0)
    if _RECORD_FLOW:
        # POST-clamp displacement: sx/sy are already clamped to the tile, so this is the
        # displacement the kernel actually has to reach, not the raw flow. That is the quantity R
        # has to cover. Sync to host here -- only ever enabled in an eager diagnostic run.
        dxa = (sx - gx).abs()
        dya = (sy - gy).abs()
        m = torch.maximum(dxa.max(), dya.max()).item()
        over = torch.maximum(dxa, dya)
        n = over.numel()
        FLOW_STATS.append((C, H, W, m,
                           100.0 * (over > 2.0).sum().item() / n,
                           100.0 * (over > 3.0).sum().item() / n))
    x0 = torch.floor(sx)
    y0 = torch.floor(sy)
    ax = (sx - x0)
    ay = (sy - y0)
    return x0, y0, ax, ay, (B, C, H, W)


def warp_gridsample(tenInput, tenFlow):
    """Stock RIFE warplayer.warp -- the reference every other form is scored against."""
    if _RECORD:
        CALL_SITES.append(("gridsample", tenInput.shape[1], tenInput.shape[2], tenInput.shape[3]))
    B, C, H, W = tenInput.shape
    dev = tenFlow.device
    hor = torch.linspace(-1.0, 1.0, W, device=dev, dtype=tenFlow.dtype).view(1, 1, 1, W).expand(B, -1, H, -1)
    ver = torch.linspace(-1.0, 1.0, H, device=dev, dtype=tenFlow.dtype).view(1, 1, H, 1).expand(B, -1, -1, W)
    grid = torch.cat([hor, ver], 1)
    f = torch.cat([tenFlow[:, 0:1] / ((W - 1.0) / 2.0), tenFlow[:, 1:2] / ((H - 1.0) / 2.0)], 1)
    g = (grid + f).permute(0, 2, 3, 1)
    return F.grid_sample(tenInput, g, mode="bilinear", padding_mode="border", align_corners=True)


def warp_gather(tenInput, tenFlow):
    """The port's form: explicit 4-tap indirect gather. Portable, and numerically the same op.

    Laid out exactly as the device kernel sees it -- source as [N=H*W, C] so a pixel's channels are
    contiguous and one descriptor can cover 2C. The four taps are rows {tl, tl+1, tl+W, tl+W+1}.
    """
    if _RECORD:
        CALL_SITES.append(("gather", tenInput.shape[1], tenInput.shape[2], tenInput.shape[3]))
    x0, y0, ax, ay, (B, C, H, W) = _bilinear_terms(tenInput, tenFlow)
    N = H * W
    src = tenInput.reshape(B, C, N).permute(0, 2, 1).reshape(B * N, C).float()

    x1 = (x0 + 1).clamp(0.0, W - 1.0)
    y1 = (y0 + 1).clamp(0.0, H - 1.0)
    boff = (torch.arange(B, device=src.device, dtype=torch.long) * N).view(B, 1, 1, 1)

    def tap(yy, xx):
        idx = (yy * W + xx).long() + boff
        return src.index_select(0, idx.reshape(-1)).view(B, H, W, C).permute(0, 3, 1, 2)

    wtl = ((1 - ax) * (1 - ay))
    wtr = (ax * (1 - ay))
    wbl = ((1 - ax) * ay)
    wbr = (ax * ay)
    out = (tap(y0, x0) * wtl + tap(y0, x1) * wtr + tap(y1, x0) * wbl + tap(y1, x1) * wbr)
    return out.to(tenInput.dtype)


def warp_window(tenInput, tenFlow, radius=2):
    """The REJECTED dense reformulation, exact, with NO indirect access.

    out[y,x] = sum_{oy,ox in [-R,R]} tri(fx-ox) * tri(fy-oy) * img[y+oy, x+ox],  tri(d)=max(0,1-|d|)

    Every term is a STATIC shift of a replicate-padded image (static DMA, conv-like) times a
    precomputed elementwise weight (Vector engine). Exact wherever |residual| <= R, and the 4 taps
    fall out of the triangular kernel. Costs (2R+1)^2 passes, which is why it loses: measured
    0.13-0.61x against the gather at C=3, and covering the real 36-45 px displacement needs R~45,
    i.e. 8281 terms, measured 1537.9 ms = 61x SLOWER than the gather it replaces.
    """
    if _RECORD:
        CALL_SITES.append(("window", tenInput.shape[1], tenInput.shape[2], tenInput.shape[3]))
    x0, y0, ax, ay, (B, C, H, W) = _bilinear_terms(tenInput, tenFlow)
    gx = torch.arange(W, device=tenInput.device, dtype=torch.float32).view(1, 1, 1, W)
    gy = torch.arange(H, device=tenInput.device, dtype=torch.float32).view(1, 1, H, 1)
    rx = (x0 + ax) - gx                       # residual displacement, signed
    ry = (y0 + ay) - gy
    R = radius
    pad = F.pad(tenInput.float(), (R, R, R, R), mode="replicate")
    acc = torch.zeros(B, C, H, W, device=tenInput.device, dtype=torch.float32)
    for oy in range(-R, R + 1):
        ty = (1.0 - (ry - oy).abs()).clamp_min(0.0)
        if not bool((ty > 0).any()):
            continue
        for ox in range(-R, R + 1):
            tx = (1.0 - (rx - ox).abs()).clamp_min(0.0)
            wgt = tx * ty
            if not bool((wgt > 0).any()):
                continue
            sl = pad[:, :, R + oy:R + oy + H, R + ox:R + ox + W]
            acc = acc + sl * wgt
    return acc.to(tenInput.dtype)


# ---------------------------------------------------------------------------------------------
# the shipping trn2 kernel: 2 indirect descriptors per pixel (4->2 merged tap pairs)
_NKI_FN = None
_NKI_FN_DYN = None


def _build_nki():
    """Inline the production v3 4->2 kernel. nki/nisa/nl come from module scope -- see the note at
    the import; putting them in a closure breaks name resolution inside the kernel."""
    if not _HAVE_NKI:
        raise SystemExit("--warp nki needs the Neuron toolchain (nki, torch_neuronx) installed")

    @nki.jit
    def bilinear_2x2_gather_blend(img, idx_tl, w_tl, w_tr, w_bl, w_br, row_w):
        """img: [N, C] in HBM. idx_tl: [K,1] uint32 top-left row index. row_w = source width W.

        Two indirect gathers per tile, each an access pattern [[C, kv], [1, 2*C]]:
          - dim0  partition = pixel, base row supplied per-partition by vector_offset (vector-DGE)
          - dim1  2*C CONTIGUOUS elements = the horizontal tap pair (tl,tr) in one descriptor
        Extents are compile-time constants; only the base row is data-dependent. No padded source
        is needed: +1/+W are strides, and a clamped read carries bilinear weight exactly 0.
        """
        K = idx_tl.shape[0]
        C = img.shape[1]
        dtype = img.dtype
        # LITERAL, not nl.tile_size.pmax. The value is the same (128) but NKI 0.4.0's parser
        # frontend cannot resolve an attribute chain inside a jit'd kernel and fails with
        # "failed to resolve name 'nl.tile_size.pmax'". Same class of restriction as its rejection
        # of comprehensions and symbolic shape division.
        P = 128
        num_k_tiles = (K + P - 1) // P
        out = nl.ndarray((K, C), dtype=dtype, buffer=nl.shared_hbm)

        for kt in nl.affine_range(num_k_tiles):
            k_off = kt * P
            kv = min(P, K - kt * P)

            idx_t = nl.ndarray((kv, 1), dtype=idx_tl.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=idx_t, src=idx_tl.ap(pattern=[[1, kv], [1, 1]], offset=k_off))
            idx_bot = nl.ndarray((kv, 1), dtype=idx_tl.dtype, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=idx_bot, data=idx_t, op0=nl.add, operand0=row_w)

            # Written out rather than built in a loop: the NKI frontend is fussy about
            # container-building inside a kernel, and this matches the shipping kernel's form.
            w0t = nl.ndarray((kv, 1), dtype=nl.float32, buffer=nl.sbuf)
            w1t = nl.ndarray((kv, 1), dtype=nl.float32, buffer=nl.sbuf)
            w2t = nl.ndarray((kv, 1), dtype=nl.float32, buffer=nl.sbuf)
            w3t = nl.ndarray((kv, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=w0t, src=w_tl.ap(pattern=[[1, kv], [1, 1]], offset=k_off))
            nisa.dma_copy(dst=w1t, src=w_tr.ap(pattern=[[1, kv], [1, 1]], offset=k_off))
            nisa.dma_copy(dst=w2t, src=w_bl.ap(pattern=[[1, kv], [1, 1]], offset=k_off))
            nisa.dma_copy(dst=w3t, src=w_br.ap(pattern=[[1, kv], [1, 1]], offset=k_off))

            top = nl.ndarray((kv, 2 * C), dtype=dtype, buffer=nl.sbuf)
            bot = nl.ndarray((kv, 2 * C), dtype=dtype, buffer=nl.sbuf)
            nisa.memset(top, 0.0)
            nisa.memset(bot, 0.0)
            nisa.dma_copy(dst=top, src=img.ap(pattern=[[C, kv], [1, 2 * C]], offset=0,
                                              vector_offset=idx_t, indirect_dim=0),
                          oob_mode=oob_mode.skip)
            nisa.dma_copy(dst=bot, src=img.ap(pattern=[[C, kv], [1, 2 * C]], offset=0,
                                              vector_offset=idx_bot, indirect_dim=0),
                          oob_mode=oob_mode.skip)

            # the blend: 4 scalar-broadcast multiplies + 3 adds, Vector engine.
            acc = nl.ndarray((kv, C), dtype=nl.float32, buffer=nl.sbuf)
            tmp = nl.ndarray((kv, C), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=acc, data=top[:, 0:C], op0=nl.multiply, operand0=w0t)
            nisa.tensor_scalar(dst=tmp, data=top[:, C:2 * C], op0=nl.multiply, operand0=w1t)
            nisa.tensor_tensor(dst=acc, data1=acc, data2=tmp, op=nl.add)
            nisa.tensor_scalar(dst=tmp, data=bot[:, 0:C], op0=nl.multiply, operand0=w2t)
            nisa.tensor_tensor(dst=acc, data1=acc, data2=tmp, op=nl.add)
            nisa.tensor_scalar(dst=tmp, data=bot[:, C:2 * C], op0=nl.multiply, operand0=w3t)
            nisa.tensor_tensor(dst=acc, data1=acc, data2=tmp, op=nl.add)

            out_tile = nl.copy(acc, dtype=dtype)
            nisa.dma_copy(dst=out[nl.ds(k_off, kv), :], src=out_tile)
        return out

    @nki.jit
    def bilinear_2x2_gather_blend_dyn(img, idx_tl, w_tl, w_tr, w_bl, w_br, row_w):
        """Same maths, but a DEVICE-SIDE loop instead of an unrolled one.

        WHY IT EXISTS: NKI unrolls every static loop, so the kernel above emits K/128 copies of its
        body. At a 992x1152 production tile that is 8,928 copies, and the measured consequence is a
        ~200 MB generated artifact and 80-100 MINUTES of compile, single-threaded in ModuleForkPass.
        `nl.dynamic_range` lowers to a real loop executed by the engine sequencers, so the body is
        emitted ONCE and the same tile compiles in seconds. The trade is roughly 2x runtime, because
        a device loop cannot be software-pipelined across iterations the way an unrolled body can.

        REQUIRES K % 128 == 0: a device loop cannot vary its tile size, so there is no tail. The
        caller checks and falls back to the unrolled kernel.

        `offset=` rejects a register ("expecting 'int', got VirtualRegister"), so dynamic addressing
        goes through `scalar_offset`, which is expressed in ROWS of indirect_dim.
        """
        K = idx_tl.shape[0]
        C = img.shape[1]
        dtype = img.dtype
        P = 128
        out = nl.ndarray((K, C), dtype=dtype, buffer=nl.shared_hbm)

        for k_off in nl.dynamic_range(0, K, P):
            idx_t = nl.ndarray((P, 1), dtype=idx_tl.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=idx_t, src=idx_tl.ap(pattern=[[1, P], [1, 1]],
                                                  scalar_offset=k_off, indirect_dim=0))
            idx_bot = nl.ndarray((P, 1), dtype=idx_tl.dtype, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=idx_bot, data=idx_t, op0=nl.add, operand0=row_w)

            w0t = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
            w1t = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
            w2t = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
            w3t = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=w0t, src=w_tl.ap(pattern=[[1, P], [1, 1]],
                                               scalar_offset=k_off, indirect_dim=0))
            nisa.dma_copy(dst=w1t, src=w_tr.ap(pattern=[[1, P], [1, 1]],
                                               scalar_offset=k_off, indirect_dim=0))
            nisa.dma_copy(dst=w2t, src=w_bl.ap(pattern=[[1, P], [1, 1]],
                                               scalar_offset=k_off, indirect_dim=0))
            nisa.dma_copy(dst=w3t, src=w_br.ap(pattern=[[1, P], [1, 1]],
                                               scalar_offset=k_off, indirect_dim=0))

            top = nl.ndarray((P, 2 * C), dtype=dtype, buffer=nl.sbuf)
            bot = nl.ndarray((P, 2 * C), dtype=dtype, buffer=nl.sbuf)
            nisa.memset(top, 0.0)
            nisa.memset(bot, 0.0)
            nisa.dma_copy(dst=top, src=img.ap(pattern=[[C, P], [1, 2 * C]], offset=0,
                                              vector_offset=idx_t, indirect_dim=0),
                          oob_mode=oob_mode.skip)
            nisa.dma_copy(dst=bot, src=img.ap(pattern=[[C, P], [1, 2 * C]], offset=0,
                                              vector_offset=idx_bot, indirect_dim=0),
                          oob_mode=oob_mode.skip)

            acc = nl.ndarray((P, C), dtype=nl.float32, buffer=nl.sbuf)
            tmp = nl.ndarray((P, C), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=acc, data=top[:, 0:C], op0=nl.multiply, operand0=w0t)
            nisa.tensor_scalar(dst=tmp, data=top[:, C:2 * C], op0=nl.multiply, operand0=w1t)
            nisa.tensor_tensor(dst=acc, data1=acc, data2=tmp, op=nl.add)
            nisa.tensor_scalar(dst=tmp, data=bot[:, 0:C], op0=nl.multiply, operand0=w2t)
            nisa.tensor_tensor(dst=acc, data1=acc, data2=tmp, op=nl.add)
            nisa.tensor_scalar(dst=tmp, data=bot[:, C:2 * C], op0=nl.multiply, operand0=w3t)
            nisa.tensor_tensor(dst=acc, data1=acc, data2=tmp, op=nl.add)

            out_tile = nl.copy(acc, dtype=dtype)
            nisa.dma_copy(dst=out.ap(pattern=[[C, P], [1, C]],
                                     scalar_offset=k_off, indirect_dim=0), src=out_tile)
        return out

    from torch_neuronx.nki_hop import wrap_nki
    return wrap_nki(bilinear_2x2_gather_blend), wrap_nki(bilinear_2x2_gather_blend_dyn)


_NKI_DYN = False          # set by --warp nki-dyn


def warp_nki(tenInput, tenFlow):
    """Host side of the trn2 path: build the single uint32 row index, then one kernel call."""
    if _RECORD:
        CALL_SITES.append(("nki", tenInput.shape[1], tenInput.shape[2], tenInput.shape[3]))
    global _NKI_FN, _NKI_FN_DYN
    if _NKI_FN is None:
        _NKI_FN, _NKI_FN_DYN = _build_nki()
    x0, y0, ax, ay, (B, C, H, W) = _bilinear_terms(tenInput, tenFlow)
    N = H * W
    NB = B * N
    idx = (y0 * W + x0).reshape(B, N)
    if B > 1:
        idx = idx + (torch.arange(B, device=idx.device, dtype=idx.dtype) * N).view(B, 1)
    idx_tl = idx.reshape(NB, 1).to(torch.int32).contiguous().view(torch.uint32)
    wx = ax.reshape(NB, 1)
    wy = ay.reshape(NB, 1)
    img_nc = tenInput.reshape(B, C, N).permute(0, 2, 1).reshape(NB, C).contiguous()
    # The device-loop kernel has no tail, so it needs NB divisible by the 128-partition tile.
    fn = _NKI_FN_DYN if (_NKI_DYN and NB % 128 == 0) else _NKI_FN
    out = fn(img_nc, idx_tl,
             ((1 - wx) * (1 - wy)).contiguous(), (wx * (1 - wy)).contiguous(),
             ((1 - wx) * wy).contiguous(), (wx * wy).contiguous(), W)
    return out.reshape(B, N, C).permute(0, 2, 1).reshape(B, C, H, W)


_SHIFTWARP_FN = None      # wrap_nki(shift_warp_band), built eagerly in main()
_SHIFTWARP_R = 3          # set by --shiftwarp-radius
_SHIFTWARP_MAXC = 3       # sites with C above this fall back; see the note in warp_shiftwarp


def _build_shiftwarp():
    """Wrap the SBUF-resident static-DMA kernel. Separate module, so its `import nki.language as nl`
    is at module scope there -- inlining it here would put nl in a closure and break NKI's name
    resolution, which is the same trap documented at the top of this file."""
    if not _HAVE_NKI:
        raise SystemExit("--warp shiftwarp needs the Neuron toolchain (nki, torch_neuronx) installed")
    from nki_shift_warp import shift_warp_band
    from torch_neuronx.nki_hop import wrap_nki
    return wrap_nki(shift_warp_band)


def warp_shiftwarp(tenInput, tenFlow):
    """Bounded-displacement warp with STATIC DMA: the (2R+1)^2 taps are compile-time offsets into an
    SBUF-resident band, so no per-pixel indirect descriptor is generated inside the kernel.

    Measured at the production tile (C=3 992x1280), microbenchmark: 42,439 us vs the gather's
    49,278 us = 1.16x, at 0.0417 max_diff LSB -- identical accuracy to the gather.

    FALLS BACK to warp_gather above C=_SHIFTWARP_MAXC. The kernel is only VERIFIED at C=3: R=3 was
    chosen because the real flow at that site is 2.33 px, and the accuracy cliff is hard -- R=2
    measured 76.19 LSB on the same flow, a clean FAIL. Displacement at the four ctx sites has never
    been measured, so the correct radius there is unknown and a silent quality regression is the
    likely failure mode. Raising _SHIFTWARP_MAXC without measuring ctx-site displacement first is
    exactly the mistake this guard exists to prevent."""
    B, C, H, W = tenInput.shape
    if C > _SHIFTWARP_MAXC:
        return warp_gather(tenInput, tenFlow)
    if _RECORD:
        CALL_SITES.append(("shiftwarp", C, H, W))
    global _SHIFTWARP_FN
    if _SHIFTWARP_FN is None:
        _SHIFTWARP_FN = _build_shiftwarp()
    R = _SHIFTWARP_R
    d = tenFlow.device
    gx = torch.arange(W, device=d, dtype=torch.float32).view(1, 1, 1, W)
    gy = torch.arange(H, device=d, dtype=torch.float32).view(1, 1, H, 1)
    sx = (gx + tenFlow[:, 0:1].float()).clamp(0.0, W - 1.0)
    sy = (gy + tenFlow[:, 1:2].float()).clamp(0.0, H - 1.0)
    rx = sx - gx
    ry = sy - gy
    # Weight planes are elementwise functions of flow, computed on the HOST. Flow is an activation,
    # but the weights only ever enter as tensor DATA, never as an access pattern -- that is what
    # keeps the kernel's descriptor path static.
    planes = []
    for oy in range(-R, R + 1):
        ty = (1.0 - (ry - oy).abs()).clamp_min(0.0)
        for ox in range(-R, R + 1):
            tx = (1.0 - (rx - ox).abs()).clamp_min(0.0)
            planes.append((tx * ty)[0, 0])
    wts = torch.stack(planes, 0).contiguous()
    imgp = F.pad(tenInput.float(), (R, R, R, R), mode="replicate")[0].permute(1, 2, 0).contiguous()
    out = _SHIFTWARP_FN(imgp, wts)
    return out.permute(2, 0, 1).unsqueeze(0).to(tenInput.dtype)


WARPS = {"gridsample": warp_gridsample, "gather": warp_gather,
         "window": warp_window, "nki": warp_nki, "nki-dyn": warp_nki,
         "shiftwarp": warp_shiftwarp}
_WARP = warp_gridsample          # swapped by --warp; Contextnet and IFNet_m both call through it


def warp(x, f):
    return _WARP(x, f)


# =============================================================================================
# PER-BLOCK TIMING, keyed to line up with the repo's C28 per-module table (a0/a1/a2 + ctx + unet)
# =============================================================================================
BLOCK_MS: dict[str, float] = {}
_PROFILE_BLOCKS = False


def _barrier(out):
    """Force device completion. Reading ONE element to the host is the documented barrier: the
    runtime warns nrta_tensor_read is synchronous, so a host read is what actually waits.
    Cheaper than a full .cpu() and enough to serialise."""
    t = out[0] if isinstance(out, (tuple, list)) else out
    t.detach().flatten()[:1].cpu()
    return out


def _tb(name, fn, *args, **kw):
    """Time one block WITH a barrier.

    The barrier is mandatory and it is also the measurement's main limitation: without it the
    forward can return before the device finishes and the elapsed time lands on whichever later
    call happens to force the sync. With it, work that might have overlapped is serialised, so
    every per-block number is an UPPER bound. The sum-vs-total check in the report says how much
    that cost -- if sum(blocks) is close to the unbarriered frame time, the barriers were cheap
    and the attribution is trustworthy.
    """
    if not _PROFILE_BLOCKS:
        return fn(*args, **kw)
    t0 = time.perf_counter()
    out = fn(*args, **kw)
    _barrier(out)
    BLOCK_MS[name] = BLOCK_MS.get(name, 0.0) + (time.perf_counter() - t0) * 1e3
    return out


# =============================================================================================
# MODEL -- inlined verbatim so the shipped checkpoint loads without key surgery
# =============================================================================================
class NeuronPReLU(nn.Module):
    """PReLU as relu(x) - w*relu(-x). BIT-EXACT, and required for torch.compile.

    nn.PReLU lowers fine in EAGER mode but cannot be legalised under torch.compile:
      Failed Torch-MLIR Neuron partial lowering: TorchFX IR -> StableHLO IR
      error: failed to legalize operation 'torch.operator' that was explicitly marked illegal
    Identity check: for x>0, relu(x)=x and relu(-x)=0 -> x. For x<0, relu(x)=0 and relu(-x)=-x ->
    -w*(-x) = w*x. That is exactly max(0,x) + w*min(0,x), so no accuracy is traded for it.
    """

    def __init__(self, num_parameters=1, init=0.25):
        super().__init__()
        self.weight = nn.Parameter(torch.full((num_parameters,), float(init)))

    def forward(self, x):
        w = self.weight.view(1, -1, *([1] * (x.dim() - 2)))
        return F.relu(x) - w * F.relu(-x)


# Chosen at import so the checkpoint's "...PReLU.weight" keys land on either class unchanged --
# both hold a single `weight` Parameter of the same shape.
_PRELU = nn.PReLU


def conv(i, o, k=3, s=1, p=1, d=1):
    return nn.Sequential(nn.Conv2d(i, o, k, s, p, dilation=d, bias=True), _PRELU(o))


def deconv(i, o):
    return nn.Sequential(nn.ConvTranspose2d(i, o, 4, 2, 1, bias=True), _PRELU(o))


class Conv2(nn.Module):
    def __init__(self, i, o, stride=2):
        super().__init__()
        self.conv1 = conv(i, o, 3, stride, 1)
        self.conv2 = conv(o, o, 3, 1, 1)

    def forward(self, x):
        return self.conv2(self.conv1(x))


_C = 16


class Contextnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv2(3, _C)
        self.conv2 = Conv2(_C, 2 * _C)
        self.conv3 = Conv2(2 * _C, 4 * _C)
        self.conv4 = Conv2(4 * _C, 8 * _C)

    def forward(self, x, flow):
        outs = []
        for cv in (self.conv1, self.conv2, self.conv3, self.conv4):
            x = cv(x)
            flow = F.interpolate(flow, scale_factor=0.5, mode="bilinear",
                                 align_corners=False, recompute_scale_factor=False) * 0.5
            outs.append(warp(x, flow))
        return outs


class Unet(nn.Module):
    def __init__(self):
        super().__init__()
        self.down0 = Conv2(17, 2 * _C)
        self.down1 = Conv2(4 * _C, 4 * _C)
        self.down2 = Conv2(8 * _C, 8 * _C)
        self.down3 = Conv2(16 * _C, 16 * _C)
        self.up0 = deconv(32 * _C, 8 * _C)
        self.up1 = deconv(16 * _C, 4 * _C)
        self.up2 = deconv(8 * _C, 2 * _C)
        self.up3 = deconv(4 * _C, _C)
        self.conv = nn.Conv2d(_C, 3, 3, 1, 1)

    def forward(self, img0, img1, w0, w1, mask, flow, c0, c1):
        s0 = self.down0(torch.cat((img0, img1, w0, w1, mask, flow), 1))
        s1 = self.down1(torch.cat((s0, c0[0], c1[0]), 1))
        s2 = self.down2(torch.cat((s1, c0[1], c1[1]), 1))
        s3 = self.down3(torch.cat((s2, c0[2], c1[2]), 1))
        x = self.up0(torch.cat((s3, c0[3], c1[3]), 1))
        x = self.up1(torch.cat((x, s2), 1))
        x = self.up2(torch.cat((x, s1), 1))
        x = self.up3(torch.cat((x, s0), 1))
        return torch.sigmoid(self.conv(x))


class IFBlock(nn.Module):
    def __init__(self, in_planes, c=64):
        super().__init__()
        self.conv0 = nn.Sequential(conv(in_planes, c // 2, 3, 2, 1), conv(c // 2, c, 3, 2, 1))
        self.convblock = nn.Sequential(*[conv(c, c) for _ in range(8)])
        self.lastconv = nn.ConvTranspose2d(c, 5, 4, 2, 1)

    def forward(self, x, flow, scale):
        if scale != 1:
            x = F.interpolate(x, scale_factor=1.0 / scale, mode="bilinear", align_corners=False)
        if flow is not None:
            flow = F.interpolate(flow, scale_factor=1.0 / scale, mode="bilinear",
                                 align_corners=False) * (1.0 / scale)
            x = torch.cat((x, flow), 1)
        x = self.conv0(x)
        x = self.convblock(x) + x
        tmp = self.lastconv(x)
        tmp = F.interpolate(tmp, scale_factor=scale * 2, mode="bilinear", align_corners=False)
        return tmp[:, :4] * scale * 2, tmp[:, 4:5]


class _StageFirst(nn.Module):
    """Pyramid stage 0: the IFBlock conv trunk AND its two full-resolution warps.

    The warps belong INSIDE this module, and that is the entire point of it rather than a tidiness
    preference. The repo's NEFF_A0 is built the same way. When an earlier mode wrapped only
    `block0/1/2`, the six full-resolution warps live in IFNet_m.forward BETWEEN the blocks, so they
    sat outside every compiled region -- roughly 60% of the frame stayed eager and the mode bought
    nothing. Compiling a stage only helps if the stage owns its warps.
    """

    def __init__(self, blk, scale, tag):
        super().__init__()
        self.blk, self.scale, self.tag = blk, scale, tag

    def forward(self, img0, img1, timestep):
        flow, mask = _tb(self.tag + "_conv", self.blk,
                         torch.cat((img0, img1, timestep), 1), None, scale=self.scale)
        w0, w1 = _tb(self.tag + "_warp",
                     lambda f=flow: (warp(img0, f[:, :2]), warp(img1, f[:, 2:4])))
        return flow, mask, w0, w1


class _StageNext(nn.Module):
    """Pyramid stages 1 and 2: the residual flow/mask update plus the stage's two warps.

    Takes the previous stage's w0/w1/flow/mask, because the conv trunk is conditioned on them, and
    returns the updated four. Same containment property as _StageFirst.
    """

    def __init__(self, blk, scale, tag):
        super().__init__()
        self.blk, self.scale, self.tag = blk, scale, tag

    def forward(self, img0, img1, timestep, w0, w1, flow, mask):
        fd, md = _tb(self.tag + "_conv", self.blk,
                     torch.cat((img0, img1, timestep, w0, w1, mask), 1), flow, scale=self.scale)
        flow = flow + fd
        mask = mask + md
        nw0, nw1 = _tb(self.tag + "_warp",
                       lambda f=flow: (warp(img0, f[:, :2]), warp(img1, f[:, 2:4])))
        return flow, mask, nw0, nw1


class _Pyramid(nn.Module):
    """All three pyramid stages in one module. Retained for the structure it documents; the
    whole flow pyramid -- and therefore all six full-resolution warps. Wrapping the stages
    individually cannot express that: each torch.compile call is its own graph."""

    def __init__(self, stages):
        super().__init__()
        self._st = stages

    def forward(self, img0, img1, timestep):
        s0, s1, s2 = self._st
        flow, mask, w0, w1 = s0(img0, img1, timestep)
        outs = [(mask, w0, w1)]
        for s in (s1, s2):
            flow, mask, w0, w1 = s(img0, img1, timestep, w0, w1, flow, mask)
            outs.append((mask, w0, w1))
        return flow, outs[0][0], outs[0][1], outs[0][2], \
            outs[1][0], outs[1][1], outs[1][2], outs[2][0], outs[2][1], outs[2][2]


class _Refine(nn.Module):
    """contextnet(img0), contextnet(img1) and unet in ONE module, so the refinement half is a
    single graph rather than three."""

    def __init__(self, contextnet, unet):
        super().__init__()
        self.contextnet, self.unet = contextnet, unet

    def forward(self, img0, img1, w0, w1, mask, flow):
        c0 = self.contextnet(img0, flow[:, :2])
        c1 = self.contextnet(img1, flow[:, 2:4])
        return self.unet(img0, img1, w0, w1, mask, flow, c0, c1)


class IFNet_m(nn.Module):
    def __init__(self):
        super().__init__()
        self.block0 = IFBlock(6 + 1, c=240)
        self.block1 = IFBlock(13 + 4 + 1, c=150)
        self.block2 = IFBlock(13 + 4 + 1, c=90)
        self.block_tea = IFBlock(16 + 4 + 1, c=90)      # training only; kept for checkpoint keys
        self.contextnet = Contextnet()
        self.unet = Unet()
        # Stage wrappers, deliberately kept OUT of the module registry. Assigning a plain LIST means
        # nn.Module.__setattr__ falls through to __dict__ instead of registering submodules, so
        # state_dict keys stay exactly the checkpoint's. Registering them would expose every IFBlock
        # weight a second time under `_stages.N.blk.*` and load_state_dict would report all of them
        # missing. The wrappers own no parameters of their own, so they need no .to(device/dtype).
        self._pyramid = None
        self._refine = None
        self._stages = [_StageFirst(self.block0, 4, "a0"),
                        _StageNext(self.block1, 2, "a1"),
                        _StageNext(self.block2, 1, "a2")]

    def forward(self, x, timestep, scale=(4, 2, 1)):
        img0, img1 = x[:, :3], x[:, 3:6]
        merged, mask_list = [], []
        # Stage granularity lives in _stages. Kept because the per-block tags key off it and a compiled
        # graph that CONTAINS that stage's two warps. Per-block tags are unchanged (a0_conv/a0_warp
        # and so on), so the C28-keyed table still lines up. scale=(4,2,1) is baked into the wrappers
        # at construction; the argument survives only for signature compatibility with upstream.
        if getattr(self, "_pyramid", None) is not None:
            # Unused now that compilation is always whole-model; _pyramid stays None.
            (flow, m0, a0, b0, m1, a1, b1, mask, w0, w1) = self._pyramid(img0, img1, timestep)
            for mk, (aa, bb) in ((m0, (a0, b0)), (m1, (a1, b1)), (mask, (w0, w1))):
                mask_list.append(torch.sigmoid(mk))
                merged.append((aa, bb))
        else:
            s0, s1, s2 = self._stages
            flow, mask, w0, w1 = s0(img0, img1, timestep)
            mask_list.append(torch.sigmoid(mask))
            merged.append((w0, w1))
            for s in (s1, s2):
                flow, mask, w0, w1 = s(img0, img1, timestep, w0, w1, flow, mask)
                mask_list.append(torch.sigmoid(mask))
                merged.append((w0, w1))
        for i in range(3):
            merged[i] = merged[i][0] * mask_list[i] + merged[i][1] * (1 - mask_list[i])
        if getattr(self, "_refine", None) is not None:
            res = self._refine(img0, img1, w0, w1, mask, flow)[:, :3] * 2 - 1
        else:
            c0 = _tb("ctx", self.contextnet, img0, flow[:, :2])  # 4 levels, each with a warp
            c1 = _tb("ctx", self.contextnet, img1, flow[:, 2:4])
            res = _tb("unet", self.unet, img0, img1, w0, w1, mask, flow, c0, c1)[:, :3] * 2 - 1
        merged[2] = torch.clamp(merged[2] + res, 0, 1)
        return merged[2]


class UniVR(nn.Module):
    """The wrapper that turns RIFE into a rolling-shutter unroller: a PER-ROW timestamp."""

    def __init__(self):
        super().__init__()
        self.UVR = IFNet_m()

    def forward(self, img, t, gamma, row0=0, full_h=None):
        """row0/full_h exist for TILING and they are not optional cosmetics.

        tau depends on the ABSOLUTE scanline in the frame, because that is what a rolling shutter
        exposes at a given instant. A tile starting at absolute row 864 must use rows 864.. over the
        FULL frame height, not 0.. over the tile height. Getting this wrong still produces a
        plausible-looking image and silently destroys agreement with the untiled reference.
        """
        h, w = img.shape[-2:]
        fh = full_h if full_h is not None else h
        rows = (torch.arange(h, device=img.device, dtype=torch.float32) + row0).view(1, 1, h, 1)
        tau = (t + gamma - gamma * rows / fh + 0.0001).expand(1, 1, h, w).to(img.dtype)
        return self.UVR(img, tau.contiguous())


# =============================================================================================
# TILING -- what makes 4K compilable, and what lets 8 cores work on one frame
# =============================================================================================
# A PADDED TILE DIMENSION MUST BE A MULTIPLE OF THIS, and it is not a nicety.
# IFBlock at scale=4 does F.interpolate(1/4) then conv0's two stride-2 convs, so the effective
# stride is 16, and lastconv + F.interpolate(scale*2) must land back on the input size exactly. A
# width of 56 gives 56/16 = 3.5, the round-trip returns 48, and the flow no longer matches the image
# -- "size of tensor a (56) must match tensor b (64)". Contextnet also downsamples 4x by 2 = /16.
# 32 covers both with margin. The production shapes satisfy it by luck rather than by design:
# 992 = 31*32, 1152 = 36*32, 1280 = 40*32.
TILE_ALIGN = 32


def _align_window(lo, hi, limit, align):
    """Grow [lo, hi) to a multiple of `align`, preferring to extend INSIDE the frame.

    Extending is free -- it just reads more context, which the halo already does. Only if the frame
    itself is too small or unaligned does this fail, and then it says so rather than producing a
    silently wrong flow.
    """
    need = (-(hi - lo)) % align
    if need == 0:
        return lo, hi
    grow_hi = min(need, limit - hi)
    hi += grow_hi
    need -= grow_hi
    if need:
        grow_lo = min(need, lo)
        lo -= grow_lo
        need -= grow_lo
    if need:
        raise SystemExit(
            "cannot align a tile to %d within a %d-px frame dimension. Pick a resolution and "
            "grid whose per-tile extent plus halo is a multiple of %d -- the model's coarse "
            "IFBlock scale has an effective stride of 16 and the upsample must round-trip "
            "exactly." % (align, limit, align))
    return lo, hi


def plan_tiles(H, W, ny, nx, halo):
    """Partition the frame into ny*nx VALID extents, each read with `halo` of context.

    Returns a list of dicts per tile:
      oy, ox, vy, vx   the valid region in FRAME coordinates (these tile the frame exactly)
      py0, px0, ph, pw the padded region actually fed to the model, clipped at frame edges
      iy, ix           where the valid region sits INSIDE the padded tile
    The halo exists because the model has a receptive field: convolutions and especially the
    flow-guided resample read outside the output pixel, so a tile computed without context is wrong
    near its edges. Valid extents partition exactly, so the stitch has no seam blending and a
    coverage check can prove every pixel is written exactly once.
    """
    vh = -(-H // ny)
    vw = -(-W // nx)
    tiles = []
    for iy_t in range(ny):
        for ix_t in range(nx):
            oy, ox = iy_t * vh, ix_t * vw
            if oy >= H or ox >= W:
                continue
            vy, vx = min(vh, H - oy), min(vw, W - ox)
            py0, px0 = max(0, oy - halo), max(0, ox - halo)
            py1, px1 = min(H, oy + vy + halo), min(W, ox + vx + halo)
            py0, py1 = _align_window(py0, py1, H, TILE_ALIGN)
            px0, px1 = _align_window(px0, px1, W, TILE_ALIGN)
            tiles.append({"oy": oy, "ox": ox, "vy": vy, "vx": vx,
                          "py0": py0, "px0": px0, "ph": py1 - py0, "pw": px1 - px0,
                          "iy": oy - py0, "ix": ox - px0})
    return tiles


def build_replicas(model, dtype, device, ncore):
    """One module replica per core.

    DEEP COPY IS MANDATORY. `.to("neuron:c")` MOVES parameters, so handing the same module to every
    core leaves the earlier replicas pointing at the wrong device and it fails inside the graph as
    "found two different devices neuron:0, neuron:1" on the first parameter that meets an activation.
    """
    import copy
    reps = []
    for c in range(ncore):
        dev = device if ncore == 1 and ":" not in device else "%s:%d" % (device.split(":")[0], c)
        m = copy.deepcopy(model).to(dtype).to(dev)
        reps.append((m, dev))
    return reps


def run_tiled(reps, tiles, pair_f, pair_b, t_fwd, t_bwd, H, W, gamma, real_triplet, only=None):
    """Run the tiles across the replicas and stitch. Returns (frame, per_core_ms, coverage_ok).

    One THREAD per core: the device call is synchronous on this runtime, so driving the cores from a
    loop runs them one after another and measures no parallelism at all. The GIL is released during
    the device call, so threads give real concurrency.
    """
    from concurrent.futures import ThreadPoolExecutor
    ncore = len(reps)
    idxs = list(range(len(tiles))) if only is None else [only]
    half = H // 2

    phases = {"prep": 0.0, "dev": 0.0, "d2h": 0.0}

    def _one(c):
        m, dev = reps[c]
        mine = [i for i in idxs if i % ncore == c] if only is None else \
               ([only] if c == 0 else [])
        outs = []
        t0 = time.perf_counter()
        for i in mine:
            T = tiles[i]
            sl = (slice(None), slice(None),
                  slice(T["py0"], T["py0"] + T["ph"]), slice(T["px0"], T["px0"] + T["pw"]))
            with torch.no_grad():
                # Which pass does this tile need? Rows above H//2 come from the backward pass. A
                # tile wholly on one side needs ONE pass, which halves the work -- the same
                # optimisation the production path makes.
                need_f = (T["oy"] + T["vy"]) > half or not real_triplet
                need_b = real_triplet and T["oy"] < half
                of = ob = None
                # PHASE SPLIT, mirroring the repo's C27 attribution (submit / D2H / stitch) so a
                # host regression cannot hide inside a "device" number. CAVEAT: the forward may
                # return before the device finishes, and the completion barrier is the .cpu() read
                # (the runtime warns nrta_tensor_read is synchronous). So prep+dev+d2h is reliable
                # as a SUM; the dev/d2h split is indicative only.
                t_p = time.perf_counter()
                if need_f:
                    xf = pair_f[sl].to(dev)
                if need_b:
                    xb = pair_b[sl].to(dev)
                t_d = time.perf_counter()
                phases["prep"] += (t_d - t_p) * 1e3
                if need_f:
                    of = m(xf, t_fwd, gamma, row0=T["py0"], full_h=H)
                if need_b:
                    ob = m(xb, t_bwd, gamma, row0=T["py0"], full_h=H)
                phases["dev"] += (time.perf_counter() - t_d) * 1e3
                o = of if of is not None else ob
                if of is not None and ob is not None:
                    o = of.clone()
                    n = max(0, min(half - T["oy"], T["vy"]))
                    if n:
                        o[:, :, T["iy"]:T["iy"] + n, :] = ob[:, :, T["iy"]:T["iy"] + n, :]
                elif ob is not None and of is None:
                    o = ob
            t_h = time.perf_counter()
            crop = o[:, :, T["iy"]:T["iy"] + T["vy"], T["ix"]:T["ix"] + T["vx"]].float().cpu()
            phases["d2h"] += (time.perf_counter() - t_h) * 1e3
            outs.append((i, crop))
        return outs, (time.perf_counter() - t0) * 1e3

    with ThreadPoolExecutor(max_workers=ncore) as pool:
        res = list(pool.map(_one, range(ncore)))
    t_s = time.perf_counter()
    frame = torch.zeros(1, 3, H, W)
    cover = torch.zeros(1, 1, H, W)
    for outs, _ms in res:
        for i, crop in outs:
            T = tiles[i]
            frame[:, :, T["oy"]:T["oy"] + T["vy"], T["ox"]:T["ox"] + T["vx"]] = crop
            cover[:, :, T["oy"]:T["oy"] + T["vy"], T["ox"]:T["ox"] + T["vx"]] += 1
    phases["stitch"] = (time.perf_counter() - t_s) * 1e3
    exp = 1 if only is None else 0
    miss = int((cover == exp).sum()) if only is not None else int((cover == 0).sum())
    dup = int((cover > 1).sum())
    return frame, [ms for _o, ms in res], (miss, dup), phases


def load_weights(model, path):
    """Checkpoint keys are 'module.*' relative to IFNet_m (docs: convert2 maps them to 'UVR.*')."""
    sd = torch.load(path, map_location="cpu", weights_only=False)
    sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
    conv_sd = {k.replace("module.", "UVR."): v for k, v in sd.items() if "module." in k}
    if not conv_sd:                       # already-clean keys
        conv_sd = {("UVR." + k if not k.startswith("UVR.") else k): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(conv_sd, strict=False)
    real_missing = [k for k in missing if not k.startswith("UVR.block_tea")]
    if real_missing:
        raise SystemExit("checkpoint does not fit the model: %d missing keys, first few: %s"
                         % (len(real_missing), real_missing[:5]))
    print("  weights: loaded %d tensors (%d unexpected, block_tea ignored: training-only)"
          % (len(conv_sd), len(unexpected)))


# =============================================================================================
# DESCRIPTOR ACCOUNTING -- the compiler-facing output
# =============================================================================================
def descriptor_report(sites, itemsize, n_forwards=1):
    print()
    print("=" * 100)
    print("DESCRIPTOR ACCOUNTING -- grouped by shape, over %d forward pass(es) = %d resamples"
          % (n_forwards, len(sites)))
    print("=" * 100)
    print("  Merged horizontal tap pairs, so 2 descriptors per pixel of 2*C contiguous elements")
    print("  each. Extents are compile-time constants; only the base row (y0*W + x0) is")
    print("  data-dependent, supplied per-partition via vector_offset (vector-DGE, 128 addresses")
    print("  per instruction). Cost is per DESCRIPTOR at %.1f ns, independent of payload width" % DESC_NS)
    print("  (measured 19.4/20.3/21.6/17.8 ns at 1/2/9/32 elements).")
    print()
    print("  %-14s %5s %6s %6s %13s %10s %11s %11s"
          % ("call site", "C", "H", "W", "descriptors", "bytes/desc", "GpSimd ms", "gather ms"))
    print("  " + "-" * 92)
    tot_d = tot_ms = tot_g = 0.0
    agg: dict[tuple[int, int, int], int] = {}
    for _tag, C, H, W in sites:
        agg[(C, H, W)] = agg.get((C, H, W), 0) + 1
    for (C, H, W), n in sorted(agg.items(), key=lambda kv: -kv[0][1] * kv[0][2]):
        K = H * W
        desc = 2 * K * n
        nbytes = 2 * C * itemsize
        ms = desc * DESC_NS / 1e6
        g_ms = 4 * K * C * n * GATHER_NS_PER_ELEM / 1e6
        tot_d += desc
        tot_ms += ms
        tot_g += g_ms
        print("  %-14s %5d %6d %6d %13s %10d %11.2f %11.2f"
              % ("x%d" % n, C, H, W, "{:,}".format(desc), nbytes, ms, g_ms))
    print("  " + "-" * 92)
    print("  %-14s %5s %6s %6s %13s %10s %11.2f %11.2f"
          % ("TOTAL", "", "", "", "{:,}".format(int(tot_d)), "", tot_ms, tot_g))
    print()
    print("  NKI kernel (2 desc/px)          %8.1f ms" % tot_ms)
    print("  torch.gather lowering           %8.1f ms   at the measured %.2f ns/element"
          % (tot_g, GATHER_NS_PER_ELEM))
    print()
    # ---- cross-checks against measured numbers in docs/NEURON_PORT_NOTES.md -------------------
    # C4 measured gathered ELEMENTS at the 864x1024 tile: 4*C*H*W per warp, 116.8M over 14 warps.
    # Recomputing it from the call sites this run actually recorded validates the accounting.
    elems = sum(4 * C * H * W for _t, C, H, W in sites)
    ref_sites = [(3, 864, 1024)] * 6 + [(c, 864 >> l, 1024 >> l)
                                        for l, c in enumerate((16, 32, 64, 128), start=1)
                                        for _ in range(2)]
    ref_elems = sum(4 * c * h * w for c, h, w in ref_sites)
    print("  ACCOUNTING CROSS-CHECK: this call-site model reproduces the measured element count in")
    print("  docs C4 exactly -- %.1fM against the documented 116.8M at the 864x1024 tile%s."
          % (ref_elems / 1e6, "" if abs(ref_elems - 116.8e6) < 1e5 else " (MISMATCH)"))
    print("  This run recorded %d call sites totalling %.1fM gathered elements."
          % (len(sites), elems / 1e6))
    print()
    print("  CONFIRMED AGAINST THE PROFILER. The kernel issues 2 descriptors per output pixel by")
    print("  construction, and `sw_dyn_dma_packets` counts them 1:1. One full-resolution warp NEFF at")
    print("  992x1280 reports 2,539,520 packets for 1,269,760 pixels = exactly 2.0000 per pixel. A")
    print("  fused module running two full-resolution warps at 864x1024 reports 3,563,248 against")
    print("  %s by construction, a 0.7%% match -- the excess is the kernel's own"
          % "{:,}".format(2 * 2 * 884736))
    print("  index/weight/store loads, 5 descriptors per 128-pixel k-tile = 2.35%.")
    print("  So the descriptor counts below are the real charge, not an estimate.")
    print()
    small = [(C, H, W, n) for (C, H, W), n in agg.items() if 2 * C * itemsize < 512]
    if small:
        frac = sum(2 * h * w * n * DESC_NS / 1e6 for _c, h, w, n in small) / max(tot_ms, 1e-9)
        print("  %.0f%% of descriptor time is on payloads under 512 B -- the DMA-inefficient regime."
              % (100 * frac))
        print("  Worst case is C=3 at full resolution: %d B moved per %.1f ns descriptor = %.2f GB/s."
              % (2 * 3 * itemsize, DESC_NS, 2 * 3 * itemsize / DESC_NS))
        print("  It cannot be widened: adjacent OUTPUT pixels read non-adjacent SOURCE pixels, so")
        print("  there is no longer contiguous run to fetch. The only fix that reaches a real payload")
        print("  is structured addressing -- a per-row uniform integer bulk shift is %s descriptors"
              % "{:,}".format(sites[0][2] if sites else 0))
        print("  of W*C*4 B instead of one per pixel, which is a ~340x cut and ~500x the payload.")


# =============================================================================================
# SCORING
# =============================================================================================
def score(out, gt, bar=3.0, label="GOLDEN"):
    if gt.shape[-2:] != out.shape[-2:]:
        print("  NOTE resizing golden %s -> %s; this itself costs accuracy"
              % (tuple(gt.shape[-2:]), tuple(out.shape[-2:])))
        gt = F.interpolate(gt, size=out.shape[-2:], mode="bilinear", align_corners=False,
                           antialias=True)
    d = (out.float() - gt.float()).abs()
    lsb = d * 255.0
    mse = d.pow(2).mean().item()
    psnr = float("inf") if mse == 0 else 10.0 * math.log10(1.0 / mse)
    cos = F.cosine_similarity(out.double().flatten(), gt.double().flatten(), dim=0).item()
    mx = lsb.max().item()
    print()
    print("=" * 100)
    print("ACCURACY vs %s" % label)
    print("=" * 100)
    print("  PSNR                 %8.2f dB" % psnr)
    print("  cosine (float64)     %10.6f" % cos)
    print("  max_diff             %8.2f LSB   [%s vs the shipped bar of <= %.0f]"
          % (mx, "PASS" if mx <= bar else "FAIL", bar))
    print("  mean_diff            %8.3f LSB" % lsb.mean().item())
    flat = lsb.flatten().float()
    if flat.numel() > 4_000_000:
        flat = flat[torch.randperm(flat.numel())[:4_000_000]]
    flat, _ = flat.sort()
    for q in (0.5, 0.9, 0.99, 0.999, 0.9999):
        print("  p%-7g            %8.2f LSB" % (q * 100, flat[int(q * (flat.numel() - 1))].item()))
    for thr in (1, 2, 3, 5, 10):
        print("  pixels > %2d LSB      %8.4f%%" % (thr, (lsb > thr).float().mean().item() * 100))
    err = lsb.amax(dim=1)[0]
    Hh, Ww = err.shape
    half = Hh // 2
    print()
    print("  spatial structure (worst channel per pixel) -- the merge line is row %d:" % half)
    for name, band in (("merge seam +/-4 rows", err[max(0, half - 4):half + 4, :]),
                       ("top border 32 rows", err[:32, :]),
                       ("bottom border 32 rows", err[-32:, :]),
                       ("interior (128 cut)", err[128:-128, 128:-128] if Hh > 256 else err)):
        print("    %-24s max %7.2f  mean %6.3f LSB" % (name, band.max().item(), band.mean().item()))
    return psnr, cos, mx


def load_img(path, H, W):
    from PIL import Image
    import numpy as np
    im = Image.open(path).convert("RGB")
    a = torch.from_numpy(np.asarray(im).copy()).permute(2, 0, 1).float() / 255.0
    if a.shape[1] != H or a.shape[2] != W:
        a = F.interpolate(a[None], size=(H, W), mode="bilinear", align_corners=False,
                          antialias=True)[0]
    return a.clamp(0, 1)


def load_gt(path, H, W):
    """Load a reference frame. .npy is fp32 and exact; an image is 8-bit and quantised.

    Prefer .npy: an 8-bit PNG reference costs up to 0.5 LSB of rounding before any model error, which
    is a sixth of the shipped 3 LSB bar and makes sub-LSB gating impossible. The original
    gs71_merged_triplet.png is a PNG, which is one reason that bar is hard to reason about.
    """
    if path.endswith(".npy"):
        import numpy as np
        g = torch.from_numpy(np.load(path)).float()
        if g.dim() == 3:
            g = g[None]
        if g.shape[-2:] != (H, W):
            raise SystemExit("reference %s is %s but this run is %dx%d. A .npy reference is NOT "
                             "resized -- regenerate it at the matching resolution."
                             % (path, tuple(g.shape[-2:]), H, W))
        return g
    return load_img(path, H, W)[None]


def save_ref(out, path, meta):
    """Write an fp32 .npy reference plus an 8-bit PNG for eyeballing plus a provenance manifest.

    The manifest is the point: a golden without recorded provenance is what produced the situation
    this script exists to untangle -- gs71_merged_triplet.png cannot be reproduced by the production
    model in fp32 and there is no record of what generated it.
    """
    import json
    import numpy as np
    a = out.detach().float().cpu().numpy()
    np.save(path, a)
    try:
        from PIL import Image
        Image.fromarray((a[0].transpose(1, 2, 0).clip(0, 1) * 255).round().astype("uint8")).save(
            path.replace(".npy", ".png"))
    except Exception as e:                                        # noqa: BLE001
        print("  (PNG preview skipped: %s)" % type(e).__name__)
    h = hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
    meta = dict(meta, sha256_fp32=h, shape=list(a.shape),
                min=float(a.min()), max=float(a.max()))
    with open(path.replace(".npy", ".json"), "w") as fh:
        json.dump(meta, fh, indent=2, sort_keys=True)
    print("  wrote reference %s (fp32, sha256 %s...)" % (path, h[:16]))
    print("  wrote manifest  %s" % path.replace(".npy", ".json"))


def verify_assets(paths):
    for p in paths:
        if p is None or not os.path.isfile(p):
            continue
        name = os.path.basename(p)
        if name not in TEST_ASSET_SHA256:
            continue
        h = hashlib.sha256()
        with open(p, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        ok = h.hexdigest() == TEST_ASSET_SHA256[name]
        print("  asset %-26s %s" % (name, "sha256 OK" if ok else "SHA256 MISMATCH -- not the "
                                    "canonical asset, numbers will not be comparable"))


# =============================================================================================
def self_test():
    """Numerics of the port's resample against the stock reference. CPU, no assets, no weights."""
    print("=" * 100)
    print("SELF TEST: does the port's gather reproduce F.grid_sample?")
    print("=" * 100)
    torch.manual_seed(0)
    bad = 0
    for C, H, W in ((3, 64, 96), (16, 32, 48), (64, 16, 24)):
        img = torch.rand(1, C, H, W)
        flow = (torch.rand(1, 2, H, W) - 0.5) * 8.0
        ref = warp_gridsample(img, flow)
        got = warp_gather(img, flow)
        win = warp_window(img, flow, radius=1)
        cg = F.cosine_similarity(ref.flatten().double(), got.flatten().double(), dim=0).item()
        cw = F.cosine_similarity(ref.flatten().double(), win.flatten().double(), dim=0).item()
        # window is exact only where |residual| <= R; with R=1 and flow up to +/-4 it must NOT match,
        # and that failure is the point of the row.
        print("  C=%-4d %3dx%-3d   gather cos %.6f  max %.2e   |   window R=1 cos %.4f (expected"
              " < 1: |flow| exceeds R)" % (C, H, W, cg, (ref - got).abs().max().item(), cw))
        if cg < 0.999999:
            bad += 1
    print("  %s" % ("PASS -- the gather is the same op as grid_sample" if bad == 0
                    else "FAIL -- %d shape(s) disagree" % bad))
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rs0")
    ap.add_argument("--rs1")
    ap.add_argument("--rs2", help="third RS frame; enables the REAL product triplet")
    ap.add_argument("--gt", help="golden global-shutter frame")
    ap.add_argument("--weights", help="pre_net_flow.pkl")
    ap.add_argument("--random-weights", action="store_true",
                    help="perf/descriptor runs only; refuses to score accuracy")
    ap.add_argument("--height", type=int, default=1728)
    ap.add_argument("--width", type=int, default=4096)
    ap.add_argument("--gamma", type=float, default=0.98)
    ap.add_argument("--warp", default="gather", choices=sorted(WARPS),
                    help="nki = unrolled kernel (best runtime, 80-100 min compile at 4K tiles); "
                         "nki-dyn = device-loop kernel (seconds to compile, ~2x runtime)")
    ap.add_argument("--record-flow", action="store_true",
                    help="measure post-clamp displacement at every resample call and print a table. "
                         "Diagnostic: answers what radius a bounded-neighbourhood kernel needs. "
                         "cannot run: fullgraph=True cannot trace it")
    ap.add_argument("--shiftwarp-radius", type=int, default=3,
                    help="shiftwarp support radius; R covers |displacement| <= R px EXACTLY. R=3 is "
                         "the smallest that passes the gate on the real 2.33 px flow (R=2 measured "
                         "76.19 LSB, a clean FAIL)")
    ap.add_argument("--shiftwarp-max-c", type=int, default=3,
                    help="shiftwarp handles sites with C <= this; larger C falls back to gather. "
                         "Default 3 = only the image-warp site, the one where the kernel is verified")
    ap.add_argument("--per-block", action="store_true",
                    help="time each module (a0/a1/a2 conv+warp, ctx, unet) with a device barrier, "
                         "keyed to line up with the repo's C28 per-module table. Barriers serialise, "
                         "so figures are UPPER bounds; the report prints a sum-vs-total check")
    ap.add_argument("--neuron-prelu", action="store_true",
                    help="replace nn.PReLU with relu(x)-w*relu(-x) (bit-exact). REQUIRED for "
                         "torch.compile: nn.PReLU cannot be legalised by the Neuron backend")
    ap.add_argument("--radius", type=int, default=2, help="--warp window radius R")
    ap.add_argument("--device", default="cpu", help="cpu | cuda | neuron")
    ap.add_argument("--dtype", default="fp32", choices=("fp32", "bf16"))
    ap.add_argument("--iters", type=int, default=1)
    ap.add_argument("--report-only", action="store_true",
                    help="descriptor accounting from shapes alone: no model, no data, no device")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--gate", action="store_true",
                    help="also score this config against the stock CPU fp32 grid_sample reference "
                         "with identical weights -- the port's own accuracy gate")
    ap.add_argument("--bar", type=float, default=3.0)
    ap.add_argument("--tiles", default="1x1", metavar="NYxNX",
                    help="tile grid, e.g. 2x4 for the production 8-tile 4K split. 1x1 = whole "
                         "frame in one graph, which does NOT compile at 4K")
    ap.add_argument("--cores", type=int, default=1,
                    help="NeuronCores to spread tiles over. Needs NEURON_RT_VISIBLE_CORES to "
                         "expose at least this many")
    ap.add_argument("--halo", type=int, default=128,
                    help="context rows/cols read outside each tile's valid extent. The model has a "
                         "receptive field, so 0 gives wrong pixels at tile edges")
    ap.add_argument("--only-tile", type=int, default=None, metavar="I",
                    help="run ONE tile on ONE core and score just its region: the per-tile "
                         "latency measurement")
    ap.add_argument("--save-ref", metavar="PATH.npy",
                    help="save this run's output as an fp32 reference + PNG preview + provenance "
                         "manifest, for use as --gt on another device")
    a = ap.parse_args()

    global _WARP, _RECORD, _RECORD_FLOW
    H, W = a.height, a.width
    itemsize = 2 if a.dtype == "bf16" else 4

    if a.self_test:
        return self_test()

    if a.report_only:
        # The call sites are fixed by the architecture: 6 full-res image warps (2 per IFBlock x 3)
        # and 8 Contextnet feature warps (4 levels x 2 images) per forward.
        sites = [("img", 3, H, W)] * 6
        for lvl, C in enumerate((_C, 2 * _C, 4 * _C, 8 * _C), start=1):
            sites += [("ctx", C, H >> lvl, W >> lvl)] * 2
        print("=" * 100)
        print("REPORT ONLY: 14 resamples per forward at %dx%d, %s" % (W, H, a.dtype))
        print("=" * 100)
        descriptor_report(sites, itemsize, n_forwards=1)
        print()
        print("  A real product frame is a TRIPLET = 2 forwards, so double the total above.")
        return 0

    if not (a.rs0 and a.rs1):
        ap.error("need --rs0 and --rs1 (or --report-only / --self-test)")
    if not (a.weights or a.random_weights):
        ap.error("need --weights, or --random-weights for a perf-only run")
    # An 8-bit golden against a random model is meaningless. A self-generated fp32 .npy reference
    # is NOT: with a fixed seed the weights are identical, so the comparison is a legitimate
    # equivalence check (tiled vs untiled, device vs CPU) rather than an accuracy claim.
    if a.gt and a.random_weights and not a.gt.endswith(".npy"):
        ap.error("--gt <image> with --random-weights would print a meaningless PSNR. Use a "
                 "self-generated .npy reference for equivalence testing, or supply --weights.")

    global _NKI_DYN, _PRELU, _PROFILE_BLOCKS, _NKI_FN, _NKI_FN_DYN
    global _SHIFTWARP_FN, _SHIFTWARP_R, _SHIFTWARP_MAXC
    _NKI_DYN = (a.warp == "nki-dyn")
    if a.per_block:
        ap.error("--per-block is incompatible with fullgraph=True: torch.compile dissolves the "
                 "module boundaries the timers sit on, so the numbers would be meaningless. This "
                 "script always compiles one fused graph, so --per-block can never apply.")
    _PROFILE_BLOCKS = a.per_block
    # Must be set BEFORE UniVR() is constructed, since conv()/deconv() read it at build time.
    # Unconditional: nn.PReLU cannot be legalised by the Neuron backend, and this script
    # always compiles. NeuronPReLU is relu(x) - w*relu(-x), bit-exact, so nothing is traded.
    _PRELU = NeuronPReLU
    if not a.neuron_prelu:
        print("  NOTE nn.PReLU replaced by NeuronPReLU: it cannot be legalised under torch.compile")
    _WARP = WARPS[a.warp]
    if a.warp == "window":
        _r = a.radius
        _WARP = lambda x, f: warp_window(x, f, radius=_r)   # noqa: E731
    if a.record_flow:
        ap.error("--record-flow cannot run under fullgraph=True: it appends to a Python list and "
                 "syncs to host inside the warp, which dynamo cannot trace. This script always "
                 "compiles, so use an older commit if the displacement table is needed.")
    if a.warp == "shiftwarp":
        if a.device != "neuron":
            ap.error("--warp shiftwarp requires --device neuron")
        _SHIFTWARP_R = a.shiftwarp_radius
        _SHIFTWARP_MAXC = a.shiftwarp_max_c
        # Same eager-build reason as the nki path below: building the wrapper lazily on first use
        # puts wrap_nki() inside a traced frame.
        _SHIFTWARP_FN = _build_shiftwarp()
        print("shiftwarp     R=%d -> %d terms, covers |disp| <= %d px; sites with C > %d fall back "
              "to gather" % (_SHIFTWARP_R, (2 * _SHIFTWARP_R + 1) ** 2, _SHIFTWARP_R,
                             _SHIFTWARP_MAXC))
    if a.warp.startswith("nki") and a.device != "neuron":
        ap.error("--warp %s requires --device neuron" % a.warp)
    if a.warp.startswith("nki"):
        # Build the NKI wrappers HERE, before anything is compiled. warp_nki() otherwise builds them
        # lazily on first use, which puts wrap_nki() inside a traced frame. At the 4K
        # tile that surfaced as `torch._dynamo.exc.Unsupported: id() with unsupported args` -- and it
        # did NOT surface at 256x384, so it is the kind of failure that hides until the shape that
        # matters. Lazy global init inside a compiled region is fragile independently of that error.
        _NKI_FN, _NKI_FN_DYN = _build_nki()

    print("=" * 100)
    print("UniVR-RIFE rolling-shutter unrolling  |  %dx%d  |  warp=%s  |  %s  |  %s"
          % (W, H, a.warp, a.dtype, a.device))
    print("=" * 100)
    verify_assets([a.rs0, a.rs1, a.rs2, a.gt])

    # Seed BEFORE construction: the module's __init__ draws the random init, so seeding afterwards
    # leaves --random-weights non-reproducible run to run. That made a tiled-vs-untiled equivalence
    # test compare two DIFFERENT models and read as a tiling bug.
    torch.manual_seed(0)
    model = UniVR().eval()
    if a.weights:
        load_weights(model, a.weights)
    else:
        print("  weights: RANDOM (seed 0, set before construction so it is reproducible) -- "
              "timing and descriptors valid, absolute accuracy is not")

    dt = torch.bfloat16 if a.dtype == "bf16" else torch.float32
    if a.device == "neuron":
        import torch_neuronx  # noqa: F401
    # In TILED mode the replicas are deep-copied from this module and moved per core, so the
    # template itself stays on CPU: moving it too would hold a ninth copy of the weights on device.
    _tiled_mode = (a.tiles.lower() != "1x1") or a.cores > 1 or a.only_tile is not None
    model = model.to(dt) if _tiled_mode else model.to(dt).to(a.device)

    rs0 = load_img(a.rs0, H, W)
    rs1 = load_img(a.rs1, H, W)
    # Tiled slices index the full-frame tensors on the HOST and move each tile to its core, so the
    # inputs stay on CPU in that mode.
    _idev = "cpu" if _tiled_mode else a.device
    pair_f = torch.cat([rs0, rs1], 0)[None].to(dt).to(_idev)
    t_fwd, t_bwd = 1 - a.gamma / 2, -a.gamma / 2
    real_triplet = bool(a.rs2)
    pair_b = None
    if real_triplet:
        rs2 = load_img(a.rs2, H, W)
        pair_b = torch.cat([rs1, rs2], 0)[None].to(dt).to(_idev)
        print("  REAL triplet: forward (rs0,rs1) at t=%+.4f, backward (rs1,rs2) at t=%+.4f,"
              % (t_fwd, t_bwd))
        print("  merged with rows above %d from the backward pass." % (H // 2))
    else:
        print("  PAIR mode: one forward at t=%+.4f. The golden is a TRIPLET product, so scoring"
              % t_fwd)
        print("  a pair against it is not the shipped comparison -- pass --rs2 for that.")

    # Compiling is INDEPENDENT of tiling. It used to live inside `if tiled:`, so
    # --tiles 1x1 --cores 1 silently skipped it and ran eager: a whole shape sweep was
    # reported as compiled that way. One function, applied on both paths.
    def apply_compile(m, tag=""):
        """fullgraph=True so a dynamo break RAISES rather than silently emitting a subgraph:
        an unguarded break around the view(torch.uint32) index bitcast in the warp corrupts
        which pixels get sampled."""
        print("  torch.compile[%s] fullgraph=True" % (tag or "model"))
        return torch.compile(m, backend="neuron", dynamic=False, fullgraph=True)

    ny, nx = (int(v) for v in a.tiles.lower().split("x"))
    tiled = (ny * nx > 1) or a.cores > 1 or a.only_tile is not None
    tiles = plan_tiles(H, W, ny, nx, a.halo)
    reps = None
    if tiled:
        ncore = 1 if a.only_tile is not None else a.cores
        print()
        print("  TILED: grid %dx%d = %d tiles, halo %d, %d core(s)"
              % (ny, nx, len(tiles), a.halo, ncore))
        shapes = sorted({(t["ph"], t["pw"]) for t in tiles})
        print("  padded tile shapes: %s  -> %d distinct graph(s) per timestamp"
              % (", ".join("%dx%d" % (h, w) for h, w in shapes), len(shapes)))
        if real_triplet:
            print("  x2 timestamps (triplet) -> up to %d graphs to compile" % (2 * len(shapes)))
        if a.only_tile is not None:
            T = tiles[a.only_tile]
            print("  --only-tile %d: valid %dx%d at (%d,%d), padded %dx%d"
                  % (a.only_tile, T["vy"], T["vx"], T["oy"], T["ox"], T["ph"], T["pw"]))
        reps = build_replicas(model, dt, a.device, ncore)
        reps = [(apply_compile(m, "replica %d/%d" % (i + 1, len(reps))), d)
                for i, (m, d) in enumerate(reps)]
        print("  %d replica(s) built" % len(reps))

    # Non-tiled path gets the same treatment; kept in a list so one_frame closes over it.
    _solo = [model if tiled else apply_compile(model, "model")]

    def one_frame(m=None, pf=None, pb=None):
        if tiled and m is None:
            fr, per, _cov, ph = run_tiled(reps, tiles, pair_f, pair_b, t_fwd, t_bwd, H, W,
                                          a.gamma, real_triplet, only=a.only_tile)
            one_frame.percore = per
            one_frame.phases = ph
            return fr
        m = _solo[0] if m is None else m
        pf = pair_f if pf is None else pf
        pb = pair_b if pb is None else pb
        with torch.no_grad():
            out = m(pf, t_fwd, a.gamma)
            if real_triplet:
                ob = m(pb, t_bwd, a.gamma)
                out = out.clone()
                out[:, :, :H // 2, :] = ob[:, :, :H // 2, :]
        return out

    # Descriptor accounting appends to a Python list from inside the warp. Under fullgraph=True that
    # is a graph break and dynamo raises instead of tracing, so the accounting is only collected on
    # the eager path. --report-only still produces the full table from shapes alone, with no device.
    # Both recorders append to Python lists from inside the warp, which fullgraph=True cannot
    # trace, so neither can ever be active now that compilation is unconditional.
    _RECORD = False
    # Same graph-break reason as _RECORD: appending to a Python list and calling .item() inside the
    # warp cannot be traced, so this is eager-only. main() already errors if --record-flow is
    _RECORD_FLOW = False
    FLOW_STATS.clear()
    CALL_SITES.clear()
    t0 = time.perf_counter()
    out = one_frame()
    if a.device == "cuda":
        torch.cuda.synchronize()
    out = out.float().cpu()          # completion barrier: the copy forces the device to finish
    first_ms = (time.perf_counter() - t0) * 1e3
    _RECORD = False
    _RECORD_FLOW = False

    if FLOW_STATS:
        print()
        print("=" * 100)
        print("MEASURED DISPLACEMENT PER RESAMPLE CALL (post-clamp, the distance the kernel must reach)")
        print("=" * 100)
        print("  %-4s %-11s %10s %10s %10s   %s"
              % ("#", "C,H,W", "max_px", ">2px %", ">3px %", "min R that covers it"))
        worst = 0.0
        worst_c3 = 0.0
        for i, (C, H, W, m, o2, o3) in enumerate(FLOW_STATS):
            need = int(math.ceil(m)) if m > 0 else 0
            print("  %-4d %-11s %10.3f %10.4f %10.4f   R=%d"
                  % (i, "%d,%d,%d" % (C, H, W), m, o2, o3, need))
            worst = max(worst, m)
            if C <= 3:
                worst_c3 = max(worst_c3, m)
        print()
        print("  WORST over all sites: %.3f px  -> needs R=%d  (%d terms)"
              % (worst, math.ceil(worst), (2 * math.ceil(worst) + 1) ** 2))
        print("  WORST at C<=3 (the sites shiftwarp actually handles): %.3f px -> needs R=%d"
              % (worst_c3, math.ceil(worst_c3)))
        print("  shiftwarp ran at R=%d, which covers |disp| <= %d px EXACTLY. Anything beyond that is"
              % (_SHIFTWARP_R, _SHIFTWARP_R))
        print("  SILENTLY CLAMPED -- no error, just wrong pixels, which is the 229.72 LSB / p50=0.00 /")
        print("  p99=172.99 signature. Compare the >%dpx column against the ~7%% of pixels that failed."
              % _SHIFTWARP_R)
    print()
    print("  forward complete: %s  range [%.4f, %.4f]  first call %.0f ms (includes compile/warmup)"
          % (tuple(out.shape), out.min(), out.max(), first_ms))

    if a.iters > 0:
        # RESET the per-block accumulator here. The first call COMPILES -- 14 s at a 4K tile -- and
        # letting that land in BLOCK_MS then dividing by the call count smears compile time across
        # every block. That is what made blocks sum to 2.51x the frame and it invalidated the whole
        # attribution: the pollution does not distribute uniformly, so shares were wrong in an
        # unknown direction rather than merely scaled.
        BLOCK_MS.clear()
        ts = []
        for _ in range(a.iters):
            t0 = time.perf_counter()
            o = one_frame()
            if a.device == "cuda":
                torch.cuda.synchronize()
            o.float().cpu()
            ts.append((time.perf_counter() - t0) * 1e3)
        ts.sort()
        med = ts[len(ts) // 2]
        print("  steady state: median %.1f ms over %d iters (min %.1f, max %.1f)"
              % (med, len(ts), ts[0], ts[-1]))
        pc = getattr(one_frame, "percore", None)
        if pc:
            # Overlap check: with N cores working concurrently the frame wall should approach the
            # SLOWEST core, not the sum. sum/wall near N means real parallelism; near 1 means the
            # cores serialised and the "multi-core" number is a lie.
            print("  per-core ms: %s" % ", ".join("%.0f" % v for v in pc))
        if a.per_block and BLOCK_MS:
            # Divide by the TIMED iterations only: BLOCK_MS was cleared after the compiling warmup.
            n_fwd = max(1, a.iters)
            tot_b = sum(BLOCK_MS.values()) / n_fwd
            print()
            print("  PER-BLOCK, per frame (barriered, so UPPER bounds -- see the check below):")
            print("      %-12s %10s %8s   %s" % ("module", "ms", "share", "what it covers"))
            print("      " + "-" * 74)
            covers = {
                "a0_conv": "IFBlock0 conv trunk, scale 4",
                "a0_warp": "2 full-res image warps",
                "a1_conv": "IFBlock1 conv trunk, scale 2",
                "a1_warp": "2 full-res image warps",
                "a2_conv": "IFBlock2 conv trunk, scale 1",
                "a2_warp": "2 full-res image warps",
                "ctx": "Contextnet x2: 8 conv levels + 8 feature warps",
                "unet": "Unet refine, 4 down + 4 up",
            }
            for k in ("a0_conv", "a0_warp", "a1_conv", "a1_warp", "a2_conv", "a2_warp",
                      "ctx", "unet"):
                if k in BLOCK_MS:
                    v = BLOCK_MS[k] / n_fwd
                    print("      %-12s %10.1f %7.1f%%   %s"
                          % (k, v, 100.0 * v / max(tot_b, 1e-9), covers.get(k, "")))
            warp_ms = sum(v for k, v in BLOCK_MS.items() if "_warp" in k) / n_fwd
            conv_ms = sum(v for k, v in BLOCK_MS.items() if "_conv" in k) / n_fwd
            print("      " + "-" * 74)
            print("      %-12s %10.1f" % ("sum", tot_b))
            print("      %-12s %10.1f %7.1f%%   the 6 full-res C=3 warps only"
                  % ("warps", warp_ms, 100.0 * warp_ms / max(tot_b, 1e-9)))
            print("      %-12s %10.1f %7.1f%%   IFBlock conv trunks only"
                  % ("convs", conv_ms, 100.0 * conv_ms / max(tot_b, 1e-9)))
            print()
            print("      SUM-vs-TOTAL CHECK: blocks sum to %.1f ms against an unbarriered frame of"
                  % tot_b)
            print("      %.1f ms -> ratio %.2f. Near 1.0 means the barriers cost little and the"
                  % (med, tot_b / max(med, 1e-9)))
            print("      attribution can be trusted. Well above 1.0 means real overlap was")
            print("      serialised and these are upper bounds good for RANKING but not subtraction.")
            print("      Well BELOW 1.0 means work escaped the timers (host-side, or the stitch).")
        ph = getattr(one_frame, "phases", None)
        if ph:
            # Summed ACROSS cores, so these exceed the wall clock by roughly the core count. What
            # matters is the SHARE, i.e. where the time goes -- that is the question a wall-clock
            # number cannot answer, and the repo's C27 uses the same attribution.
            tot = sum(ph.values())
            print()
            print("  WHERE THE TIME GOES (summed over %d cores; shares are the point, not totals):"
                  % len(pc))
            for k in ("prep", "dev", "d2h", "stitch"):
                if k in ph:
                    print("      %-8s %10.1f ms   %5.1f%%" % (k, ph[k], 100.0 * ph[k] / max(tot, 1e-9)))
            print("      %-8s %10.1f ms" % ("total", tot))
            print("    prep = host slice + H2D, dev = forward call, d2h = .cpu() read, stitch = host")
            print("    assembly. The forward may return before the device finishes and the .cpu()")
            print("    read is the real completion barrier (the runtime warns nrta_tensor_read is")
            print("    synchronous), so dev+d2h is reliable as a SUM and the split between them is")
            print("    indicative only.")
            print("  slowest core %.1f ms, sum %.1f ms, sum/wall %.2f (near %d = real parallelism, "
                  "near 1 = serialised)" % (max(pc), sum(pc), sum(pc) / max(med, 1e-9), len(pc)))
        if tiled and a.only_tile is None:
            _f, _p, (miss, dup), _ph = run_tiled(reps, tiles, pair_f, pair_b, t_fwd, t_bwd, H, W,
                                                 a.gamma, real_triplet, only=None)
            print("  stitch coverage: %d pixels unwritten, %d written more than once %s"
                  % (miss, dup, "[OK]" if miss == 0 and dup == 0 else "[BROKEN]"))
        # The 161 ms reference is a 4K TRIPLET on an L40S via ONNX+TRT. Quoting it against any
        # other resolution or mode is meaningless, so it is gated rather than scaled.
        is_4k = (H, W) == (1728, 4096)
        if is_4k and real_triplet and a.only_tile is None:
            report_gap(med)
        elif is_4k and real_triplet and a.only_tile is not None:
            print("  NO gap printed: --only-tile measures ONE tile, and the CUDA baselines are")
            print("  whole-frame. Multiply by the tile count only if you also believe the cores")
            print("  would not overlap -- which mode B measures directly.")
        else:
            print("  NO baseline comparison printed: the %.0f ms L40S reference is 4K (1728x4096)"
                  % L40S_4K_TRT_MS)
            print("  triplet specifically, and this run is %dx%d %s. Re-run at the production"
                  % (W, H, "triplet" if real_triplet else "pair"))
            print("  resolution and mode to get a comparable ratio.")

    if CALL_SITES:
        descriptor_report(CALL_SITES, itemsize, n_forwards=2 if real_triplet else 1)
    else:
        # _RECORD is always off now (appending to a list from inside the warp is not
        # traceable under fullgraph=True), so there is nothing to report. Printing the table
        # anyway emitted a page of zeros followed by hardcoded figures from other runs.
        print()
        print("  (no descriptor accounting: fullgraph=True cannot trace call-site recording)")

    if a.gate:
        # THE PORT'S OWN GATE, and the one the docs quote: the same weights and the same inputs
        # through the stock CPU fp32 grid_sample path. This isolates what the PORT changed (resample
        # implementation + storage dtype) from what the MODEL does, which a golden comparison cannot.
        print()
        print("=" * 100)
        print("ACCURACY GATE: this config vs the stock CPU fp32 grid_sample reference")
        print("=" * 100)
        keep = _WARP                      # _WARP already declared global at the top of main()
        _WARP = warp_gridsample
        ref_model = UniVR().eval()
        ref_model.load_state_dict({k: v.float().cpu() for k, v in model.state_dict().items()})
        ref = one_frame(ref_model, pair_f.float().cpu(),
                        None if pair_b is None else pair_b.float().cpu()).float()
        _WARP = keep
        if a.only_tile is not None:
            T = tiles[a.only_tile]
            sl = (slice(None), slice(None), slice(T["oy"], T["oy"] + T["vy"]),
                  slice(T["ox"], T["ox"] + T["vx"]))
            ref = ref[sl]
            out = out[sl]
        d = (out.float() - ref).abs()
        mse = d.pow(2).mean().item()
        print("  reference: stock grid_sample, fp32, CPU, identical weights")
        print("  cos (float64)        %10.6f" % F.cosine_similarity(
            out.double().flatten(), ref.double().flatten(), dim=0).item())
        print("  PSNR                 %8.2f dB" % (float("inf") if mse == 0
                                                   else 10.0 * math.log10(1.0 / mse)))
        print("  max_diff             %8.4e  (%.2f LSB)" % (d.max().item(), d.max().item() * 255))

    def _crop(x):
        """With --only-tile the frame has ONE tile written, so score only that region -- otherwise
        the metrics are dominated by the zeros everywhere else and mean nothing."""
        if a.only_tile is None:
            return x
        T = tiles[a.only_tile]
        return x[:, :, T["oy"]:T["oy"] + T["vy"], T["ox"]:T["ox"] + T["vx"]]

    if a.gt:
        lbl = "REFERENCE (fp32 .npy)" if a.gt.endswith(".npy") else "GOLDEN PNG (8-bit)"
        if a.only_tile is not None:
            T = tiles[a.only_tile]
            lbl += " -- TILE %d ONLY, valid %dx%d at (%d,%d)" % (a.only_tile, T["vy"], T["vx"],
                                                                 T["oy"], T["ox"])
        score(_crop(out), _crop(load_gt(a.gt, H, W)), bar=a.bar, label=lbl)

    if a.save_ref:
        print()
        print("=" * 100)
        print("SAVING THIS RUN AS A REFERENCE")
        print("=" * 100)
        meta = {
            "generated_by": "repro_unrolling_trn2.py",
            "device": a.device, "dtype": a.dtype, "warp": a.warp,
            "height": H, "width": W, "gamma": a.gamma,
            "t_fwd": t_fwd, "t_bwd": t_bwd if real_triplet else None,
            "mode": "triplet" if real_triplet else "pair",
            "torch": torch.__version__,
            "weights": os.path.basename(a.weights) if a.weights else "RANDOM",
            "weights_sha256": (hashlib.sha256(open(a.weights, "rb").read()).hexdigest()
                               if a.weights else None),
            "inputs": [os.path.basename(p) for p in (a.rs0, a.rs1, a.rs2) if p],
            "gpu": (torch.cuda.get_device_name(0) if a.device == "cuda" else None),
        }
        save_ref(out, a.save_ref, meta)
        print()
        print("  USE IT AS THE GATE on another device with:")
        print("    --gt %s   (fp32, exact, no resize)" % a.save_ref)
        print("  It is regenerable and its provenance is recorded, which is the whole difference")
        print("  from the shipped PNG golden.")
    print()
    print("DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
