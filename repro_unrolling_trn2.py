
from __future__ import annotations

import argparse
import hashlib
import math
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F


for _n, _v in (("cache_size_limit", 64), ("accumulated_cache_size_limit", 256),
               ("recompile_limit", 64), ("accumulated_recompile_limit", 256)):
    if hasattr(torch._dynamo.config, _n):
        setattr(torch._dynamo.config, _n, _v)
_eff = [getattr(torch._dynamo.config, _n)
        for _n in ("cache_size_limit", "recompile_limit")
        if hasattr(torch._dynamo.config, _n)]
if not _eff or min(_eff) < 64:
    raise SystemExit(
        "dynamo recompile limit is %r, expected >= 64. The config attribute was renamed and "
        "this build exposes neither a settable cache_size_limit nor recompile_limit. Fix the "
        "name here BEFORE running: under fullgraph=True this is a hard error that only shows up "
        "at 8 cores, after every compile is already paid for." % (_eff,))


print("  dynamo recompile limit: %d (must exceed the per-tile recompile count, observed 22)"
      % min(_eff), file=sys.stderr)


try:
    import nki
    import nki.isa as nisa
    import nki.language as nl
    from nki.isa.constants import oob_mode
    _HAVE_NKI = True
except ImportError:
    _HAVE_NKI = False


DESC_NS = 26.5
GATHER_NS_PER_ELEM = 11.35


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
    for k, (ms, _src, _why) in sorted(CUDA_BASELINE_4K_TRIPLET.items(), key=lambda kv: kv[1][0]):
        print("  gap vs %-18s %8.1f ms  %.2fx" % (k, ms, trn2_ms / ms))

TEST_ASSET_SHA256 = {
    "rs70.png": "d71d165877c25bf915409eb44ba318bac7cab5ff3666d37b9d9c5626e62bdacb",
    "rs71.png": "4a90f52bbbc3e17d4afb595d7a5a579db67b2994a650d25949203227daa5cd59",
    "rs72.png": "e4744b5bb618c40ac09ba22b2cc8a0b95246d23efbae98b1b6f98b2e5a6fceb4",
    "gs71_merged_triplet.png": "df0b97b396f17b09e22218202de6bc6208253d5c205703a495c07dc4ee2c6dd8",
}


CALL_SITES: list[tuple[str, int, int, int]] = []
_RECORD = False


FLOW_STATS: list[tuple[int, int, int, float, float, float]] = []
_RECORD_FLOW = False


def _bilinear_terms(tenInput, tenFlow):
    B, C, H, W = tenInput.shape
    dev, dt = tenFlow.device, torch.float32
    gx = torch.arange(W, device=dev, dtype=dt).view(1, 1, 1, W)
    gy = torch.arange(H, device=dev, dtype=dt).view(1, 1, H, 1)
    sx = (gx + tenFlow[:, 0:1].to(dt)).clamp(0.0, W - 1.0)
    sy = (gy + tenFlow[:, 1:2].to(dt)).clamp(0.0, H - 1.0)
    if _RECORD_FLOW:


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
    if _RECORD:
        CALL_SITES.append(("window", tenInput.shape[1], tenInput.shape[2], tenInput.shape[3]))
    x0, y0, ax, ay, (B, C, H, W) = _bilinear_terms(tenInput, tenFlow)
    gx = torch.arange(W, device=tenInput.device, dtype=torch.float32).view(1, 1, 1, W)
    gy = torch.arange(H, device=tenInput.device, dtype=torch.float32).view(1, 1, H, 1)
    rx = (x0 + ax) - gx
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


_NKI_FN = None
_NKI_FN_DYN = None


def _build_nki():
    if not _HAVE_NKI:
        raise SystemExit("--warp nki needs the Neuron toolchain (nki, torch_neuronx) installed")

    @nki.jit
    def bilinear_2x2_gather_blend(img, idx_tl, w_tl, w_tr, w_bl, w_br, row_w):
        K = idx_tl.shape[0]
        C = img.shape[1]
        dtype = img.dtype


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


_NKI_DYN = False


def warp_nki(tenInput, tenFlow):
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

    fn = _NKI_FN_DYN if (_NKI_DYN and NB % 128 == 0) else _NKI_FN
    out = fn(img_nc, idx_tl,
             ((1 - wx) * (1 - wy)).contiguous(), (wx * (1 - wy)).contiguous(),
             ((1 - wx) * wy).contiguous(), (wx * wy).contiguous(), W)
    return out.reshape(B, N, C).permute(0, 2, 1).reshape(B, C, H, W)


_NKL_GS_FN = None
_NKL_GATHER_METHOD = None
_NKL_MAX_INDICES = None


def _build_gridsample_nkl():
    try:
        from nkilib.experimental.indirect.grid_sample import grid_sample
    except ImportError as e:
        raise SystemExit(
            "--warp gridsample-nkl needs nkilib.experimental.indirect.grid_sample from "
            "KaenaNeuronKernelLibrary (CR-288764575, OPEN at revision 3 -- NOT merged, so it is "
            "not in a released image). Import failed: %s\n"
            "Put the package on PYTHONPATH or bake it into the image. Check the import BEFORE "
            "spending a compile budget: this raises in a second, a bad run costs hours." % e)
    from torch_neuronx.nki_hop import wrap_nki
    return wrap_nki(grid_sample)


def warp_gridsample_nkl(tenInput, tenFlow):
    if _RECORD:
        CALL_SITES.append(("gridsample-nkl", tenInput.shape[1], tenInput.shape[2],
                           tenInput.shape[3]))
    global _NKL_GS_FN
    if _NKL_GS_FN is None:
        _NKL_GS_FN = _build_gridsample_nkl()
    B, C, H, W = tenInput.shape
    dev = tenFlow.device


    hor = torch.linspace(-1.0, 1.0, W, device=dev, dtype=tenFlow.dtype).view(1, 1, 1, W).expand(B, -1, H, -1)
    ver = torch.linspace(-1.0, 1.0, H, device=dev, dtype=tenFlow.dtype).view(1, 1, H, 1).expand(B, -1, -1, W)
    grid = torch.cat([hor, ver], 1)
    f = torch.cat([tenFlow[:, 0:1] / ((W - 1.0) / 2.0), tenFlow[:, 1:2] / ((H - 1.0) / 2.0)], 1)
    g = (grid + f).permute(0, 2, 3, 1).contiguous()

    gm = _NKL_GATHER_METHOD or ("transpose" if tenInput.element_size() == 2 else "copy")
    value = tenInput.permute(0, 2, 3, 1).contiguous()
    out = _NKL_GS_FN(value, g,
                     sampling_mode="bilinear",
                     coord_mode="minus_one_one",
                     input_layout="NHWC",
                     align_corners=True,
                     padding_mode="border",
                     max_indices_per_indirect=_NKL_MAX_INDICES,
                     gather_method=gm)
    return out.permute(0, 3, 1, 2)


_SHIFTWARP_FN = None
_SHIFTWARP_R = 3
_SHIFTWARP_MAXC = 3


def _build_shiftwarp():
    if not _HAVE_NKI:
        raise SystemExit("--warp shiftwarp needs the Neuron toolchain (nki, torch_neuronx) installed")
    from nki_shift_warp import shift_warp_band
    from torch_neuronx.nki_hop import wrap_nki
    return wrap_nki(shift_warp_band)


def warp_shiftwarp(tenInput, tenFlow):
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
         "shiftwarp": warp_shiftwarp, "gridsample-nkl": warp_gridsample_nkl}
_WARP = warp_gridsample


def warp(x, f):
    return _WARP(x, f)


def build_warp_region(base, region, device, annotate=False):
    if region == "eager":
        if annotate:
            try:
                from torch.profiler import record_function
            except Exception:
                from torch.autograd.profiler import record_function
            def _eager_region(x, f):

                with record_function("resample:C%d:%dx%d" % (x.shape[1], x.shape[2], x.shape[3])):
                    return base(x, f)
            fn = _eager_region
        else:
            fn = base
    elif region == "cpu":
        inner = torch.compile(base, dynamic=False)
        def _cpu_region(x, f):
            dev = x.device
            return inner(x.cpu(), f.cpu()).to(dev)
        fn = _cpu_region
    else:
        fn = torch.compile(base, backend="neuron", dynamic=False)
    try:
        return torch._dynamo.disable(fn)
    except Exception as e:
        print("  WARNING: torch._dynamo.disable unavailable (%s); the region may be INLINED into "
              "the outer graph, which would silently defeat the split" % type(e).__name__)
        return fn


BLOCK_MS: dict[str, float] = {}
_PROFILE_BLOCKS = False


def _barrier(out):
    t = out[0] if isinstance(out, (tuple, list)) else out
    t.detach().flatten()[:1].cpu()
    return out


def _tb(name, fn, *args, **kw):
    if not _PROFILE_BLOCKS:
        return fn(*args, **kw)
    t0 = time.perf_counter()
    out = fn(*args, **kw)
    _barrier(out)
    BLOCK_MS[name] = BLOCK_MS.get(name, 0.0) + (time.perf_counter() - t0) * 1e3
    return out


class NeuronPReLU(nn.Module):

    def __init__(self, num_parameters=1, init=0.25):
        super().__init__()
        self.weight = nn.Parameter(torch.full((num_parameters,), float(init)))

    def forward(self, x):
        w = self.weight.view(1, -1, *([1] * (x.dim() - 2)))
        return F.relu(x) - w * F.relu(-x)


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
            flow = do_down(flow, 2) * 0.5
            outs.append(warp(x, flow))
        return outs


_PRECOMPUTE_RESIZE = False

_RESIZE_TAPS: dict = {}


def _resize_taps(in_sz, f, dtype, device):
    key = (int(in_sz), float(f), str(dtype), str(device))
    e = _RESIZE_TAPS.get(key)
    if e is None:
        out_sz = int(in_sz * f)
        o = torch.arange(out_sz, dtype=torch.float64)
        src = ((o + 0.5) / f - 0.5).clamp(0, in_sz - 1)
        i0 = src.floor().to(torch.long)
        i1 = torch.minimum(i0 + 1, torch.tensor(in_sz - 1))
        wr = src - i0
        e = (i0.to(device), i1.to(device),
             (1.0 - wr).to(dtype).to(device), wr.to(dtype).to(device), out_sz)
        _RESIZE_TAPS[key] = e
    return e


def resize_precomputed(x, f):
    hi0, hi1, hwl, hwr, Ho = _resize_taps(x.shape[2], f, x.dtype, x.device)
    wi0, wi1, wwl, wwr, Wo = _resize_taps(x.shape[3], f, x.dtype, x.device)
    t = x.index_select(2, hi0) * hwl.view(1, 1, Ho, 1) + \
        x.index_select(2, hi1) * hwr.view(1, 1, Ho, 1)
    return t.index_select(3, wi0) * wwl.view(1, 1, 1, Wo) + \
        t.index_select(3, wi1) * wwr.view(1, 1, 1, Wo)


def do_down(x, inv):
    if _PRECOMPUTE_RESIZE:
        return resize_precomputed(x, 1.0 / inv)
    return F.interpolate(x, scale_factor=1.0 / inv, mode="bilinear", align_corners=False)


def do_up(x, s):
    if _PRECOMPUTE_RESIZE:
        return resize_precomputed(x, float(s))
    return F.interpolate(x, scale_factor=s, mode="bilinear", align_corners=False)


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
            x = do_down(x, scale)
        if flow is not None:


            if not (_PRECOMPUTE_RESIZE and scale == 1):
                flow = do_down(flow, scale) * (1.0 / scale)
            x = torch.cat((x, flow), 1)
        x = self.conv0(x)
        x = self.convblock(x) + x
        tmp = self.lastconv(x)
        tmp = do_up(tmp, scale * 2)
        return tmp[:, :4] * scale * 2, tmp[:, 4:5]


class _StageFirst(nn.Module):

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
        self.block_tea = IFBlock(16 + 4 + 1, c=90)
        self.contextnet = Contextnet()
        self.unet = Unet()


        self._pyramid = None
        self._refine = None
        self._stages = [_StageFirst(self.block0, 4, "a0"),
                        _StageNext(self.block1, 2, "a1"),
                        _StageNext(self.block2, 1, "a2")]

    def forward(self, x, timestep, scale=(4, 2, 1)):
        img0, img1 = x[:, :3], x[:, 3:6]
        merged, mask_list = [], []


        if getattr(self, "_pyramid", None) is not None:

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
            c0 = _tb("ctx", self.contextnet, img0, flow[:, :2])
            c1 = _tb("ctx", self.contextnet, img1, flow[:, 2:4])
            res = _tb("unet", self.unet, img0, img1, w0, w1, mask, flow, c0, c1)[:, :3] * 2 - 1
        merged[2] = torch.clamp(merged[2] + res, 0, 1)
        return merged[2]


class UniVR(nn.Module):

    def __init__(self):
        super().__init__()
        self.UVR = IFNet_m()

    def forward(self, img, t, gamma, row0=0, full_h=None):
        h, w = img.shape[-2:]
        fh = full_h if full_h is not None else h
        rows = (torch.arange(h, device=img.device, dtype=torch.float32) + row0).view(1, 1, h, 1)
        tau = (t + gamma - gamma * rows / fh + 0.0001).expand(1, 1, h, w).to(img.dtype)
        return self.UVR(img, tau.contiguous())


TILE_ALIGN = 32


def _align_window(lo, hi, limit, align):
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
    import copy
    reps = []
    for c in range(ncore):
        dev = device if ncore == 1 and ":" not in device else "%s:%d" % (device.split(":")[0], c)
        m = copy.deepcopy(model).to(dtype).to(dev)
        reps.append((m, dev))
    return reps


def run_tiled(reps, tiles, pair_f, pair_b, t_fwd, t_bwd, H, W, gamma, real_triplet, only=None):
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


                need_f = (T["oy"] + T["vy"]) > half or not real_triplet
                need_b = real_triplet and T["oy"] < half
                of = ob = None


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

    if ncore == 1:


        res = [_one(0)]
    else:
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
    sd = torch.load(path, map_location="cpu", weights_only=False)
    sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
    conv_sd = {k.replace("module.", "UVR."): v for k, v in sd.items() if "module." in k}
    if not conv_sd:
        conv_sd = {("UVR." + k if not k.startswith("UVR.") else k): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(conv_sd, strict=False)
    real_missing = [k for k in missing if not k.startswith("UVR.block_tea")]
    if real_missing:
        raise SystemExit("checkpoint does not fit the model: %d missing keys, first few: %s"
                         % (len(real_missing), real_missing[:5]))


def descriptor_report(sites, itemsize, n_forwards=1):
    print("DESCRIPTOR ACCOUNTING -- grouped by shape, over %d forward pass(es) = %d resamples"
          % (n_forwards, len(sites)))
    print("  data-dependent, supplied per-partition via vector_offset (vector-DGE, 128 addresses")
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


    elems = sum(4 * C * H * W for _t, C, H, W in sites)
    ref_sites = [(3, 864, 1024)] * 6 + [(c, 864 >> l, 1024 >> l)
                                        for l, c in enumerate((16, 32, 64, 128), start=1)
                                        for _ in range(2)]
    ref_elems = sum(4 * c * h * w for c, h, w in ref_sites)
    print("  docs C4 exactly -- %.1fM against the documented 116.8M at the 864x1024 tile%s."
          % (ref_elems / 1e6, "" if abs(ref_elems - 116.8e6) < 1e5 else " (MISMATCH)"))
    print("  construction, and `sw_dyn_dma_packets` counts them 1:1. One full-resolution warp NEFF at")
    print("  fused module running two full-resolution warps at 864x1024 reports 3,563,248 against")
    print("  index/weight/store loads, 5 descriptors per 128-pixel k-tile = 2.35%.")
    small = [(C, H, W, n) for (C, H, W), n in agg.items() if 2 * C * itemsize < 512]
    if small:
        frac = sum(2 * h * w * n * DESC_NS / 1e6 for _c, h, w, n in small) / max(tot_ms, 1e-9)
        print("  %.0f%% of descriptor time is on payloads under 512 B -- the DMA-inefficient regime."
              % (100 * frac))


def score(out, gt, bar=3.0, label="GOLDEN"):
    if gt.shape[-2:] != out.shape[-2:]:
        gt = F.interpolate(gt, size=out.shape[-2:], mode="bilinear", align_corners=False,
                           antialias=True)
    d = (out.float() - gt.float()).abs()
    lsb = d * 255.0
    mse = d.pow(2).mean().item()
    psnr = float("inf") if mse == 0 else 10.0 * math.log10(1.0 / mse)
    cos = F.cosine_similarity(out.double().flatten(), gt.double().flatten(), dim=0).item()
    mx = lsb.max().item()
    print("  PSNR                 %8.2f dB" % psnr)
    print("  max_diff             %8.2f LSB   [%s vs the shipped bar of <= %.0f]"
          % (mx, "PASS" if mx <= bar else "FAIL", bar))
    flat = lsb.flatten().float()
    if flat.numel() > 4_000_000:
        flat = flat[torch.randperm(flat.numel())[:4_000_000]]
    flat, _ = flat.sort()
    for q in (0.5, 0.9, 0.99, 0.999, 0.9999):
        pass
    for thr in (1, 2, 3, 5, 10):
        pass
    err = lsb.amax(dim=1)[0]
    Hh, Ww = err.shape
    half = Hh // 2
    for name, band in (("merge seam +/-4 rows", err[max(0, half - 4):half + 4, :]),
                       ("top border 32 rows", err[:32, :]),
                       ("bottom border 32 rows", err[-32:, :]),
                       ("interior (128 cut)", err[128:-128, 128:-128] if Hh > 256 else err)):
        pass
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
    import json
    import numpy as np
    a = out.detach().float().cpu().numpy()
    np.save(path, a)
    try:
        from PIL import Image
        Image.fromarray((a[0].transpose(1, 2, 0).clip(0, 1) * 255).round().astype("uint8")).save(
            path.replace(".npy", ".png"))
    except Exception as e:
        pass
    h = hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
    meta = dict(meta, sha256_fp32=h, shape=list(a.shape),
                min=float(a.min()), max=float(a.max()))
    with open(path.replace(".npy", ".json"), "w") as fh:
        json.dump(meta, fh, indent=2, sort_keys=True)


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


def self_test():
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


        if cg < 0.999999:
            bad += 1
    print("  %s" % ("PASS -- the gather is the same op as grid_sample" if bad == 0
                    else "FAIL -- %d shape(s) disagree" % bad))


    tol = 1e-5
    x = torch.randn(1, 5, 704, 768)
    small = torch.randn(1, 5, 88, 96)

    for t, f, ftol in ((x, 0.5, tol), (x, 0.25, tol), (small, 2.0, tol), (small, 8.0, tol),
                       (torch.randn(1, 5, 96, 96), 1.0 / 3, tol),
                       (torch.randn(1, 5, 64, 64), 2.5, 1e-4)):
        ref = F.interpolate(t, scale_factor=f, mode="bilinear", align_corners=False)
        got = resize_precomputed(t, f)
        d = (ref - got).abs().max().item() if ref.shape == got.shape else float("inf")
        print("  taps f=%-6.4f %s -> %s  max|diff| %.2e  %s"
              % (f, tuple(t.shape[2:]), tuple(ref.shape[2:]), d, "OK" if d < ftol else "FAIL"))
        bad += d >= ftol
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
    ap.add_argument("--nkl-gather-method", choices=("transpose", "copy"), default=None,
                    help="gridsample-nkl only. Default picks by dtype, because the kernel asserts "
                         "gather_method='transpose' requires a 2-byte dtype -- so fp32 gets "
                         "'copy'. Override only to test the assert or to force the slower path.")
    ap.add_argument("--nkl-max-indices", type=int, default=None,
                    help="gridsample-nkl only: max_indices_per_indirect, the cap on gather indices "
                         "per batched indirect gather (None disables the batched path). This is "
                         "THE tuning knob at our scale: the kernel's tests sample to 64x64 (~4k "
                         "queries) while a 704x768 tile is ~540k, so the default may not fit SBUF.")
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
    ap.add_argument("--compile", type=int, choices=(0, 1), default=1,
                    help="1 (default) = torch.compile the model. 0 = run EAGER, every native "
                         "PyTorch op dispatching individually. Use 0 with --perfetto: under "
                         "torch.compile the ops are fused into 'Torch-Compiled Region' entries and "
                         "the individual convs / grid_sample / interpolate DO NOT APPEAR in the "
                         "trace at all -- a measured trace held 1,411 Python frames and only 24 "
                         "aten events, with 99.6%% of its 760 ms inside one opaque .cpu() wait. "
                         "Eager is the only way a profiler can see the operations. Latency from an "
                         "eager run is NOT comparable to a compiled one.")
    ap.add_argument("--resize", choices=("interpolate", "precomputed"), default="interpolate",
                    help="how the FIXED-FACTOR resizes in IFBlock are expressed. interpolate "
                         "(default) = F.interpolate, which derives the coordinates in-graph and "
                         "lowers to SWDGE indirect DMA. precomputed = resize_precomputed: the "
                         "coordinates are computed once on the host and baked in as constant "
                         "index vectors, so the reads lower to static DMA. Both are exact "
                         "(self-test asserts ~1e-7); this is a lowering choice, not an "
                         "approximation. MEASURED at tile 9 halo 128: interpolate's dominant NEFF "
                         "is 301.7 ms at gpsimd 66%% / tensor 14%%, precomputed is 41.4 ms at "
                         "gpsimd 2%% / tensor 29%%, and it also takes max_diff from 22.38 to 0.06 "
                         "LSB because interpolate's UPSAMPLE was the accuracy bug.")
    ap.add_argument("--warp-region", choices=("none", "neuron", "cpu", "eager"), default="none",
                    help="compile the RESAMPLE as its own region with its own backend, separate "
                         "from the model. none = inline in the model graph (default). neuron = its "
                         "own neuron graph. cpu = inductor on the HOST, tensors hopping device -> "
                         "host -> device on each of the 14 calls per tile, with the rest of the "
                         "model still on neuron -- which is how 'is gridsample better on CPU' gets "
                         "measured without moving the whole model. eager = the resample is NOT "
                         "compiled, still on the device, while the convs around it stay compiled: "
                         "the only way a profiler can NAME the resample (aten::grid_sampler_2d "
                         "with its shapes and 14 calls/tile) instead of one opaque "
                         "'Torch-Compiled Region', and it also gives the resample its OWN NEFFs so "
                         "its DEVICE share is attributable. Pair it with --perfetto. All three "
                         "force fullgraph=0.")
    ap.add_argument("--fullgraph", type=int, choices=(0, 1), default=1,
                    help="1 (default) = torch.compile(fullgraph=True), one fused graph per tile "
                         "shape. 0 = allow graph breaks, which is what makes --warp gridsample "
                         "usable: F.grid_sample issues 3-205x more DMA descriptors than gather "
                         "(3.0 pkt/px at C=3 up to 205 at C=128) and its FUSED graph exceeds "
                         "compiler memory (OOM at 169 and 94 min at 1500Gi). gridsample is the "
                         "resample REFERENCE every other warp is scored against, and the resample "
                         "is 63.1%% of device-active time (614.2 of 973.2 ms) against 36.9%% for "
                         "all 54 convs -- so it is the largest device cost, in its most expensive "
                         "form. NOT usable with --warp nki/nki-dyn: see the guard below.")
    ap.add_argument("--perfetto", metavar="PATH.json",
                    help="wrap the timed loop in torch.profiler and export a Chrome/Perfetto "
                         "trace to PATH (drag the .gz into ui.perfetto.dev). This is the ONLY "
                         "artifact that attributes device time back to a Python line: the runtime "
                         "NEURON_RT_INSPECT bundle tells you a DMA happened, this tells you which "
                         "line issued it. Adds real overhead, so never read latency from this run.")
    a = ap.parse_args()

    global _WARP, _RECORD, _RECORD_FLOW
    H, W = a.height, a.width
    itemsize = 2 if a.dtype == "bf16" else 4

    if a.self_test:
        return self_test()

    if a.report_only:


        sites = [("img", 3, H, W)] * 6
        for lvl, C in enumerate((_C, 2 * _C, 4 * _C, 8 * _C), start=1):
            sites += [("ctx", C, H >> lvl, W >> lvl)] * 2
        descriptor_report(sites, itemsize, n_forwards=1)
        return 0

    if not (a.rs0 and a.rs1):
        ap.error("need --rs0 and --rs1 (or --report-only / --self-test)")
    if not (a.weights or a.random_weights):
        ap.error("need --weights, or --random-weights for a perf-only run")


    if a.gt and a.random_weights and not a.gt.endswith(".npy"):
        ap.error("--gt <image> with --random-weights would print a meaningless PSNR. Use a "
                 "self-generated .npy reference for equivalence testing, or supply --weights.")

    global _NKI_DYN, _PRELU, _PROFILE_BLOCKS, _NKI_FN, _NKI_FN_DYN, _PRECOMPUTE_RESIZE
    global _SHIFTWARP_FN, _SHIFTWARP_R, _SHIFTWARP_MAXC
    global _NKL_GATHER_METHOD, _NKL_MAX_INDICES
    _NKI_DYN = (a.warp == "nki-dyn")
    _NKL_GATHER_METHOD = a.nkl_gather_method
    _NKL_MAX_INDICES = a.nkl_max_indices
    if a.warp == "gridsample-nkl":


        _build_gridsample_nkl()
        print("  --warp gridsample-nkl: NKL grid_sample imported and wrapped OK")
    if a.per_block:
        ap.error("--per-block is incompatible with torch.compile: it dissolves the module "
                 "boundaries the timers sit on, so the numbers would be meaningless. --fullgraph 0 "
                 "does NOT rescue it -- breaks fall wherever dynamo puts them, not on module "
                 "boundaries, so the attribution would be arbitrary rather than merely coarse. "
                 "That is how an earlier table came to sum to 2.51x the frame.")
    _PROFILE_BLOCKS = a.per_block


    _PRECOMPUTE_RESIZE = (a.resize == "precomputed")
    print("  resize        : %s" % ("resize_precomputed (host-baked constant indices)"
                                    if _PRECOMPUTE_RESIZE else "F.interpolate (indices in-graph)"))
    _PRELU = NeuronPReLU
    if not a.neuron_prelu:
        pass
    _WARP = WARPS[a.warp]
    if a.warp_region != "none":


        if a.fullgraph:
            print("  --warp-region %s forces --fullgraph 0 (the region boundary IS a graph break)"
                  % a.warp_region)
            a.fullgraph = 0
        if a.warp_region == "cpu" and a.warp == "gridsample-nkl":
            ap.error("--warp-region cpu with --warp gridsample-nkl is contradictory: the NKL "
                     "kernel is an NKI device kernel and cannot run on the host. Use "
                     "--warp gridsample for the cpu region.")
        if a.warp == "window":


            ap.error("--warp-region is not wired for --warp window: the radius rebind below would "
                     "silently overwrite the region wrapper.")
        _WARP = build_warp_region(WARPS[a.warp], a.warp_region, a.device,
                                  annotate=bool(a.perfetto))
        print("  warp region   : %s (%s)"
              % (a.warp_region,
                 {"cpu": "own torch.compile, backend inductor/cpu",
                  "neuron": "own torch.compile, backend neuron",
                  "eager": "NOT compiled -- ATen dispatches one op at a time on %s, so the "
                           "profiler names it" % a.device}[a.warp_region]))
        if a.warp_region == "eager" and not a.perfetto:
            print("    NOTE no --perfetto: the region is open but nothing is recording it. The "
                  "point of this mode is the trace.")
    if a.warp == "window":
        _r = a.radius
        _WARP = lambda x, f: warp_window(x, f, radius=_r)


    if a.fullgraph == 0 and a.warp in ("nki", "nki-dyn"):
        ap.error("--fullgraph 0 is unsafe with --warp %s: a graph break around the "
                 "view(torch.uint32) index bitcast silently corrupts which pixels are sampled. "
                 "Use --warp gridsample or gather with --fullgraph 0, or keep --fullgraph 1."
                 % a.warp)
    if a.record_flow and a.fullgraph == 1 and a.compile:
        ap.error("--record-flow cannot run under fullgraph=True: it appends to a Python list and "
                 "syncs to host inside the warp, which dynamo cannot trace. Pass --fullgraph 0 or "
                 "--compile 0 to re-enable it -- eager needs no tracing at all.")
    if a.warp == "shiftwarp":
        if a.device != "neuron":
            ap.error("--warp shiftwarp requires --device neuron")
        _SHIFTWARP_R = a.shiftwarp_radius
        _SHIFTWARP_MAXC = a.shiftwarp_max_c


        _SHIFTWARP_FN = _build_shiftwarp()
    if a.warp.startswith("nki") and a.device != "neuron":
        ap.error("--warp %s requires --device neuron" % a.warp)


    if a.warp == "gridsample-nkl" and a.device != "neuron":
        ap.error("--warp gridsample-nkl is an NKI kernel and requires --device neuron; use "
                 "--warp gridsample for the host arm (same op, same semantics)")
    if a.warp.startswith("nki"):


        _NKI_FN, _NKI_FN_DYN = _build_nki()

    verify_assets([a.rs0, a.rs1, a.rs2, a.gt])


    torch.manual_seed(0)
    model = UniVR().eval()
    if a.weights:
        load_weights(model, a.weights)
    else:
        pass

    dt = torch.bfloat16 if a.dtype == "bf16" else torch.float32
    if a.device == "neuron":
        import torch_neuronx


    _tiled_mode = (a.tiles.lower() != "1x1") or a.cores > 1 or a.only_tile is not None
    model = model.to(dt) if _tiled_mode else model.to(dt).to(a.device)

    rs0 = load_img(a.rs0, H, W)
    rs1 = load_img(a.rs1, H, W)


    _idev = "cpu" if _tiled_mode else a.device
    pair_f = torch.cat([rs0, rs1], 0)[None].to(dt).to(_idev)
    t_fwd, t_bwd = 1 - a.gamma / 2, -a.gamma / 2
    real_triplet = bool(a.rs2)
    pair_b = None
    if real_triplet:
        rs2 = load_img(a.rs2, H, W)
        pair_b = torch.cat([rs1, rs2], 0)[None].to(dt).to(_idev)
        print("  merged with rows above %d from the backward pass." % (H // 2))
    else:
        print("  a pair against it is not the shipped comparison -- pass --rs2 for that.")


    def apply_compile(m, tag=""):
        if not a.compile:


            return m
        fg = bool(a.fullgraph)


        if a.device == "cpu":
            return torch.compile(m, dynamic=False, fullgraph=fg)
        return torch.compile(m, backend="neuron", dynamic=False, fullgraph=fg)

    ny, nx = (int(v) for v in a.tiles.lower().split("x"))
    tiled = (ny * nx > 1) or a.cores > 1 or a.only_tile is not None
    tiles = plan_tiles(H, W, ny, nx, a.halo)
    reps = None
    if tiled:
        ncore = 1 if a.only_tile is not None else a.cores
        print("  TILED: grid %dx%d = %d tiles, halo %d, %d core(s)"
              % (ny, nx, len(tiles), a.halo, ncore))
        shapes = sorted({(t["ph"], t["pw"]) for t in tiles})
        print("  padded tile shapes: %s  -> %d distinct graph(s) per timestamp"
              % (", ".join("%dx%d" % (h, w) for h, w in shapes), len(shapes)))
        print("  x2 timestamps (triplet) -> up to %d graphs to compile" % (2 * len(shapes)))
        if real_triplet:
            pass
        if a.only_tile is not None:
            T = tiles[a.only_tile]
            print("  --only-tile %d: valid %dx%d at (%d,%d), padded %dx%d"
                  % (a.only_tile, T["vy"], T["vx"], T["oy"], T["ox"], T["ph"], T["pw"]))
        reps = build_replicas(model, dt, a.device, ncore)
        reps = [(apply_compile(m, "replica %d/%d" % (i + 1, len(reps))), d)
                for i, (m, d) in enumerate(reps)]
        print("  %d replica(s) built" % len(reps))


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


    _RECORD = False
    _RECORD_FLOW = bool(a.record_flow)
    FLOW_STATS.clear()
    CALL_SITES.clear()
    t0 = time.perf_counter()
    out = one_frame()
    if a.device == "cuda":
        torch.cuda.synchronize()
    out = out.float().cpu()
    first_ms = (time.perf_counter() - t0) * 1e3
    _RECORD = False
    _RECORD_FLOW = False

    if FLOW_STATS:
        worst = 0.0
        worst_c3 = 0.0
        for i, (C, H, W, m, o2, o3) in enumerate(FLOW_STATS):
            need = int(math.ceil(m)) if m > 0 else 0
            worst = max(worst, m)
            if C <= 3:
                worst_c3 = max(worst_c3, m)
        print("  SILENTLY CLAMPED -- no error, just wrong pixels, which is the 229.72 LSB / p50=0.00 /")
        print("  p99=172.99 signature. Compare the >%dpx column against the ~7%% of pixels that failed."
              % _SHIFTWARP_R)
    print("  forward complete: %s  range [%.4f, %.4f]  first call %.0f ms (includes compile/warmup)"
          % (tuple(out.shape), out.min(), out.max(), first_ms))

    if a.iters > 0:


        BLOCK_MS.clear()
        ts = []


        _prof = _prof_kind = None
        if a.perfetto:


            import inspect
            for _mod, _sym in (("torch_neuronx.profiling", "NeuronProfiler"),
                               ("torch_neuronx.experimental.profiler", "NeuronProfiler"),
                               ("torch_neuronx.profiler", "NeuronProfiler"),
                               ("torch_neuronx.experimental.profiler", "profile")):
                try:
                    _m = __import__(_mod, fromlist=[_sym])
                    _cls = getattr(_m, _sym)


                    _want = {"record_shapes": True, "with_stack": True}
                    try:
                        _params = inspect.signature(_cls).parameters
                        _kw = {k: v for k, v in _want.items() if k in _params}
                    except (TypeError, ValueError):
                        _kw = {}
                    _cand = _cls(**_kw)


                    _missing = [n for n in ("__enter__", "__exit__", "export_chrome_trace")
                                if not hasattr(_cand, n)]
                    if _missing:
                        print("  --perfetto: %s.%s constructed but lacks %s -- skipping"
                              % (_mod, _sym, ", ".join(_missing)))
                        continue
                    _prof = _cand
                    _dropped = sorted(set(_want) - set(_kw))
                    _prof_kind = "%s.%s (device rows)%s" % (
                        _mod, _sym,
                        "" if not _dropped else "  [it does not accept %s, so shapes/source lines "
                                                "may be absent]" % ", ".join(_dropped))
                    break
                except Exception as _e:
                    print("  --perfetto: %s.%s unavailable (%s: %s)"
                          % (_mod, _sym, type(_e).__name__, _e))
            if _prof is None:
                from torch.profiler import profile, ProfilerActivity
                _prof = profile(activities=[ProfilerActivity.CPU],
                                record_shapes=True, with_stack=True)
                _prof_kind = "torch.profiler (CPU rows ONLY -- no device timeline)"
                print("  --perfetto: NO NeuronProfiler found. This trace will name Python lines")
                print("    system trace converted with `neuron-explorer view -d <dir>"
                      " --output-format perfetto`.")
            print("  --perfetto: tracing the timed loop with %s" % _prof_kind)
            print("  NOTE this run's median is NOT a latency measurement: the profiler is on.")
            _prof.__enter__()
        for _ in range(a.iters):
            t0 = time.perf_counter()
            o = one_frame()
            if a.device == "cuda":
                torch.cuda.synchronize()
            o.float().cpu()
            ts.append((time.perf_counter() - t0) * 1e3)
        if _prof is not None:
            _prof.__exit__(None, None, None)


            import gzip
            import shutil
            try:
                _prof.export_chrome_trace(a.perfetto)
                with open(a.perfetto, "rb") as _f, gzip.open(a.perfetto + ".gz", "wb") as _g:
                    shutil.copyfileobj(_f, _g)
                os.remove(a.perfetto)
                print("  --perfetto: wrote %s.gz (%.1f MB) -- drag into ui.perfetto.dev"
                      % (a.perfetto, os.path.getsize(a.perfetto + ".gz") / 1e6))
            except Exception as e:
                print("  --perfetto: export FAILED (%s: %s)" % (type(e).__name__, e))
        ts.sort()
        med = ts[len(ts) // 2]
        print("  steady state: median %.1f ms over %d iters (min %.1f, max %.1f)"
              % (med, len(ts), ts[0], ts[-1]))
        pc = getattr(one_frame, "percore", None)
        if pc:


            print("  per-core ms: %s" % ", ".join("%.0f" % v for v in pc))
        if a.per_block and BLOCK_MS:

            n_fwd = max(1, a.iters)
            tot_b = sum(BLOCK_MS.values()) / n_fwd
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
            warp_ms = sum(v for k, v in BLOCK_MS.items() if "_warp" in k) / n_fwd
            conv_ms = sum(v for k, v in BLOCK_MS.items() if "_conv" in k) / n_fwd
        ph = getattr(one_frame, "phases", None)
        if ph:


            tot = sum(ph.values())
            for k in ("prep", "dev", "d2h", "stitch"):
                if k in ph:
                    pass
            print("    synchronous), so dev+d2h is reliable as a SUM and the split between them is")
            print("  slowest core %.1f ms, sum %.1f ms, sum/wall %.2f (near %d = real parallelism, "
                  "near 1 = serialised)" % (max(pc), sum(pc), sum(pc) / max(med, 1e-9), len(pc)))
        if tiled and a.only_tile is None:
            _f, _p, (miss, dup), _ph = run_tiled(reps, tiles, pair_f, pair_b, t_fwd, t_bwd, H, W,
                                                 a.gamma, real_triplet, only=None)


        is_4k = (H, W) == (1728, 4096)
        if is_4k and real_triplet and a.only_tile is None:
            report_gap(med)
        elif is_4k and real_triplet and a.only_tile is not None:
            print("  NO gap printed: --only-tile measures ONE tile, and the CUDA baselines are")
            print("  whole-frame. Multiply by the tile count only if you also believe the cores")
        else:
            pass

    if CALL_SITES:
        descriptor_report(CALL_SITES, itemsize, n_forwards=2 if real_triplet else 1)
    else:


        pass

    if a.gate:


        keep = _WARP
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
        print("  PSNR                 %8.2f dB" % (float("inf") if mse == 0
                                                   else 10.0 * math.log10(1.0 / mse)))
        print("  max_diff             %8.4e  (%.2f LSB)" % (d.max().item(), d.max().item() * 255))

    def _crop(x):
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
        print("    --gt %s   (fp32, exact, no resize)" % a.save_ref)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
