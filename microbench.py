#!/usr/bin/env python3
import argparse
import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SHAPES = [
    (3, 256, 384, 6),
    (16, 128, 192, 2),
    (32, 64, 96, 2),
    (64, 32, 48, 2),
    (128, 16, 24, 2),
]


def warp_gridsample(x, flow):
    B, C, H, W = x.shape
    d = flow.device
    hor = torch.linspace(-1.0, 1.0, W, device=d, dtype=flow.dtype).view(1, 1, 1, W).expand(B, -1, H, -1)
    ver = torch.linspace(-1.0, 1.0, H, device=d, dtype=flow.dtype).view(1, 1, H, 1).expand(B, -1, -1, W)
    grid = torch.cat([hor, ver], 1)
    f = torch.cat([flow[:, 0:1] / ((W - 1.0) / 2.0), flow[:, 1:2] / ((H - 1.0) / 2.0)], 1)
    g = (grid + f).permute(0, 2, 3, 1)
    return F.grid_sample(x, g, mode="bilinear", padding_mode="border", align_corners=True)


def warp_index_select(x, flow):
    B, C, H, W = x.shape
    d = flow.device
    gx = torch.arange(W, device=d, dtype=torch.float32).view(1, 1, 1, W)
    gy = torch.arange(H, device=d, dtype=torch.float32).view(1, 1, H, 1)
    sx = (gx + flow[:, 0:1].float()).clamp(0.0, W - 1.0)
    sy = (gy + flow[:, 1:2].float()).clamp(0.0, H - 1.0)
    x0 = torch.floor(sx)
    y0 = torch.floor(sy)
    ax = sx - x0
    ay = sy - y0
    x1 = (x0 + 1).clamp(0.0, W - 1.0)
    y1 = (y0 + 1).clamp(0.0, H - 1.0)
    N = H * W
    src = x.reshape(B, C, N).permute(0, 2, 1).reshape(B * N, C).float()
    boff = (torch.arange(B, device=d, dtype=torch.long) * N).view(B, 1, 1, 1)

    def tap(yy, xx):
        idx = (yy * W + xx).long() + boff
        return src.index_select(0, idx.reshape(-1)).view(B, H, W, C).permute(0, 3, 1, 2)

    out = (tap(y0, x0) * ((1 - ax) * (1 - ay)) + tap(y0, x1) * (ax * (1 - ay))
           + tap(y1, x0) * ((1 - ax) * ay) + tap(y1, x1) * (ax * ay))
    return out.to(x.dtype)


def transpose_only(x, flow):
    B, C, H, W = x.shape
    return x.reshape(B, C, H * W).permute(0, 2, 1).reshape(B * H * W, C).float()


def shift_only(x, flow):
    return torch.roll(x, shifts=(4, 4), dims=(2, 3))


def warp_window(x, flow, radius=1):
    B, C, H, W = x.shape
    d = flow.device
    gx = torch.arange(W, device=d, dtype=torch.float32).view(1, 1, 1, W)
    gy = torch.arange(H, device=d, dtype=torch.float32).view(1, 1, H, 1)
    sx = (gx + flow[:, 0:1].float()).clamp(0.0, W - 1.0)
    sy = (gy + flow[:, 1:2].float()).clamp(0.0, H - 1.0)
    rx = sx - gx
    ry = sy - gy
    R = radius
    pad = F.pad(x.float(), (R, R, R, R), mode="replicate")
    acc = torch.zeros(B, C, H, W, device=d, dtype=torch.float32)
    for oy in range(-R, R + 1):
        ty = (1.0 - (ry - oy).abs()).clamp_min(0.0)
        for ox in range(-R, R + 1):
            tx = (1.0 - (rx - ox).abs()).clamp_min(0.0)
            acc = acc + pad[:, :, R + oy:R + oy + H, R + ox:R + ox + W] * (tx * ty)
    return acc.to(x.dtype)


def warp_window2(x, flow):
    return warp_window(x, flow, radius=2)


def warp_shiftmatmul(x, flow, radius=2, bulk=(0, 0)):
    B, C, H, W = x.shape
    d = flow.device
    R = radius
    by, bx = bulk
    gx = torch.arange(W, device=d, dtype=torch.float32).view(1, 1, 1, W)
    gy = torch.arange(H, device=d, dtype=torch.float32).view(1, 1, H, 1)
    sx = (gx + flow[:, 0:1].float()).clamp(0.0, W - 1.0)
    sy = (gy + flow[:, 1:2].float()).clamp(0.0, H - 1.0)
    rx = sx - gx - float(bx)
    ry = sy - gy - float(by)
    base = x.float()
    roll_shifts = tuple(s for s in (by, bx) if s != 0)
    roll_dims = tuple(d for s, d in ((by, 2), (bx, 3)) if s != 0)
    if roll_dims:
        base = torch.roll(base, shifts=roll_shifts, dims=roll_dims)
    pad = F.pad(base, (R, R, R, R), mode="replicate")
    shifts = []
    wts = []
    for oy in range(-R, R + 1):
        ty = (1.0 - (ry - oy).abs()).clamp_min(0.0)
        for ox in range(-R, R + 1):
            tx = (1.0 - (rx - ox).abs()).clamp_min(0.0)
            shifts.append(pad[:, :, R + oy:R + oy + H, R + ox:R + ox + W])
            wts.append(tx * ty)
    acc = shifts[0] * wts[0]
    for i in range(1, len(shifts)):
        acc = acc + shifts[i] * wts[i]
    return acc.to(x.dtype)


def warp_nkishift(x, flow, radius=2):
    B, C, H, W = x.shape
    d = flow.device
    R = radius
    from nki_shift_warp import shift_warp_band
    from torch_neuronx.nki_hop import wrap_nki
    gx = torch.arange(W, device=d, dtype=torch.float32).view(1, 1, 1, W)
    gy = torch.arange(H, device=d, dtype=torch.float32).view(1, 1, H, 1)
    sx = (gx + flow[:, 0:1].float()).clamp(0.0, W - 1.0)
    sy = (gy + flow[:, 1:2].float()).clamp(0.0, H - 1.0)
    rx = sx - gx
    ry = sy - gy
    planes = []
    for oy in range(-R, R + 1):
        ty = (1.0 - (ry - oy).abs()).clamp_min(0.0)
        for ox in range(-R, R + 1):
            tx = (1.0 - (rx - ox).abs()).clamp_min(0.0)
            planes.append((tx * ty)[0, 0])
    wts = torch.stack(planes, 0).contiguous()
    imgp = F.pad(x.float(), (R, R, R, R), mode="replicate")[0].permute(1, 2, 0).contiguous()
    out = wrap_nki(shift_warp_band)(imgp, wts)
    return out.permute(2, 0, 1).unsqueeze(0).to(x.dtype)


_NKL_GS_FN = None
# Set from --nkl-max-indices / --nkl-gather-method in main(), read by the warp below.
# max_indices_per_indirect=None DISABLES the kernel's batched indirect gather -- the feature its
# own description leads with. Measured with batching OFF it reached hardware_dynamic_dma 5.8%
# (vs index_select's 0.10%) at 44.2 ns/descriptor against index_select's 40.4, i.e. 9% slower overall despite
# matching index_select's 1.006 desc/px. Raising the hardware-DGE share is the only lever left, so this
# knob must be reachable rather than hardcoded.
_NKL_MAX_IDX = None
_NKL_GM = None


def ensure_nkl():
    """Import and wrap the NKL kernel. MUST be called BEFORE torch.compile, never from inside the
    traced function.

    The first version did the import inside warp_gridsample_nkl, which torch.compile traces. Dynamo
    then raised its OWN error before the except clause could run:
        torch._dynamo.exc.Unsupported: Import failure
          module_name: nkilib.experimental.indirect.grid_sample
    which is unhelpful twice over -- it masks whether the module is merely absent, and it looks
    like a tracing limitation rather than a missing dependency. Hoisting the import out means a
    missing package says so plainly, and a present one is wrapped once.
    """
    global _NKL_GS_FN
    if _NKL_GS_FN is None:
        from nkilib.experimental.indirect.grid_sample import grid_sample
        from torch_neuronx.nki_hop import wrap_nki
        _NKL_GS_FN = wrap_nki(grid_sample)
    return _NKL_GS_FN


def warp_gridsample_nkl(x, flow):
    """NKL grid_sample (KaenaNeuronKernelLibrary, CR-288764575) behind F.grid_sample semantics.

    Grid math is copied verbatim from warp_gridsample above so this is a true A/B against the
    reference -- the microbench already scores every op against warp_gridsample on CPU, so any
    accuracy delta here is the kernel and not the coordinates.

    Two kernel asserts shape this call: input_layout must be NHWC (the NCHW option in the
    docstring is unimplemented) so the value is permuted in and out, and gather_method
    "transpose" requires a 2-byte dtype, so fp32 takes the "copy" path.
    """
    B, C, H, W = x.shape
    d = flow.device
    hor = torch.linspace(-1.0, 1.0, W, device=d, dtype=flow.dtype).view(1, 1, 1, W).expand(B, -1, H, -1)
    ver = torch.linspace(-1.0, 1.0, H, device=d, dtype=flow.dtype).view(1, 1, H, 1).expand(B, -1, -1, W)
    grid = torch.cat([hor, ver], 1)
    f = torch.cat([flow[:, 0:1] / ((W - 1.0) / 2.0), flow[:, 1:2] / ((H - 1.0) / 2.0)], 1)
    g = (grid + f).permute(0, 2, 3, 1).contiguous()
    # fp32 forces "copy": the kernel asserts gather_method="transpose" needs a 2-byte dtype, and
    # bf16 is not an escape here (it fails the model's quality gate at 23.31 dB).
    gm = _NKL_GM or ("transpose" if x.element_size() == 2 else "copy")
    # The already-wrapped kernel, resolved by ensure_nkl() BEFORE torch.compile. Referencing the
    # global directly rather than calling ensure_nkl() here keeps the import out of the traced
    # region entirely -- that is what produced "torch._dynamo.exc.Unsupported: Import failure".
    out = _NKL_GS_FN(x.permute(0, 2, 3, 1).contiguous(), g,
                     sampling_mode="bilinear", coord_mode="minus_one_one",
                     input_layout="NHWC", align_corners=True,
                     padding_mode="border", max_indices_per_indirect=_NKL_MAX_IDX,
                     gather_method=gm)
    return out.permute(0, 3, 1, 2)


def warp_nki_repo(x, flow):
    """The repo's WORKING NKI resample -- `bilinear_2x2_gather_blend` from the model, reached
    through the model's own `warp_nki` host wrapper so index construction cannot drift.

    THIS IS THE CONFIG THAT WAS MISSING, and its absence made every earlier comparison misleading.
    index_select is only the argparse default and "the port's form"; it is not the reference and not the
    fastest. At the model level nki-dyn measured 3820.4 ms against index_select's 3900.5, and the
    README's 3673.3 ms baseline used --warp nki. So the bar for any new resample kernel is
    nki-dyn, not index_select.

    NOT to be confused with `nkishift`, which is the DEAD shiftwarp kernel: 1.16x at the op level
    and then 229.72 LSB in the model, because it clamps at R=3 px while measured displacement is
    29.02 px. This one is accurate -- index_select and nki-dyn agreed to every digit in the fused model.
    """
    import repro_unrolling_trn2 as M
    M._NKI_DYN = _NKI_DYN_SEL
    return M.warp_nki(x, flow)


def warp_nki_dyn_repo(x, flow):
    """The device-loop variant: compiles in seconds where the unrolled one takes 80-100 min at a
    4K tile, at ~1.1-1.3x its runtime. The FASTEST resample on record at the model level."""
    return warp_nki_repo(x, flow)


_NKI_DYN_SEL = False    # set from --op in main(): nki-dyn picks the device-loop kernel


OPS = {
    "gridsample": warp_gridsample,
    "nki": warp_nki_repo,
    "nki-dyn": warp_nki_dyn_repo,
    "gridsample-nkl": warp_gridsample_nkl,
    "index_select": warp_index_select,
    "transpose": transpose_only,
    "shift": shift_only,
    "window1": warp_window,
    "window2": warp_window2,
    "shiftmatmul": warp_shiftmatmul,
    "nkishift": warp_nkishift,
}


def roofline(C, H, W, itemsize=4):
    px = H * W
    return {
        "output_px": px,
        "output_bytes": px * C * itemsize,
        "src_bytes": px * C * itemsize,
        "taps": 4,
        "tap_bytes": 4 * px * C * itemsize,
        "min_desc_1_per_px": px,
        "min_desc_2_per_px": 2 * px,
        "bytes_per_desc_at_2C": 2 * C * itemsize,
        "index_bytes": 2 * px * itemsize,
        "weight_bytes": 4 * px * itemsize,
    }


_C = 16


def _prelu(o):
    class P(nn.Module):
        def __init__(self, n):
            super().__init__()
            self.weight = nn.Parameter(torch.full((n,), 0.25))

        def forward(self, x):
            w = self.weight.view(1, -1, *([1] * (x.dim() - 2)))
            return F.relu(x) - w * F.relu(-x)
    return P(o)


# =============================================================================================
# THE REAL RESAMPLE INVENTORY -- what a 4K frame actually warps, and at what shapes
# =============================================================================================
# This exists because the synthetic configs were WRONG in the way that matters. They swept
# C=3,16,32,64,128 all at the FULL padded tile shape, but the model never does that: Contextnet
# is four Conv2(stride=2) levels, so every step DOUBLES the channel count and QUARTERS the area.
# C=128 runs at ph/16 x pw/16 -- 44x48 = 2,112 px for a 704x768 tile, not 540,672. Benchmarking
# C=128 at full tile resolution measures a shape that does not exist, 256x too large.
#
# Consequence for the gridsample-vs-NKI question: the headline "3.0 pkt/px at C=3 rising to 205
# at C=128" is real per call, but C=128 is only ~0.12% of the frame's pixel-warps. Weighted by
# the actual inventory, gridsample is roughly 3.2x index_select in total descriptors, not 100x, and the
# dominant site is C=3 at FULL resolution (54.5% of channel-pixels, 6 calls per tile) where
# gridsample is only 3.0 vs index_select's ~2.0. Any claim about the kernel's headroom has to be made
# against this table, not against the per-call peak.
#
# The 14 sites per tile match the count in STATE.md: 6 full-resolution pyramid warps (3 stages x
# {img0, img1}) plus 4 Contextnet levels x {img0, img1}. 32 tiles x 14 = 448 warp calls a frame.
#
# The padded shapes come from the MODEL's own plan_tiles, imported rather than copied. An earlier
# version hardcoded the 4x8 halo-128 set, which was wrong on two counts: it silently excluded every
# other geometry this project has run, and a hardcoded table drifts from the tiling code the moment
# TILE_ALIGN or the clipping changes. --grid/--halo now select any of them:
#     4x8 halo 128  the config PROVEN fully fusable (all 8 slots, job k2zwh)
#     4x8 halo  64  the config that produced the measured 3900.5 / 3820.4 ms fused frame
#     2x8 halo  64  928x640 = 593,920 px, the measured compile OOM -- included because its
#                   resample inventory is still the right question even where it cannot compile
# Border tiles get their halo clipped, so each grid has several distinct padded shapes, and each
# tile runs exactly ONE pass (H/2 = 864 falls on a row boundary at 4 rows, so need_f and need_b
# are never both true) -- which makes the tile count the call multiplier.
def tile_shapes(grid="4x8", halo=128, H=1728, W=4096):
    """[(ph, pw, n_tiles)] for a grid, via the model's plan_tiles. Single source of truth."""
    from repro_unrolling_trn2 import plan_tiles           # safe: guarded by __name__ == __main__
    ny, nx = (int(v) for v in grid.lower().split("x"))
    counts = {}
    for T in plan_tiles(H, W, ny, nx, halo):
        counts[(T["ph"], T["pw"])] = counts.get((T["ph"], T["pw"]), 0) + 1
    return sorted(((ph, pw, n) for (ph, pw), n in counts.items()), key=lambda r: -r[0] * r[1])


def warp_inventory(ph, pw):
    """Resample sites for ONE tile of padded size (ph, pw): (name, C, H, W, calls_per_tile).

    Six at full resolution from the flow pyramid (_StageFirst + 2x _StageNext, each warping img0
    and img1), then one per Contextnet level. Level lvl outputs _C * 2**(lvl-1) channels at
    ph/2**lvl because Conv2's first conv has stride 2, and is called twice (img0, img1).
    """
    sites = [("ifnet_pyramid", 3, ph, pw, 6)]
    for lvl in range(1, 5):
        sites.append(("ctx%d" % lvl, _C * 2 ** (lvl - 1), ph >> lvl, pw >> lvl, 2))
    return sites


def op_sites(grid="4x8", halo=128, warps=("index_select",)):
    """EVERY op the model performs, at the dimensions it performs them, weighted by calls/frame.

    Nothing here is synthetic. Warp dims come from warp_inventory (which reads the model's real
    call sites) and conv dims from conv_inventory, both evaluated at every padded tile shape the
    grid produces. Weight is calls/frame times the cost driver for that op CLASS -- pixel-warps
    for the resample (measured descriptor-bound: 1.006 desc/px at 40.40 ns, MFU 0.00%) and MACs
    for the convs (dense compute) -- so the two are ranked within their class, never against each
    other, because a pixel-warp and a MAC are not comparable units.

    Returns [(cls, op, shape_str, calls, weight, unit)] sorted by weight within each class,
    where `unit` names the piece of MODEL work the row measures -- the shape for a warp (two
    implementations of one dim measure the same work) and site@shape for a conv (eight cb convs
    share a shape but are eight distinct sites). Coverage is summed over units, never rows.
    """
    rows = []
    shapes = tile_shapes(grid, halo)
    wq = {}
    for ph, pw, ntiles in shapes:
        for _n, c, h, w, calls in warp_inventory(ph, pw):
            wq[(c, h, w)] = wq.get((c, h, w), 0) + calls * ntiles
    for (c, h, w), calls in wq.items():
        shape = "%d,%d,%d" % (c, h, w)
        for op in warps:
            # unit = shape: two impls of one dim are two measurements of ONE piece of model work.
            rows.append(("warp", op, shape, calls, float(h) * w * calls, shape))
    cq = {}
    for ph, pw, ntiles in shapes:
        for name, grp, i, o, ih, iw, st in conv_inventory(ph, pw):
            # Contextnet runs twice per forward (img0 and img1); everything else once per tile.
            k = ntiles * (2 if grp == "ctx" else 1)
            key = (name, ih, iw)
            cq[key] = (cq.get(key, (0, 0.0))[0] + k, conv_macs(i, o, ih, iw, st))
    for (name, ih, iw), (calls, macs) in cq.items():
        # unit = (site, shape): eight cb convs share one shape but are eight distinct sites.
        rows.append(("conv", name, "%d,%d" % (ih, iw), calls, macs * calls, "%s@%d,%d" % (name, ih, iw)))
    out = []
    for cls in ("warp", "conv"):
        sel = sorted((r for r in rows if r[0] == cls), key=lambda r: -r[4])
        out.extend(sel)
    return out


def print_op_sites(grid="4x8", halo=128, warps=("index_select",), top=0, specs_only=False, only="both"):
    """The config list, derived. With specs_only emit `op:shape` lines the job consumes directly."""
    rows = op_sites(grid, halo, warps)
    if only != "both":
        want = "warp" if only == "warps" else "conv"
        rows = [r for r in rows if r[0] == want]
    # Coverage is per DIMENSION, not per config: measuring one dim with two implementations is two
    # configs but ONE unit of the model's work, so summing config weights would report 200% coverage.
    dim_w = {}
    for cls, _o, _shape, _k, wt, unit in rows:
        dim_w[(cls, unit)] = wt
    tot = {}
    for (cls, _s), wt in dim_w.items():
        tot[cls] = tot.get(cls, 0.0) + wt
    # top applies to DIMS, keeping every implementation of each kept dim so the A/B stays paired.
    keep = rows
    if top:
        keep = []
        for cls in ("warp", "conv"):
            dims = [u for (c, u), _w in sorted(dim_w.items(), key=lambda kv: -kv[1]) if c == cls]
            sel = set(dims[:top])
            keep.extend(r for r in rows if r[0] == cls and r[5] in sel)
    if specs_only:
        for _cls, op, shape, _k, _wt, _u in keep:
            print("%s:%s" % (op, shape))
        return 0
    print("grid %s halo %d -- EVERY op the model runs, at the dims it runs them" % (grid, halo))
    print("%-6s %-16s %-12s %8s %14s %8s %8s"
          % ("class", "op", "shape", "calls/fr", "weight", "share%", "cum%"))
    for cls in ("warp", "conv"):
        cum = 0.0
        seen = set()
        for _c, op, shape, k, wt, unit in [x for x in keep if x[0] == cls]:
            share = 100.0 * wt / tot[cls]
            if unit not in seen:                   # count each unit of model work once
                cum += share
                seen.add(unit)
            print("%-6s %-16s %-12s %8d %14.4g %8.2f %8.2f" % (cls, op, shape, k, wt, share, cum))
        ndim = len([1 for (c, _s) in dim_w if c == cls])
        if top and len(seen) < ndim:
            print("  ^ %d of %d %s DIMS measured -- %.1f%% of %s weight NOT measured"
                  % (len(seen), ndim, cls, 100.0 - cum, cls))
    print()
    print("configs selected: %d of %d" % (len(keep), len(rows)))
    return 0


def print_warp_sites(grid="4x8", halo=128, specs_only=False):
    """The frame-wide inventory. With specs_only, emit `C,H,W:calls_per_frame` for the job to
    consume, deduplicated across shapes, so the config list is derived rather than hand-written."""
    shapes = tile_shapes(grid, halo)
    weight = {}
    for ph, pw, ntiles in shapes:
        for _n, c, h, w, calls in warp_inventory(ph, pw):
            weight[(c, h, w)] = weight.get((c, h, w), 0) + calls * ntiles
    if specs_only:
        for (c, h, w), n in sorted(weight.items(), key=lambda kv: -kv[0][1] * kv[0][2] * kv[1]):
            print("%d,%d,%d:%d" % (c, h, w, n))
        return 0
    tot_px = sum(h * w * n for (_c, h, w), n in weight.items())
    tot_cpx = sum(c * h * w * n for (c, h, w), n in weight.items())
    print("grid %s halo %d -> %d tiles, %d distinct padded shapes"
          % (grid, halo, sum(n for _p, _q, n in shapes), len(shapes)))
    print("%-14s %5s %10s %7s %8s %13s %7s" % ("site", "C", "HxW", "px", "calls", "ch_px", "share%"))
    for ph, pw, ntiles in shapes:
        print("--- padded %dx%d  x%d tiles" % (ph, pw, ntiles))
        for n, c, h, w, calls in warp_inventory(ph, pw):
            cpx = c * h * w * calls * ntiles
            print("%-14s %5d %10s %7d %8d %13d %7.2f"
                  % (n, c, "%dx%d" % (h, w), h * w, calls * ntiles, cpx, 100.0 * cpx / tot_cpx))
    print()
    print("frame: %d warp calls, %.2fM pixel-warps, %.2fM channel-pixels, %d distinct (C,H,W)"
          % (sum(weight.values()), tot_px / 1e6, tot_cpx / 1e6, len(weight)))
    byc = {}
    for (c, h, w), n in weight.items():
        byc[c] = byc.get(c, 0) + c * h * w * n
    for c in sorted(byc):
        print("  C=%-4d %7.2fM ch_px  %5.1f%% of frame" % (c, byc[c] / 1e6, 100.0 * byc[c] / tot_cpx))
    return 0


def conv_inventory(H, W):
    C = _C
    ops = []
    for name, inp, c, sc in (("a0", 7, 240, 4), ("a1", 18, 150, 2), ("a2", 18, 90, 1)):
        h, w = H // sc, W // sc
        ops.append((name + "_conv0a", "ifblock", inp, c // 2, h, w, 2))
        ops.append((name + "_conv0b", "ifblock", c // 2, c, h // 2, w // 2, 2))
        h2, w2 = h // 4, w // 4
        for i in range(8):
            ops.append(("%s_cb%d" % (name, i), "ifblock", c, c, h2, w2, 1))
        ops.append((name + "_lastconv", "ifblock", c, 5, h2, w2, -2))
    h, w = H, W
    for lvl, (i, o) in enumerate(((3, C), (C, 2 * C), (2 * C, 4 * C), (4 * C, 8 * C)), start=1):
        ops.append(("ctx%d_c1" % lvl, "ctx", i, o, h, w, 2))
        h, w = h // 2, w // 2
        ops.append(("ctx%d_c2" % lvl, "ctx", o, o, h, w, 1))
    h, w = H, W
    for n, i, o in (("down0", 17, 2 * C), ("down1", 4 * C, 4 * C),
                    ("down2", 8 * C, 8 * C), ("down3", 16 * C, 16 * C)):
        ops.append(("u_%s_c1" % n, "unet", i, o, h, w, 2))
        h, w = h // 2, w // 2
        ops.append(("u_%s_c2" % n, "unet", o, o, h, w, 1))
    for n, i, o in (("up0", 32 * C, 8 * C), ("up1", 16 * C, 4 * C),
                    ("up2", 8 * C, 2 * C), ("up3", 4 * C, C)):
        ops.append(("u_%s" % n, "unet", i, o, h, w, -2))
        h, w = h * 2, w * 2
    ops.append(("u_final", "unet", C, 3, h, w, 1))
    return ops


def conv_macs(i, o, ih, iw, s):
    oh, ow = (ih * 2, iw * 2) if s < 0 else (ih // s, iw // s)
    return i * o * 9 * oh * ow


def build_conv(i, o, s):
    if s < 0:
        return nn.Sequential(nn.ConvTranspose2d(i, o, 4, 2, 1, bias=True), _prelu(o))
    return nn.Sequential(nn.Conv2d(i, o, 3, s, 1, bias=True), _prelu(o))


FUSED_4x8_H64_SHAPES = ["512,576", "512,640", "576,576", "576,640"]


def run_sequence(a):
    """`--op sequence`: the model's WHOLE op sequence for one tile, compiled, timed.

    The other ops here measure ONE call in isolation. That answers "how many descriptors does a
    resample issue" -- and it did: index_select 1.006 desc/px at 40.40 ns, gridsample 3.004 at 39.81.
    It cannot answer "what does swapping the resample do to the model", because an isolated op has
    no pipelining, no layout reuse, and no overlap with the convs around it; because absolute
    single-io microseconds are noise across runs; and because a swapped resample changes the
    LAYOUT its neighbours see -- the NKL kernel asserts NHWC while the model is NCHW, so it adds
    two permutes per call that no single-op config measures.

    So this runs the real sequence: IFNet's three pyramid stages with their six full-resolution
    warps, Contextnet's four levels x {img0, img1}, and the Unet -- 14 warps and 54 convs -- as
    ONE compiled graph at a real padded tile shape. That median IS comparable between warps.

    It reuses the MODEL'S OWN modules, so the sequence is identical by construction rather than by
    transcription; a hand-copied version would drift and still look authoritative.

    Random weights, so PSNR against the golden would be meaningless and none is printed.
    Correctness is an equivalence check BETWEEN warps at a fixed seed: same weights, same input,
    so two resample implementations must agree to fp32 rounding. --save-out then --cmp.
    """
    import time
    import repro_unrolling_trn2 as M

    H, W = 1728, 4096
    ny, nx = (int(v) for v in a.grid.lower().split("x"))
    tiles = M.plan_tiles(H, W, ny, nx, a.halo)
    T = tiles[a.tile]
    ph, pw, py0 = T["ph"], T["pw"], T["py0"]
    half = H // 2
    need_f = (T["oy"] + T["vy"]) > half
    t = (1 - a.gamma / 2) if need_f else (-a.gamma / 2)

    print("op            sequence (14 warps + 54 convs, one graph)")
    print("grid/halo     %s halo %d tile %d" % (a.grid, a.halo, a.tile))
    print("padded        %dx%d = %d px   row0 %d   pass %s (t=%+.4f)"
          % (ph, pw, ph * pw, py0, "forward" if need_f else "backward", t))
    print("warp          %s" % a.warp)
    for name, c, h, w, calls in warp_inventory(ph, pw):
        print("  site        %-8s C=%-4d %dx%d  x%d" % (name, c, h, w, calls))

    torch.manual_seed(a.seed)
    dt = torch.bfloat16 if a.dtype == "bf16" else torch.float32
    # Globals MUST be set before UniVR() is built: conv()/deconv() read _PRELU at build time.
    M._PRELU = M.NeuronPReLU
    M._WARP = M.WARPS[a.warp]
    M._NKI_DYN = (a.warp == "nki-dyn")
    M._NKL_GATHER_METHOD = a.nkl_gather_method
    M._NKL_MAX_INDICES = a.nkl_max_indices
    if a.warp == "gridsample-nkl":
        M._build_gridsample_nkl()        # fail in a second, not after a compile
        print("nkl kernel    imported and wrapped OK")
    if a.warp == "shiftwarp":
        M._SHIFTWARP_FN = M._build_shiftwarp()

    model = M.UniVR().to(dt).to(a.device).eval()
    x = torch.rand(1, 6, ph, pw, dtype=dt).to(a.device)
    print("torch.compile fullgraph=%s" % bool(a.fullgraph))
    fn = torch.compile(model, backend="neuron", dynamic=False, fullgraph=bool(a.fullgraph))

    with torch.no_grad():
        t0 = time.perf_counter()
        out = fn(x, t, a.gamma, row0=py0, full_h=H)
        out.float().cpu()
        print("ran           first call %.0f ms (compile+warmup)  out %s"
              % ((time.perf_counter() - t0) * 1e3, tuple(out.shape)))
        ts = []
        for _ in range(a.iters):
            t0 = time.perf_counter()
            o = fn(x, t, a.gamma, row0=py0, full_h=H)
            o.float().cpu()
            ts.append((time.perf_counter() - t0) * 1e3)
    ts.sort()
    med = ts[len(ts) // 2]
    print("TILE MEDIAN   %.2f ms over %d iters (min %.2f, max %.2f)" % (med, len(ts), ts[0], ts[-1]))
    same = sum(1 for q in tiles if (q["ph"], q["pw"]) == (ph, pw))
    print("frame extrap  %d of %d tiles share this shape -> %.0f ms tile work, %.0f ms across 8 cores"
          % (same, len(tiles), med * same, med * same / 8))
    print("              (assumes equal cost per tile and perfect 8-way overlap: a LOWER bound)")

    got = o.float().cpu().numpy()
    if a.save_out:
        np.save(a.save_out, got)
        print("saved         %s  (compare another warp with --cmp)" % a.save_out)
    if a.cmp:
        ref = np.load(a.cmp)
        if ref.shape != got.shape:
            print("EQUIVALENCE   n/a: %s vs %s" % (ref.shape, got.shape))
        else:
            d = np.abs(ref - got)
            lsb = float(d.max()) * 255.0
            mse = float((d ** 2).mean())
            psnr = float("inf") if mse == 0 else 10.0 * math.log10(1.0 / mse)
            print("EQUIVALENCE   max_diff %.4f LSB   PSNR %.2f dB   [%s vs %s]"
                  % (lsb, psnr, "AGREE" if lsb <= 3.0 else "DISAGREE", a.cmp))
            print("              same seed, so this isolates the RESAMPLE. NOT accuracy vs the")
            print("              golden -- random weights make that meaningless, so none is shown.")
    return 0


def run_conv(name, group, i, o, ih, iw, st, a):
    dt = torch.bfloat16 if a.dtype == "bf16" else torch.float32
    torch.manual_seed(0)
    m = build_conv(i, o, st).to(dt)
    x = torch.rand(1, i, ih, iw, dtype=dt)
    oh, ow = (ih * 2, iw * 2) if st < 0 else (ih // st, iw // st)
    macs = conv_macs(i, o, ih, iw, st)

    print("op            %s" % name)
    print("group         %s" % group)
    print("in            1x%dx%dx%d  dtype=%s" % (i, ih, iw, a.dtype))
    print("out           1x%dx%dx%d" % (o, oh, ow))
    print("stride        %d%s" % (st, "  (ConvTranspose2d 4,2,1)" if st < 0 else ""))
    print("MMAC          %.1f" % (macs / 1e6))
    print("MFLOP         %.1f   (2 x MACs)" % (2 * macs / 1e6))

    with torch.no_grad():
        ref = m(x).float()

    print("torch.compile fullgraph=True")
    md = m.to(a.device)
    xd = x.to(a.device)
    fn_ = torch.compile(md, backend="neuron", dynamic=False, fullgraph=True)
    with torch.no_grad():
        got = fn_(xd).float().cpu()
    print("ran           output %s" % (tuple(got.shape),))

    if got.shape != ref.shape:
        print("ACCURACY      n/a: device %s vs cpu %s" % (tuple(got.shape), tuple(ref.shape)))
        return 1
    diff = (ref - got).abs()
    amax = diff.max().item()
    rel = amax / max(ref.abs().max().item(), 1e-12)
    mse = (diff ** 2).mean().item()
    psnr = float("inf") if mse == 0 else 10.0 * math.log10(
        max(ref.abs().max().item() ** 2, 1e-12) / mse)
    cos = torch.nn.functional.cosine_similarity(
        ref.flatten().double(), got.flatten().double(), dim=0).item()
    ok = rel <= 1e-3
    print("ACCURACY      max_abs %.3e  rel %.3e  PSNR %.2f dB  cos %.8f  [%s vs rel 1e-3]"
          % (amax, rel, psnr, cos, "PASS" if ok else "FAIL"))
    print()
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--op", required=True)
    ap.add_argument("--shape")
    ap.add_argument("--grid", default="4x8",
                    help="tile grid for --list-warp-sites/--warp-site-specs. 4x8 = the only grid "
                         "under the compiler memory ceiling and the one both fused configs use; "
                         "2x8/2x4/3x4 are measured OOMs but their inventories are still printable.")
    ap.add_argument("--halo", type=int, default=128,
                    help="halo for --list-warp-sites/--warp-site-specs. 128 = the production halo, "
                         "proven fusable on all 8 slots; 64 = the config that measured 3900.5 ms.")
    ap.add_argument("--list-convs", action="store_true")
    ap.add_argument("--list-warp-sites", action="store_true",
                    help="print the resample inventory a 4K frame actually performs: the real "
                         "(C,H,W) of every warp call across all four padded tile shapes, weighted "
                         "by call count and tile multiplicity. Read this BEFORE trusting any warp "
                         "config -- the Contextnet levels halve resolution as they double channels, "
                         "so C=128 lives at ph/16 and a C=128 config at full tile size is fiction.")
    ap.add_argument("--list-op-sites", action="store_true",
                    help="EVERY op the model runs (warps AND convs) at the dims it runs them, "
                         "weight-ordered with cumulative coverage. Nothing synthetic.")
    ap.add_argument("--op-site-specs", action="store_true",
                    help="machine-readable --list-op-sites: `op:shape` lines for the job.")
    ap.add_argument("--warp-impls", default="index_select",
                    help="comma-separated warp implementations to emit configs for, e.g. "
                         "index_select,gridsample-nkl")
    ap.add_argument("--op-site-class", default="both", choices=("warps", "convs", "both"),
                    help="restrict --list-op-sites/--op-site-specs to one op class.")
    ap.add_argument("--top", type=int, default=0,
                    help="keep only the top N configs per class by weight (0 = all). A truncated run "
                         "PRINTS what it left out -- never a silent cap.")
    ap.add_argument("--warp-site-specs", action="store_true",
                    help="machine-readable form of --list-warp-sites: `C,H,W:calls_per_frame`, "
                         "deduplicated. The job builds its config list from this.")
    ap.add_argument("--fused-shapes", action="store_true")
    ap.add_argument("--device", default="neuron")
    ap.add_argument("--flow-mag", type=float, default=8.0)
    ap.add_argument("--flow-smooth", action="store_true")
    ap.add_argument("--radius", type=int, default=3)
    ap.add_argument("--dtype", default="fp32", choices=("fp32", "bf16"))
    ap.add_argument("--int-flow", action="store_true")
    # --op sequence only
    ap.add_argument("--tile", type=int, default=9,
                    help="--op sequence: which tile. 9 is the LARGEST shape at 4x8; tile 1 is NOT "
                         "the largest and testing it once produced a false FUSES verdict.")
    ap.add_argument("--warp", default="index_select",
                    help="--op sequence: the resample implementation inside the sequence")
    ap.add_argument("--iters", type=int, default=5,
                    help="timed iterations after the compile/warmup call, for both the single-op "
                         "configs and --op sequence")
    ap.add_argument("--fullgraph", type=int, choices=(0, 1), default=1)
    ap.add_argument("--gamma", type=float, default=0.98)
    ap.add_argument("--seed", type=int, default=0,
                    help="fixes weights AND input, so two warps are a true equivalence test")
    ap.add_argument("--nkl-gather-method", choices=("transpose", "copy"), default=None)
    ap.add_argument("--nkl-max-indices", type=int, default=None)
    ap.add_argument("--save-out", metavar="PATH.npy")
    ap.add_argument("--cmp", metavar="PATH.npy")
    a = ap.parse_args()

    if a.list_op_sites or a.op_site_specs:
        return print_op_sites(a.grid, a.halo, tuple(a.warp_impls.split(",")), a.top,
                              specs_only=a.op_site_specs, only=a.op_site_class)

    if a.list_warp_sites or a.warp_site_specs:
        return print_warp_sites(a.grid, a.halo, specs_only=a.warp_site_specs)

    if a.list_convs:
        H, W = (int(v) for v in (a.shape or "576,640").split(","))
        inv = conv_inventory(H, W)
        tot = sum(conv_macs(i, o, ih, iw, st) for _n, _g, i, o, ih, iw, st in inv)
        ctx = sum(conv_macs(i, o, ih, iw, st) for _n, g, i, o, ih, iw, st in inv if g == "ctx")
        print("%-16s %-8s %6s %6s %11s %7s %12s" %
              ("op", "group", "in_ch", "out_ch", "in_hxw", "stride", "MMAC"))
        for n, g, i, o, ih, iw, st in inv:
            print("%-16s %-8s %6d %6d %11s %7d %12.1f"
                  % (n, g, i, o, "%dx%d" % (ih, iw), st, conv_macs(i, o, ih, iw, st) / 1e6))
        print()
        print("ops %d   MACs single pass %.2f G   per forward (ctx x2) %.2f G"
              % (len(inv), tot / 1e9, (tot + ctx) / 1e9))
        return 0

    global _NKL_MAX_IDX, _NKL_GM, _NKI_DYN_SEL
    _NKI_DYN_SEL = (a.op == "nki-dyn")
    _NKL_MAX_IDX = a.nkl_max_indices
    _NKL_GM = a.nkl_gather_method
    if a.op == "gridsample-nkl":
        print("nkl config    gather_method=%s  max_indices_per_indirect=%s"
              % (_NKL_GM or "auto(copy for fp32)", _NKL_MAX_IDX))

    if a.op == "sequence":
        if a.fullgraph == 0 and a.warp in ("nki", "nki-dyn"):
            ap.error("--fullgraph 0 is unsafe with --warp %s: a graph break around the "
                     "view(torch.uint32) index bitcast silently corrupts which pixels are sampled."
                     % a.warp)
        if a.device == "neuron":
            import torch_neuronx  # noqa: F401
        return run_sequence(a)

    if not a.shape and not a.fused_shapes:
        ap.error("need --shape, or --fused-shapes to sweep the 4x8 halo-64 shapes")

    if a.device == "neuron":
        import torch_neuronx  # noqa: F401

    if a.op.startswith("conv:") or any(a.op == n for n, *_ in conv_inventory(576, 640)):
        shapes = FUSED_4x8_H64_SHAPES if a.fused_shapes else [a.shape]
        rcs = 0
        for sh in shapes:
            H, W = (int(v) for v in sh.split(","))
            inv = conv_inventory(H, W)
            if a.op.startswith("conv:"):
                grp = a.op.split(":", 1)[1]
                sel = [o for o in inv if o[1] == grp]
                if not sel:
                    ap.error("no conv group %r; groups are ifblock, ctx, unet" % grp)
            else:
                sel = [o for o in inv if o[0] == a.op]
            for n, g, i, o, ih, iw, st in sel:
                rcs |= run_conv(n, g, i, o, ih, iw, st, a)
        return rcs

    C, H, W = (int(v) for v in a.shape.split(","))
    dt = torch.bfloat16 if a.dtype == "bf16" else torch.float32

    torch.manual_seed(0)
    x = torch.rand(1, C, H, W, dtype=dt)
    if a.flow_smooth:
        ch, cw = max(2, H // 16), max(2, W // 16)
        ctrl = (torch.rand(1, 2, ch, cw) * 2 - 1)
        flow = F.interpolate(ctrl, size=(H, W), mode="bicubic", align_corners=True)
        flow = flow / flow.abs().max().clamp_min(1e-6) * a.flow_mag
    else:
        flow = (torch.rand(1, 2, H, W) * 2 - 1) * a.flow_mag
    if a.int_flow:
        flow = flow.round()
    flow = flow.to(dt)

    r = roofline(C, H, W, 2 if a.dtype == "bf16" else 4)
    print("op            %s" % a.op)
    print("shape         C=%d H=%d W=%d  dtype=%s" % (C, H, W, a.dtype))
    print("output_px     %d" % r["output_px"])
    print("output_bytes  %d" % r["output_bytes"])
    print("tap_bytes     %d   (4 taps x C x px)" % r["tap_bytes"])
    print("index_bytes   %d" % r["index_bytes"])
    print("weight_bytes  %d" % r["weight_bytes"])
    print("min_desc      %d at 1/px, %d at 2/px" % (r["min_desc_1_per_px"], r["min_desc_2_per_px"]))
    print("B_per_desc    %d at 2C contiguous" % r["bytes_per_desc_at_2C"])
    print("flow_mag      %g%s%s" % (a.flow_mag, " (integer)" if a.int_flow else "",
                                    " SMOOTH" if a.flow_smooth else " uniform-random"))
    if a.op in ("nkishift", "shiftmatmul"):
        print("radius        R=%d  -> %d terms  (covers |disp| <= %d px)"
              % (a.radius, (2 * a.radius + 1) ** 2, a.radius))
    print("flow_maxabs   %.4f px   max neighbour jump %.4f px"
          % (flow.float().abs().max(),
             (flow.float()[:, :, :, 1:] - flow.float()[:, :, :, :-1]).abs().max()))

    x = x.to(a.device)
    flow = flow.to(a.device)

    bulk = (0, 0)
    if a.op == "shiftmatmul":
        by = int(flow[:, 1:2].float().mean().round().item())
        bx = int(flow[:, 0:1].float().mean().round().item())
        bulk = (by, bx)
        print("bulk_shift    (dy=%d, dx=%d)  computed on host, baked as a constant" % bulk)

    op = OPS[a.op]
    if a.op == "shiftmatmul":
        _b = bulk
        _r = a.radius
        op = lambda t, f: warp_shiftmatmul(t, f, radius=_r, bulk=_b)
    elif a.op == "nkishift":
        _r = a.radius
        op = lambda t, f: warp_nkishift(t, f, radius=_r)
    elif a.op == "gridsample-nkl":
        # Resolve the kernel BEFORE torch.compile so a missing package reports itself plainly
        # instead of surfacing as a dynamo "Import failure" graph break from inside tracing.
        try:
            ensure_nkl()
        except ImportError as e:                                  # noqa: BLE001
            raise SystemExit(
                "op gridsample-nkl needs nkilib.experimental.indirect.grid_sample from "
                "KaenaNeuronKernelLibrary (CR-288764575, OPEN at rev 3 -- unmerged, so absent "
                "from released images). Import failed: %s\n"
                "Put nkilib on PYTHONPATH; the job can restore it from a single tar object on "
                "the PVC." % e)
        print("nkl kernel    imported and wrapped OK (gather_method picked by dtype)")
    # backend="neuron" is only valid ON neuron. A CPU config exists so gridsample can be compared
    # across BACKENDS -- same op, same semantics, three lowerings -- which is the only truly
    # apples-to-apples comparison available: index_select is a different formulation (4-tap indirect
    # index_select), numerically equivalent but not the same code path, and mixing it in is what made
    # the earlier baseline arbitrary.
    if a.device == "cpu":
        try:
            fn = torch.compile(op, dynamic=False, fullgraph=True)   # inductor
            print("torch.compile inductor (cpu), fullgraph=True")
        except Exception as e:                                      # noqa: BLE001
            fn = op
            print("cpu EAGER (inductor unavailable: %s) -- not a compiled number" % type(e).__name__)
    else:
        fn = torch.compile(op, backend="neuron", dynamic=False, fullgraph=True)
        print("torch.compile fullgraph=True")

    # WARM WALL-CLOCK MEDIAN. This path previously ran the op exactly ONCE and never timed it --
    # every latency number came from the neuron-only profiler (total_active_time), so a CPU config
    # would have produced no number at all and the three-backend comparison was impossible.
    # The first call compiles, so it is discarded; the median of the rest is the comparable figure.
    import time as _time
    with torch.no_grad():
        t0 = _time.perf_counter()
        out = fn(x, flow)
        out.float().cpu()
        first = (_time.perf_counter() - t0) * 1e3
        ts = []
        for _ in range(max(1, a.iters)):
            t0 = _time.perf_counter()
            out = fn(x, flow)
            out.float().cpu()
            ts.append((_time.perf_counter() - t0) * 1e3)
    ts.sort()
    got = out.float().cpu()
    print("ran           output %s   first call %.1f ms (compile+warmup)"
          % (tuple(out.shape), first))
    print("wall          %.3f ms  median of %d (min %.3f, max %.3f)"
          % (ts[len(ts) // 2], len(ts), ts[0], ts[-1]))

    with torch.no_grad():
        ref = warp_gridsample(x.float().cpu(), flow.float().cpu())
    if ref.shape == got.shape:
        diff = (ref - got).abs()
        lsb = diff.max().item() * 255.0
        mse = (diff ** 2).mean().item()
        psnr = float("inf") if mse == 0 else 10.0 * math.log10(1.0 / mse)
        cos = torch.nn.functional.cosine_similarity(
            ref.flatten().double(), got.flatten().double(), dim=0).item()
        print("ACCURACY      max_diff %.4f LSB   PSNR %.2f dB   cos %.6f   [%s vs bar 3]"
              % (lsb, psnr, cos, "PASS" if lsb <= 3.0 else "FAIL"))
    else:
        print("ACCURACY      n/a: shape %s is not comparable to grid_sample %s"
              % (tuple(got.shape), tuple(ref.shape)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
