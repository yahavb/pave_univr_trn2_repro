#!/usr/bin/env python3
"""Benchmark the model's CONV inventory, the 54 ops the warp microbench never covered.

WHY THIS EXISTS
`warp_op_bench.py` measured 14 of the model's ~68 ops -- every resample and nothing else. That
produced a contradiction nobody has resolved:

  sum of the 14 warp sites (from measured active_us)   ~614 ms per tile-triplet
  README --per-block, resample = 62.1% of a forward    ~2562 ms against the measured 4125 ms

4.2x apart. One is wrong, and "the resample dominates" -- the premise behind every kernel attempt
in this repo -- rests on the larger one. The gap cannot be closed without measuring the other 54
ops, because only then does `sum(convs) + sum(warps)` become checkable against the wall clock.

Likely source of the disagreement, unverified: the warp numbers are `total_active_time` from a
device profile (device-busy), while --per-block uses barriered host timers (wall, and it only runs
under --compile none). Those measure different things. This bench uses the SAME profile-based path
as the warp bench so the two sums are commensurable.

THE INVENTORY is derived from the model source, not guessed:
  3 IFBlocks   conv0 (2 strided convs) + convblock (8 convs) + ConvTranspose2d,
               c = 240/150/90 at scales 4/2/1, in_planes 7/18/18
  Contextnet   4 levels of Conv2 (stride-2 conv + stride-1 conv), 3->16->32->64->128, run TWICE
  Unet         4x Conv2 down (17->32, 64->64, 128->128, 256->256), 4x deconv up
               (512->128, 256->64, 128->32, 64->16), final Conv2d(16,3)
Total 54 conv/deconv ops per forward, 184.19 GMAC at the 992x1280 tile.

Usage:
    python conv_op_bench.py --list                       # print the inventory and exit
    python conv_op_bench.py --op a0_cb0 --tile 992,1280  # one op, profiled like the warp bench
    python conv_op_bench.py --group ifblock --tile 992,1280
"""
from __future__ import annotations

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F


def _prelu(o):
    """relu(x) - w*relu(-x). The model substitutes this for nn.PReLU under torch.compile because
    PReLU cannot be legalised by the Neuron backend; the identity is exact, so timing it here
    matches what the model actually compiles."""
    class P(nn.Module):
        def __init__(self, n):
            super().__init__()
            self.weight = nn.Parameter(torch.full((n,), 0.25))

        def forward(self, x):
            w = self.weight.view(1, -1, *([1] * (x.dim() - 2)))
            return F.relu(x) - w * F.relu(-x)
    return P(o)


def inventory(H, W):
    """Every conv/deconv the forward runs, as (name, group, in_ch, out_ch, in_h, in_w, stride).

    stride > 0 is a Conv2d with that stride; stride == -2 is a ConvTranspose2d(4, 2, 1), which
    doubles the spatial dims. in_h/in_w are the INPUT dims, so output is derived per op.
    """
    C = 16
    ops = []
    # ---- pyramid: 3 IFBlocks. Input is downsampled by `scale` first, then conv0 halves twice.
    for name, inp, c, sc in (("a0", 7, 240, 4), ("a1", 18, 150, 2), ("a2", 18, 90, 1)):
        h, w = H // sc, W // sc
        ops.append((name + "_conv0a", "ifblock", inp, c // 2, h, w, 2))
        ops.append((name + "_conv0b", "ifblock", c // 2, c, h // 2, w // 2, 2))
        h2, w2 = h // 4, w // 4
        for i in range(8):
            ops.append(("%s_cb%d" % (name, i), "ifblock", c, c, h2, w2, 1))
        ops.append((name + "_lastconv", "ifblock", c, 5, h2, w2, -2))
    # ---- Contextnet, run once per image so every op below executes TWICE per forward.
    h, w = H, W
    for lvl, (i, o) in enumerate(((3, C), (C, 2 * C), (2 * C, 4 * C), (4 * C, 8 * C)), start=1):
        ops.append(("ctx%d_c1" % lvl, "ctx", i, o, h, w, 2))
        h, w = h // 2, w // 2
        ops.append(("ctx%d_c2" % lvl, "ctx", o, o, h, w, 1))
    # ---- Unet encoder, then decoder. Channel counts include the skip concatenations.
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


def macs(i, o, ih, iw, s):
    oh, ow = (ih * 2, iw * 2) if s < 0 else (ih // s, iw // s)
    return i * o * 9 * oh * ow


def build(i, o, s):
    if s < 0:
        return nn.Sequential(nn.ConvTranspose2d(i, o, 4, 2, 1, bias=True), _prelu(o))
    return nn.Sequential(nn.Conv2d(i, o, 3, s, 1, bias=True), _prelu(o))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tile", default="992,1280", help="padded tile H,W")
    ap.add_argument("--op", help="single op by name")
    ap.add_argument("--group", choices=("ifblock", "ctx", "unet"), help="all ops in a group")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--device", default="neuron")
    ap.add_argument("--dtype", default="fp32", choices=("fp32", "bf16"))
    ap.add_argument("--iters", type=int, default=0,
                    help="0 = compile only. The profile is captured externally from the NEFF, the "
                         "same way the warp bench does it, so no host timing is needed here")
    a = ap.parse_args()

    H, W = (int(v) for v in a.tile.split(","))
    ops = inventory(H, W)

    if a.list:
        tot = 0
        print("%-16s %-8s %6s %6s %11s %8s %12s" %
              ("op", "group", "in_ch", "out_ch", "in_hxw", "stride", "MMAC"))
        for n, g, i, o, ih, iw, s in ops:
            m = macs(i, o, ih, iw, s)
            tot += m
            print("%-16s %-8s %6d %6d %11s %8d %12.1f"
                  % (n, g, i, o, "%dx%d" % (ih, iw), s, m / 1e6))
        # Contextnet runs twice per forward (once per image), so its MACs are doubled in the total.
        ctx = sum(macs(i, o, ih, iw, s) for n, g, i, o, ih, iw, s in ops if g == "ctx")
        print()
        print("ops                        %d" % len(ops))
        print("MACs, single pass          %.2f G" % (tot / 1e9))
        print("MACs per forward (ctx x2)  %.2f G" % ((tot + ctx) / 1e9))
        print("MACs per tile-triplet      %.2f G" % (2 * (tot + ctx) / 1e9))
        print()
        print("For comparison, the 14 warps at this tile were measured at ~307 ms per forward")
        print("(sum of active_us x calls). If the convs land far below the tensor engine's peak,")
        print("that is where the unexplained 4.2x between the warp sum and --per-block lives.")
        return 0

    sel = [o for o in ops if (a.op and o[0] == a.op) or (a.group and o[1] == a.group)]
    if not sel:
        raise SystemExit("no op matched --op/--group; use --list to see names")

    if a.device == "neuron":
        import torch_neuronx  # noqa: F401
    dt = torch.bfloat16 if a.dtype == "bf16" else torch.float32

    for n, g, i, o, ih, iw, s in sel:
        m = build(i, o, s).to(dt).to(a.device)
        x = torch.rand(1, i, ih, iw, dtype=dt).to(a.device)
        oh, ow = (ih * 2, iw * 2) if s < 0 else (ih // s, iw // s)
        print("op            %s" % n)
        print("group         %s" % g)
        print("in            1x%dx%dx%d  dtype=%s" % (i, ih, iw, a.dtype))
        print("out           1x%dx%dx%d" % (o, oh, ow))
        print("stride        %d%s" % (s, "  (ConvTranspose2d 4,2,1)" if s < 0 else ""))
        print("MMAC          %.1f" % (macs(i, o, ih, iw, s) / 1e6))
        cc = dict(backend="neuron", dynamic=False, fullgraph=True)
        print("torch.compile %s" % cc)
        fn = torch.compile(m, **cc) if a.device == "neuron" else m
        with torch.no_grad():
            y = fn(x)
        print("ran           output %s" % (tuple(y.shape),))
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
