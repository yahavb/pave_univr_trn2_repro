#!/usr/bin/env python3
import argparse
import time

import torch
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


def warp_gather(x, flow):
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


def warp_shift(x, flow):
    B, C, H, W = x.shape
    dx = flow[:, 0:1].mean().round().int().item()
    dy = flow[:, 1:2].mean().round().int().item()
    return torch.roll(x, shifts=(dy, dx), dims=(2, 3))


OPS = {"gridsample": warp_gridsample, "gather": warp_gather, "shift": warp_shift}


def bench(fn, x, flow, iters, dev):
    with torch.no_grad():
        o = fn(x, flow)
    o.float().cpu()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        with torch.no_grad():
            o = fn(x, flow)
        o.float().cpu()
        ts.append((time.perf_counter() - t0) * 1e3)
    ts.sort()
    return ts[len(ts) // 2], o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="neuron")
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--ops", default="gridsample,gather")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--flow-mag", type=float, default=8.0)
    a = ap.parse_args()

    ops = [o for o in a.ops.split(",") if o]
    torch.manual_seed(0)

    print("%-11s %5s %5s %5s %4s %10s %12s %11s %9s" % (
        "op", "C", "H", "W", "n", "px", "median_ms", "per_call_ms", "MB"))
    totals = {}
    for name in ops:
        fn = OPS[name]
        if a.compile:
            fn = torch.compile(fn, backend="neuron", dynamic=False, fullgraph=True)
        tot = 0.0
        for (C, H, W, n) in SHAPES:
            x = torch.rand(1, C, H, W)
            flow = (torch.rand(1, 2, H, W) * 2 - 1) * a.flow_mag
            x = x.to(a.device)
            flow = flow.to(a.device)
            ms, o = bench(fn, x, flow, a.iters, a.device)
            mb = 4 * 4 * C * H * W / 1e6
            tot += ms * n
            print("%-11s %5d %5d %5d %4d %10d %12.3f %11.3f %9.2f" % (
                name, C, H, W, n, H * W, ms, ms, mb))
        totals[name] = tot
        print("%-11s %s per forward (14 sites): %.2f ms" % (name, " " * 32, tot))
        print()

    if len(totals) > 1:
        base = totals[ops[0]]
        print("relative to %s:" % ops[0])
        for k, v in totals.items():
            print("  %-11s %8.2f ms  %5.2fx" % (k, v, v / base))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
