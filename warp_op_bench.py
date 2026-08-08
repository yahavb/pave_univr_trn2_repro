#!/usr/bin/env python3
import argparse
import math
import os

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
    base = torch.roll(x.float(), shifts=(by, bx), dims=(2, 3))
    pad = F.pad(base, (R, R, R, R), mode="replicate")
    shifts = []
    wts = []
    for oy in range(-R, R + 1):
        ty = (1.0 - (ry - oy).abs()).clamp_min(0.0)
        for ox in range(-R, R + 1):
            tx = (1.0 - (rx - ox).abs()).clamp_min(0.0)
            shifts.append(pad[:, :, R + oy:R + oy + H, R + ox:R + ox + W])
            wts.append(tx * ty)
    S = torch.stack(shifts, 0)
    Wt = torch.stack(wts, 0)
    return (S * Wt).sum(0).to(x.dtype)


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


OPS = {
    "gridsample": warp_gridsample,
    "gather": warp_gather,
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--op", required=True, choices=sorted(OPS))
    ap.add_argument("--shape", required=True, help="C,H,W")
    ap.add_argument("--device", default="neuron")
    ap.add_argument("--flow-mag", type=float, default=8.0)
    ap.add_argument("--dtype", default="fp32", choices=("fp32", "bf16"))
    ap.add_argument("--int-flow", action="store_true")
    a = ap.parse_args()

    if a.device == "neuron":
        import torch_neuronx  # noqa: F401

    C, H, W = (int(v) for v in a.shape.split(","))
    dt = torch.bfloat16 if a.dtype == "bf16" else torch.float32

    torch.manual_seed(0)
    x = torch.rand(1, C, H, W, dtype=dt)
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
    print("flow_mag      %g%s" % (a.flow_mag, " (integer)" if a.int_flow else ""))

    x = x.to(a.device)
    flow = flow.to(a.device)

    bulk = (0, 0)
    if a.op == "shiftmatmul":
        by = int(flow[:, 1:2].float().mean().round().item())
        bx = int(flow[:, 0:1].float().mean().round().item())
        bulk = (by, bx)
        print("bulk_shift    (dy=%d, dx=%d)  computed on host, baked as a constant" % bulk)

    cc = dict(backend="neuron", dynamic=False, fullgraph=True)
    print("torch.compile %s" % cc)
    op = OPS[a.op]
    if a.op == "shiftmatmul":
        _b = bulk
        op = lambda t, f: warp_shiftmatmul(t, f, radius=2, bulk=_b)
    fn = torch.compile(op, **cc)

    with torch.no_grad():
        out = fn(x, flow)
    got = out.float().cpu()
    print("ran           output %s" % (tuple(out.shape),))

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
