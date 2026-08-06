#!/usr/bin/env python3
import argparse
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


OPS = {
    "gridsample": warp_gridsample,
    "gather": warp_gather,
    "transpose": transpose_only,
    "shift": shift_only,
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
    a = ap.parse_args()

    if a.device == "neuron":
        import torch_neuronx  # noqa: F401

    C, H, W = (int(v) for v in a.shape.split(","))
    dt = torch.bfloat16 if a.dtype == "bf16" else torch.float32

    torch.manual_seed(0)
    x = torch.rand(1, C, H, W, dtype=dt)
    flow = ((torch.rand(1, 2, H, W) * 2 - 1) * a.flow_mag).to(dt)

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

    x = x.to(a.device)
    flow = flow.to(a.device)

    cc = dict(backend="neuron", dynamic=False, fullgraph=True)
    print("torch.compile %s" % cc)
    fn = torch.compile(OPS[a.op], **cc)

    with torch.no_grad():
        out = fn(x, flow)
    out.float().cpu()
    print("ran           output %s" % (tuple(out.shape),))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
