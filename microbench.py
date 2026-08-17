#!/usr/bin/env python3
import argparse
import math
import os

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


def warp_gridsample_nkl(x, flow):
    """NKL grid_sample (KaenaNeuronKernelLibrary, CR-288764575) behind F.grid_sample semantics.

    Grid math is copied verbatim from warp_gridsample above so this is a true A/B against the
    reference -- the microbench already scores every op against warp_gridsample on CPU, so any
    accuracy delta here is the kernel and not the coordinates.

    Two kernel asserts shape this call: input_layout must be NHWC (the NCHW option in the
    docstring is unimplemented) so the value is permuted in and out, and gather_method
    "transpose" requires a 2-byte dtype, so fp32 takes the "copy" path.
    """
    try:
        from nkilib.experimental.indirect.grid_sample import grid_sample
    except ImportError as e:                                      # noqa: BLE001
        raise SystemExit(
            "op gridsample-nkl needs nkilib.experimental.indirect.grid_sample from "
            "KaenaNeuronKernelLibrary (CR-288764575, OPEN at rev 3 -- unmerged, so absent from "
            "released images). Import failed: %s" % e)
    from torch_neuronx.nki_hop import wrap_nki
    B, C, H, W = x.shape
    d = flow.device
    hor = torch.linspace(-1.0, 1.0, W, device=d, dtype=flow.dtype).view(1, 1, 1, W).expand(B, -1, H, -1)
    ver = torch.linspace(-1.0, 1.0, H, device=d, dtype=flow.dtype).view(1, 1, H, 1).expand(B, -1, -1, W)
    grid = torch.cat([hor, ver], 1)
    f = torch.cat([flow[:, 0:1] / ((W - 1.0) / 2.0), flow[:, 1:2] / ((H - 1.0) / 2.0)], 1)
    g = (grid + f).permute(0, 2, 3, 1).contiguous()
    gm = "transpose" if x.element_size() == 2 else "copy"
    out = wrap_nki(grid_sample)(x.permute(0, 2, 3, 1).contiguous(), g,
                                sampling_mode="bilinear", coord_mode="minus_one_one",
                                input_layout="NHWC", align_corners=True,
                                padding_mode="border", max_indices_per_indirect=None,
                                gather_method=gm)
    return out.permute(0, 3, 1, 2)


OPS = {
    "gridsample": warp_gridsample,
    "gridsample-nkl": warp_gridsample_nkl,
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
    ap.add_argument("--list-convs", action="store_true")
    ap.add_argument("--fused-shapes", action="store_true")
    ap.add_argument("--device", default="neuron")
    ap.add_argument("--flow-mag", type=float, default=8.0)
    ap.add_argument("--flow-smooth", action="store_true")
    ap.add_argument("--radius", type=int, default=3)
    ap.add_argument("--dtype", default="fp32", choices=("fp32", "bf16"))
    ap.add_argument("--int-flow", action="store_true")
    a = ap.parse_args()

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

    print("torch.compile fullgraph=True")
    op = OPS[a.op]
    if a.op == "shiftmatmul":
        _b = bulk
        _r = a.radius
        op = lambda t, f: warp_shiftmatmul(t, f, radius=_r, bulk=_b)
    elif a.op == "nkishift":
        _r = a.radius
        op = lambda t, f: warp_nkishift(t, f, radius=_r)
    fn = torch.compile(op, backend="neuron", dynamic=False, fullgraph=True)

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
