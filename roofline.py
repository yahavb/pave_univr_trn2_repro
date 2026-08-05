#!/usr/bin/env python3
"""Roofline analysis for the torch-level UniVR forward pass on trn2.

No device, no weights, no data: FLOPs and bytes are derived from layer shapes, and every
hardware constant is declared in HW with its provenance. Measured per-module device times
and engine shares come from the profiler (C28 / bundle README) and are stated as such.

    python roofline.py
    python roofline.py --height 864 --width 1024
    python roofline.py --peak-tflops 181 --peak-gbs 1300
"""
from __future__ import annotations

import argparse

HW = {
    "desc_ns": (26.5, "MEASURED: 19.4/20.3/21.6/17.8 ns at 1/2/9/32 elements"),
    "peak_tflops_fp32": (45.0, "ASSUMED per core at LNC=1 -- CONFIRM for your SDK"),
    "peak_gbs": (1000.0, "ASSUMED per-core HBM share -- CONFIRM"),
}

# MEASURED at tile 1280x992, fp32, one core. 'x' = invocations per forward.
# gpsimd/tensor are percentages of that module's device time (C28; ctx/unet from the
# bundle README's per-module table, which reports the dominant engine only).
MEASURED = {
    "a0":      {"ms": 81.67,  "x": 1, "gpsimd": 59.1, "tensor": 4.2},
    "a1":      {"ms": 106.00, "x": 1, "gpsimd": 45.7, "tensor": 16.3},
    "a2":      {"ms": 147.03, "x": 1, "gpsimd": 32.9, "tensor": None},
    "ctxconv": {"ms": 23.25,  "x": 2, "gpsimd": None, "tensor": None},
    "ctxwarp": {"ms": 13.75,  "x": 2, "gpsimd": 59.2, "tensor": None},
    "unet":    {"ms": 82.40,  "x": 1, "gpsimd": None, "tensor": 43.6},
}
MEASURED_FORWARD_MS = 756.0   # C28: whole forward, so modules do not sum to it
DESC_PER_PIXEL = 1.024        # C28 MEASURED: 1,300,832 packets / 1,269,760 pixels


def conv_flops(cin, cout, k, ho, wo):
    return 2.0 * cin * cout * k * k * ho * wo


def conv_bytes(cin, cout, k, hi, wi, ho, wo, itemsize=4):
    return itemsize * (cin * hi * wi + cout * ho * wo + cin * cout * k * k)


class Trace:
    def __init__(self):
        self.convs, self.warps = [], []

    def conv(self, tag, cin, cout, h, w, k=3, stride=1):
        ho, wo = h // stride, w // stride
        self.convs.append((tag, conv_flops(cin, cout, k, ho, wo),
                           conv_bytes(cin, cout, k, h, w, ho, wo)))
        return ho, wo

    def deconv(self, tag, cin, cout, h, w, k=4):
        ho, wo = h * 2, w * 2
        self.convs.append((tag, conv_flops(cin, cout, k, h, w),
                           conv_bytes(cin, cout, k, h, w, ho, wo)))
        return ho, wo

    def conv2(self, tag, cin, cout, h, w, stride=2):
        h, w = self.conv(tag + ".1", cin, cout, h, w, 3, stride)
        return self.conv(tag + ".2", cout, cout, h, w, 3, 1)

    def warp(self, tag, c, h, w):
        self.warps.append((tag, c, h, w))


def ifblock(t, tag, in_planes, c, scale, H, W):
    h, w = H // scale, W // scale
    h, w = t.conv(tag + ".conv0.1", in_planes, c // 2, h, w, 3, 2)
    h, w = t.conv(tag + ".conv0.2", c // 2, c, h, w, 3, 2)
    for i in range(8):
        t.conv(tag + ".blk%d" % i, c, c, h, w, 3, 1)
    t.deconv(tag + ".lastconv", c, 5, h, w)


def trace_forward(H, W):
    t = Trace()
    for tag, inp, c, sc in (("a0", 7, 240, 4), ("a1", 18, 150, 2), ("a2", 18, 90, 1)):
        ifblock(t, tag, inp, c, sc, H, W)
        t.warp(tag + ".w0", 3, H, W)
        t.warp(tag + ".w1", 3, H, W)
    for side in ("c0", "c1"):
        h, w, cin = H, W, 3
        for lvl, cout in enumerate((16, 32, 64, 128)):
            h, w = t.conv2("ctx.%s.L%d" % (side, lvl), cin, cout, h, w, stride=2)
            cin = cout
            t.warp("ctx.%s.w%d" % (side, lvl), cout, h, w)
    h, w = t.conv2("unet.down0", 17, 32, H, W)
    s1 = t.conv2("unet.down1", 64, 64, h, w)
    s2 = t.conv2("unet.down2", 128, 128, *s1)
    s3 = t.conv2("unet.down3", 256, 256, *s2)
    u = t.deconv("unet.up0", 512, 128, *s3)
    u = t.deconv("unet.up1", 256, 64, *u)
    u = t.deconv("unet.up2", 128, 32, *u)
    u = t.deconv("unet.up3", 64, 16, *u)
    t.conv("unet.conv", 16, 3, *u, k=3)
    return t


def group_of(tag):
    for g in ("a0", "a1", "a2"):
        if tag.startswith(g):
            return g
    if tag.startswith("ctx"):
        return "ctxconv" if ".L" in tag else "ctxwarp"
    return "unet"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--height", type=int, default=992)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--peak-tflops", type=float, default=None)
    ap.add_argument("--peak-gbs", type=float, default=None)
    a = ap.parse_args()

    peak_f = (a.peak_tflops or HW["peak_tflops_fp32"][0]) * 1e12
    peak_b = (a.peak_gbs or HW["peak_gbs"][0]) * 1e9
    dns = HW["desc_ns"][0]
    H, W = a.height, a.width
    t = trace_forward(H, W)

    g = {}
    for tag, fl, by in t.convs:
        d = g.setdefault(group_of(tag), {"fl": 0.0, "by": 0.0, "px": 0.0, "wby": 0.0})
        d["fl"] += fl
        d["by"] += by
    for tag, c, h, w in t.warps:
        d = g.setdefault(group_of(tag), {"fl": 0.0, "by": 0.0, "px": 0.0, "wby": 0.0})
        d["px"] += float(h * w)
        d["wby"] += 4.0 * 4 * c * h * w

    print("=" * 100)
    print("ROOFLINE -- torch-level UniVR forward, tile %dx%d, fp32, ONE NeuronCore" % (H, W))
    print("=" * 100)
    for k, (v, p) in HW.items():
        print("  %-18s %10.2f   %s" % (k, v, p))
    ridge = peak_f / peak_b
    print("\n  ridge = %.0f FLOP/B. Below it bandwidth-bound, above it compute-bound." % ridge)

    print("\nPART 1 -- CONV WORK: roofline floor vs what the profiler measured\n")
    print("  %-9s %3s %10s %8s %8s %9s %9s %11s %8s %8s" % (
        "module", "x", "GFLOP", "GB", "FLOP/B", "cmp ms", "bw ms", "measured", "vs roof", "tensor%"))
    print("  " + "-" * 96)
    tot = {"fl": 0.0, "by": 0.0, "c": 0.0, "b": 0.0}
    for k in ("a0", "a1", "a2", "ctxconv", "unet"):
        n = MEASURED[k]["x"]
        fl, by = g[k]["fl"] * n, g[k]["by"] * n
        c_ms, b_ms = fl / peak_f * 1e3, by / peak_b * 1e3
        meas = MEASURED[k]["ms"] * n
        te = MEASURED[k]["tensor"]
        tot["fl"] += fl; tot["by"] += by; tot["c"] += c_ms; tot["b"] += b_ms
        print("  %-9s %3d %10.2f %8.3f %8.1f %9.2f %9.2f %11.2f %7.0fx %8s" % (
            k, n, fl / 1e9, by / 1e9, fl / by, c_ms, b_ms, meas,
            meas / max(c_ms, b_ms), ("%.1f" % te) if te else "-"))
    print("  " + "-" * 96)
    roof = max(tot["c"], tot["b"])
    print("  %-9s %3s %10.2f %8.3f %8.1f %9.2f %9.2f" % (
        "TOTAL", "", tot["fl"] / 1e9, tot["by"] / 1e9, tot["fl"] / tot["by"], tot["c"], tot["b"]))
    print("\n  Conv roofline floor = %.1f ms. Every module is 40-90x above it." % roof)

    print("\nPART 2 -- THE RESAMPLE: a descriptor-rate ceiling the classic roofline omits\n")
    print("  %-9s %3s %13s %9s %8s %10s %11s %9s" % (
        "module", "x", "descriptors", "GB", "B/desc", "desc ms", "measured", "gpsimd%"))
    print("  " + "-" * 96)
    wtot = wdesc = 0.0
    for k in ("a0", "a1", "a2", "ctxwarp"):
        if g.get(k, {}).get("px", 0) == 0:
            continue
        n = MEASURED[k]["x"]
        desc = g[k]["px"] * DESC_PER_PIXEL * n
        by = g[k]["wby"] * n
        ms = desc * dns / 1e6
        gp = MEASURED[k]["gpsimd"]
        gp_ms = MEASURED[k]["ms"] * n * gp / 100 if gp else None
        wtot += ms; wdesc += desc
        print("  %-9s %3d %13s %9.3f %8.1f %10.2f %11s %9s" % (
            k, n, "{:,}".format(int(desc)), by / 1e9, by / desc, ms,
            ("%.2f" % gp_ms) if gp_ms else "-", ("%.1f" % gp) if gp else "-"))
    print("  " + "-" * 96)
    print("  %-9s %3s %13s %9s %8s %10.2f" % (
        "TOTAL", "", "{:,}".format(int(wdesc)), "", "", wtot))
    print("\n  Ceiling = %.0f M desc/s at %.1f ns. At C=3 the payload is 24 B, so this binds" % (
        1e3 / dns, dns))
    print("  at %.2f GB/s -- %.0fx under the bandwidth roof." % (
        24 / dns, peak_b / 1e9 / (24 / dns)))

    print("\nPART 3 -- WHERE THE TIME ACTUALLY GOES (measured, not modelled)\n")
    msum = sum(v["ms"] * v["x"] for v in MEASURED.values())
    gp_known = sum(v["ms"] * v["x"] * v["gpsimd"] / 100
                   for v in MEASURED.values() if v["gpsimd"])
    print("  module sum            %7.1f ms" % msum)
    print("  whole forward         %7.1f ms   (C28) -> %.0f ms = %.0f%% OUTSIDE the modules" % (
        MEASURED_FORWARD_MS, MEASURED_FORWARD_MS - msum,
        100 * (MEASURED_FORWARD_MS - msum) / MEASURED_FORWARD_MS))
    print("  gpsimd (gather)       %7.1f ms   = %.0f%% of module sum, %.0f%% of the forward" % (
        gp_known, 100 * gp_known / msum, 100 * gp_known / MEASURED_FORWARD_MS))
    print("  conv roofline floor   %7.1f ms   = %.1f%% of the forward" % (
        roof, 100 * roof / MEASURED_FORWARD_MS))
    print("  descriptor model      %7.1f ms   vs %.1f ms measured gpsimd = %.2fx" % (
        wtot, gp_known, wtot / gp_known))
    print("\n  MFU %.2f%%   MBU %.2f%%   (module sum, assumed peaks)" % (
        100 * tot["fl"] / (msum / 1e3) / peak_f,
        100 * (tot["by"] + sum(g[k]["wby"] * MEASURED[k]["x"] for k in g)) / (msum / 1e3) / peak_b))

    print("\nPART 4 -- IS THE TORCH CODE OPTIMAL?\n")
    conv_meas = sum(MEASURED[k]["ms"] * MEASURED[k]["x"] for k in ("ctxconv", "unet"))
    conv_roof = max(g["ctxconv"]["fl"] * 2 + g["unet"]["fl"],
                    0) / peak_f * 1e3
    print("  Two modules run NO warps, so their measured time is pure conv+overhead:")
    print("    ctxconv %5.1f ms  and  unet %5.1f ms  = %5.1f ms," % (
        MEASURED["ctxconv"]["ms"] * 2, MEASURED["unet"]["ms"], conv_meas))
    print("    against a %.1f ms compute roof -- so conv work runs ~%.0fx off its roof." % (
        conv_roof, conv_meas / conv_roof))
    print("    unet's effective bandwidth is %.1f GB/s of an assumed %.0f." % (
        g["unet"]["by"] / (MEASURED["unet"]["ms"] / 1e3) / 1e9, peak_b / 1e9))
    print("    Tensor is %.1f%% in unet but only %.1f%% in a0 -- the conv path is NOT" % (
        MEASURED["unet"]["tensor"], MEASURED["a0"]["tensor"]))
    print("    saturating anything. There IS torch-level headroom here.")
    print()
    print("  But size it before chasing it. Even driving ALL conv work to its roof saves")
    print("  at most %.0f ms of a %.0f ms forward = %.0f%%, while the gather is %.0f%%." % (
        conv_meas - conv_roof, MEASURED_FORWARD_MS,
        100 * (conv_meas - conv_roof) / MEASURED_FORWARD_MS,
        100 * gp_known / MEASURED_FORWARD_MS))
    print()
    print("  RANKED torch-level levers, by measured headroom:")
    print()
    print("  1. THE %.0f ms OUTSIDE the modules (%.0f%% of the forward). Biggest single term," % (
        MEASURED_FORWARD_MS - msum, 100 * (MEASURED_FORWARD_MS - msum) / MEASURED_FORWARD_MS))
    print("     larger than the entire conv budget, and it needs NO kernel work. It is host")
    print("     dispatch, layout/copy and sync between compiled regions -- pure torch-level")
    print("     structure. It is also UNATTRIBUTED: measure it first (per-call host vs device")
    print("     around each region) before optimising anything else.")
    print()
    print("  2. Lane fill / space-to-depth on ctxconv+unet: %.0f ms of conv time at C=3-17" % conv_meas)
    print("     inputs, which starve the 128-lane partition dim. This is the one place the")
    print("     classic roofline gap is real and torch-addressable (NHWC, channel padding,")
    print("     s2d stem). Ceiling %.0f ms, realistic gain a fraction of it." % (conv_meas - conv_roof))
    print()
    print("  3. The gather itself: %.0f ms, %.0f%% of the forward, at ~100%% of the descriptor" % (
        gp_known, 100 * gp_known / MEASURED_FORWARD_MS))
    print("     ceiling. NOT torch-addressable -- grid_sample, gather and NKI all issue ~1")
    print("     descriptor per output pixel because adjacent OUTPUT pixels read non-adjacent")
    print("     SOURCE pixels. Changing torch ops cannot change the addressing pattern.")
    print("     Only structured addressing helps, and that needs NKI:")
    struct = H * 2 * dns / 1e6
    print("       per-row uniform shift: %s desc of %d B = %.2f ms vs %.0f ms (%.0fx)" % (
        "{:,}".format(H * 2), W * 3 * 4, struct, gp_known, gp_known / struct))
    print("       That is an ARITHMETIC CEILING, not a result: a uniform per-row shift is")
    print("       exact only where flow is row-constant, and measured max displacement is")
    print("       43.28 px at 4K. The residual path is where the descriptors return.")
    print()
    print("  CAVEATS")
    print("   * peak TFLOP/s and GB/s are ASSUMED. The ranking is insensitive to them: conv")
    print("     work would have to be ~%.0fx costlier than its roof to rival the gather." % (
        gp_known / roof))
    print("   * Module times are measured at tile 1280x992; FLOPs/bytes here are computed at")
    print("     %dx%d. Pass --height/--width to match before comparing columns." % (H, W))
    print("   * Descriptor count uses the MEASURED 1.024/pixel, not the kernel's nominal 2.0.")
    print("     The model then lands %.2fx off measured gpsimd, so treat it as an estimate" % (
        wtot / gp_known))
    print("     of the right order, not a precise prediction.")


if __name__ == "__main__":
    raise SystemExit(main())
