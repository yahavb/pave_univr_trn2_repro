#!/usr/bin/env python3
"""The model's op SEQUENCE for one tile, compiled and timed, with the resample swappable.

WHY THIS EXISTS, and why it is not another microbenchmark.

`microbench.py` times ONE op per invocation. That was the right tool for "how many DMA
descriptors does a resample issue", and it answered it: at the dominant site (C=3, 704x768)
gather runs 21.98 ms at 1.006 desc/px and gridsample 64.66 ms at 3.004, with an identical ~40 ns
per descriptor. But isolated ops cannot answer "what does swapping the resample do to the
model", because:

  * a single-op capture has no pipelining, no layout reuse, and no scheduling overlap with the
    convs that surround it, so per-op sums do not add up to the sequence
  * absolute single-io microseconds are noise across runs (a NEFF once measured 535,000 us in one
    capture and vanished in the next); only shares and counts are stable
  * a swapped resample changes the LAYOUT its neighbours see. The NKL kernel needs NHWC while the
    model is NCHW, so it adds two permutes per call that no per-op arm measures

So this runs the real sequence -- IFNet's three pyramid stages with their six full-resolution
warps, Contextnet's four levels x {img0, img1}, and the Unet -- as ONE compiled graph, at a real
padded tile shape, and reports the median. That number IS comparable between warps.

IT IMPORTS THE MODEL'S OWN MODULES. UniVR, IFNet_m, Contextnet, Unet, Conv2, the warp registry
and plan_tiles all come from repro_unrolling_trn2. Nothing is re-implemented, so the op sequence
is identical BY CONSTRUCTION rather than by transcription -- a hand-copied "distilled" model
would drift the first time either file changed, and a drifted sequence is worse than no
measurement because it looks authoritative.

What is deliberately dropped: weight loading, accuracy-vs-golden scoring, the 32-tile split, the
8-core threading and the stitch. One tile, one core, one graph, random weights.

RANDOM WEIGHTS MEAN PSNR AGAINST THE GOLDEN IS MEANINGLESS -- and this script never prints one.
Correctness is instead an EQUIVALENCE check between warps, which is the question that actually
matters here: with the same seed the weights and inputs are identical, so two resample
implementations must agree to fp32 rounding. Use --save-out with one warp and --cmp with the
other. That is a real correctness signal without needing the golden, and it is exactly how
gather and nki-dyn were shown to agree to every digit.

Usage:
    # baseline, save its output
    python distill_tile.py --grid 4x8 --halo 128 --tile 9 --warp gather \
        --iters 5 --save-out /tmp/gather.npy
    # the kernel under test, compared against it
    python distill_tile.py --grid 4x8 --halo 128 --tile 9 --warp gridsample-nkl \
        --iters 5 --cmp /tmp/gather.npy
"""
import argparse
import time

import numpy as np
import torch

import repro_unrolling_trn2 as M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="4x8",
                    help="tile grid, which decides the padded shapes. 4x8 is the only grid under "
                         "the compiler memory ceiling; 2x8/2x4/3x4 OOM as a fused model but a "
                         "single tile of theirs still compiles here.")
    ap.add_argument("--halo", type=int, default=128,
                    help="128 = production, proven fusable on all 8 slots. 64 = the config that "
                         "measured 3900.5 / 3820.4 ms and is 28%% less resample work.")
    ap.add_argument("--tile", type=int, default=9,
                    help="which tile. 9 is the LARGEST shape at 4x8 (704x768); --only-tile 1 is "
                         "NOT the largest and testing it once produced a false FUSES verdict.")
    ap.add_argument("--warp", default="gather", choices=sorted(M.WARPS),
                    help="the resample implementation under test -- the whole point of the script")
    ap.add_argument("--iters", type=int, default=5,
                    help="timed iterations after the compile/warmup call. Median is reported.")
    ap.add_argument("--dtype", default="fp32", choices=("fp32", "bf16"))
    ap.add_argument("--device", default="neuron")
    ap.add_argument("--fullgraph", type=int, choices=(0, 1), default=1,
                    help="0 allows graph breaks, which is what makes --warp gridsample usable: "
                         "ATen grid_sample's FUSED graph exceeds compiler memory. Refused for the "
                         "NKI warps, whose uint32 bitcast corrupts sampling across a break.")
    ap.add_argument("--gamma", type=float, default=0.98)
    ap.add_argument("--seed", type=int, default=0,
                    help="fixes weights AND input, so two warps are a true equivalence test")
    ap.add_argument("--nkl-gather-method", choices=("transpose", "copy"), default=None)
    ap.add_argument("--nkl-max-indices", type=int, default=None)
    ap.add_argument("--save-out", metavar="PATH.npy",
                    help="save this run's output for a later --cmp")
    ap.add_argument("--cmp", metavar="PATH.npy",
                    help="compare against a saved output: max_diff in LSB, PSNR, cosine. With the "
                         "same --seed this is an equivalence test between warp implementations.")
    ap.add_argument("--print-ops", action="store_true",
                    help="print the op sequence with its real dims and exit without running")
    a = ap.parse_args()

    if a.fullgraph == 0 and a.warp in ("nki", "nki-dyn"):
        ap.error("--fullgraph 0 is unsafe with --warp %s: a graph break around the "
                 "view(torch.uint32) index bitcast silently corrupts which pixels are sampled."
                 % a.warp)

    H, W = 1728, 4096
    ny, nx = (int(v) for v in a.grid.lower().split("x"))
    tiles = M.plan_tiles(H, W, ny, nx, a.halo)
    if not 0 <= a.tile < len(tiles):
        ap.error("--tile %d out of range: %s has %d tiles" % (a.tile, a.grid, len(tiles)))
    T = tiles[a.tile]
    ph, pw, py0 = T["ph"], T["pw"], T["py0"]

    # Which pass this tile needs, by the model's own rule. H/2 = 864 falls on the row1/row2
    # boundary at 4 rows, so no tile needs both and each compiles exactly ONE graph.
    half = H // 2
    need_f = (T["oy"] + T["vy"]) > half
    t_fwd, t_bwd = 1 - a.gamma / 2, -a.gamma / 2
    t = t_fwd if need_f else t_bwd

    print("=" * 96)
    print("DISTILLED TILE SEQUENCE  grid %s halo %d tile %d" % (a.grid, a.halo, a.tile))
    print("=" * 96)
    print("  padded %dx%d = %s px   valid %dx%d at (%d,%d)   row0 %d   pass %s (t=%+.4f)"
          % (ph, pw, format(ph * pw, ","), T["vy"], T["vx"], T["oy"], T["ox"], py0,
             "forward" if need_f else "backward", t))

    if a.print_ops:
        print()
        print("  RESAMPLE SEQUENCE (14 calls, the dims the model actually uses):")
        print("    %-16s %5s %12s %8s" % ("site", "C", "HxW", "calls"))
        for name, c, h, w, calls in _warp_seq(ph, pw):
            print("    %-16s %5d %12s %8d" % (name, c, "%dx%d" % (h, w), calls))
        print()
        print("  plus 54 conv sites per tile (IFBlock a0/a1/a2, Contextnet x2, Unet).")
        print("  Sequence: [pyramid stage0 conv -> 2 warps] -> [stage1] -> [stage2]")
        print("            -> Contextnet(img0) 4x[Conv2 -> warp] -> Contextnet(img1) 4x[...]")
        print("            -> Unet down0..3 / up0..3 -> final conv")
        return 0

    torch.manual_seed(a.seed)
    dt = torch.bfloat16 if a.dtype == "bf16" else torch.float32

    # Globals MUST be set before UniVR() is constructed: conv()/deconv() read _PRELU at build
    # time, and the warp registry is read through the module-level `warp` indirection.
    M._PRELU = M.NeuronPReLU
    M._WARP = M.WARPS[a.warp]
    M._NKI_DYN = (a.warp == "nki-dyn")
    M._NKL_GATHER_METHOD = a.nkl_gather_method
    M._NKL_MAX_INDICES = a.nkl_max_indices
    if a.warp == "gridsample-nkl":
        M._build_gridsample_nkl()      # fail in a second, not after a compile
        print("  NKL grid_sample imported and wrapped OK")
    if a.warp == "shiftwarp":
        M._SHIFTWARP_FN = M._build_shiftwarp()

    if a.device == "neuron":
        import torch_neuronx  # noqa: F401

    model = M.UniVR().to(dt).to(a.device).eval()
    x = torch.rand(1, 6, ph, pw, dtype=dt).to(a.device)

    print("  warp %s   dtype %s   device %s   fullgraph %s"
          % (a.warp, a.dtype, a.device, bool(a.fullgraph)))
    compiled = torch.compile(model, backend="neuron", dynamic=False, fullgraph=bool(a.fullgraph))

    with torch.no_grad():
        t0 = time.perf_counter()
        out = compiled(x, t, a.gamma, row0=py0, full_h=H)
        out.float().cpu()
        first = (time.perf_counter() - t0) * 1e3
    print("  first call %.0f ms (compile + warmup)   out %s" % (first, tuple(out.shape)))

    ts = []
    with torch.no_grad():
        for _ in range(a.iters):
            t0 = time.perf_counter()
            o = compiled(x, t, a.gamma, row0=py0, full_h=H)
            o.float().cpu()
            ts.append((time.perf_counter() - t0) * 1e3)
    ts.sort()
    med = ts[len(ts) // 2]
    print()
    print("  TILE MEDIAN  %.2f ms over %d iters (min %.2f, max %.2f)"
          % (med, len(ts), ts[0], ts[-1]))

    # Frame extrapolation. Honest about what it assumes: every tile of this shape costs the same,
    # and the 8 cores overlap perfectly. Both are optimistic, so treat it as a LOWER bound.
    same = sum(1 for q in tiles if (q["ph"], q["pw"]) == (ph, pw))
    print("  %d of %d tiles share this shape -> %.0f ms of tile work, %.0f ms across 8 cores"
          % (same, len(tiles), med * same, med * same / 8))
    print("  (extrapolation assumes equal cost per tile and perfect 8-way overlap: a LOWER bound)")

    got = o.float().cpu().numpy()
    if a.save_out:
        np.save(a.save_out, got)
        print("  saved output -> %s  (compare another warp with --cmp)" % a.save_out)
    if a.cmp:
        ref = np.load(a.cmp)
        if ref.shape != got.shape:
            print("  CMP n/a: shape %s vs %s" % (ref.shape, got.shape))
        else:
            d = np.abs(ref - got)
            lsb = float(d.max()) * 255.0
            mse = float((d ** 2).mean())
            psnr = float("inf") if mse == 0 else 10.0 * np.log10(1.0 / mse)
            cos = float(np.dot(ref.ravel(), got.ravel())
                        / (np.linalg.norm(ref.ravel()) * np.linalg.norm(got.ravel()) + 1e-30))
            print("  EQUIVALENCE vs %s: max_diff %.4f LSB   PSNR %.2f dB   cos %.6f   [%s]"
                  % (a.cmp, lsb, psnr, cos, "AGREE" if lsb <= 3.0 else "DISAGREE"))
            print("  Same seed, so weights and input are identical: this compares the RESAMPLE")
            print("  implementations only. It is NOT an accuracy claim against the golden --")
            print("  random weights make any PSNR-vs-golden meaningless, which is why none is shown.")
    return 0


def _warp_seq(ph, pw):
    """The 14 resample calls one tile performs, read off the model's structure.

    Six at full resolution: three pyramid stages (_StageFirst + 2x _StageNext), each warping img0
    and img1. Then one per Contextnet level x {img0, img1}: level lvl outputs _C * 2**(lvl-1)
    channels at ph/2**lvl, because Conv2's first conv has stride 2.
    """
    sites = [("ifnet_pyramid", 3, ph, pw, 6)]
    for lvl in range(1, 5):
        sites.append(("ctx%d" % lvl, M._C * 2 ** (lvl - 1), ph >> lvl, pw >> lvl, 2))
    return sites


if __name__ == "__main__":
    raise SystemExit(main())
