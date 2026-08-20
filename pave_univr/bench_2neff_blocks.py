"""Per-block timing breakdown of the 2-NEFF UniVR graph, with the quality gate attached.

Reports NEFF_A (IFBlocks + pyramid resamples + 6 full-res warps + merge) and NEFF_B
(Contextnet + Unet + 8 warps) separately, plus the full forward, and runs ablations so
each cost can be ATTRIBUTED rather than guessed:

  resample : bilinear (stock semantics) vs fast (avg_pool/nearest)  -> cost of correctness
  warp     : NKI v3 indirect-DMA gather vs torch.gather             -> cost of the gather
  pad_ctx  : Contextnet stem Cin 3->16 zero-padded                  -> the conv fast-path lever

Every row carries cos/PSNR vs the stock CPU fp32 reference computed from the same
weights, because a runtime number without a passing gate is meaningless (the
avg_pool/nearest config is 2-3x faster and 15 dB wrong -- that is exactly the trap).

Timing uses a real device->host completion barrier and a t(reps)=fixed+reps*device fit,
so the reported per-block cost is marginal device time, not per-launch overhead.

Usage: NEURON_LOGICAL_NC_CONFIG=1 NEURON_RT_VISIBLE_CORES=16 \
       NEURON_CC_FLAGS='--target trn2 --lnc 1' \
       python -u pave_univr/bench_2neff_blocks.py --height 256 --width 384
"""
from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import statistics as st
import sys
import time

os.environ.setdefault("NEURON_RT_NUM_CORES", "1")
_PKG = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _PKG)

import torch  # noqa: E402

from univr_utils import setup_univr_paths  # noqa: E402

setup_univr_paths(model_type="rife")

import network_UniVR as netmod  # noqa: E402


def metrics(ref, test):
    r, t = ref.float().flatten(), test.float().flatten()
    cos = torch.nn.functional.cosine_similarity(r, t, dim=0).item()
    mse = ((r - t) ** 2).mean().item()
    psnr = float("inf") if mse == 0 else 10 * math.log10(1.0 / mse)
    return cos, psnr, (r - t).abs().max().item()


def timed(fn, args, iters, reps=1):
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(reps):
                out = fn(*args)
        (out[0] if isinstance(out, tuple) else out).flatten()[:1].cpu()
        ts.append(time.perf_counter() - t0)
    return st.median(ts), out


def fit(fn, args, iters):
    xs, ys = [], []
    for r in (1, 2, 4):
        m, _ = timed(fn, args, max(4, iters // 3), reps=r)
        xs.append(float(r)); ys.append(m)
    n = len(xs); mx = sum(xs) / n; my = sum(ys) / n
    slope = (sum((a - mx) * (b - my) for a, b in zip(xs, ys))
             / sum((a - mx) ** 2 for a in xs))
    return slope, my - slope * mx


def cpu_reference(H, W, t, gamma):
    torch.manual_seed(0)
    m = netmod.UniVR(device=torch.device("cpu")).to("cpu").eval()
    im = torch.rand(1, 6, H, W).clamp(0, 1)
    rows = torch.arange(H, dtype=torch.float32).view(1, 1, H, 1)
    tof = (t + gamma - gamma * rows / H + 0.0001).expand(1, 1, H, W).contiguous()
    with torch.no_grad():
        _, _, merged = m.UVR(torch.cat((im, torch.zeros(1, 0, H, W)), 1), tof)
    return im, merged[2].float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--height", type=int, default=256)
    ap.add_argument("--width", type=int, default=384)
    ap.add_argument("--gamma", type=float, default=0.98)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--gate", type=float, default=0.999)
    ap.add_argument("--configs", default="bilinear/3/1",
                    help="resample/warp/padctx. ONE per process: re-importing the NKI "
                         "modules in-process breaks torch_neuronx's kernel registration "
                         "(AssertionError: Unexpected type NKIHOPCaller), so the shell "
                         "driver launches a separate process per config.")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"],
                    help="storage dtype for both NEFFs")
    ap.add_argument("--split", action="store_true",
                    help="4-NEFF form: NEFF_A split per pyramid stage (A0/A1/A2) + B, "
                         "timed per stage. Tests whether the 40x fp32 NEFF_A blow-up is a "
                         "working-set problem.")
    ap.add_argument("--json-out", default="")
    args = ap.parse_args()
    H, W, g = args.height, args.width, args.gamma
    t = 1 - g / 2

    import torch_neuronx  # noqa: F401

    print(f"=== 2-NEFF per-block breakdown | {W}x{H} bf16 | "
          f"core={os.environ.get('NEURON_RT_VISIBLE_CORES','?')} "
          f"lnc={os.environ.get('NEURON_LOGICAL_NC_CONFIG','?')} | gate cos>{args.gate} ===",
          flush=True)
    im, ref = cpu_reference(H, W, t, g)
    print(f"CPU fp32 stock reference ready {tuple(ref.shape)}", flush=True)

    rows = []
    for spec in [s.strip() for s in args.configs.split(",") if s.strip()]:
        rs, warp, pad = spec.split("/")
        os.environ["PAVE_UNIVR_RESAMPLE"] = rs
        os.environ["PAVE_UNIVR_NKI_WARP"] = warp
        os.environ["PAVE_UNIVR_PAD_CTX_CIN"] = pad
        for mod in ("univr_2neff", "nki_dma_warp", "nki_dma_warp_v2", "nki_dma_warp_v3"):
            sys.modules.pop(mod, None)
        U = importlib.import_module("univr_2neff")
        dt = torch.bfloat16 if args.dtype == "bf16" else torch.float32
        label = f"{rs}/warp{warp}/pad{pad}/{args.dtype}"
        if args.split:
            label += "/split4"
            try:
                torch.manual_seed(0)
                m2 = netmod.UniVR(device=torch.device("cpu")).to("cpu").eval()
                ifn = m2.UVR
                U.replace_prelu(ifn)
                a0 = U.NEFF_A0(ifn.block0, H, W).to(dt).eval().to("neuron")
                a1 = U.NEFF_A1(ifn.block1, H, W).to(dt).eval().to("neuron")
                a2 = U.NEFF_A2(ifn.block2, H, W).to(dt).eval().to("neuron")
                b = U.NEFF_B(ifn.contextnet, ifn.unet, H, W).to(dt).eval().to("neuron")
                f0, f1, f2, fb = (U._compile(a0), U._compile(a1), U._compile(a2),
                                  U._compile(b))
                imd = im.to(dt).to("neuron")
                i0, i1 = imd[:, :3], imd[:, 3:6]
                row_idx = torch.arange(H, dtype=torch.float32).view(1, 1, H, 1)
                tofd = ((t + g - g * row_idx / H + 0.0001).expand(1, 1, H, W)
                        .contiguous().to(dt).to("neuron"))
                tc0 = time.perf_counter()
                with torch.no_grad():
                    fl, mk, w0, w1 = f0(i0, i1, tofd)
                    fl, mk, w0, w1 = f1(i0, i1, tofd, w0, w1, mk, fl)
                    mg, fl, mk, w0, w1 = f2(i0, i1, tofd, w0, w1, mk, fl)
                    fb(i0, i1, w0, w1, mk, fl, mg).flatten()[:1].cpu()
                compile_s = time.perf_counter() - tc0

                args0 = (i0, i1, tofd)
                with torch.no_grad():
                    fl0, mk0, w00, w10 = f0(*args0)
                args1 = (i0, i1, tofd, w00, w10, mk0, fl0)
                with torch.no_grad():
                    fl1, mk1, w01, w11 = f1(*args1)
                args2 = (i0, i1, tofd, w01, w11, mk1, fl1)
                with torch.no_grad():
                    mg2, fl2, mk2, w02, w12 = f2(*args2)
                argsb = (i0, i1, w02, w12, mk2, fl2, mg2)
                d0, x0 = fit(f0, args0, args.iters)
                d1, x1 = fit(f1, args1, args.iters)
                d2, x2 = fit(f2, args2, args.iters)
                db, xb = fit(fb, argsb, args.iters)

                with torch.no_grad():
                    fl, mk, w0, w1 = f0(i0, i1, tofd)
                    fl, mk, w0, w1 = f1(i0, i1, tofd, w0, w1, mk, fl)
                    mg, fl, mk, w0, w1 = f2(i0, i1, tofd, w0, w1, mk, fl)
                    out = fb(i0, i1, w0, w1, mk, fl, mg).float().cpu()
                cos, psnr, mxd = metrics(ref, out)
                ok = cos > args.gate
                tot = (d0 + d1 + d2 + db) * 1e3
                rows_out = {"config": label, "a0_ms": d0 * 1e3, "a1_ms": d1 * 1e3,
                            "a2_ms": d2 * 1e3, "neff_b_ms": db * 1e3, "total_ms": tot,
                            "cos": cos, "psnr_db": psnr, "max_diff": mxd,
                            "gate_pass": ok, "compile_s": compile_s}
                rows.append(rows_out)
                print(f"[{label:30s}] A0 {d0*1e3:7.2f} | A1 {d1*1e3:7.2f} | "
                      f"A2 {d2*1e3:7.2f} | B {db*1e3:7.2f} | fwd {tot:8.2f} ms | "
                      f"cos {cos:.6f} PSNR {psnr:6.2f} dB "
                      f"[{'PASS' if ok else 'FAIL'}] | compile {compile_s:.0f}s", flush=True)
                del a0, a1, a2, b, f0, f1, f2, fb, m2
            except Exception as e:  # noqa: BLE001
                print(f"[{label:30s}] FAILED: {type(e).__name__}: {str(e)[:300]}",
                      flush=True)
                rows.append({"config": label, "error": f"{type(e).__name__}: {str(e)[:300]}"})
            continue
        try:
            torch.manual_seed(0)
            m2 = netmod.UniVR(device=torch.device("cpu")).to("cpu").eval()
            ifn = m2.UVR
            U.replace_prelu(ifn)
            a = U.NEFF_A(ifn.block0, ifn.block1, ifn.block2, H, W).to(dt).eval().to("neuron")
            b = U.NEFF_B(ifn.contextnet, ifn.unet, H, W).to(dt).eval().to("neuron")
            fa, fb = U._compile(a), U._compile(b)

            imd = im.to(dt).to("neuron")
            t0 = time.perf_counter()
            with torch.no_grad():
                outa = fa(imd, t, g)
                merged, flow, mask, wi0, wi1 = outa
                outb = fb(imd[:, :3], imd[:, 3:6], wi0, wi1, mask, flow, merged)
                outb.flatten()[:1].cpu()
            compile_s = time.perf_counter() - t0

            a_args = (imd, t, g)
            b_args = (imd[:, :3], imd[:, 3:6], wi0, wi1, mask, flow, merged)
            _ = timed(fa, a_args, 2); _ = timed(fb, b_args, 2)
            da, fxa = fit(fa, a_args, args.iters)
            db, fxb = fit(fb, b_args, args.iters)

            with torch.no_grad():
                mg, fl, mk, w0, w1 = fa(imd, t, g)
                out = fb(imd[:, :3], imd[:, 3:6], w0, w1, mk, fl, mg).float().cpu()
            cos, psnr, mxd = metrics(ref, out)
            ok = cos > args.gate
            tot = (da + db) * 1e3
            row = {"config": label, "resample": rs, "warp": warp, "pad_ctx": pad,
                   "neff_a_ms": da * 1e3, "neff_b_ms": db * 1e3, "total_ms": tot,
                   "fixed_a_ms": fxa * 1e3, "fixed_b_ms": fxb * 1e3,
                   "cos": cos, "psnr_db": psnr, "max_diff": mxd, "gate_pass": ok,
                   "compile_s": compile_s}
            rows.append(row)
            print(f"[{label:22s}] NEFF_A {da*1e3:8.2f} ms | NEFF_B {db*1e3:8.2f} ms | "
                  f"fwd {tot:8.2f} ms | cos {cos:.6f} PSNR {psnr:6.2f} dB "
                  f"[{'PASS' if ok else 'FAIL'}] | compile {compile_s:.0f}s", flush=True)
            del a, b, fa, fb, m2
        except Exception as e:  # noqa: BLE001
            print(f"[{label:22s}] FAILED: {type(e).__name__}: {str(e)[:300]}", flush=True)
            rows.append({"config": label, "error": f"{type(e).__name__}: {str(e)[:300]}"})

    base = next((r for r in rows if r.get("config") == "bilinear/warp3/pad1"
                 and "total_ms" in r), None)
    if base:
        print("\n--- attribution (vs bilinear/warp3/pad1, the gated production config) ---")
        for r in rows:
            if "total_ms" in r and r is not base:
                d = r["total_ms"] - base["total_ms"]
                print(f"  {r['config']:22s} {r['total_ms']:8.2f} ms "
                      f"({d:+8.2f} ms, {base['total_ms']/r['total_ms']:.2f}x) "
                      f"gate {'PASS' if r['gate_pass'] else 'FAIL'}")
    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump({"config": vars(args), "rows": rows}, f, indent=1)
        print(f"wrote {args.json_out}")
    print("DONE")


if __name__ == "__main__":
    main()
