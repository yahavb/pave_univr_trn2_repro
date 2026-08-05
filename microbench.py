#!/usr/bin/env python3
import argparse
import time

import torch

import repro_unrolling_trn2 as R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--rs0", required=True)
    ap.add_argument("--rs1", required=True)
    ap.add_argument("--height", type=int, default=256)
    ap.add_argument("--width", type=int, default=384)
    ap.add_argument("--gamma", type=float, default=0.98)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--warp", default="gridsample",
                    choices=("gridsample", "gather", "nki", "nki-dyn"))
    a = ap.parse_args()

    R._PRELU = R.NeuronPReLU
    R._WARP = R.WARPS[a.warp]
    if a.warp.startswith("nki"):
        R._NKI_DYN = (a.warp == "nki-dyn")
        R._NKI_FN, R._NKI_FN_DYN = R._build_nki()

    torch.manual_seed(0)
    model = R.UniVR().eval()
    R.load_weights(model, a.weights)
    model = model.to(torch.float32).to("neuron")

    img = torch.cat([R.load_img(a.rs0, a.height, a.width),
                     R.load_img(a.rs1, a.height, a.width)], 0)[None]
    img = img.to(torch.float32).to("neuron")
    t = 1 - a.gamma / 2

    cc = dict(backend="neuron", dynamic=False, fullgraph=True)
    print("shape         %dx%d" % (a.height, a.width))
    print("warp          %s" % a.warp)
    print("torch.compile %s" % cc)
    compiled = torch.compile(model, **cc)

    t0 = time.perf_counter()
    with torch.no_grad():
        out = compiled(img, t, a.gamma)
    out.float().cpu()
    print("first call    %.0f ms" % ((time.perf_counter() - t0) * 1e3))

    ts = []
    for _ in range(a.iters):
        t1 = time.perf_counter()
        with torch.no_grad():
            o = compiled(img, t, a.gamma)
        o.float().cpu()
        ts.append((time.perf_counter() - t1) * 1e3)
    ts.sort()
    print("median        %.1f ms over %d iters (min %.1f max %.1f)"
          % (ts[len(ts) // 2], len(ts), ts[0], ts[-1]))
    print("output        %s range [%.4f, %.4f]" % (tuple(o.shape), o.min(), o.max()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
