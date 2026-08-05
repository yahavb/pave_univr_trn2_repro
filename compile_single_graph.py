#!/usr/bin/env python3
"""Compile the whole UniVR model as ONE graph on Neuron and run it once.

No tiling, no granularity options, no accounting. One graph, fullgraph=True.

    python compile_single_graph.py --weights pre_net_flow.pth --rs0 rs70.png --rs1 rs71.png
"""
import argparse
import sys
import time

import torch

import repro_unrolling_trn2 as R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--rs0", required=True)
    ap.add_argument("--rs1", required=True)
    ap.add_argument("--height", type=int, default=32)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--gamma", type=float, default=0.98)
    ap.add_argument("--device", default="neuron")
    a = ap.parse_args()

    if a.height % 32 or a.width % 32:
        raise SystemExit("height and width must be multiples of 32")

    # PReLU cannot be legalised under torch.compile; the swap is bit-exact.
    R._PRELU = R.NeuronPReLU
    R._WARP = R.warp_nki
    R._NKI_DYN = True
    R._NKI_FN, R._NKI_FN_DYN = R._build_nki()

    torch.manual_seed(0)
    model = R.UniVR().eval()
    R.load_weights(model, a.weights)

    img = torch.cat([R.load_img(a.rs0, a.height, a.width),
                     R.load_img(a.rs1, a.height, a.width)], 0)[None]
    t = 1 - a.gamma / 2

    print("shape        %dx%d" % (a.height, a.width))
    print("device       %s" % a.device)
    model = model.to(torch.float32).to(a.device)
    img = img.to(torch.float32).to(a.device)

    cc = dict(backend="neuron", dynamic=False, fullgraph=True)
    print("torch.compile %s" % cc)
    compiled = torch.compile(model, **cc)

    t0 = time.perf_counter()
    with torch.no_grad():
        out = compiled(img, t, a.gamma)
    out = out.float().cpu()
    ms = (time.perf_counter() - t0) * 1e3

    print("OK           output %s  range [%.4f, %.4f]" % (tuple(out.shape), out.min(), out.max()))
    print("first call   %.0f ms (compile + execute)" % ms)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
