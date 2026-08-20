"""NKI bilinear-resample (warp / grid_sample replacement) for trn2.

The slow part of the PyTorch warp is the 4-tap gather, which lowers to
software-dynamic DMA (device profile: 99% swDMA, 0.1% TensorE, MBU 0.10% —
random-access latency-bound). This kernel does the 4-tap gather + bilinear
blend using the HARDWARE vector-dynamic-access indirect DMA (the proven pattern
from torch_neuronx's klir_gather: `tensor.ap(..., vector_offset=idx, indirect_dim=0)`),
which is the efficient indirect path rather than the swDMA decomposition.

Layout: image as [K, C] with K = H*W rows (the spatial axis we gather along,
dim=0) and C channels in the contiguous free dim. For each output pixel k the 4
bilinear neighbours are 4 source rows; we gather each and blend with per-pixel
weights (broadcast over C). Cheap index/weight arithmetic (floor/clamp/frac)
stays as torch elementwise ops (those lower fine; they are not the bottleneck).

NKI 0.4.0 API.
"""
import os
import sys

_PKG = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _PKG)

import torch  # noqa: E402

import nki  # noqa: E402
import nki.isa as nisa  # noqa: E402
import nki.language as nl  # noqa: E402
from nki.isa.constants import oob_mode  # noqa: E402
from torch_neuronx.nki_hop import wrap_nki  # noqa: E402


def _div_ceil(n, d):
    return (n + d - 1) // d


@nki.jit
def resample_gather_blend(img, idx0, idx1, idx2, idx3, w0, w1, w2, w3):
    """out[k, :] = sum_t w_t[k] * img[idx_t[k], :]   (bilinear 4-tap blend).

    img:   [K, C] fp32 (HBM), K = H*W spatial rows, C channels (free).
    idx_t: [K, 1] int32 (HBM) — source row index for tap t.
    w_t:   [K, 1] fp32  (HBM) — bilinear weight for tap t (broadcast over C).
    Returns out [K, C] fp32.
    """
    K, C = img.shape
    dtype = img.dtype  # bf16 or fp32 — gather + output stay in the data dtype
    out = nl.ndarray((K, C), dtype=dtype, buffer=nl.shared_hbm)
    KT = nl.tile_size.pmax  # 128 rows per tile
    n_tiles = _div_ceil(K, KT)
    idxs = (idx0, idx1, idx2, idx3)
    ws = (w0, w1, w2, w3)

    for t in nl.affine_range(n_tiles):
        k_off = t * KT
        kv = min(KT, K - t * KT)
        # Blend accumulates in fp32 for accuracy even when data is bf16.
        acc = nl.ndarray((kv, C), dtype=nl.float32, buffer=nl.sbuf)
        for tap in range(4):
            idx_tile = nl.ndarray((kv, 1), dtype=nl.int32, buffer=nl.sbuf)
            nisa.dma_copy(dst=idx_tile,
                          src=idxs[tap].ap(pattern=[[1, kv], [1, 1]], offset=k_off))
            w_tile = nl.ndarray((kv, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=w_tile,
                          src=ws[tap].ap(pattern=[[1, kv], [1, 1]], offset=k_off))
            # hardware indirect gather: row p reads img[idx_tile[p], 0:C] (native dtype)
            g = nl.ndarray((kv, C), dtype=dtype, buffer=nl.sbuf)
            nisa.dma_copy(
                dst=g,
                src=img.ap(pattern=[[C, kv], [1, C]], offset=0,
                           vector_offset=idx_tile, indirect_dim=0),
                oob_mode=oob_mode.error,
            )
            # multiply by per-row weight (broadcast over C) into fp32 acc.
            gw = nl.ndarray((kv, C), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=gw, data=g, op0=nl.multiply, operand0=w_tile)
            if tap == 0:
                nisa.tensor_copy(dst=acc, src=gw)
            else:
                nisa.tensor_tensor(dst=acc, data1=acc, data2=gw, op=nl.add)
        # cast the fp32 blend back to the data dtype on store
        out_tile = nl.copy(acc, dtype=dtype)
        nisa.dma_copy(dst=out[nl.ds(k_off, kv), 0:C], src=out_tile)

    return out


_wrapped = wrap_nki(resample_gather_blend)


def _indices_weights(flow, H, W):
    """4 neighbour flat row-indices + bilinear weights from flow.
    align_corners=True, padding_mode='border'. Returns idx_t [HW,1] int32,
    w_t [HW,1] fp32 (N=1).

    Coordinate/index math is done in fp32 regardless of the flow dtype: bf16's
    8-bit mantissa cannot represent integer coords/indices up to H*W, so a bf16
    flow would produce wrong gather indices. Weights returned fp32 (the kernel
    blends in fp32)."""
    dev = flow.device
    ft = torch.float32  # force fp32 coord/index math (bf16-safe)
    ys = torch.arange(H, device=dev, dtype=ft).view(H, 1).expand(H, W)
    xs = torch.arange(W, device=dev, dtype=ft).view(1, W).expand(H, W)
    sx = (xs + flow[0, 0].to(ft)).clamp(0, W - 1)
    sy = (ys + flow[0, 1].to(ft)).clamp(0, H - 1)
    x0 = torch.floor(sx); y0 = torch.floor(sy)
    wx = sx - x0; wy = sy - y0
    x0i = x0.clamp(0, W - 1); x1i = (x0 + 1).clamp(0, W - 1)
    y0i = y0.clamp(0, H - 1); y1i = (y0 + 1).clamp(0, H - 1)

    def flat(yi, xi):
        return (yi * W + xi).reshape(H * W, 1).to(torch.int32)

    idx = (flat(y0i, x0i), flat(y1i, x0i), flat(y0i, x1i), flat(y1i, x1i))
    wxf = wx.reshape(H * W, 1); wyf = wy.reshape(H * W, 1)
    w = ((1 - wxf) * (1 - wyf), (1 - wxf) * wyf, wxf * (1 - wyf), wxf * wyf)
    return idx, w


def warp_resample_nki(tenInput, tenFlow):
    """Drop-in for the gather-based warp_neuron. tenInput [1,C,H,W], tenFlow [1,2,H,W].

    Works for fp32 or bf16 image data: the gather/blend run in the image dtype
    (blend internally accumulates fp32), while indices/weights are computed in
    fp32 for exactness."""
    N, C, H, W = tenInput.shape
    img_kc = tenInput.permute(0, 2, 3, 1).reshape(H * W, C).contiguous()  # [HW, C]
    idx, w = _indices_weights(tenFlow, H, W)
    out_kc = _wrapped(img_kc, idx[0], idx[1], idx[2], idx[3], w[0], w[1], w[2], w[3])
    return out_kc.reshape(H, W, C).permute(2, 0, 1).reshape(N, C, H, W)


if __name__ == "__main__":
    import argparse
    import time

    import torch.nn.functional as F
    import torch_neuronx  # noqa: F401

    from univr_neuron import warp_neuron, precompute_warp_grids  # noqa: E402

    def cos(a, b):
        a, b = a.flatten().float(), b.flatten().float()
        return torch.nn.functional.cosine_similarity(a, b, dim=0).item()

    def ref_warp(img, flow):
        """fp32 grid_sample golden (CPU)."""
        H, W = img.shape[-2:]
        ys = torch.linspace(-1, 1, H).view(1, 1, H, 1).expand(img.shape[0], 1, H, W)
        xs = torch.linspace(-1, 1, W).view(1, 1, 1, W).expand(img.shape[0], 1, H, W)
        base = torch.cat([xs, ys], 1)
        fl = torch.cat([flow[:, 0:1] / ((W - 1) / 2), flow[:, 1:2] / ((H - 1) / 2)], 1)
        g = (base + fl).permute(0, 2, 3, 1)
        return F.grid_sample(img, g, mode="bilinear", padding_mode="border", align_corners=True)

    ap = argparse.ArgumentParser()
    ap.add_argument("--height", type=int, default=256)
    ap.add_argument("--width", type=int, default=384)
    ap.add_argument("--channels", type=int, default=16)
    ap.add_argument("--iters", type=int, default=20)
    args = ap.parse_args()
    C, H, W = args.channels, args.height, args.width
    torch.manual_seed(0)
    img = torch.rand(1, C, H, W)
    flow = (torch.rand(1, 2, H, W) - 0.5) * 8
    ref = ref_warp(img, flow)  # fp32 golden
    dev = torch.device("neuron")

    def timeit(fn, n):
        for _ in range(3):
            fn(); torch_neuronx.synchronize()
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        torch_neuronx.synchronize()
        return (time.perf_counter() - t0) / n * 1000

    print(f"WARP A/B  C={C} H={H} W={W}  (accuracy vs fp32 grid_sample, time over {args.iters} iters)")
    print(f"{'variant':>22} | {'cos_sim':>9} | {'max_diff':>10} | {'ms':>8}")
    print("-" * 60)

    results = {}
    for dt_name, dt in (("fp32", torch.float32), ("bf16", torch.bfloat16)):
        imgd = img.to(dt).to(dev)
        flowd = flow.to(dt).to(dev)
        # precompute base grids for warp_neuron at this dtype/res (eager fallback also works)
        precompute_warp_grids((H, W), dev, dt, n_levels=1)

        # --- torch.gather baseline (our warp_neuron), compiled ---
        ctorch = torch.compile(warp_neuron, backend="neuron", fullgraph=True, dynamic=False)
        out_t = ctorch(imgd, flowd).cpu().float()
        ms_t = timeit(lambda: ctorch(imgd, flowd), args.iters)
        results[f"gather_{dt_name}"] = (cos(ref, out_t), (ref - out_t).abs().max().item(), ms_t)
        print(f"{'torch.gather '+dt_name:>22} | {results[f'gather_{dt_name}'][0]:>9.6f} | "
              f"{results[f'gather_{dt_name}'][1]:>10.3e} | {ms_t:>8.3f}", flush=True)

        # --- NKI DMA-DGE gather ---
        out_n = warp_resample_nki(imgd, flowd).cpu().float()
        ms_n = timeit(lambda: warp_resample_nki(imgd, flowd), args.iters)
        results[f"nki_{dt_name}"] = (cos(ref, out_n), (ref - out_n).abs().max().item(), ms_n)
        print(f"{'NKI-DMA-DGE '+dt_name:>22} | {results[f'nki_{dt_name}'][0]:>9.6f} | "
              f"{results[f'nki_{dt_name}'][1]:>10.3e} | {ms_n:>8.3f}", flush=True)

    # Summary: speedups vs the current baseline (gather_fp32)
    base = results["gather_fp32"][2]
    print("\nSpeedup vs current baseline (torch.gather fp32 = 1.00x):")
    for k, (_, _, ms) in results.items():
        print(f"  {k:>16}: {base/ms:>5.2f}x  ({ms:.3f} ms)")
    print("DONE", flush=True)
