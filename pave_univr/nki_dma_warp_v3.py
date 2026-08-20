"""DMA-DGE bilinear warp v3 — TRUE Direction 1: collapse 4 gathers into 1.

The four bilinear corners (tl, tr, bl, br) of pixel k live at rows
{tl, tl+1, tl+W, tl+W+1} in the [N=H*W, C] row-major source. A SINGLE indirect
gather with a 2-level free access pattern reads the whole 2x2xC neighborhood:

    free pattern = [[W*C, 2], [1, 2*C]]
      -> chunk0 (2C) = rows tl, tl+1  (top pair: tl, tr)
      -> chunk1 (2C) = rows tl+W, tl+W+1 (bottom pair: bl, br)

This issues ONE descriptor per pixel instead of four -> 4x fewer SWDGE
descriptors on GpSimd (the profiled bottleneck). No padded source needed:
the +1/+W offsets are DMA strides, and edge correctness is guaranteed because
the bilinear weight is exactly 0 at clamped boundaries (wrong-but-finite * 0 = 0).
OOB at the final pixel is handled with oob_mode.skip + zero-init gather tile.

Fallback (V3_TWO_GATHER=True): two gathers (top pair, bottom pair), each a plain
[1, 2*C] free pattern (definitely supported) -> 2x descriptor cut.
"""

import os
import nki
import nki.isa as nisa
import nki.language as nl
from nki.isa.constants import oob_mode

_TWO_GATHER = os.environ.get("V3_TWO_GATHER", "0") == "1"
# Mixed strategy: use the single strided 2x2 gather (4->1) only when the contiguous
# payload per row (C) is large enough to amortize the strided transfer; otherwise use
# two contiguous-pair gathers (4->2). Crossover measured between C=16 (4->1 loses) and
# C=64 (4->1 wins). C is known at trace time so this is a compile-time decision.
_C_THRESHOLD = int(os.environ.get("PAVE_UNIVR_WARP_CTHRESH", "48"))


@nki.jit
def bilinear_2x2_gather_blend(img, idx_tl, w_tl, w_tr, w_bl, w_br, row_w):
    """Single 2x2 gather + bilinear blend. img: [N, C] HBM. idx_tl: [K,1] uint32.
    row_w = W (source width in pixels), so bottom row = tl + W."""
    K = idx_tl.shape[0]
    C = img.shape[1]
    dtype = img.dtype

    P = nl.tile_size.pmax
    num_k_tiles = (K + P - 1) // P
    out = nl.ndarray((K, C), dtype=dtype, buffer=nl.shared_hbm)

    for kt in nl.affine_range(num_k_tiles):
        k_off = kt * P
        kv = min(P, K - kt * P)

        idx_t = nl.ndarray((kv, 1), dtype=idx_tl.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=idx_t, src=idx_tl.ap(pattern=[[1, kv], [1, 1]], offset=k_off))

        # idx for bottom row = tl + W (computed in SBUF)
        idx_bot = nl.ndarray((kv, 1), dtype=idx_tl.dtype, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=idx_bot, data=idx_t, op0=nl.add, operand0=row_w)

        # weights
        w_tl_t = nl.ndarray((kv, 1), dtype=nl.float32, buffer=nl.sbuf)
        w_tr_t = nl.ndarray((kv, 1), dtype=nl.float32, buffer=nl.sbuf)
        w_bl_t = nl.ndarray((kv, 1), dtype=nl.float32, buffer=nl.sbuf)
        w_br_t = nl.ndarray((kv, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=w_tl_t, src=w_tl.ap(pattern=[[1, kv], [1, 1]], offset=k_off))
        nisa.dma_copy(dst=w_tr_t, src=w_tr.ap(pattern=[[1, kv], [1, 1]], offset=k_off))
        nisa.dma_copy(dst=w_bl_t, src=w_bl.ap(pattern=[[1, kv], [1, 1]], offset=k_off))
        nisa.dma_copy(dst=w_br_t, src=w_br.ap(pattern=[[1, kv], [1, 1]], offset=k_off))

        acc = nl.ndarray((kv, C), dtype=nl.float32, buffer=nl.sbuf)
        tmp = nl.ndarray((kv, C), dtype=nl.float32, buffer=nl.sbuf)

        # Mixed: small C -> two contiguous-pair gathers (4->2); large C -> single
        # strided 2x2 gather (4->1). Decided at trace time from C.
        # V3_TWO_GATHER=1 forces 4->2 for benchmarking.
        if _TWO_GATHER or C < _C_THRESHOLD:
            # Two gathers: top pair (tl,tr) and bottom pair (bl,br), each [1, 2C]
            top = nl.ndarray((kv, 2 * C), dtype=dtype, buffer=nl.sbuf)
            bot = nl.ndarray((kv, 2 * C), dtype=dtype, buffer=nl.sbuf)
            nisa.memset(top, 0.0)
            nisa.memset(bot, 0.0)
            nisa.dma_copy(dst=top, src=img.ap(pattern=[[C, kv], [1, 2 * C]], offset=0,
                                              vector_offset=idx_t, indirect_dim=0),
                          oob_mode=oob_mode.skip)
            nisa.dma_copy(dst=bot, src=img.ap(pattern=[[C, kv], [1, 2 * C]], offset=0,
                                              vector_offset=idx_bot, indirect_dim=0),
                          oob_mode=oob_mode.skip)
            c_tl = top[:, 0:C]
            c_tr = top[:, C:2 * C]
            c_bl = bot[:, 0:C]
            c_br = bot[:, C:2 * C]
        else:
            # Single 2x2 gather: free pattern [[W*C, 2], [1, 2C]] -> 4C per pixel
            blk = nl.ndarray((kv, 2, 2 * C), dtype=dtype, buffer=nl.sbuf)
            nisa.memset(blk, 0.0)
            nisa.dma_copy(dst=blk, src=img.ap(
                pattern=[[C, kv], [row_w * C, 2], [1, 2 * C]], offset=0,
                vector_offset=idx_t, indirect_dim=0), oob_mode=oob_mode.skip)
            c_tl = blk[:, 0, 0:C]
            c_tr = blk[:, 0, C:2 * C]
            c_bl = blk[:, 1, 0:C]
            c_br = blk[:, 1, C:2 * C]

        nisa.tensor_scalar(dst=acc, data=c_tl, op0=nl.multiply, operand0=w_tl_t)
        nisa.tensor_scalar(dst=tmp, data=c_tr, op0=nl.multiply, operand0=w_tr_t)
        nisa.tensor_tensor(dst=acc, data1=acc, data2=tmp, op=nl.add)
        nisa.tensor_scalar(dst=tmp, data=c_bl, op0=nl.multiply, operand0=w_bl_t)
        nisa.tensor_tensor(dst=acc, data1=acc, data2=tmp, op=nl.add)
        nisa.tensor_scalar(dst=tmp, data=c_br, op0=nl.multiply, operand0=w_br_t)
        nisa.tensor_tensor(dst=acc, data1=acc, data2=tmp, op=nl.add)

        out_tile = nl.copy(acc, dtype=dtype)
        nisa.dma_copy(dst=out[nl.ds(k_off, kv), :], src=out_tile)

    return out


def _emit_tile(img, idx_tl, w_tl, w_tr, w_bl, w_br, out, k_off, kv, C, row_w):
    """Emit one tile's gather+blend+write (trace-time helper). Always 4->2 path."""
    idx_t = nl.ndarray((kv, 1), dtype=idx_tl.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=idx_t, src=idx_tl.ap(pattern=[[1, kv], [1, 1]], offset=k_off))
    idx_bot = nl.ndarray((kv, 1), dtype=idx_tl.dtype, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=idx_bot, data=idx_t, op0=nl.add, operand0=row_w)

    w_tl_t = nl.ndarray((kv, 1), dtype=nl.float32, buffer=nl.sbuf)
    w_tr_t = nl.ndarray((kv, 1), dtype=nl.float32, buffer=nl.sbuf)
    w_bl_t = nl.ndarray((kv, 1), dtype=nl.float32, buffer=nl.sbuf)
    w_br_t = nl.ndarray((kv, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=w_tl_t, src=w_tl.ap(pattern=[[1, kv], [1, 1]], offset=k_off))
    nisa.dma_copy(dst=w_tr_t, src=w_tr.ap(pattern=[[1, kv], [1, 1]], offset=k_off))
    nisa.dma_copy(dst=w_bl_t, src=w_bl.ap(pattern=[[1, kv], [1, 1]], offset=k_off))
    nisa.dma_copy(dst=w_br_t, src=w_br.ap(pattern=[[1, kv], [1, 1]], offset=k_off))

    top = nl.ndarray((kv, 2 * C), dtype=img.dtype, buffer=nl.sbuf)
    bot = nl.ndarray((kv, 2 * C), dtype=img.dtype, buffer=nl.sbuf)
    nisa.memset(top, 0.0)
    nisa.memset(bot, 0.0)
    nisa.dma_copy(dst=top, src=img.ap(pattern=[[C, kv], [1, 2 * C]], offset=0,
                                      vector_offset=idx_t, indirect_dim=0),
                  oob_mode=oob_mode.skip)
    nisa.dma_copy(dst=bot, src=img.ap(pattern=[[C, kv], [1, 2 * C]], offset=0,
                                      vector_offset=idx_bot, indirect_dim=0),
                  oob_mode=oob_mode.skip)
    acc = nl.ndarray((kv, C), dtype=nl.float32, buffer=nl.sbuf)
    tmp = nl.ndarray((kv, C), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=acc, data=top[:, 0:C], op0=nl.multiply, operand0=w_tl_t)
    nisa.tensor_scalar(dst=tmp, data=top[:, C:2 * C], op0=nl.multiply, operand0=w_tr_t)
    nisa.tensor_tensor(dst=acc, data1=acc, data2=tmp, op=nl.add)
    nisa.tensor_scalar(dst=tmp, data=bot[:, 0:C], op0=nl.multiply, operand0=w_bl_t)
    nisa.tensor_tensor(dst=acc, data1=acc, data2=tmp, op=nl.add)
    nisa.tensor_scalar(dst=tmp, data=bot[:, C:2 * C], op0=nl.multiply, operand0=w_br_t)
    nisa.tensor_tensor(dst=acc, data1=acc, data2=tmp, op=nl.add)
    out_tile = nl.copy(acc, dtype=img.dtype)
    nisa.dma_copy(dst=out[nl.ds(k_off, kv), :], src=out_tile)


@nki.jit
def bilinear_2x2_gather_blend_pipe(img, idx_tl, w_tl, w_tr, w_bl, w_br, row_w, unroll):
    """U-tile-unrolled variant: emits `unroll` independent tiles per affine_range
    iteration, exposing ILP so the scheduler can overlap tile B's gather (GpSimd)
    with tile A's blend (Vector). Requires K % (P*unroll) == 0 (else falls back)."""
    K = idx_tl.shape[0]
    C = img.shape[1]
    dtype = img.dtype
    P = nl.tile_size.pmax
    num_k_tiles = (K + P - 1) // P
    out = nl.ndarray((K, C), dtype=dtype, buffer=nl.shared_hbm)

    U = unroll if (K % P == 0 and num_k_tiles % unroll == 0) else 1
    num_iters = num_k_tiles // U
    for it in nl.affine_range(num_iters):
        for u in range(U):  # trace-time unroll -> U independent tiles in the body
            kt = it * U + u
            k_off = kt * P
            _emit_tile(img, idx_tl, w_tl, w_tr, w_bl, w_br, out, k_off, P, C, row_w)
    return out


_warp_fn_v3 = None
_warp_fn_v3_pipe = None
_V3_PIPELINE = os.environ.get("V3_PIPELINE", "0") == "1"
_V3_UNROLL = int(os.environ.get("V3_UNROLL", "2"))


def dma_warp_v3(tenInput, tenFlow, grid_y, grid_x):
    """Drop-in bilinear warp v3: single 2x2 gather, no pad, edge weights = 0.

    tenInput: (1,C,H,W), tenFlow: (1,2,H,W), grid_y/grid_x: base grids.
    V3_PIPELINE=1 -> use the U-tile-unrolled (option 3) kernel.
    """
    import torch
    from torch_neuronx.nki_hop import wrap_nki

    global _warp_fn_v3, _warp_fn_v3_pipe
    if _warp_fn_v3 is None:
        _warp_fn_v3 = wrap_nki(bilinear_2x2_gather_blend)
    if _V3_PIPELINE and _warp_fn_v3_pipe is None:
        _warp_fn_v3_pipe = wrap_nki(bilinear_2x2_gather_blend_pipe)

    B, C, H, W = tenInput.shape
    N = H * W
    NB = B * N          # batched: the B images stack along the flat row axis

    fx = grid_x.to(torch.float32) + tenFlow[:, 0:1].to(torch.float32)
    fy = grid_y.to(torch.float32) + tenFlow[:, 1:2].to(torch.float32)
    sx = fx.clamp(0.0, W - 1.0)
    sy = fy.clamp(0.0, H - 1.0)
    x0 = sx.floor()
    y0 = sy.floor()
    wx = (sx - x0).reshape(NB, 1)
    wy = (sy - y0).reshape(NB, 1)

    # Single top-left flat index into [B*N, C] (no pad; +1/+W are kernel strides).
    # For B>1 each image's indices are shifted by b*N so ONE gather covers the whole batch
    # (no per-image kernel dispatch). A read that runs past an image's last row carries a
    # bilinear weight of exactly 0 -- the same property this kernel already relies on at
    # image edges -- so batch boundaries need no special handling.
    idx_flat = (y0 * W + x0).reshape(B, N)
    if B > 1:
        off = (torch.arange(B, device=idx_flat.device, dtype=idx_flat.dtype) * N).view(B, 1)
        idx_flat = idx_flat + off
    idx_tl = idx_flat.reshape(NB, 1).to(torch.int32).contiguous().view(torch.uint32)

    w_tl = ((1 - wx) * (1 - wy)).contiguous()
    w_tr = (wx * (1 - wy)).contiguous()
    w_bl = ((1 - wx) * wy).contiguous()
    w_br = (wx * wy).contiguous()

    img_nc = tenInput.reshape(B, C, N).permute(0, 2, 1).reshape(NB, C).contiguous()
    if _V3_PIPELINE:
        out_nc = _warp_fn_v3_pipe(img_nc, idx_tl, w_tl, w_tr, w_bl, w_br, W, _V3_UNROLL)
    else:
        out_nc = _warp_fn_v3(img_nc, idx_tl, w_tl, w_tr, w_bl, w_br, W)
    return out_nc.reshape(B, N, C).permute(0, 2, 1).reshape(B, C, H, W)
