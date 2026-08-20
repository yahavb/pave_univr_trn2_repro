"""DMA-DGE bilinear warp NKI kernel — gather runs on DMA engine, NOT GpSimd.

The 4-corner bilinear gather is done via explicit `nisa.dma_copy(src.ap(vector_offset=...))`
indirect DMA (the DMA-engine DGE path), and the bilinear blend in SBUF on the Vector engine.
Index/weight math stays in the torch graph (fp32, exact); this kernel does the 4 gathers +
weighted blend. Layout: source image (N=H*W, C), channels contiguous so each gather moves a
full C-vector per pixel. Validated cos 0.999999 vs grid_sample. (Ported from the mainline
benchmarks/neuron port; the proven 2-NEFF fast path uses this.)
"""

import nki
import nki.isa as nisa
import nki.language as nl
from nki.isa.constants import oob_mode


@nki.jit
def bilinear_gather_blend(img, idx_tl, idx_tr, idx_bl, idx_br, w_tl, w_tr, w_bl, w_br):
    """out[k,:] = sum_corner w_corner[k] * img[idx_corner[k], :]  (4-tap bilinear)."""
    K = idx_tl.shape[0]
    C = img.shape[1]
    dtype = img.dtype

    P = nl.tile_size.pmax  # 128
    num_k_tiles = (K + P - 1) // P
    out = nl.ndarray((K, C), dtype=dtype, buffer=nl.shared_hbm)

    for kt in nl.affine_range(num_k_tiles):
        k_off = kt * P
        k_valid = min(P, K - kt * P)

        idx_tl_t = nl.ndarray((k_valid, 1), dtype=idx_tl.dtype, buffer=nl.sbuf)
        idx_tr_t = nl.ndarray((k_valid, 1), dtype=idx_tr.dtype, buffer=nl.sbuf)
        idx_bl_t = nl.ndarray((k_valid, 1), dtype=idx_bl.dtype, buffer=nl.sbuf)
        idx_br_t = nl.ndarray((k_valid, 1), dtype=idx_br.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=idx_tl_t, src=idx_tl.ap(pattern=[[1, k_valid], [1, 1]], offset=k_off))
        nisa.dma_copy(dst=idx_tr_t, src=idx_tr.ap(pattern=[[1, k_valid], [1, 1]], offset=k_off))
        nisa.dma_copy(dst=idx_bl_t, src=idx_bl.ap(pattern=[[1, k_valid], [1, 1]], offset=k_off))
        nisa.dma_copy(dst=idx_br_t, src=idx_br.ap(pattern=[[1, k_valid], [1, 1]], offset=k_off))

        w_tl_t = nl.ndarray((k_valid, 1), dtype=nl.float32, buffer=nl.sbuf)
        w_tr_t = nl.ndarray((k_valid, 1), dtype=nl.float32, buffer=nl.sbuf)
        w_bl_t = nl.ndarray((k_valid, 1), dtype=nl.float32, buffer=nl.sbuf)
        w_br_t = nl.ndarray((k_valid, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=w_tl_t, src=w_tl.ap(pattern=[[1, k_valid], [1, 1]], offset=k_off))
        nisa.dma_copy(dst=w_tr_t, src=w_tr.ap(pattern=[[1, k_valid], [1, 1]], offset=k_off))
        nisa.dma_copy(dst=w_bl_t, src=w_bl.ap(pattern=[[1, k_valid], [1, 1]], offset=k_off))
        nisa.dma_copy(dst=w_br_t, src=w_br.ap(pattern=[[1, k_valid], [1, 1]], offset=k_off))

        c_tl = nl.ndarray((k_valid, C), dtype=dtype, buffer=nl.sbuf)
        c_tr = nl.ndarray((k_valid, C), dtype=dtype, buffer=nl.sbuf)
        c_bl = nl.ndarray((k_valid, C), dtype=dtype, buffer=nl.sbuf)
        c_br = nl.ndarray((k_valid, C), dtype=dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=c_tl, src=img.ap(pattern=[[C, k_valid], [1, C]], offset=0,
                                           vector_offset=idx_tl_t, indirect_dim=0),
                      oob_mode=oob_mode.error)
        nisa.dma_copy(dst=c_tr, src=img.ap(pattern=[[C, k_valid], [1, C]], offset=0,
                                           vector_offset=idx_tr_t, indirect_dim=0),
                      oob_mode=oob_mode.error)
        nisa.dma_copy(dst=c_bl, src=img.ap(pattern=[[C, k_valid], [1, C]], offset=0,
                                           vector_offset=idx_bl_t, indirect_dim=0),
                      oob_mode=oob_mode.error)
        nisa.dma_copy(dst=c_br, src=img.ap(pattern=[[C, k_valid], [1, C]], offset=0,
                                           vector_offset=idx_br_t, indirect_dim=0),
                      oob_mode=oob_mode.error)

        acc = nl.ndarray((k_valid, C), dtype=nl.float32, buffer=nl.sbuf)
        tmp = nl.ndarray((k_valid, C), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=acc, data=c_tl, op0=nl.multiply, operand0=w_tl_t)
        nisa.tensor_scalar(dst=tmp, data=c_tr, op0=nl.multiply, operand0=w_tr_t)
        nisa.tensor_tensor(dst=acc, data1=acc, data2=tmp, op=nl.add)
        nisa.tensor_scalar(dst=tmp, data=c_bl, op0=nl.multiply, operand0=w_bl_t)
        nisa.tensor_tensor(dst=acc, data1=acc, data2=tmp, op=nl.add)
        nisa.tensor_scalar(dst=tmp, data=c_br, op0=nl.multiply, operand0=w_br_t)
        nisa.tensor_tensor(dst=acc, data1=acc, data2=tmp, op=nl.add)

        out_tile = nl.copy(acc, dtype=dtype)
        nisa.dma_copy(dst=out[nl.ds(k_off, k_valid), :], src=out_tile)

    return out


_warp_fn = None


def dma_warp(tenInput, tenFlow, grid_y, grid_x):
    """Drop-in warp: computes idx/weights in torch (fp32), gathers+blends via NKI DMA-DGE.
    tenInput (1,C,H,W), tenFlow (1,2,H,W), grid_y/grid_x precomputed base grids."""
    import torch
    from torch_neuronx.nki_hop import wrap_nki

    global _warp_fn
    if _warp_fn is None:
        _warp_fn = wrap_nki(bilinear_gather_blend)

    B, C, H, W = tenInput.shape
    N = H * W
    fx = grid_x.to(torch.float32) + tenFlow[:, 0:1].to(torch.float32)
    fy = grid_y.to(torch.float32) + tenFlow[:, 1:2].to(torch.float32)
    sx = fx.clamp(0.0, W - 1.0)
    sy = fy.clamp(0.0, H - 1.0)
    x0 = sx.floor(); y0 = sy.floor()
    x1 = (x0 + 1).clamp(max=W - 1)
    y1 = (y0 + 1).clamp(max=H - 1)
    wx = sx - x0
    wy = sy - y0

    def flat(yc, xc):
        return (yc * W + xc).reshape(N, 1).to(torch.int32).view(torch.uint32)

    idx_tl = flat(y0, x0); idx_tr = flat(y0, x1)
    idx_bl = flat(y1, x0); idx_br = flat(y1, x1)
    wxf = wx.reshape(N, 1); wyf = wy.reshape(N, 1)
    w_tl = ((1 - wxf) * (1 - wyf)).contiguous()
    w_tr = (wxf * (1 - wyf)).contiguous()
    w_bl = ((1 - wxf) * wyf).contiguous()
    w_br = (wxf * wyf).contiguous()

    img_nc = tenInput.reshape(C, N).transpose(0, 1).contiguous()
    out_nc = _warp_fn(img_nc, idx_tl, idx_tr, idx_bl, idx_br, w_tl, w_tr, w_bl, w_br)
    return out_nc.transpose(0, 1).reshape(1, C, H, W)
