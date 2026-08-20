"""DMA-DGE bilinear warp NKI kernel v2 — reduced descriptor count.

Direction 1 optimization: replicate-pad the source image so all 4 corners are
valid at idx_tl+{0,1,W_pad,W_pad+1}. Only idx_tl is passed from torch; the other
3 indices are computed in SBUF via tensor_scalar add (Vector engine, fast).
Only wx,wy are passed; the 4 bilinear weights are computed in-kernel.
This cuts HBM→SBUF index DMA from 4 to 1 and weight DMA from 4 to 2 per tile.

The 4 indirect gathers remain (hardware constraint: vector_offset = 1 row/partition),
but the reduced index/weight traffic + eliminated clamp ops lighten the surrounding
graph and reduce overall SWDGE descriptor setup overhead.
"""

import nki
import nki.isa as nisa
import nki.language as nl
from nki.isa.constants import oob_mode


@nki.jit
def bilinear_gather_blend_v2(img_padded, idx_tl, wx, wy, w_pad):
    """Bilinear gather+blend with padded source. Only idx_tl needed.

    img_padded: ((H+1)*(W+1), C) — replicate-padded source, row-major.
    idx_tl: (K, 1) uint32 — top-left flat index into padded image.
    wx: (K, 1) float32 — fractional x weight.
    wy: (K, 1) float32 — fractional y weight.
    w_pad: int — padded width (W+1), used to compute bl = tl + w_pad.
    """
    K = idx_tl.shape[0]
    C = img_padded.shape[1]
    dtype = img_padded.dtype

    P = nl.tile_size.pmax  # 128
    num_k_tiles = (K + P - 1) // P
    out = nl.ndarray((K, C), dtype=dtype, buffer=nl.shared_hbm)

    for kt in nl.affine_range(num_k_tiles):
        k_off = kt * P
        k_valid = min(P, K - kt * P)

        # Load idx_tl tile from HBM (only 1 index tensor needed)
        idx_tl_t = nl.ndarray((k_valid, 1), dtype=idx_tl.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=idx_tl_t,
                      src=idx_tl.ap(pattern=[[1, k_valid], [1, 1]], offset=k_off))

        # Compute idx_tr, idx_bl, idx_br from idx_tl in SBUF (Vector engine, fast)
        one_u32 = nl.ndarray((k_valid, 1), dtype=nl.uint32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=one_u32, data=idx_tl_t, op0=nl.add, operand0=1)
        idx_tr_t = one_u32  # tl + 1

        wpad_u32 = nl.ndarray((k_valid, 1), dtype=nl.uint32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=wpad_u32, data=idx_tl_t, op0=nl.add, operand0=w_pad)
        idx_bl_t = wpad_u32  # tl + W_pad

        wpad1_u32 = nl.ndarray((k_valid, 1), dtype=nl.uint32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=wpad1_u32, data=idx_tl_t, op0=nl.add,
                           operand0=w_pad + 1)
        idx_br_t = wpad1_u32  # tl + W_pad + 1

        # Load wx, wy tiles (only 2 weight tensors, not 4)
        wx_t = nl.ndarray((k_valid, 1), dtype=nl.float32, buffer=nl.sbuf)
        wy_t = nl.ndarray((k_valid, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=wx_t,
                      src=wx.ap(pattern=[[1, k_valid], [1, 1]], offset=k_off))
        nisa.dma_copy(dst=wy_t,
                      src=wy.ap(pattern=[[1, k_valid], [1, 1]], offset=k_off))

        # Compute bilinear weights in SBUF: w_tl=(1-wx)(1-wy), etc.
        one_f = nl.ndarray((k_valid, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=one_f, data=wx_t, op0=nl.multiply, operand0=0.0)
        nisa.tensor_scalar(dst=one_f, data=one_f, op0=nl.add, operand0=1.0)

        # 1-wx
        inv_wx = nl.ndarray((k_valid, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=inv_wx, data1=one_f, data2=wx_t, op=nl.subtract)
        # 1-wy
        inv_wy = nl.ndarray((k_valid, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=inv_wy, data1=one_f, data2=wy_t, op=nl.subtract)

        # w_tl = (1-wx)*(1-wy)
        w_tl_t = nl.ndarray((k_valid, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=w_tl_t, data1=inv_wx, data2=inv_wy, op=nl.multiply)
        # w_tr = wx*(1-wy)
        w_tr_t = nl.ndarray((k_valid, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=w_tr_t, data1=wx_t, data2=inv_wy, op=nl.multiply)
        # w_bl = (1-wx)*wy
        w_bl_t = nl.ndarray((k_valid, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=w_bl_t, data1=inv_wx, data2=wy_t, op=nl.multiply)
        # w_br = wx*wy
        w_br_t = nl.ndarray((k_valid, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=w_br_t, data1=wx_t, data2=wy_t, op=nl.multiply)

        # 4 indirect gathers (still required — 1 row per partition per call)
        c_tl = nl.ndarray((k_valid, C), dtype=dtype, buffer=nl.sbuf)
        c_tr = nl.ndarray((k_valid, C), dtype=dtype, buffer=nl.sbuf)
        c_bl = nl.ndarray((k_valid, C), dtype=dtype, buffer=nl.sbuf)
        c_br = nl.ndarray((k_valid, C), dtype=dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=c_tl, src=img_padded.ap(
            pattern=[[C, k_valid], [1, C]], offset=0,
            vector_offset=idx_tl_t, indirect_dim=0), oob_mode=oob_mode.error)
        nisa.dma_copy(dst=c_tr, src=img_padded.ap(
            pattern=[[C, k_valid], [1, C]], offset=0,
            vector_offset=idx_tr_t, indirect_dim=0), oob_mode=oob_mode.error)
        nisa.dma_copy(dst=c_bl, src=img_padded.ap(
            pattern=[[C, k_valid], [1, C]], offset=0,
            vector_offset=idx_bl_t, indirect_dim=0), oob_mode=oob_mode.error)
        nisa.dma_copy(dst=c_br, src=img_padded.ap(
            pattern=[[C, k_valid], [1, C]], offset=0,
            vector_offset=idx_br_t, indirect_dim=0), oob_mode=oob_mode.error)

        # Weighted blend in SBUF (Vector engine)
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


# ─── Torch caller (replicate-pad + simplified index/weight math) ───────────

_warp_fn_v2 = None


def dma_warp_v2(tenInput, tenFlow, grid_y, grid_x):
    """Drop-in bilinear warp v2: padded source, single idx_tl, in-kernel weight compute.

    tenInput: (1,C,H,W), tenFlow: (1,2,H,W), grid_y/grid_x: precomputed base grids.
    """
    import torch
    from torch_neuronx.nki_hop import wrap_nki

    global _warp_fn_v2
    if _warp_fn_v2 is None:
        _warp_fn_v2 = wrap_nki(bilinear_gather_blend_v2)

    B, C, H, W = tenInput.shape
    N = H * W
    W_pad = W + 1  # padded width

    # Replicate-pad the source image by 1 on right and bottom
    # img shape: (1,C,H,W) → pad to (1,C,H+1,W+1) then flatten to ((H+1)*(W+1), C)
    img_padded = torch.nn.functional.pad(tenInput, (0, 1, 0, 1), mode='replicate')
    # Flatten to (N_pad, C) row-major
    img_nc = img_padded.reshape(C, (H + 1) * W_pad).transpose(0, 1).contiguous()

    # Compute sampling coordinates (fp32)
    fx = grid_x.to(torch.float32) + tenFlow[:, 0:1].to(torch.float32)
    fy = grid_y.to(torch.float32) + tenFlow[:, 1:2].to(torch.float32)
    # Clamp to [0, W-1] / [0, H-1] so floor gives valid tl in [0,W-1]x[0,H-1]
    # With padding, tl+1 and tl+W_pad are always in-bounds (no further clamp needed)
    sx = fx.clamp(0.0, W - 1.0)
    sy = fy.clamp(0.0, H - 1.0)
    x0 = sx.floor()
    y0 = sy.floor()

    # Fractional weights
    wx = (sx - x0).reshape(N, 1).contiguous()
    wy = (sy - y0).reshape(N, 1).contiguous()

    # Flat index into padded image: y0 * W_pad + x0
    idx_tl = (y0 * W_pad + x0).reshape(N, 1).to(torch.int32).view(torch.uint32)

    out_nc = _warp_fn_v2(img_nc, idx_tl, wx, wy, W_pad)
    return out_nc.transpose(0, 1).reshape(1, C, H, W)
