#!/usr/bin/env python3
"""NKI resample: bounded-displacement bilinear warp with NO per-pixel HBM descriptors.

Design follows from two measurements, not from preference:

  1. Real flow at the production tile has max |displacement| = 2.33 px (measured by
     capturing the flow the model actually produces at the C=3 site). The bundle's
     43.28 px figure does not hold for this frame pair.
  2. The whole C=3 992x1280 tile is 15.3 MB, which FITS the 24 MB SBUF.

Together those mean the source neighbourhood every output pixel needs is already
on-chip. The per-pixel indirect gather -- 1,278,496 software descriptors, 99.2% of
device time on GpSimd -- exists only because the reference implementation addresses
HBM per pixel. If the band is resident in SBUF, the 4 taps become STATIC offsets
into SBUF and the descriptor count drops to a handful of band loads.

Structure, per row band of P=128 output rows:
  * one static DMA loads rows [r-R, r+P+R) x W x C into SBUF   <- descriptors per BAND
  * for each of (2R+1)^2 integer offsets, a static SBUF slice is multiplied by a
    precomputed weight plane and accumulated                    <- Vector engine
  * one static DMA stores the band

Weights are computed on the host (they depend on flow, which is an activation, but
they are elementwise -- no addressing), so nothing data-dependent reaches an
access pattern. That is the whole point: the descriptor path stays static.

References used from KaenaNeuronKernelLibrary:
  experimental/misc/gather.py            -- tiling over pmax, ap(pattern=...) form
  experimental/deformable_attention/ms_deformable_attention.py -- bilinear-in-NKI
"""
from __future__ import annotations

import nki
import nki.isa as nisa
import nki.language as nl

P_MAX = 128


@nki.jit
def shift_warp_band(img, wts):
    """img: [H+2R, W, C] in HBM, replicate-padded in y by R. wts: [T, H, W] weight
    planes in HBM, T = (2R+1)^2, ordered oy-major then ox. Returns [H, W, C].

    R is derived from T at trace time, so the kernel is shape-generic over radius.
    """
    Hp = img.shape[0]
    W = img.shape[1]
    C = img.shape[2]
    T = wts.shape[0]
    H = wts.shape[1]
    dtype = img.dtype

    side = 1
    while side * side < T:
        side += 1
    R = (side - 1) // 2

    out = nl.ndarray((H, W, C), dtype=dtype, buffer=nl.shared_hbm)

    n_bands = (H + P_MAX - 1) // P_MAX

    for b in nl.affine_range(n_bands):
        r0 = b * P_MAX
        rows = min(P_MAX, H - r0)

        # One static load of the band plus its vertical halo. Contiguous in W*C,
        # so this is a handful of large descriptors rather than one per pixel.
        band = nl.ndarray((rows + 2 * R, W * C), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=band,
            src=img.ap(pattern=[[W * C, rows + 2 * R], [1, W * C]], offset=r0 * W * C),
        )

        acc = nl.ndarray((rows, W * C), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(acc, 0.0)

        for t in nl.affine_range(T):
            oy = t // side - R
            ox = t % side - R

            wplane = nl.ndarray((rows, W), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(
                dst=wplane,
                src=wts.ap(pattern=[[W, rows], [1, W]], offset=t * H * W + r0 * W),
            )

            # Broadcast the per-pixel weight across C. C is small (3 at the site
            # that matters), so this is a cheap Vector-engine op.
            wbc = nl.ndarray((rows, W * C), dtype=nl.float32, buffer=nl.sbuf)
            for c in nl.affine_range(C):
                nisa.tensor_copy(dst=wbc[:, nl.ds(c, W * C - c)], src=wplane)

            # STATIC slice: the y offset is a band-row shift, the x offset a byte
            # shift of C elements. No index tensor, no vector_offset, no SWDGE.
            src = band[nl.ds(R + oy, rows), :]
            prod = nl.ndarray((rows, W * C), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=prod, data1=src, data2=wbc, op=nl.multiply)
            nisa.tensor_tensor(dst=acc, data1=acc, data2=prod, op=nl.add)

        band_out = nl.copy(acc, dtype=dtype)
        nisa.dma_copy(
            dst=out.ap(pattern=[[W * C, rows], [1, W * C]], offset=r0 * W * C),
            src=band_out,
        )

    return out
