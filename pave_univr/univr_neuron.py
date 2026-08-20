"""UniVR-RIFE optimized for inference on AWS Trainium2/Inferentia2.

Mirrors the VGGT Neuron port methodology
(`PAVEFasterGS/_vggt_neuron_clone/vggt/models/vggt_neuron.py`): keep the stock
model intact, add a thin Neuron wrapper that (1) precomputes all static grids so
the runtime graphs are static-shape, and (2) compiles the conv-heavy sub-graphs
with `torch.compile(backend="neuron", fullgraph=True)`.

The one op the Neuron `torch.compile` backend will NOT lower is `F.grid_sample`
(the RIFE `warp`) — confirmed on trn2 by `_neuron_warp_probe.py`:
`failed to legalize operation 'torch.operator'`. The probe also confirmed that a
hand-rolled bilinear-gather sampler DOES compile in fullgraph and matches
grid_sample to cos_sim 0.999999. So this module replaces `warp` with
`warp_neuron` (design §3 path b) and monkey-patches it into the stock
`IFNet_m` / `refine` modules — exactly how the inductor backend swaps in
`warp_compilable`.

This module must remain importable on a plain CPU/torch install (no
`torch_neuronx`) so the correctness harness can build the eager reference side;
`torch.compile(backend="neuron")` is only invoked from `compile_ifnet_neuron()`.
"""
from __future__ import annotations

import logging
import sys
from typing import Dict, List, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def _floor_nonneg(x):
    """floor() for x >= 0 via int truncation. Avoids torch.floor (a Neuron
    CPU_FALLBACK op in eager) so the warp/resize stay fully on-device in eager
    mode. Valid only where x has been clamped to >= 0 first (both call sites do)."""
    return x.to(torch.int32).to(x.dtype)


# ---------------------------------------------------------------------------
# Manual bilinear-gather warp (replaces F.grid_sample / RIFE `warp`)
# ---------------------------------------------------------------------------
# Registry of precomputed identity base grids keyed by (device, dtype, H, W).
# Populated by `precompute_warp_grids()` BEFORE compilation so the compiled
# graphs read a constant tensor (no linspace/meshgrid in the traced graph).
_BASE_GRIDS: Dict[Tuple[str, torch.dtype, int, int], torch.Tensor] = {}


def _grid_key(device, dtype, H: int, W: int):
    return (str(device), dtype, H, W)


def make_base_grid(H: int, W: int, device, dtype) -> torch.Tensor:
    """Normalized identity sampling grid, shape [1, 2, H, W] (channel 0 = x).

    Matches stock `warplayer.warp`: x = linspace(-1, 1, W), y = linspace(-1, 1, H),
    used with grid_sample(align_corners=True).
    """
    xs = torch.linspace(-1.0, 1.0, W, device=device, dtype=dtype)
    ys = torch.linspace(-1.0, 1.0, H, device=device, dtype=dtype)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([gx, gy], dim=0).unsqueeze(0).contiguous()


def precompute_warp_grids(input_hw: Tuple[int, int], device, dtype,
                          n_levels: int = 5) -> List[Tuple[int, int]]:
    """Precompute identity base grids for the resolution levels the RIFE forward
    actually warps at: full (IFNet main-loop warps) and /2, /4, /8, /16
    (Contextnet pyramid warps). Returns the list of (H, W) levels registered.
    """
    H, W = input_hw
    levels = []
    for i in range(n_levels):
        h, w = H >> i, W >> i
        if h < 1 or w < 1:
            break
        key = _grid_key(device, dtype, h, w)
        if key not in _BASE_GRIDS:
            _BASE_GRIDS[key] = make_base_grid(h, w, device, dtype)
        levels.append((h, w))
    logger.info("Precomputed warp base grids for levels: %s (device=%s dtype=%s)",
                levels, device, dtype)
    return levels


def warp_neuron(tenInput: torch.Tensor, tenFlow: torch.Tensor) -> torch.Tensor:
    """Compile-friendly bilinear backward-warp — drop-in for RIFE `warp`.

    Equivalent to
        F.grid_sample(tenInput, base+flow, mode='bilinear',
                      padding_mode='border', align_corners=True)
    but expressed as floor/frac + gather + 4-tap weighted sum, which the Neuron
    `torch.compile` backend lowers (grid_sample does not). Validated on trn2 to
    match grid_sample at cos_sim 0.999999.

    tenInput: [N, C, H, W]
    tenFlow:  [N, 2, H, W]  (pixel units; channel 0 = dx, channel 1 = dy)
    """
    N, C, H, W = tenInput.shape
    key = _grid_key(tenInput.device, tenInput.dtype, H, W)
    base_grid = _BASE_GRIDS.get(key)
    if base_grid is None:
        # Eager fallback: build on the fly (kept off the compiled path because
        # precompute_warp_grids() pre-populates every level used at compile time).
        base_grid = make_base_grid(H, W, tenInput.device, tenInput.dtype)
        _BASE_GRIDS[key] = base_grid

    # Normalize flow to [-1, 1] grid units (stock warplayer convention).
    # Coordinate/index math is done in fp32 regardless of data dtype: bf16's
    # 8-bit mantissa cannot represent integer coords/indices up to H*W, which
    # would corrupt the gather. Data/gather/blend stay in the input dtype.
    f32 = torch.float32
    fl = torch.cat([tenFlow[:, 0:1].to(f32) / ((W - 1.0) / 2.0),
                    tenFlow[:, 1:2].to(f32) / ((H - 1.0) / 2.0)], 1)
    g = base_grid.to(f32) + fl  # [N, 2, H, W] fp32

    # De-normalize to pixel coordinates (align_corners=True), fp32.
    gx = (g[:, 0] + 1.0) * 0.5 * (W - 1.0)  # [N, H, W]
    gy = (g[:, 1] + 1.0) * 0.5 * (H - 1.0)

    # Clamp source coords to the valid range BEFORE flooring (padding_mode=
    # 'border'). This lets us use int-truncation for floor (floor==trunc for
    # x>=0), which avoids torch.floor — a Neuron CPU_FALLBACK op — so the warp
    # stays fully on-device in eager mode too. Matches grid_sample border.
    gx = gx.clamp(0.0, W - 1.0)
    gy = gy.clamp(0.0, H - 1.0)
    x0 = _floor_nonneg(gx)
    y0 = _floor_nonneg(gy)
    x1 = x0 + 1.0
    y1 = y0 + 1.0
    wx1 = gx - x0
    wy1 = gy - y0
    wx0 = 1.0 - wx1
    wy0 = 1.0 - wy1

    # indices clamp (border).
    x0c = x0.clamp(0, W - 1).long()
    x1c = x1.clamp(0, W - 1).long()
    y0c = y0.clamp(0, H - 1).long()
    y1c = y1.clamp(0, H - 1).long()

    img_flat = tenInput.reshape(N, C, H * W)

    def gather(yc, xc):
        idx = (yc * W + xc).reshape(N, 1, H * W).expand(N, C, H * W)
        return torch.gather(img_flat, 2, idx).reshape(N, C, H, W)

    Ia = gather(y0c, x0c)
    Ib = gather(y1c, x0c)
    Ic = gather(y0c, x1c)
    Id = gather(y1c, x1c)
    # weights computed in fp32, cast to data dtype for the blend (keeps the
    # output in the input dtype so it feeds the next conv without a type clash).
    dt = tenInput.dtype
    wa = (wx0 * wy0).unsqueeze(1).to(dt)
    wb = (wx0 * wy1).unsqueeze(1).to(dt)
    wc = (wx1 * wy0).unsqueeze(1).to(dt)
    wd = (wx1 * wy1).unsqueeze(1).to(dt)
    return Ia * wa + Ib * wb + Ic * wc + Id * wd


def patch_warp_for_neuron(use_nki: bool = False) -> List[str]:
    """Monkey-patch `warp` -> warp implementation in the loaded stock RIFE modules.

    The stock `IFNet_m` and `refine` modules do `from warplayer import warp`, so
    the name `warp` lives in each module's global namespace. We rebind both,
    mirroring `UnrollingProvider._compile_model`'s warp->warp_compilable swap.

    Args:
        use_nki: if True, install the DMA-DGE NKI gather kernel
            (`resample_nki.warp_resample_nki`) — the 4-tap gather runs on the
            hardware DMA engine instead of GpSimd (the profiled 89% bottleneck).
            If False (default), install the torch.gather `warp_neuron` baseline.

    Returns the list of modules patched (must be IFNet_m and refine).
    """
    if use_nki:
        try:
            from .resample_nki import warp_resample_nki as warp_impl
        except ImportError:
            # imported as a top-level module (sys.path insert), not a package
            from resample_nki import warp_resample_nki as warp_impl
        impl_name = "warp_resample_nki (DMA-DGE)"
    else:
        warp_impl = warp_neuron
        impl_name = "warp_neuron (torch.gather)"
    patched = []
    for mod_name in ("IFNet_m", "refine"):
        mod = sys.modules.get(mod_name)
        if mod is not None and hasattr(mod, "warp"):
            mod.warp = warp_impl
            patched.append(mod_name)
    if len(patched) != 2:
        raise RuntimeError(
            f"Expected to patch warp in IFNet_m and refine, only found: {patched}. "
            "Ensure setup_univr_paths() ran and the model was imported first."
        )
    logger.info("Patched warp -> %s in modules: %s", impl_name, patched)
    return patched


# ---------------------------------------------------------------------------
# UniVRNeuron wrapper
# ---------------------------------------------------------------------------

class UniVRNeuron(nn.Module):
    """Neuron-optimized wrapper around a stock `IFNet_m` (the RIFE backbone).

    Holds the stock `IFNet_m` (with loaded weights), precomputes the per-row
    time-offset map and the warp base grids as buffers, and exposes a forward
    matching `network_UniVR.UniVR.forward`'s contract: returns (merged, flow,
    mask) where `merged[2]` is the final corrected frame.

    Construction order matters (VGGT lesson): load weights into `ifnet` BEFORE
    wrapping/compiling so compiled `_orig_mod.` key prefixes never clash.
    """

    def __init__(self, ifnet: nn.Module, input_hw: Tuple[int, int],
                 gamma: float, dtype: torch.dtype = torch.bfloat16,
                 device: str = "neuron"):
        super().__init__()
        self.UVR = ifnet
        self.gamma = gamma
        self.dtype = dtype
        self._device = device
        H, W = input_hw
        assert H % 16 == 0 and W % 16 == 0, \
            f"input_hw must be multiples of 16, got {input_hw}"
        self.input_hw = (H, W)

        # Static per-row fraction map: grid_rows / H (was generate_2D_grid + /H).
        # Shape [1, 1, H, W], constant across columns. The runtime (t, gamma)
        # combine is a cheap elementwise op, keeping the graph static-shape.
        rows = torch.arange(0, H, dtype=torch.float32).view(1, 1, H, 1).expand(1, 1, H, W)
        row_frac = (rows / float(H)).contiguous()
        self.register_buffer("row_frac", row_frac, persistent=False)

    def _time_offset(self, im_rs: torch.Tensor, t: float, gamma: float) -> torch.Tensor:
        """Replicate network_UniVR.GS_temporal_offset:
        tau = t + gamma - gamma * grid_rows/H + 0.0001, broadcast to [N,1,H,W].
        """
        row_frac = self.row_frac.to(device=im_rs.device, dtype=im_rs.dtype)
        # Match stock op order exactly: t + gamma - gamma*row_frac + 0.0001.
        tau = t + gamma - gamma * row_frac + 0.0001
        # Match stock's (img[:, :1]*0 + 1) * tau batch/device/dtype broadcast.
        ones = im_rs[:, :1] * 0 + 1
        return ones * tau

    def forward(self, im_rs: torch.Tensor, t: float = None, gamma: float = None):
        """Run the RIFE forward.

        im_rs: [N, 6, H, W] (two stacked RS frames) in [0, 1].
        Returns (merged, flow, mask) like network_UniVR.UniVR.forward.
        """
        if gamma is None:
            gamma = self.gamma
        time_offset = self._time_offset(im_rs, t, gamma)
        # scale_list copied fresh: UniVR.forward mutates it in place.
        flow, mask, merged = self.UVR.forward(im_rs, time_offset, [4, 2, 1])
        return merged, flow, mask


# ---------------------------------------------------------------------------
# Compile-safe PReLU (the neuron backend will NOT lower nn.PReLU)
# ---------------------------------------------------------------------------
# Confirmed on trn2: compiling an IFBlock fails with
#   failed to legalize operation 'torch.operator' ... PReLU
# PReLU is pervasive (every conv() in IFNet_m/refine wraps Conv2d + PReLU), so
# we replace it with the exact algebraic equivalent expressed in relu/mul, which
# the StableHLO lowering accepts:
#   prelu(x) = max(0, x) + w * min(0, x) = relu(x) - w * relu(-x)

class NeuronPReLU(nn.Module):
    """Drop-in for nn.PReLU using only relu/mul so it lowers in a neuron NEFF.

    Reuses the stock PReLU's learned per-channel `weight` (no value change).
    """

    def __init__(self, prelu: nn.PReLU):
        super().__init__()
        # keep the original learned slope as a parameter (shape [C] or [1])
        self.weight = nn.Parameter(prelu.weight.detach().clone())

    def forward(self, x):
        if x.dim() == 4:
            w = self.weight.view(1, -1, 1, 1)
        elif x.dim() == 2:
            w = self.weight.view(1, -1)
        else:
            w = self.weight
        return torch.relu(x) - w * torch.relu(-x)


def patch_prelu_for_neuron(module: nn.Module) -> int:
    """Recursively replace every nn.PReLU in `module` with NeuronPReLU.

    Returns the count replaced. Must run AFTER weights are loaded (it copies the
    learned slope) and BEFORE compilation.
    """
    n = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.PReLU):
            setattr(module, name, NeuronPReLU(child))
            n += 1
        else:
            n += patch_prelu_for_neuron(child)
    return n


# ---------------------------------------------------------------------------
# Compile-safe F.interpolate (scale_factor -> explicit size)
# ---------------------------------------------------------------------------
# Confirmed on trn2: compiling an IFBlock fails at
#   F.interpolate(tmp, scale_factor=scale*2, mode="bilinear")  (IFNet_m.py:48)
# with "upsample_bilinear2d() received an invalid combination of arguments
# (FakeTensor, NoneType, bool, list of SymInt)". The neuron backend lowers
# bilinear upsample only with an explicit integer output_size, not a
# scale_factor. Our input resolution is fixed, so every output size is static;
# this shim rewrites scale_factor calls to size= calls (numerically identical:
# PyTorch derives size = floor(in*scale_factor), exact for our /2,/4 ratios).
import torch.nn.functional as _F  # noqa: E402

_ORIG_INTERPOLATE = _F.interpolate


def _bilinear_resize(x, out_h, out_w, align_corners=False):
    """Manual bilinear resize using index_select + elementwise (no upsample op).

    The Neuron backend's bilinear `upsample`/`interpolate` lowering is incorrect
    under fullgraph compile (bisect: cos_sim ~0.4-0.5 vs CPU), so we express the
    resize with supported ops only — mirroring `warp_neuron`. Matches
    torch.nn.functional.interpolate(mode='bilinear') for align_corners False/True.
    """
    N, C, H, W = x.shape
    dev = x.device
    # Coordinate/index math MUST be fp32 even when x is bf16: bf16 has an 8-bit
    # mantissa, so integers above 256 are not exactly representable and a 384- or
    # 1024-wide axis silently samples the wrong columns. MEASURED: doing this in
    # bf16 costs cos 0.9335 / PSNR 14.5 dB end-to-end (the same failure mode the
    # warp already guards against). Only the final blend weights are cast back.
    dt = torch.float32
    oy = torch.arange(out_h, device=dev, dtype=dt)
    ox = torch.arange(out_w, device=dev, dtype=dt)
    if align_corners and out_h > 1:
        sy = oy * ((H - 1.0) / (out_h - 1.0))
    else:
        sy = (oy + 0.5) * (H / out_h) - 0.5
    if align_corners and out_w > 1:
        sx = ox * ((W - 1.0) / (out_w - 1.0))
    else:
        sx = (ox + 0.5) * (W / out_w) - 0.5
    # border behavior: clamp source coords to valid range (matches torch edges)
    sy = sy.clamp(0.0, H - 1.0)
    sx = sx.clamp(0.0, W - 1.0)
    # MUST be torch.floor, NOT _floor_nonneg: inside a compiled graph the neuron
    # backend folds the int32 round-trip `x.to(int32).to(float32)` away as if it were
    # an identity, so `s - floor(s)` becomes 0, every interpolation weight collapses to
    # zero and the bilinear resize silently degenerates to NEAREST. MEASURED on trn2
    # (probe_resize_device.py): cos 0.500 for downsamples -- exactly the correlation of
    # a 4-tap mean with a single tap on noise -- and 0.38-0.40 for upsamples, in both
    # bf16 and fp32. `index_select` and the broadcast multiply are each individually
    # correct on device (probe_upsample_device.py §A), which is what isolated the floor.
    # torch.floor lowers fine in a compiled graph; _floor_nonneg exists only to keep the
    # EAGER path off a CPU_FALLBACK op, so it stays available for that use.
    y0 = torch.floor(sy)
    x0 = torch.floor(sx)
    wy = (sy - y0).view(1, 1, out_h, 1).to(x.dtype)
    wx = (sx - x0).view(1, 1, 1, out_w).to(x.dtype)
    y0i = y0.long().clamp(0, H - 1)
    y1i = (y0 + 1).long().clamp(0, H - 1)
    x0i = x0.long().clamp(0, W - 1)
    x1i = (x0 + 1).long().clamp(0, W - 1)

    top = x.index_select(2, y0i)          # [N, C, out_h, W]
    bot = x.index_select(2, y1i)
    tl = top.index_select(3, x0i)         # [N, C, out_h, out_w]
    tr = top.index_select(3, x1i)
    bl = bot.index_select(3, x0i)
    br = bot.index_select(3, x1i)
    top_row = tl * (1.0 - wx) + tr * wx
    bot_row = bl * (1.0 - wx) + br * wx
    return top_row * (1.0 - wy) + bot_row * wy


def _interp_size(input, size=None, scale_factor=None, mode="nearest",
                 align_corners=None, recompute_scale_factor=None, antialias=False):
    """Replacement for F.interpolate: bilinear resizes use the manual
    index_select path (the neuron bilinear upsample lowering is wrong); other
    modes defer to the original op."""
    if size is None and scale_factor is not None:
        H, W = int(input.shape[-2]), int(input.shape[-1])
        if isinstance(scale_factor, (list, tuple)):
            sh, sw = float(scale_factor[0]), float(scale_factor[1])
        else:
            sh = sw = float(scale_factor)
        out_h, out_w = int(round(H * sh)), int(round(W * sw))
    elif size is not None:
        if isinstance(size, (list, tuple)):
            out_h, out_w = int(size[0]), int(size[1])
        else:
            out_h = out_w = int(size)
    else:
        return _ORIG_INTERPOLATE(input, size=size, scale_factor=scale_factor,
                                 mode=mode, align_corners=align_corners)
    if mode == "bilinear":
        return _bilinear_resize(input, out_h, out_w,
                                align_corners=bool(align_corners))
    return _ORIG_INTERPOLATE(input, size=(out_h, out_w), mode=mode,
                             align_corners=align_corners)


def patch_interpolate_for_neuron():
    """Globally rebind torch.nn.functional.interpolate to the size-based shim so
    the stock RIFE `F.interpolate(scale_factor=...)` calls lower on Neuron.
    Idempotent; `unpatch_interpolate` restores the original."""
    _F.interpolate = _interp_size


def unpatch_interpolate():
    _F.interpolate = _ORIG_INTERPOLATE


# ---------------------------------------------------------------------------
# Compilation (task 3)
# ---------------------------------------------------------------------------

def compile_ifnet_neuron(ifnet: nn.Module, compile_contextnet: bool = True) -> List[str]:
    """Compile the conv-heavy RIFE sub-graphs with the Neuron backend.

    Mirrors VGGT's `_compile_blocks`: each sub-module becomes a fused NEFF.
    The `warp` calls in `IFNet_m.forward` sit on the seams BETWEEN the compiled
    IFBlocks (so the blocks compile regardless of warp); the warps INSIDE
    `Contextnet` only compile because `warp` is `warp_neuron` (the bilinear-
    gather sampler the probe proved lowers).

    Requirements:
      - `patch_warp_for_neuron()` already ran (so Contextnet uses warp_neuron).
      - `precompute_warp_grids()` already populated every level used, so the
        compiled graphs read constant base grids (no linspace in the trace).

    Args:
        ifnet: stock `IFNet_m` instance (weights already loaded).
        compile_contextnet: compile Contextnet too. If the in-graph warp causes
            a compile failure on a given SDK, set False to keep Contextnet eager
            (block0/1/2 + unet are the heavy stages and still compile).

    Returns the list of compiled sub-module names.
    """
    # nn.PReLU does not lower in the neuron backend; swap to the relu/mul
    # equivalent before compiling (exact, reuses the learned slope).
    n_prelu = patch_prelu_for_neuron(ifnet)
    logger.info("Replaced %d PReLU -> NeuronPReLU before compile", n_prelu)

    # F.interpolate(scale_factor=...) does not lower; rebind to the size-based
    # shim so the IFBlock/Contextnet upsample/downsample calls compile.
    patch_interpolate_for_neuron()

    compiled = []

    for name in ("block0", "block1", "block2", "unet"):
        sub = getattr(ifnet, name, None)
        if sub is not None:
            setattr(ifnet, name, torch.compile(sub, backend="neuron",
                                               fullgraph=True, dynamic=False))
            compiled.append(name)

    if compile_contextnet and getattr(ifnet, "contextnet", None) is not None:
        ifnet.contextnet = torch.compile(
            ifnet.contextnet, backend="neuron", fullgraph=True, dynamic=False
        )
        compiled.append("contextnet")

    logger.info("Compiled RIFE sub-graphs (backend=neuron, fullgraph): %s", compiled)
    return compiled


def compile_ifnet_whole(ifnet: nn.Module):
    """Compile the ENTIRE IFNet_m.forward as one fullgraph NEFF.

    Unlike `compile_ifnet_neuron` (which compiles sub-modules and leaves the
    IFNet_m.forward glue — the 6 full-res warps, cat/sigmoid/clamp — running
    EAGER on the device, where CPU_FALLBACK ops like aten::floor force host
    round-trips), this places EVERY op on device in a single graph. fullgraph
    success is itself the proof that no op falls back to CPU.

    Returns the compiled module; the caller must rebind it, e.g.
        wrap.UVR = compile_ifnet_whole(wrap.UVR)

    Prereqs (same as the sub-module path): weights loaded, warp patched,
    grids precomputed. PReLU swap + interpolate shim are applied here.
    """
    patch_prelu_for_neuron(ifnet)
    patch_interpolate_for_neuron()
    compiled = torch.compile(ifnet, backend="neuron", fullgraph=True, dynamic=False)
    logger.info("Compiled WHOLE IFNet_m forward (backend=neuron, fullgraph)")
    return compiled
