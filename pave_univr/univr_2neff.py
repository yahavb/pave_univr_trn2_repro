"""UniVR-RIFE as 2 fused NEFFs with the warp INLINED in-graph — the fast path.

This is the production form of the optimization arc (see docs/NEURON_PORT_NOTES.md):
  284ms (sub-module) -> 128ms (--whole) -> 73.5ms (this 2-NEFF + DMA-DGE warp), 3.9x.

NEFF_A = IFBlocks + time_offset + merge ; NEFF_B = Contextnet + Unet + residual + clamp.
Every warp is inlined inside the compiled NEFF (zero eager dispatch — the lesson that the
eager-warp path regressed 8.7x). With PAVE_UNIVR_NKI_WARP=1 the 4-tap gather runs on the
DMA engine (off GpSimd) via nki_dma_warp.dma_warp; else an inline torch.gather warp.

`ModelUniVR2NEFF` exposes the same `set_input()` / `forward(t, gamma)` contract as the stock
`ModelUniVR`, so the provider can swap it in transparently.
"""
from __future__ import annotations

import logging
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch._dynamo

logger = logging.getLogger(__name__)
torch._dynamo.config.cache_size_limit = 128

_NKI_WARP_MODE = os.environ.get("PAVE_UNIVR_NKI_WARP", "3")  # 0=torch,1=v1,2=v2,3=v3(default,best)
_USE_DMA_WARP = _NKI_WARP_MODE in ("1", "2", "3")

# Resample mode for the IFBlock pyramid up/down samples.
#   "bilinear" (default) : stock RIFE semantics via the proven index_select resize
#   "fast"               : avg_pool2d down / nearest up -- FASTER but WRONG
# MEASURED: "fast" costs cos 0.9400 / PSNR 14.96 dB vs the stock CPU fp32 reference
# (identical for the NKI and torch.gather warps, and reproduced on CPU in fp32 with
# shared weights -> the resamples are the whole error). Restoring bilinear gives
# cos 0.999999 / PSNR 110 dB. See check_2neff_semantics_cpu.py.
_RESAMPLE = os.environ.get("PAVE_UNIVR_RESAMPLE", "bilinear").lower()

# Pad the Contextnet stem's input channels 3 -> 16. MEASURED at tile 864x1024:
# Conv2d(3->16,s2) = 13.03 ms vs the SAME conv widened to Conv2d(16->16,s2) = 1.72 ms
# (7.6x faster for 5.4x more FLOPs; Cin=8 is 26.9 ms, so the fast conv path is gated at
# 16 channels). Output is mathematically identical because the added weights are zero.
_PAD_CTX_CIN = os.environ.get("PAVE_UNIVR_PAD_CTX_CIN", "1") == "1"
_CTX_CIN = 16
# Padding the stem's input channels 3->16 inflates the stage's LARGEST tensor 5.3x
# (864x1024 fp32: 10.6 MB -> 56.6 MB, against ~28 MB of SBUF per NeuronCore-v3 at LNC=1).
# It was adopted because an ISOLATED Conv2d(3->16,s2) measured 7.6x slower than Conv2d(16->16,s2).
# MEASURED IN-GRAPH (fp32, --model-type unet-inference, device time from neuron-profile):
#   tile        pad ON            pad OFF           spill ON -> OFF
#   216x256      3.69 ms           2.67 ms          16.9 ->   9.4 MB
#   432x512      7.92 ms           3.94 ms          68.0 ->  37.8 MB
#   864x1024    31.71 ms          15.85 ms         272.8 -> 174.2 MB
# The stage is DMA/spill-bound (92-97% static DMA), so paying 5.3x the bytes on the largest
# tensor loses 1.4-2.0x at EVERY tile size -- the isolated-conv win does not survive in-graph.
# Default is therefore OFF. Kept switchable because the original win was seen in bf16, where
# the byte cost is halved; PAVE_UNIVR_PAD_MAX_MB is the max padded full-res input in MB.
_PAD_MAX_MB = float(os.environ.get("PAVE_UNIVR_PAD_MAX_MB", "0"))


def _pad_is_safe(H: int, W: int, itemsize: int = 4) -> bool:
    """Would a 16-channel full-res input still fit the padding budget?"""
    return (_CTX_CIN * H * W * itemsize) / 1e6 <= _PAD_MAX_MB

# Selective precision. bf16 storage is 15.9x faster than fp32 end-to-end but loses 87 dB
# (110.11 -> 23.31 dB). The loss is not spread evenly: `flow`, `mask` and `time_offset` are
# tiny tensors (4/1/1 channels) that are ACCUMULATED across the three pyramid stages and
# then used as sampling coordinates, so an 8-bit mantissa on them perturbs where the warp
# reads. Keeping just those in fp32 while images and convs stay bf16 costs almost no
# memory. `PAVE_UNIVR_FP32_FLOW` selects which of them to protect:
#   0     : none (pure bf16)
#   flow  : flow + mask
#   tof   : time_offset
#   all   : flow + mask + time_offset  (default when enabled)
_FP32_FLOW = os.environ.get("PAVE_UNIVR_FP32_FLOW", "0").lower()
_F32_FLOW = _FP32_FLOW in ("1", "flow", "all")
_F32_TOF = _FP32_FLOW in ("1", "tof", "all")

# Resolve the DMA warp function at import time (not inside _warp, which gets traced).
_dma_warp_fn = None
if _NKI_WARP_MODE == "1":
    try:
        from .nki_dma_warp import dma_warp as _dma_warp_fn
    except ImportError:
        from nki_dma_warp import dma_warp as _dma_warp_fn
elif _NKI_WARP_MODE == "2":
    try:
        from .nki_dma_warp_v2 import dma_warp_v2 as _dma_warp_fn
    except ImportError:
        from nki_dma_warp_v2 import dma_warp_v2 as _dma_warp_fn
elif _NKI_WARP_MODE == "3":
    try:
        from .nki_dma_warp_v3 import dma_warp_v3 as _dma_warp_fn
    except ImportError:
        from nki_dma_warp_v3 import dma_warp_v3 as _dma_warp_fn


# Reuse the ONE proven lowerable bilinear resize (univr_neuron._bilinear_resize,
# validated cos~1 vs CPU; the backend's own bilinear upsample lowering is wrong).
# univr_neuron is importable on plain torch, so this adds no device dependency.
try:
    from .univr_neuron import _bilinear_resize as _bilin
except ImportError:
    from univr_neuron import _bilinear_resize as _bilin


def _down(x, k: int):
    """Downsample by even integer factor k, EXACTLY matching stock RIFE
    F.interpolate(scale_factor=1/k, mode='bilinear', align_corners=False).

    For output index i the source coordinate is (i+0.5)*k - 0.5 = k*i + (k-1)/2, whose
    fractional part is exactly 0.5 for even k. So the result is the mean of source rows
    k*i + k/2 - 1 and k*i + k/2 -- two static strided slices and an add, no clamping
    needed. Note avg_pool2d(k) is only equivalent for k=2; for k=4 the correct answer
    averages the CENTRE two rows of each 4-row block, which is why the old
    `avg_pool2d(4)` shortcut was wrong.

    Why not the index_select resize: MEASURED on trn2 (probe_resize_device.py), every
    index-based resize is mis-lowered on device -- downsample cos ~0.50, upsample cos
    0.38-0.40 vs CPU-exact -- for both bf16 and fp32 and for both index_select and
    torch.gather. This static-slice form measures cos 1.000000 (fp32) / 0.999988+ (bf16)
    on device, and is all-static addressing so it never touches the DGE path.
    """
    if _RESAMPLE == "fast":
        return F.avg_pool2d(x, k)
    assert k % 2 == 0, f"even downsample factor required, got {k}"
    h = k // 2
    y = (x[:, :, h - 1::k, :] + x[:, :, h::k, :]) * 0.5
    return (y[:, :, :, h - 1::k] + y[:, :, :, h::k]) * 0.5


# Bilinear upsample implementation.
#   "matmul" (default) : two dense matmuls with constant resample matrices -- all-static,
#                        runs on the systolic array, no gather and no dilation
#   "convt"            : fixed-weight depthwise ConvTranspose2d -- CORRECT ON CPU but the
#                        backend lowers stride-s transposed conv by DILATING the input,
#                        which blows SBUF: "[NCC_INLA001] Allocated memory out of bound
#                        {dilated_0_0_0}@SB<0,0>(5x255492)", plus NRT 1203 DMA-engine and
#                        scheduling failures on the x4/x2 variants. Do not use on device.
#   "idxsel"           : univr_neuron._bilinear_resize (correct, but on the DGE path)
# MEASURED (bench_resize_cost.py, 256x384, one core): the idxsel upsample costs 22.4 ms
# per call and 3 are needed per forward = 67 ms. Its cost is independent of INPUT size and
# of dtype (22.36/22.39/22.55 ms for x8/x4/x2; 22.9 ms in bf16) => it is descriptor-bound
# indirect DMA, not bandwidth. Per gathered element that is 11.4 ns/elem -- the same
# software-DGE rate measured for the unoptimised torch.gather warp. The tap offsets are
# compile-time constants, so none of that addressing is necessary.
_UPSAMPLE = os.environ.get("PAVE_UNIVR_UPSAMPLE", "matmul").lower()

# Pad the upsampled channel count to 16 before the transposed conv, to stay on the conv
# fast path measured in probe_ctx_conv1.py (Cin<16 falls off it).
_UP_PAD_C = os.environ.get("PAVE_UNIVR_UP_PAD_C", "0") == "1"

_UP_KERNEL_CACHE: dict = {}
_UP_MATRIX_CACHE: dict = {}


def _up_matrix(n_in: int, s: int, dtype, device):
    """Constant (n_in*s, n_in) bilinear resample matrix for one axis.

    Row j holds the two tap weights of
    F.interpolate(scale_factor=s, mode='bilinear', align_corners=False): the source
    coordinate is (j+0.5)/s - 0.5, clamped to [0, n_in-1], split between floor and
    floor+1. Border clamping falls out because both taps collapse to the same index and
    their weights sum to 1. Upsampling along an axis is therefore a plain matmul, which
    lands on the systolic array instead of the DGE (gather) or a dilated conv.
    """
    key = (n_in, s, dtype, str(device))
    P = _UP_MATRIX_CACHE.get(key)
    if P is None:
        n_out = n_in * s
        P = torch.zeros(n_out, n_in, dtype=torch.float32)
        j = torch.arange(n_out, dtype=torch.float64)
        src = ((j + 0.5) / s - 0.5).clamp(0.0, n_in - 1.0)
        i0 = src.floor()
        frac = (src - i0).to(torch.float32)
        i0c = i0.long().clamp(0, n_in - 1)
        i1c = (i0 + 1).long().clamp(0, n_in - 1)
        rows = torch.arange(n_out)
        P[rows, i0c] += 1.0 - frac
        P[rows, i1c] += frac
        P = P.to(dtype=dtype, device=device)
        _UP_MATRIX_CACHE[key] = P
    return P


def _up_kernel(C: int, s: int, dtype, device):
    """Separable bilinear kernel as a depthwise ConvTranspose2d weight, (C,1,2s,2s).

    For output j the source coordinate is (j+0.5)/s - 0.5, giving a 2-tap filter whose
    weights are periodic in j with period s. That is exactly a stride-s transposed
    convolution with the triangular kernel w[i] = max(0, 1 - |i-(s-0.5)|/s), i in [0,2s).
    """
    key = (C, s, dtype, str(device))
    w = _UP_KERNEL_CACHE.get(key)
    if w is None:
        r = torch.arange(2 * s, dtype=torch.float32)
        w1d = (1.0 - (r - (s - 0.5)).abs() / s).clamp(min=0.0)
        w2d = torch.outer(w1d, w1d)
        w = torch.zeros(C, 1, 2 * s, 2 * s, dtype=torch.float32)
        w[:, 0] = w2d
        w = w.to(dtype=dtype, device=device)
        _UP_KERNEL_CACHE[key] = w
    return w


def _up(x, s: int):
    """Upsample by integer factor s, matching stock RIFE
    F.interpolate(scale_factor=s, mode='bilinear', align_corners=False).

    The transposed convolution reproduces the interior exactly; a replicate pad of 1
    supplies the clamped border taps, and the result is cropped at s + s//2 (verified
    analytically for s=2 and s=4, and against F.interpolate on device).
    """
    if _RESAMPLE == "fast":
        return F.interpolate(x, scale_factor=float(s), mode="nearest")
    H, W = int(x.shape[-2]), int(x.shape[-1])
    if _UPSAMPLE == "matmul":
        # rows: (H*s,H) @ (N,C,H,W) -> (N,C,H*s,W) ; cols: (N,C,H*s,W) @ (W,W*s)
        Ph = _up_matrix(H, s, x.dtype, x.device)
        Pw = _up_matrix(W, s, x.dtype, x.device)
        return torch.matmul(torch.matmul(Ph, x), Pw.transpose(0, 1))
    if _UPSAMPLE != "convt":
        return _bilin(x, H * s, W * s, align_corners=False)

    C = int(x.shape[1])
    Cpad = 16 if (_UP_PAD_C and C < 16) else C
    xp = F.pad(x, (1, 1, 1, 1), mode="replicate")
    if Cpad != C:
        xp = F.pad(xp, (0, 0, 0, 0, 0, Cpad - C))
    y = F.conv_transpose2d(xp, _up_kernel(Cpad, s, x.dtype, x.device),
                           stride=s, groups=Cpad)
    o = s + s // 2
    y = y[:, :, o:o + H * s, o:o + W * s]
    return y[:, :C] if Cpad != C else y


_NO_COMPILE = os.environ.get("PAVE_UNIVR_NO_COMPILE", "0") == "1"


def _compile(mod):
    """torch.compile onto the Neuron backend, unless PAVE_UNIVR_NO_COMPILE=1.

    The escape hatch exists so the TILING LOGIC can be verified on CPU. Without it every path
    that builds a NEFF is untestable off-device -- `TiledUniVR2NEFF(device="cpu")` still
    compiles for Neuron and dies with "Stream pool not initialized for device 0" -- which means
    index arithmetic like the no-edge-pad padded-coordinate offsets could only be checked after
    an hour of compile. Eager mode makes that a two-minute CPU check.
    """
    if _NO_COMPILE:
        return mod
    return torch.compile(mod, backend="neuron", dynamic=False, fullgraph=True)


# ---------------------------------------------------------------------------
# Space-to-depth for the lane-starved stem convolutions.
# ---------------------------------------------------------------------------
# The channel dimension maps to the 128 partition lanes, so a Cin=3 conv runs 3 of 128 lanes.
# MEASURED consequence at the production tile (bench_conv_mfu): contextnet.conv1 costs 27.1 ms
# per call, x2 per forward = 54.2 ms = 56% of ALL conv time for 1.1% of the FLOPs.
#
# Space-to-depth fixes that WITHOUT changing the byte count -- it is a pure reindexing,
# [1,C,H,W] -> [1,4C,H/2,W/2] -- which is the crucial difference from zero-padding Cin 3->16.
# Padding also bought lanes but multiplied the stage's largest tensor 5.3x, and on a stage that
# is 92-97% static DMA it LOST 2.0x in-graph despite winning 7.6x on the isolated conv.
# MEASURED isolated at 864x1024: baseline 6.53 ms, padded 7.34 ms, S2D 1.71 ms (3.82x) --
# but that is an ISOLATED number and the padding precedent says isolated wins can reverse,
# hence PAVE_UNIVR_S2D_STEM to measure it in-graph.
_S2D_STEM = os.environ.get("PAVE_UNIVR_S2D_STEM", "0") == "1"


def _space_to_depth(x, k=2):
    """[B,C,H,W] -> [B,C*k*k,H/k,W/k]; channel order (c, p, q) -> c*k*k + p*k + q."""
    B, C, H, W = x.shape
    x = x.view(B, C, H // k, k, W // k, k)
    return x.permute(0, 1, 3, 5, 2, 4).reshape(B, C * k * k, H // k, W // k)


def _s2d_conv2(conv):
    """Rewrite Conv2d(C,O,3,stride=2,padding=1) as Conv2d(4C,O,2,stride=1) on space-to-depth-2.

    With padding=1 the output at (i,j) reads x[c, 2i+u-1, 2j+v-1] for u,v in 0..2. Solving
    2i+u-1 = 2a+p with FLOOR division gives u=0 -> (i-1,1), u=1 -> (i,0), u=2 -> (i,1), so the
    taps span S2D offsets {-1, 0}: a 2x2 window ABOVE-LEFT of i. The S2D tensor is therefore
    padded on TOP/LEFT, and kernel index ka = a_off + 1. Getting that backwards yields cos 0.678
    -- close enough to look plausible -- which is why bench_space_to_depth verifies the fold
    against the reference conv (measured cos 1.00000012, max_diff 2.4e-07).
    """
    Wt = conv.weight
    O, C = Wt.shape[0], Wt.shape[1]
    new = nn.Conv2d(C * 4, O, 2, 1, 0, bias=conv.bias is not None)
    with torch.no_grad():
        w = torch.zeros(O, C * 4, 2, 2, dtype=Wt.dtype)
        for u in range(3):
            a_off, pp = divmod(u - 1, 2)
            for v in range(3):
                b_off, qq = divmod(v - 1, 2)
                for c in range(C):
                    w[:, c * 4 + pp * 2 + qq, a_off + 1, b_off + 1] += Wt[:, c, u, v]
        new.weight.copy_(w)
        if conv.bias is not None:
            new.bias.copy_(conv.bias)
    return new


class S2DStem(nn.Module):
    """Drop-in for a stride-2 3x3 conv: space-to-depth then an equivalent 2x2 stride-1 conv."""

    def __init__(self, conv):
        super().__init__()
        self.conv = _s2d_conv2(conv)

    def forward(self, x):
        return self.conv(F.pad(_space_to_depth(x, 2), (1, 0, 1, 0)))


def load_real_weights(univr, path, strict=True):
    """Load a trained UniVR checkpoint into a `network_UniVR.UniVR` instance.

    WHY THIS EXISTS: `UniVR.__init__` only does `self.UVR = IFNet_m()`, i.e. RANDOM
    INITIALISATION. Nothing in the Neuron port path ever loaded a checkpoint, so every
    accuracy number produced before this was a device-vs-CPU comparison of a random model --
    valid as port fidelity, meaningless as absolute accuracy, and meaningless for anything
    that depends on the VALUES of the flow field (displacement bounds, halo sizing, the
    shift-warp support radius m).

    Two key formats exist and confusing them fails SILENTLY:
      * `pre_*_ft/pre_net_flow.pth`  -- keys already prefixed "UVR." and load directly.
      * DataParallel-saved variants  -- keys prefixed "module.", which `model_UniVR.load()`
        remaps via its `convert2`. Applying that remap to an already-"UVR."-prefixed
        checkpoint matches ZERO tensors and leaves the model random with no error raised.
    So the remap is applied only when a "module." prefix is actually present, and the loaded
    tensor count is asserted.
    """
    sd = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    if any(k.startswith("module.") for k in sd):
        sd = {k.replace("module.", "UVR."): v for k, v in sd.items() if "module." in k}
    missing, unexpected = univr.load_state_dict(sd, strict=False)
    n = len(sd)
    logger.info("weights %s: %d tensors, missing=%d unexpected=%d",
                path, n, len(missing), len(unexpected))
    if strict and (n == 0 or unexpected):
        raise RuntimeError(
            f"weight load looks wrong for {path}: matched={n} unexpected={len(unexpected)}. "
            f"Refusing to run -- a silently-random model invalidates every accuracy number.")
    return univr


class NeuronPReLU(nn.Module):
    """nn.PReLU replacement (torch.where) that lowers on the neuron backend."""

    def __init__(self, num_parameters=1):
        super().__init__()
        self.weight = nn.Parameter(torch.full((num_parameters,), 0.25))

    def forward(self, x):
        w = self.weight.view(1, -1, 1, 1)
        return torch.where(x >= 0, x, w * x)


def replace_prelu(model: nn.Module) -> int:
    count = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.PReLU):
            np_ = NeuronPReLU(module.num_parameters)
            np_.weight.data.copy_(module.weight.data)
            parts = name.split(".")
            parent = model
            for p in parts[:-1]:
                parent = parent[int(p)] if p.isdigit() else getattr(parent, p)
            if parts[-1].isdigit():
                parent[int(parts[-1])] = np_
            else:
                setattr(parent, parts[-1], np_)
            count += 1
    return count


def _make_grid(H, W):
    """Identity sampling grids, MATERIALIZED contiguous.

    `.expand()` produces a stride-0 view; registering that as a module buffer makes the
    native-eager backend re-materialize it on the host on EVERY launch, which turns the
    whole NEFF host-synchronous. MEASURED at 256x384 fp32 (diag_bisect.py): a graph
    containing nothing but a cat + strided downsample that touches one expanded buffer
    costs 679 ms submit (ratio 0.999, host-bound); adding the two warps -- which touch two
    more expanded buffers -- doubles it to 1356 ms, while convs and the upsample in between
    add nothing. `.contiguous()` is the fix.
    """
    gy = torch.arange(H, dtype=torch.float32).view(1, 1, H, 1).expand(1, 1, H, W).contiguous()
    gx = torch.arange(W, dtype=torch.float32).view(1, 1, 1, W).expand(1, 1, H, W).contiguous()
    return gy, gx


def _warp(tenInput, tenFlow, grid_y, grid_x):
    """Inline bilinear warp — compiles INTO the NEFF. fp32 coord/index math (bf16 mantissa
    cannot represent integer indices up to H*W). With PAVE_UNIVR_NKI_WARP=1 -> DMA-DGE NKI."""
    if _USE_DMA_WARP:
        return _dma_warp_fn(tenInput, tenFlow, grid_y, grid_x)

    B, C, H, W = tenInput.shape
    dt = tenInput.dtype
    fx = grid_x.to(torch.float32) + tenFlow[:, 0:1].to(torch.float32)
    fy = grid_y.to(torch.float32) + tenFlow[:, 1:2].to(torch.float32)
    sx = fx.clamp(0.0, W - 1.0)
    sy = fy.clamp(0.0, H - 1.0)
    x0 = sx.floor(); y0 = sy.floor()
    x1 = (x0 + 1).clamp(max=W - 1)
    y1 = (y0 + 1).clamp(max=H - 1)
    wx = (sx - x0).to(dt); wy = (sy - y0).to(dt)
    x0l = x0.long(); y0l = y0.long(); x1l = x1.long(); y1l = y1.long()
    inp = tenInput.reshape(B, C, H * W)

    def g(yc, xc):
        idx = (yc * W + xc).expand(B, C, H, W).reshape(B, C, H * W)
        return torch.gather(inp, 2, idx).reshape(B, C, H, W)

    tl = g(y0l, x0l); tr = g(y0l, x1l); bl = g(y1l, x0l); br = g(y1l, x1l)
    top = tl * (1 - wx) + tr * wx
    bot = bl * (1 - wx) + br * wx
    return top * (1 - wy) + bot * wy


class NEFF_A(nn.Module):
    """IFBlocks(0,1,2) + time_offset + merge. All warps inlined; no eager ops leak out."""

    def __init__(self, block0, block1, block2, H, W):
        super().__init__()
        self.block0, self.block1, self.block2 = block0, block1, block2
        gy, gx = _make_grid(H, W)
        self.register_buffer("grid_y", gy)
        self.register_buffer("grid_x", gx)
        # contiguous: a stride-0 expanded buffer costs ~679 ms/launch on the host in fp32
        # (see _make_grid docstring / diag_bisect.py)
        gr = torch.arange(H, dtype=torch.float32).view(1, 1, H, 1) \
            .expand(1, 1, H, W).contiguous()
        self.register_buffer("grid_rows", gr)

    def forward(self, img, timestep, gamma):
        B, C, H, W = img.shape
        img0, img1 = img[:, :3], img[:, 3:6]
        # time_offset is the rolling-shutter time field; in bf16 its absolute error is
        # ~4e-3 of a frame time, which shifts the interpolation target everywhere.
        tof_dt = torch.float32 if _F32_TOF else img.dtype
        gr = self.grid_rows.to(tof_dt)
        time_offset = timestep + gamma - gamma * gr / H + 0.0001
        time_offset = torch.ones(B, 1, H, W, device=img.device, dtype=tof_dt) * time_offset
        # flow/mask are accumulated across the three pyramid stages and then used as
        # sampling coordinates, so they get their own precision knob.
        acc_dt = torch.float32 if _F32_FLOW else img.dtype
        tof_c = time_offset.to(img.dtype)

        x = _down(torch.cat((img0, img1, tof_c), 1), 4)
        x = self.block0.conv0(x); x = self.block0.convblock(x) + x
        tmp = self.block0.lastconv(x); tmp = _up(tmp, 8)
        flow = tmp[:, :4].to(acc_dt) * 8.0; mask = tmp[:, 4:5].to(acc_dt)
        wi0 = _warp(img0, flow[:, :2].to(img.dtype), self.grid_y, self.grid_x)
        wi1 = _warp(img1, flow[:, 2:4].to(img.dtype), self.grid_y, self.grid_x)

        x = torch.cat((img0, img1, tof_c, wi0, wi1, mask.to(img.dtype)), 1)
        x = _down(x, 2)
        flow_down = _down(flow, 2) * 0.5
        x = torch.cat((x, flow_down.to(img.dtype)), 1)
        x = self.block1.conv0(x); x = self.block1.convblock(x) + x
        tmp = self.block1.lastconv(x); tmp = _up(tmp, 4)
        flow = flow + tmp[:, :4].to(acc_dt) * 4.0
        mask = mask + tmp[:, 4:5].to(acc_dt)
        wi0 = _warp(img0, flow[:, :2].to(img.dtype), self.grid_y, self.grid_x)
        wi1 = _warp(img1, flow[:, 2:4].to(img.dtype), self.grid_y, self.grid_x)

        x = torch.cat((img0, img1, tof_c, wi0, wi1, mask.to(img.dtype)), 1)
        x = torch.cat((x, flow.to(img.dtype)), 1)
        x = self.block2.conv0(x); x = self.block2.convblock(x) + x
        tmp = self.block2.lastconv(x); tmp = _up(tmp, 2)
        flow = flow + tmp[:, :4].to(acc_dt) * 2.0
        mask = mask + tmp[:, 4:5].to(acc_dt)
        wi0 = _warp(img0, flow[:, :2].to(img.dtype), self.grid_y, self.grid_x)
        wi1 = _warp(img1, flow[:, 2:4].to(img.dtype), self.grid_y, self.grid_x)

        mask_sig = torch.sigmoid(mask)
        merged = (wi0.to(acc_dt) * mask_sig
                  + wi1.to(acc_dt) * (1 - mask_sig)).to(img.dtype)
        return merged, flow.to(img.dtype), mask.to(img.dtype), wi0, wi1


# ---------------------------------------------------------------------------
# NEFF_A split into one NEFF per pyramid stage
# ---------------------------------------------------------------------------
# MEASURED at 256x384 with the matmul upsample: NEFF_A costs 19.64 ms in bf16 but
# 791.69 ms in fp32 -- 40x, where NEFF_B shows only the expected 3.7x PE-rate penalty
# (measured ceilings 70.7 vs 18.1 TFLOP/s = 3.9x). A 40x gap is not arithmetic, it is the
# fp32 working set no longer fitting: NEFF_A holds six full-resolution warps plus several
# full-res concatenations, and fp32 doubles every intermediate. The 2-NEFF boundary was
# chosen for bf16.
#
# Splitting per pyramid stage keeps every warp INSIDE a compiled NEFF, so this does not
# reintroduce the eager-warp dispatch regression (previously 8.7x). The stages take the
# rolling-shutter field as a TENSOR (like NEFF_A_Tiled), so gamma/timestep never become
# graph constants and one artifact serves any camera.


# Stage 0's two warps are the ONLY warps whose result is never used at full resolution:
# NEFF_A1 consumes them exclusively as _down(cat(...), 2). Because _down is channel-wise,
# _down(cat(a,b,...)) == cat(_down(a), _down(b), ...) exactly, so the pipeline only ever needs
# _down(warp(img, flow), 2).
#
# _down is a genuine 2x2 average (see _down), so the half-res result is NOT algebraically
# reducible to a single bilinear tap -- warping the downsampled image with the downsampled
# flow is an APPROXIMATION whose error scales with the local curvature of the flow field.
# It is worth measuring because it removes 3/4 of stage 0's output pixels and the warp cost is
# descriptor-bound, i.e. proportional to output pixel count: PROFILED A0 = 50.73 ms with
# tensor at 4.7%, gpsimd 65.9% and 909,040 sw-DGE descriptors at 864x1024.
# Gate it: only adopt if the fp32 frame gate holds at the ~110 dB level.
_A0_HALFWARP = os.environ.get("PAVE_UNIVR_A0_HALFWARP", "0") == "1"


class NEFF_A0(nn.Module):
    """Pyramid stage 0: block0 at 1/4 scale + its two warps.

    With PAVE_UNIVR_A0_HALFWARP=1 the warps (and the returned wi0/wi1, mask) are produced at
    HALF resolution, which is the only resolution NEFF_A1 ever reads them at.
    """

    def __init__(self, block0, H, W):
        super().__init__()
        self.block0 = block0
        gy, gx = _make_grid(H, W)
        self.register_buffer("grid_y", gy)
        self.register_buffer("grid_x", gx)
        self.halfwarp = _A0_HALFWARP
        if self.halfwarp:
            gyh, gxh = _make_grid(H // 2, W // 2)
            self.register_buffer("grid_y_h", gyh)
            self.register_buffer("grid_x_h", gxh)

    def forward(self, img0, img1, tof):
        x = _down(torch.cat((img0, img1, tof), 1), 4)
        x = self.block0.conv0(x); x = self.block0.convblock(x) + x
        tmp = _up(self.block0.lastconv(x), 8)
        flow = tmp[:, :4] * 8.0
        mask = tmp[:, 4:5]
        if self.halfwarp:
            # warp at half res: 1/4 of the output pixels => 1/4 of the DMA descriptors
            fh = _down(flow, 2) * 0.5
            wi0 = _warp(_down(img0, 2), fh[:, :2], self.grid_y_h, self.grid_x_h)
            wi1 = _warp(_down(img1, 2), fh[:, 2:4], self.grid_y_h, self.grid_x_h)
        else:
            wi0 = _warp(img0, flow[:, :2], self.grid_y, self.grid_x)
            wi1 = _warp(img1, flow[:, 2:4], self.grid_y, self.grid_x)
        return flow, mask, wi0, wi1


class NEFF_A1(nn.Module):
    """Pyramid stage 1: block1 at 1/2 scale + its two full-res warps."""

    def __init__(self, block1, H, W):
        super().__init__()
        self.block1 = block1
        gy, gx = _make_grid(H, W)
        self.register_buffer("grid_y", gy)
        self.register_buffer("grid_x", gx)
        self.halfwarp = _A0_HALFWARP

    def forward(self, img0, img1, tof, wi0, wi1, mask, flow):
        if self.halfwarp:
            # wi0/wi1 already arrive at half res. _down is channel-wise, so splitting the
            # concatenation this way preserves the EXACT channel order of the fused form:
            #   _down(cat(img0,img1,tof,wi0,wi1,mask), 2)
            #     == cat(_down(img0),_down(img1),_down(tof), wi0_h, wi1_h, _down(mask))
            x = torch.cat((_down(torch.cat((img0, img1, tof), 1), 2),
                           wi0, wi1, _down(mask, 2)), 1)
        else:
            x = _down(torch.cat((img0, img1, tof, wi0, wi1, mask), 1), 2)
        x = torch.cat((x, _down(flow, 2) * 0.5), 1)
        x = self.block1.conv0(x); x = self.block1.convblock(x) + x
        tmp = _up(self.block1.lastconv(x), 4)
        flow = flow + tmp[:, :4] * 4.0
        mask = mask + tmp[:, 4:5]
        nwi0 = _warp(img0, flow[:, :2], self.grid_y, self.grid_x)
        nwi1 = _warp(img1, flow[:, 2:4], self.grid_y, self.grid_x)
        return flow, mask, nwi0, nwi1


class NEFF_A2(nn.Module):
    """Pyramid stage 2: block2 at full scale + its two warps + the mask merge."""

    def __init__(self, block2, H, W):
        super().__init__()
        self.block2 = block2
        gy, gx = _make_grid(H, W)
        self.register_buffer("grid_y", gy)
        self.register_buffer("grid_x", gx)

    def forward(self, img0, img1, tof, wi0, wi1, mask, flow):
        x = torch.cat((img0, img1, tof, wi0, wi1, mask), 1)
        x = torch.cat((x, flow), 1)
        x = self.block2.conv0(x); x = self.block2.convblock(x) + x
        tmp = _up(self.block2.lastconv(x), 2)
        flow = flow + tmp[:, :4] * 2.0
        mask = mask + tmp[:, 4:5]
        nwi0 = _warp(img0, flow[:, :2], self.grid_y, self.grid_x)
        nwi1 = _warp(img1, flow[:, 2:4], self.grid_y, self.grid_x)
        mask_sig = torch.sigmoid(mask)
        merged = nwi0 * mask_sig + nwi1 * (1 - mask_sig)
        return merged, flow, mask, nwi0, nwi1


class _WrapperSplit(nn.Module):
    """4-NEFF form: A0 -> A1 -> A2 -> B, rolling-shutter field passed in as a tensor."""

    def __init__(self, a0, a1, a2, b):
        super().__init__()
        self.a0, self.a1, self.a2, self.b = a0, a1, a2, b

    def forward(self, img, tof):
        img0, img1 = img[:, :3], img[:, 3:6]
        flow, mask, wi0, wi1 = self.a0(img0, img1, tof)
        flow, mask, wi0, wi1 = self.a1(img0, img1, tof, wi0, wi1, mask, flow)
        merged, flow, mask, wi0, wi1 = self.a2(img0, img1, tof, wi0, wi1, mask, flow)
        return self.b(img0, img1, wi0, wi1, mask, flow, merged)


def build_split(ifnet, H, W, dtype=torch.bfloat16, device="neuron"):
    """Compile the 4-NEFF form (3 pyramid stages + Contextnet/Unet)."""
    replace_prelu(ifnet)
    a0 = NEFF_A0(ifnet.block0, H, W).to(dtype).eval().to(device)
    a1 = NEFF_A1(ifnet.block1, H, W).to(dtype).eval().to(device)
    a2 = NEFF_A2(ifnet.block2, H, W).to(dtype).eval().to(device)
    b = NEFF_B(ifnet.contextnet, ifnet.unet, H, W).to(dtype).eval().to(device)
    logger.info("split build: dtype=%s @ %dx%d", dtype, W, H)
    return _WrapperSplit(_compile(a0), _compile(a1), _compile(a2), _compile(b)).to(device)


def _widen_conv_in(conv: nn.Conv2d, new_in: int) -> nn.Conv2d:
    """Same conv with in_channels widened to `new_in`, added weights ZERO.

    Output is mathematically identical for any values in the added channels, so the
    caller can pad the input with anything (we use zeros). Lifts the Contextnet stem
    off the compiler's slow small-channel conv path -- MEASURED 7.6x on the conv.
    """
    new = nn.Conv2d(new_in, conv.out_channels, conv.kernel_size, conv.stride,
                    conv.padding, bias=conv.bias is not None)
    with torch.no_grad():
        new.weight.zero_()
        new.weight[:, :conv.in_channels] = conv.weight
        if conv.bias is not None:
            new.bias.copy_(conv.bias)
    return new


class NEFF_B(nn.Module):
    """Contextnet + Unet + residual + clamp."""

    def __init__(self, contextnet, unet, H, W):
        super().__init__()
        self.ctx_conv1 = contextnet.conv1
        self.ctx_conv2 = contextnet.conv2
        self.ctx_conv3 = contextnet.conv3
        self.ctx_conv4 = contextnet.conv4
        self.unet = unet
        self.pad_cin = 0
        if _PAD_CTX_CIN and _pad_is_safe(H, W):
            c0 = self.ctx_conv1.conv1[0]
            if isinstance(c0, nn.Conv2d) and c0.in_channels < _CTX_CIN:
                self.pad_cin = _CTX_CIN - c0.in_channels
                self.ctx_conv1.conv1[0] = _widen_conv_in(c0, _CTX_CIN)
        for s in (2, 4, 8, 16):
            gy, gx = _make_grid(H // s, W // s)
            self.register_buffer(f"gy_{s}", gy)
            self.register_buffer(f"gx_{s}", gx)

    def _ctx(self, img, flow_half):
        if self.pad_cin:
            img = F.pad(img, (0, 0, 0, 0, 0, self.pad_cin))
        cx = self.ctx_conv1(img)
        fc = _down(flow_half, 2) * 0.5
        f1 = _warp(cx, fc, self.gy_2, self.gx_2)
        cx = self.ctx_conv2(cx); fc = _down(fc, 2) * 0.5
        f2 = _warp(cx, fc, self.gy_4, self.gx_4)
        cx = self.ctx_conv3(cx); fc = _down(fc, 2) * 0.5
        f3 = _warp(cx, fc, self.gy_8, self.gx_8)
        cx = self.ctx_conv4(cx); fc = _down(fc, 2) * 0.5
        f4 = _warp(cx, fc, self.gy_16, self.gx_16)
        return f1, f2, f3, f4

    def forward(self, img0, img1, wi0, wi1, mask, flow, merged):
        f1_0, f2_0, f3_0, f4_0 = self._ctx(img0, flow[:, :2])
        f1_1, f2_1, f3_1, f4_1 = self._ctx(img1, flow[:, 2:4])
        s0 = self.unet.down0(torch.cat((img0, img1, wi0, wi1, mask, flow), 1))
        s1 = self.unet.down1(torch.cat((s0, f1_0, f1_1), 1))
        s2 = self.unet.down2(torch.cat((s1, f2_0, f2_1), 1))
        s3 = self.unet.down3(torch.cat((s2, f3_0, f3_1), 1))
        x = self.unet.up0(torch.cat((s3, f4_0, f4_1), 1))
        x = self.unet.up1(torch.cat((x, s2), 1))
        x = self.unet.up2(torch.cat((x, s1), 1))
        x = self.unet.up3(torch.cat((x, s0), 1))
        x = self.unet.conv(x)
        res = torch.sigmoid(x) * 2 - 1
        return torch.clamp(merged + res, 0, 1)


# ---------------------------------------------------------------------------
# NEFF_B split: Contextnet (reused for both images) + Unet
# ---------------------------------------------------------------------------
# PROFILED at the production tile (864x1024, fp32, one core): NEFF_B is 164.4 ms device
# with dma_active 83.3%, static_dma 58.3%, spill_reload_bytes = 1159.8 MB per forward, and
# mfu_max_achievable only 13.0%. That is a SPILL signature, not a gather one (contrast
# NEFF_A: 48% software-DGE, 315K packets, static_dma 1.9%, ceiling 38.2%): NEFF_B's working
# set does not fit SBUF at production tile size, so it round-trips through HBM
# (3967 MB read + 909 MB write for a 3.5 Mpix tile ~ 1400 B/pixel).
#
# Splitting is the documented remedy for spill, and unlike NEFF_A -- where a per-stage split
# measured WORSE because it only added handoff round-trips -- here the profiler names spill
# explicitly. Contextnet is called twice on independent images, so ONE compiled NEFF serves
# both calls (half the compile, weights stay resident).


class NEFF_B_CTX(nn.Module):
    """Contextnet for ONE image: conv1..conv4 with a warp after each level.

    Returns the 4-level feature pyramid. Compiled once, called twice (img0, img1).
    """

    def __init__(self, contextnet, H, W):
        super().__init__()
        self.ctx_conv1 = contextnet.conv1
        self.ctx_conv2 = contextnet.conv2
        self.ctx_conv3 = contextnet.conv3
        self.ctx_conv4 = contextnet.conv4
        self.pad_cin = 0
        if _PAD_CTX_CIN and _pad_is_safe(H, W):
            c0 = self.ctx_conv1.conv1[0]
            if isinstance(c0, nn.Conv2d) and c0.in_channels < _CTX_CIN:
                self.pad_cin = _CTX_CIN - c0.in_channels
                self.ctx_conv1.conv1[0] = _widen_conv_in(c0, _CTX_CIN)
        for s in (2, 4, 8, 16):
            gy, gx = _make_grid(H // s, W // s)
            self.register_buffer(f"gy_{s}", gy)
            self.register_buffer(f"gx_{s}", gx)

    def forward(self, img, flow_half):
        if self.pad_cin:
            img = F.pad(img, (0, 0, 0, 0, 0, self.pad_cin))
        cx = self.ctx_conv1(img)
        fc = _down(flow_half, 2) * 0.5
        f1 = _warp(cx, fc, self.gy_2, self.gx_2)
        cx = self.ctx_conv2(cx); fc = _down(fc, 2) * 0.5
        f2 = _warp(cx, fc, self.gy_4, self.gx_4)
        cx = self.ctx_conv3(cx); fc = _down(fc, 2) * 0.5
        f3 = _warp(cx, fc, self.gy_8, self.gx_8)
        cx = self.ctx_conv4(cx); fc = _down(fc, 2) * 0.5
        f4 = _warp(cx, fc, self.gy_16, self.gx_16)
        return f1, f2, f3, f4


class NEFF_CTXCONV(nn.Module):
    """Contextnet conv pyramid for ONE image -> (cx1, cx2, cx3, cx4). NO warps.

    These 4 feature maps depend ONLY on the image, not on timestep/flow, so for TRIPLET
    (2 forwards, same images) they are computed ONCE per image per FRAME and reused by both
    passes -- instead of the current 4 Contextnet calls/frame (2 images x 2 passes).
    PROFILED: Contextnet is 62% of NEFF_B at 0.2% MFU, and its cost is the small-channel
    convs (esp. the Cin=3 stem); halving how often they run is the largest gate-preserving
    lever found.
    """

    def __init__(self, contextnet, H=None, W=None):
        super().__init__()
        self.ctx_conv1 = contextnet.conv1
        self.ctx_conv2 = contextnet.conv2
        self.ctx_conv3 = contextnet.conv3
        self.ctx_conv4 = contextnet.conv4
        self.pad_cin = 0
        if _PAD_CTX_CIN and (H is None or _pad_is_safe(H, W)):
            c0 = self.ctx_conv1.conv1[0]
            if isinstance(c0, nn.Conv2d) and c0.in_channels < _CTX_CIN:
                self.pad_cin = _CTX_CIN - c0.in_channels
                self.ctx_conv1.conv1[0] = _widen_conv_in(c0, _CTX_CIN)
        # PAVE_UNIVR_S2D_STEM=1 replaces the Cin=3 stride-2 stem with its space-to-depth
        # equivalent. Exact (verified cos 1.00000012), byte-conserving, and 3.82x on the
        # ISOLATED conv -- this switch exists to find out whether that survives in-graph, which
        # the zero-padding precedent (7.6x isolated, 2.0x slower in-graph) says is not automatic.
        if _S2D_STEM and not self.pad_cin:
            c0 = self.ctx_conv1.conv1[0]
            if isinstance(c0, nn.Conv2d) and c0.stride == (2, 2) and c0.kernel_size == (3, 3):
                self.ctx_conv1.conv1[0] = S2DStem(c0)

    def forward(self, img):
        if self.pad_cin:
            img = F.pad(img, (0, 0, 0, 0, 0, self.pad_cin))
        cx1 = self.ctx_conv1(img)
        cx2 = self.ctx_conv2(cx1)
        cx3 = self.ctx_conv3(cx2)
        cx4 = self.ctx_conv4(cx3)
        return cx1, cx2, cx3, cx4


class NEFF_CTXWARP(nn.Module):
    """Warp a cached Contextnet feature pyramid by flow -> (f1, f2, f3, f4). t-dependent."""

    def __init__(self, H, W):
        super().__init__()
        for s in (2, 4, 8, 16):
            gy, gx = _make_grid(H // s, W // s)
            self.register_buffer(f"gy_{s}", gy)
            self.register_buffer(f"gx_{s}", gx)

    def forward(self, cx1, cx2, cx3, cx4, flow_half):
        fc = _down(flow_half, 2) * 0.5
        f1 = _warp(cx1, fc, self.gy_2, self.gx_2)
        fc = _down(fc, 2) * 0.5
        f2 = _warp(cx2, fc, self.gy_4, self.gx_4)
        fc = _down(fc, 2) * 0.5
        f3 = _warp(cx3, fc, self.gy_8, self.gx_8)
        fc = _down(fc, 2) * 0.5
        f4 = _warp(cx4, fc, self.gy_16, self.gx_16)
        return f1, f2, f3, f4


def build_ctxcache(ifnet, H, W, dtype=torch.bfloat16, device="neuron"):
    """Per-FRAME wrapper that caches the t-independent Contextnet conv pyramids.

    NEFFs: A0/A1/A2 (per stage) + NEFF_CTXCONV (img->pyramid, 2 calls/frame) +
    NEFF_CTXWARP (pyramid,flow->features, 4 calls/frame) + NEFF_B_UNET (2 calls/frame).
    Exposes forward_frame(img, gamma) returning BOTH corrected frames (t=1-g/2 and -g/2).
    """
    replace_prelu(ifnet)
    a0 = _compile(NEFF_A0(ifnet.block0, H, W).to(dtype).eval().to(device))
    a1 = _compile(NEFF_A1(ifnet.block1, H, W).to(dtype).eval().to(device))
    a2 = _compile(NEFF_A2(ifnet.block2, H, W).to(dtype).eval().to(device))
    cc = _compile(NEFF_CTXCONV(ifnet.contextnet, H, W).to(dtype).eval().to(device))
    cw = _compile(NEFF_CTXWARP(H, W).to(dtype).eval().to(device))
    uu = _compile(NEFF_B_UNET(ifnet.unet).to(dtype).eval().to(device))

    class _WFrame(nn.Module):
        def __init__(self):
            super().__init__()
            self.a0, self.a1, self.a2 = a0, a1, a2
            self.cc, self.cw, self.uu = cc, cw, uu
            gr = torch.arange(H, dtype=torch.float32).view(1, 1, H, 1) \
                .expand(1, 1, H, W).contiguous()
            self.register_buffer("grid_rows", gr.to(dtype).to(device))

        def _pass(self, i0, i1, t, gamma, p0, p1):
            tof = (t + gamma - gamma * self.grid_rows / H + 0.0001)
            fl, mk, w0, w1 = self.a0(i0, i1, tof)
            fl, mk, w0, w1 = self.a1(i0, i1, tof, w0, w1, mk, fl)
            mg, fl, mk, w0, w1 = self.a2(i0, i1, tof, w0, w1, mk, fl)
            f0 = self.cw(*p0, fl[:, :2])
            f1 = self.cw(*p1, fl[:, 2:4])
            return self.uu(i0, i1, w0, w1, mk, fl, mg, *f0, *f1)

        def forward_frame(self, img, gamma):
            i0, i1 = img[:, :3], img[:, 3:6]
            p0 = self.cc(i0)   # conv pyramids computed ONCE, reused by both passes
            p1 = self.cc(i1)
            out_fwd = self._pass(i0, i1, 1 - gamma / 2, gamma, p0, p1)
            out_bwd = self._pass(i0, i1, -gamma / 2, gamma, p0, p1)
            return out_fwd, out_bwd

        def forward(self, img, timestep, gamma):
            i0, i1 = img[:, :3], img[:, 3:6]
            p0, p1 = self.cc(i0), self.cc(i1)
            return self._pass(i0, i1, timestep, gamma, p0, p1)

    return _WFrame().to(device)


def _split_conv(conv, parts, sizes):
    """Apply `conv` to cat(parts, dim=1) WITHOUT materializing the concatenation.

    A convolution over a channel-concatenation is exactly the sum of convolutions over the
    pieces with the matching input-channel slices of the weight:
        conv(cat(A,B), W, b) == conv(A, W[:, :cA], b) + conv(B, W[:, cA:], 0)
    The bias is added once (folded into the first term). Only the fp summation ORDER changes.

    WHY: PROFILED at the production tile 864x1024 fp32, the Unet spills 614 MB (63% static
    DMA) while SBUF is ~28 MB per NeuronCore-v3 at LNC=1. Its concatenations are the largest
    tensors in the stage and each one is a pure COPY of data that is already resident:
        down0  cat(img0,img1,wi0,wi1,mask,flow) = 17ch at FULL res = 60.2 MB
        up3    cat(x, s0)                       = 64ch at H/2     = 56.6 MB
    Removing them drops the peak live set, which is what the allocator spills against.
    """
    off = 0
    out = None
    for p, c in zip(parts, sizes):
        w = conv.weight[:, off:off + c]
        y = F.conv2d(p, w, conv.bias if out is None else None,
                     stride=conv.stride, padding=conv.padding, dilation=conv.dilation)
        out = y if out is None else out + y
        off += c
    return out


_UNET_SPLITCAT = os.environ.get("PAVE_UNIVR_UNET_SPLITCAT", "0") == "1"


class NEFF_B_UNET(nn.Module):
    """Unet + residual + clamp, taking both context pyramids as inputs."""

    def __init__(self, unet):
        super().__init__()
        self.unet = unet

    def _c2(self, blk, parts):
        """Conv2 block whose first conv consumes a concatenation."""
        if not _UNET_SPLITCAT:
            return blk(torch.cat(parts, 1))
        c0, act0 = blk.conv1[0], blk.conv1[1]
        x = act0(_split_conv(c0, parts, [p.shape[1] for p in parts]))
        return blk.conv2(x)

    def _dc(self, blk, parts):
        """deconv block (ConvTranspose2d) whose input is a concatenation.

        ConvTranspose2d sums over OUTPUT channels, so the same channel-split identity holds
        on its dim-0 weight axis -- but the transposed-conv lowering on this backend dilates
        the input (NCC_INLA001), so splitting it is not obviously a win. Left concatenated
        unless explicitly enabled.
        """
        return blk(torch.cat(parts, 1))

    def forward(self, img0, img1, wi0, wi1, mask, flow, merged,
                f1_0, f2_0, f3_0, f4_0, f1_1, f2_1, f3_1, f4_1):
        s0 = self._c2(self.unet.down0, (img0, img1, wi0, wi1, mask, flow))
        s1 = self._c2(self.unet.down1, (s0, f1_0, f1_1))
        s2 = self._c2(self.unet.down2, (s1, f2_0, f2_1))
        s3 = self._c2(self.unet.down3, (s2, f3_0, f3_1))
        x = self._dc(self.unet.up0, (s3, f4_0, f4_1))
        x = self._dc(self.unet.up1, (x, s2))
        x = self._dc(self.unet.up2, (x, s1))
        x = self._dc(self.unet.up3, (x, s0))
        x = self.unet.conv(x)
        res = torch.sigmoid(x) * 2 - 1
        return torch.clamp(merged + res, 0, 1)


class _WrapperBSplit(nn.Module):
    """NEFF_A + (Contextnet NEFF x2, one compiled graph) + Unet NEFF."""

    def __init__(self, neff_a, neff_ctx, neff_unet):
        super().__init__()
        self.neff_a, self.neff_ctx, self.neff_unet = neff_a, neff_ctx, neff_unet

    def forward(self, img, timestep, gamma):
        img0, img1 = img[:, :3], img[:, 3:6]
        merged, flow, mask, wi0, wi1 = self.neff_a(img, timestep, gamma)
        a1, a2, a3, a4 = self.neff_ctx(img0, flow[:, :2])
        b1, b2, b3, b4 = self.neff_ctx(img1, flow[:, 2:4])
        return self.neff_unet(img0, img1, wi0, wi1, mask, flow, merged,
                              a1, a2, a3, a4, b1, b2, b3, b4)


def build_bsplit(ifnet, H, W, dtype=torch.bfloat16, device="neuron"):
    """NEFF_A + split NEFF_B (Contextnet NEFF reused twice + Unet NEFF)."""
    replace_prelu(ifnet)
    a = NEFF_A(ifnet.block0, ifnet.block1, ifnet.block2, H, W).to(dtype).eval().to(device)
    c = NEFF_B_CTX(ifnet.contextnet, H, W).to(dtype).eval().to(device)
    u = NEFF_B_UNET(ifnet.unet).to(dtype).eval().to(device)
    logger.info("B-split build: dtype=%s @ %dx%d", dtype, W, H)
    return _WrapperBSplit(_compile(a), _compile(c), _compile(u)).to(device)


def build_allsplit(ifnet, H, W, dtype=torch.bfloat16, device="neuron"):
    """Maximally split: A0/A1/A2 (per pyramid stage) + Contextnet NEFF (reused x2) + Unet.

    Fastest to COMPILE (each NEFF is small: 2 warps, not 6 -- bf16 measured ~10x faster
    compile than the monolithic NEFF_A) and lets the profiler attribute every stage. The
    stride-0 buffer fix removed the host penalty that made an earlier A-split look slower.
    """
    """A0/A1/A2 + Contextnet(reused) + Unet, each its own NEFF."""
    replace_prelu(ifnet)
    a0 = NEFF_A0(ifnet.block0, H, W).to(dtype).eval().to(device)
    a1 = NEFF_A1(ifnet.block1, H, W).to(dtype).eval().to(device)
    a2 = NEFF_A2(ifnet.block2, H, W).to(dtype).eval().to(device)
    c = NEFF_B_CTX(ifnet.contextnet, H, W).to(dtype).eval().to(device)
    u = NEFF_B_UNET(ifnet.unet).to(dtype).eval().to(device)
    logger.info("all-split build: dtype=%s @ %dx%d", dtype, W, H)
    a0c, a1c, a2c = _compile(a0), _compile(a1), _compile(a2)
    cc, uc = _compile(c), _compile(u)

    # tof is a (1,1,H,W) tensor built on the host from (t, gamma); passed to A0/A1/A2 so
    # gamma/timestep never become graph constants (one artifact serves any camera).
    class _W(nn.Module):
        def __init__(self):
            super().__init__()
            self.a0, self.a1, self.a2, self.ctx, self.unet = a0c, a1c, a2c, cc, uc
            gr = torch.arange(H, dtype=torch.float32).view(1, 1, H, 1) \
                .expand(1, 1, H, W).contiguous()
            self.register_buffer("grid_rows", gr.to(dtype).to(device))

        def forward(self, img, timestep, gamma):
            i0, i1 = img[:, :3], img[:, 3:6]
            tof = (timestep + gamma - gamma * self.grid_rows / H + 0.0001)
            fl, mk, w0, w1 = self.a0(i0, i1, tof)
            fl, mk, w0, w1 = self.a1(i0, i1, tof, w0, w1, mk, fl)
            mg, fl, mk, w0, w1 = self.a2(i0, i1, tof, w0, w1, mk, fl)
            p0 = self.ctx(i0, fl[:, :2]); p1 = self.ctx(i1, fl[:, 2:4])
            return self.unet(i0, i1, w0, w1, mk, fl, mg, *p0, *p1)

    return _W().to(device)


class _Wrapper(nn.Module):
    def __init__(self, neff_a, neff_b):
        super().__init__()
        self.neff_a, self.neff_b = neff_a, neff_b

    def forward(self, img, timestep, gamma):
        img0, img1 = img[:, :3], img[:, 3:6]
        merged, flow, mask, wi0, wi1 = self.neff_a(img, timestep, gamma)
        return self.neff_b(img0, img1, wi0, wi1, mask, flow, merged)


def build_2neff(ifnet: nn.Module, H: int, W: int, dtype=torch.bfloat16, device="neuron"):
    """Build the compiled 2-NEFF wrapper from an already-weight-loaded stock IFNet_m.

    PReLU is swapped for the lowerable NeuronPReLU (reusing learned slopes). The two NEFFs
    are cast to `dtype` and compiled with backend=neuron fullgraph. Returns the wrapper.
    """
    replace_prelu(ifnet)
    neff_a = NEFF_A(ifnet.block0, ifnet.block1, ifnet.block2, H, W)
    neff_b = NEFF_B(ifnet.contextnet, ifnet.unet, H, W)
    neff_a = neff_a.to(dtype).eval().to(device)
    neff_b = neff_b.to(dtype).eval().to(device)
    logger.info("2-NEFF build: dtype=%s nki_warp=%s @ %dx%d", dtype, _USE_DMA_WARP, W, H)
    wrap = _Wrapper(_compile(neff_a), _compile(neff_b)).to(device)
    return wrap


class ModelUniVR2NEFF:
    """Adapter exposing the stock ModelUniVR contract (set_input / forward) over the 2-NEFF
    wrapper, so the provider can use it transparently."""

    def __init__(self, ifnet, H, W, gamma, dtype=torch.bfloat16, device="neuron"):
        self.wrap = build_2neff(ifnet, H, W, dtype=dtype, device=device)
        self.dtype = dtype
        self.device = torch.device(device)
        self.gamma = gamma
        self.im_rs = None

    def set_input(self, inputs):
        # inputs = [im_rs, None, None, None] (matches stock ModelUniVR.set_input)
        im = inputs[0]
        self.im_rs = im.to(self.dtype).to(self.device)

    def forward(self, t, gamma=None):
        if gamma is None:
            gamma = self.gamma
        with torch.no_grad():
            return self.wrap(self.im_rs, t, gamma)
