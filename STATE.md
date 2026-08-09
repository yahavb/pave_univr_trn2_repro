# Session state

Repo: `github.com/yahavb/pave_univr_trn2_repro` (private, `main`), HEAD `65e340e`.
Local: `~/pave_univr_trn2_repro`. Everything below is committed and pushed.

## Goal

Find what hardware limit the resample op hits on trn2, using profiles of pure-torch
microbenchmarks with NO NKI kernels, then design a kernel against the known bound.
Full method in `METHOD.md`.

## Files

| file | what it is |
|---|---|
| `repro_unrolling_trn2.py` | the whole model, from the original bundle. `--height 1728 --width 4096 --tiles 2x4 --halo 128` |
| `warp_op_bench.py` | ONE op at ONE shape under `torch.compile(fullgraph=True)`. Arms: `gridsample`, `gather`, `transpose`, `shift`, `window1`, `window2`, `shiftmatmul`, `nkishift` |
| `nki_shift_warp.py` | the NKI kernel. NEVER COMPILED SUCCESSFULLY |
| `profile_roofline.py` | reads a captured `summary.json`, prints per-engine time, descriptors/pixel, ns/descriptor |
| `sweep-job.yaml` | the microbenchmark k8s job |
| `repro-job.yaml` | the whole-model k8s job, from the bundle |
| `METHOD.md` | method, tile geometry, acceptance criteria |
| `SINGLE_GRAPH_NCC_EBIR033.md` | STALE. Says single graph never compiles; it does, up to 512x768 |

Run: `kubectl delete job univr-sweep --ignore-not-found; kubectl apply -f sweep-job.yaml`
Results: `s3://621547421844-ap-southeast-4/univr_neuron/univr_sweep_<ts>.tar.gz`

## Shapes

Production frame 1728x4096 never executes as one graph. `--tiles 2x4 --halo 128`
gives 8 tiles, valid extent 864x1024, **padded 992x1152 (edge) and 992x1280
(interior)**. The padded tile is what runs. Five resample sites per forward:

| site | C | HxW at the 992x1280 tile | pixels | calls |
|---|---|---|---|---|
| image warp | 3 | 992x1280 | 1,269,760 | 6 of 14 |
| ctx L1 | 16 | 496x640 | 317,440 | 2 |
| ctx L2 | 32 | 248x320 | 79,360 | 2 |
| ctx L3 | 64 | 124x160 | 19,840 | 2 |
| ctx L4 | 128 | 62x80 | 4,960 | 2 |

## MEASURED, production tile, pure torch

| op | shape | active_us | gpsimd% | sw_packets | pkt/px |
|---|---|---|---|---|---|
| **gather (BASELINE)** | 3x992x1280 | **49,280** | 99.2 | 1,278,496 | 1.0069 |
| gridsample | 3x992x1280 | 144,462 | 99.7 | 3,815,520 | 3.0049 |
| gather | 32x248x320 | 3,635 | 90.6 | 85,008 | 1.0712 |
| gridsample | 32x248x320 | 112,483 | 99.7 | 3,050,688 | 38.4411 |
| gather | 64x124x160 | 1,701 | 91.1 | 33,888 | 1.7081 |
| gridsample | 64x124x160 | 73,316 | 99.8 | 2,031,616 | 102.4000 |
| gather | 128x62x80 | 367 | 86.3 | 8,848 | 1.7839 |
| gridsample | 128x62x80 | 37,207 | 99.8 | 1,016,576 | 204.9548 |
| shift (1 static shift) | 3x992x1280 | **657** | 20.1 | 12,336 | 0.0097 |
| transpose only | 3x992x1280 | 57 | 22.0 | 2,304 | 0.0018 |
| window1 (9 dense terms) | 3x992x1280 | 112,756 | 35.3 | 994,736 | 0.7834 |
| window2 (25 dense terms) | 3x992x1280 | 160,421 | 23.3 | 997,488 | 0.7856 |

`gather` at C=16 496x640 FAILS to compile: `NCC_EBIR033`, `in=5079040 out=128`
(= 317,440 px x 16). A one-op pure-torch reproducer of the same compiler error that
also blocks whole-model single-graph compiles.

### What that establishes

* GpSimd holds 86-99.8% of device time on every resample. MFU 0.52%, MBU 0.0026%,
  so the bound is **descriptor generation**, not compute or bandwidth.
* ns/descriptor is 36-45 across all shapes and both implementations -- flat, so cost
  is per descriptor rather than per byte.
* `grid_sample` is 3-101x worse than the explicit gather. Its packets/pixel tracks C
  (3.0 at C=3, 205 at C=128) while gather stays near 1. Stock PyTorch is the
  pathological implementation here.
* A static shift is 75x cheaper than the gather with 104x fewer descriptors, so
  shifted access is not the problem.
* But the dense window is 2.3-3.3x WORSE than the gather and still issues ~1 packet
  per pixel despite having NO indirect access. Combining static shifts in torch
  reintroduces per-pixel descriptors. Suspected cause is the per-term elementwise
  weight; UNMEASURED.
* `window2` (25 terms) issues the SAME packets as `window1` (9 terms) but takes 1.4x
  longer, so within that arm time is not tracking descriptor count. Unexplained.

## Flow displacement, measured by capturing what the model produces

| tile | max abs | mean | bulk shift | resid > 2 px |
|---|---|---|---|---|
| 256x384 | 2.02 px | 0.47 | (0,0) | 0.01% |
| **992x1280** | **2.33 px** | 0.50 | **(0,0)** | 0.43% |

The bundle's 43.28 px figure does NOT hold for this frame pair. Two consequences:
the bulk shift buys nothing, so `shiftmatmul` degenerates to `window2`; and the
needed neighbourhood is tiny, which is what makes the SBUF-resident kernel possible.

Never test a bulk-shift decomposition on random flow. Random flow gave 95% of pixels
exceeding a 2 px residual and would have rejected the design; real optical flow is
smooth.

## NOT MEASURED: the two candidate arms

`shiftmatmul` (torch) and `nkishift` (NKI) have NEVER produced a number. Four
attempts, four different failures, each fixed:

1. `shiftmatmul`: `int(...item())` inside the traced region ->
   `Unsupported builtin function: trunc`. Bulk shift now computed on the host.
2. `nkishift`: `dma_copy dst partition dimension 132 exceeds maximum 128`. Band load
   is rows+2R partitions, so the output band is now `P_MAX - 2R` = 124.
3. `shiftmatmul`: `torch.stack` of 25 tensors lowered to `stablehlo.concatenate` on
   the wrong axis (1984 = 2x992). Now accumulates in place, no stack.
4. `nkishift`: hand-built weight access pattern resolved against the whole `[T,H,W]`
   tensor -> `NCC_EBIR033 in=11280384`. Now `wts[t, nl.ds(r0, rows), :]`. Also cut
   SBUF from 82 to 51 KB/partition.

Two bugs were also found by re-reading, both of which would have produced silently
WRONG output rather than an error: the x offset was never applied (all 25 terms
sampled the same column), and the per-channel weight broadcast had a shape mismatch.

`HEAD` has never been run. The next run is the first test of fixes 3 and 4.

## Kernel design and its validation

`nki_shift_warp.py`, per band of `P_MAX - 2R` output rows:
one static DMA loads (rows+2R) x (W+2R) x C into SBUF; for each of (2R+1)^2 integer
offsets a STATIC slice is multiplied by a broadcast weight plane and accumulated on
Vector; one static DMA stores the band. Descriptors per BAND, not per pixel. Weights
are elementwise and host-precomputed, so nothing data-dependent enters an access
pattern.

Why it can work: displacement is +/-2.33 px and the C=3 992x1280 tile is 15.3 MB,
which fits the 24 MB SBUF, so the neighbourhood is already on-chip.

Validated without a device:
* index arithmetic simulated in numpy against `F.grid_sample`: 0.0009 LSB, cos
  1.000000. Covers offsets, padding, banding, channel broadcast.
* every construct checked against `~/KaenaNeuronKernelLibrary/src/nkilib_src/nkilib`.
  `data2=X.ap(...)`, in-place `dst==data1`, and pmax tiling are all idiomatic there.
  My per-channel `affine_range` loop was NOT -- the library broadcasts with a
  stride-0 access dim (`moe_cte_utils.py:780`), which is now used:
  `wplane.ap(pattern=[[W, rows], [1, W], [0, C]])`.

NOT validated: NKI API semantics, SBUF capacity, the validator. `nki` cannot be
imported on this machine. Expect iteration.

`dma_transpose`'s indirect form is deliberately unused: it requires a 2-byte dtype
and production is fp32. bf16 fails the quality gate at 23.31 dB and fp16 overflows
(flat index reaches 1.27M vs fp16's 65504 ceiling).

## Accuracy gate

Every arm now scores against `F.grid_sample` in the same run and prints max_diff LSB,
PSNR, cosine, PASS/FAIL at bar 3, with `max_LSB` and `gate` as summary columns.
Verified it separates arms: `window1` FAILs at 160.19 LSB while `gather`, `window2`
and `shiftmatmul` pass at 0.0019.

Score against the REAL captured flow, not synthetic. `window1`/`window2`/`shift` are
cost probes, not substitutes -- they are wrong at real displacement.

## Next

1. Run `HEAD`. Does `shiftmatmul` compile, and does `nkishift` compile.
2. If `shiftmatmul` runs, expect ~160,000 us (it degenerates to `window2` at bulk
   shift (0,0)), i.e. 3.3x WORSE than the gather. That would confirm the torch-level
   path is dead and the idea needs NKI.
3. Once both produce numbers, build the 3 x 5 matrix: gather / shiftmatmul / nkishift
   across all five sites. NOT built yet. `nkishift` is hardcoded R=2, valid only
   where displacement <= 2 px -- measured at C=3, NOT at the ctx sites.
4. Tile-geometry sweep (10 legal geometries, `gather` only today) is a separate
   question; it matters after a winner exists. Compare ns/pixel, not total.
5. Only after step 1 passes with the gate: wire into `repro_unrolling_trn2.py` and
   re-run `repro-job.yaml`, which scores against `ref_cuda_fp32_1728x4096.npy` at
   `--bar 3`. The resample is ~62% of a forward, so op-level gains dilute.

## Traps that cost runs

* `set -e` kills the script when a `grep`/`find` in a command substitution finds
  nothing, which is normal. Three runs died this way. Every such assignment now has
  `|| true`, and the arm loop is wrapped in `set +e` with `rc=${PIPESTATUS[0]}`.
* `warp_op_bench.py` must `import torch_neuronx` or there is no `neuron` backend and
  no `neuron` device. Standing alone it had neither and died in 5 s.
* `--compile` used to sit under `if tiled:`, so `--tiles 1x1 --cores 1` silently ran
  EAGER and reported success. A whole shape sweep was reported as compiled that way
  and had to be retracted. Compiling is now independent of tiling and every mode
  prints its config.
* `summary-json` is keyed by NEFF hash at the top level, not `"summary"`. Reading it
  wrong printed a page of zeros that looked like real data.
* `window2` at 25 terms OOM-killed a pod at 248 GB once (exit 137). Survived on a
  retry. The archive step is after the loop, so a kill anywhere loses everything.
* Never `git add -A` here: it swept 274 MB and 442 MB `.ntff` files into a commit and
  GitHub rejected the push at its 100 MB hard limit. `.gitignore` now covers
  `*.ntff`, `*.tar.gz`, `univr_*/`.
