# Method

How we find what bounds the resample on trn2, and how a proposed replacement is
accepted or rejected. Every claim here is either measured and cited, or marked
UNMEASURED.

## 1. Why the op, not the model

`repro_unrolling_trn2.py` runs the whole model: 62 conv/deconv sites and 14 resamples
per forward. A profile of that cannot say what the resample costs, so it cannot guide
a kernel.

`microbench.py` runs ONE op at ONE shape under
`torch.compile(backend="neuron", dynamic=False, fullgraph=True)`. Each invocation
produces one NEFF attributable to that op and shape. No NKI kernels: `gather` is
`index_select`, `gridsample` is `F.grid_sample`, the rest are `roll`/slice/`stack`.
Verified by tracing every executed op — 24 distinct torch ops, zero custom ops.

## 2. Spatial tiling: where the shapes come from

The production frame is **1728x4096**. It never executes at that size — it does not
compile as one graph. `--tiles 2x4 --halo 128` splits it into 8 tiles of valid extent
864x1024, each *read* with a 128 px halo for the receptive field, giving **padded
shapes 992x1152 (edge tiles) and 992x1280 (interior)**. The padded tile is what the
model executes on, so it is what we benchmark.

This is spatial tiling, not sequence parallelism: each tile is an independent region,
one per core, with no collectives between cores. The parent stitches the outputs.

The five resample sites follow from `Contextnet.forward`, which halves resolution and
doubles channels per level:

| site | C | H x W at the 992x1280 tile | pixels | calls per forward |
|---|---|---|---|---|
| image warp | 3 | 992x1280 | 1,269,760 | 6 of 14 |
| ctx level 1 | 16 | 496x640 | 317,440 | 2 |
| ctx level 2 | 32 | 248x320 | 79,360 | 2 |
| ctx level 3 | 64 | 124x160 | 19,840 | 2 |
| ctx level 4 | 128 | 62x80 | 4,960 | 2 |

C=3 992x1280 is the primary target: 6 of the 14 sites and the largest by pixels.

### Tile geometries explored

The tile was inherited, not chosen, so all legal geometries of the frame are swept at
fixed C=3, halo 128, tile dims a multiple of `TILE_ALIGN`=32:

| grid | tiles | padded (interior) | px/tile | tiles x px |
|---|---|---|---|---|
| 1x2 | 2 | 1728x2176 | 3,760,128 | 7,520,256 |
| 1x4 | 4 | 1728x1152 | 2,211,840 | 8,847,360 |
| 2x2 | 4 | 992x2176 | 2,158,592 | 8,634,368 |
| **2x4** | **8** | **992x1280** | **1,269,760** | **10,158,080** |
| 3x4 | 12 | 704x1280 | 1,064,960 | 12,779,520 |
| 2x8 | 16 | 992x768 | 761,856 | 12,189,696 |
| 4x4 | 16 | 576x1280 | 901,120 | 14,417,920 |
| 4x8 | 32 | 576x768 | 540,672 | 17,301,504 |
| 6x8 | 48 | 416x768 | 417,792 | 20,054,016 |
| 4x16 | 64 | 576x512 | 360,448 | 23,068,672 |

The last column is the trade: total pixel-work grows with tile count because every
tile re-reads its halo — 4x16 does 2.3x the work of 2x4 for the same frame. Against
that, fewer tiles means larger graphs, and large graphs stop compiling. The
comparable metric across geometries is **ns per pixel**, not total time.

## 3. What is measured, and what is not

Wall-clock is not used. At single-op scale it is dominated by dispatch and the host
read barrier, so it measures the harness. Metrics come from
`neuron-profile capture --single-io` then `view --output-format summary-json`:

| metric | source |
|---|---|
| `total_active_time` | device busy time |
| `<engine>_active_time` | per-engine share; GpSimd is where software DMA descriptors are built |
| `software_dynamic_dma_packet_count` | descriptors issued |
| descriptors per output pixel | derived |
| ns per descriptor | derived: active_time / packets |

`dma_transfer_total_bytes` reads 0 in these profiles, so achieved GB/s and the
bandwidth axis of the roofline are NOT available. Unexplained.

Each point also writes a `<op>_<CxHxW>/` directory holding the `.neff` + `.ntff`
pair plus `summary.json`, uploadable to neuron-explorer as Individual Files.

## 4. Measured so far, production tile

| op | shape | active_us | gpsimd% | sw_packets | pkt/px |
|---|---|---|---|---|---|
| gather | 3x992x1280 | **49,280** | 99.2 | 1,278,496 | 1.0069 |
| gridsample | 3x992x1280 | 144,462 | 99.7 | 3,815,520 | 3.0049 |
| gather | 32x248x320 | 3,635 | 90.6 | 85,008 | 1.0712 |
| gridsample | 32x248x320 | 112,483 | 99.7 | 3,050,688 | 38.4411 |
| gather | 128x62x80 | 367 | 86.3 | 8,848 | 1.7839 |
| gridsample | 128x62x80 | 37,207 | 99.8 | 1,016,576 | 204.9548 |
| shift (1 static shift) | 3x992x1280 | **657** | 20.1 | 12,336 | 0.0097 |
| transpose only | 3x992x1280 | 57 | 22.0 | 2,304 | 0.0018 |
| window1 (9 dense terms) | 3x992x1280 | 112,756 | 35.3 | 994,736 | 0.7834 |
| window2 (25 dense terms) | 3x992x1280 | 160,421 | 23.3 | 997,488 | 0.7856 |

Conclusions from that table:

* **GpSimd holds 86-99.8% of device time on every resample.** Compute and bandwidth
  are near zero (MFU 0.52%, MBU 0.0026%), so the bound is descriptor generation.
* **ns per descriptor is 36-45 across all shapes and both implementations** — flat, so
  cost is per descriptor rather than per byte.
* **`grid_sample` is 3-101x worse than the explicit gather**, and its packets/pixel
  tracks C (3.0 at C=3, 205 at C=128) while gather stays near 1. Stock PyTorch is the
  pathological implementation here.
* **A static shift is 75x cheaper than the gather** (657 vs 49,280 us) with 104x fewer
  descriptors. Shifted access is not the problem.
* **But the dense window is 2.3-3.3x WORSE than the gather** and issues ~1 packet per
  pixel despite having no indirect access at all. So combining static shifts in torch
  reintroduces per-pixel descriptors — the suspected cause is the per-term elementwise
  weight, UNMEASURED.
* `window2` (25 terms) issues the same packets as `window1` (9 terms) but takes 1.4x
  longer, so within that arm time is NOT tracking descriptor count. Unexplained, and it
  weakens the descriptor-bound story for the static path specifically.

`gather` at C=16 496x640 fails to compile: `NCC_EBIR033`, `in=5079040 and out=128`
(= 317,440 px x 16). A one-op pure-torch reproducer of the compiler error that also
killed whole-model single-graph compiles.

## 5. Acceptance criteria for a replacement

A candidate must be BOTH faster and correct. Speed alone is not a result — three of
the arms above are fast and wrong.

**Correctness**, scored against `F.grid_sample` on the REAL flow the model produces at
that site, not synthetic flow:

* `max_diff <= 3 LSB` — the bar the product ships against
* report PSNR and cosine alongside

Synthetic flow invalidates this test. A bulk-shift decomposition checked against
uniform random flow showed 95% of pixels exceeding a 2 px residual, which would have
rejected a viable design. The model's real flow at the C=3 site has max displacement
2.02 px and 0.00% of pixels exceeding 2 px after a bulk shift. Real optical flow is
smooth; random tensors are not.

**Speed**, all four together — the first three can improve while total time gets
worse, which is exactly what `window1` did:

| column | baseline | target |
|---|---|---|
| `active_us` | 49,280 | lower |
| `gpsimd%` | 99.2 | < 20 |
| `sw_packets` | 1,278,496 | < 10,000 |
| `pkt_per_px` | 1.0069 | < 0.01 |

## 6. Sequence

1. Op faster in isolation, at the production tile, with the accuracy gate passing.
2. Only then wire it into `repro_unrolling_trn2.py` and re-run the whole-model job,
   which scores against `ref_cuda_fp32_1728x4096.npy` at `--bar 3`.
3. A gain at step 1 that does not survive step 2 is not a gain. The resample is ~62%
   of a forward, so op-level speedup is diluted at the model level.

## 7. Candidate under test: shiftmatmul

One global integer bulk shift (`torch.roll`), then a dense `(2R+1)^2` sum of STATIC
slices times precomputed weights over the small residual. No indirect addressing.

Replaces, at C=3 992x1280:

| before | after |
|---|---|
| `reshape`+`permute` to `[N,C]` | removed |
| `floor` x2, flat index `y*W+x` x4 | `roll` (bulk shift) |
| **`index_select` x4 — the gather** | 25 static slices of a padded base |
| 4-term weighted blend | `stack` + `(S*Wt).sum(0)` |

Address math (`arange`, `+flow`, `clamp`, weight products) is unchanged.

Accuracy vs `F.grid_sample` on real captured flow at 256x384: R=2 gives 1.5082 LSB
(PASS), R=3 gives 0.0039 LSB (equivalent). **Both UNMEASURED at the production tile,
and the bundle records max displacement of 43.28 px at 4K against the 2.02 px seen at
256x384 — so R may need to be much larger there, or the arm may fail the gate
outright.**

Performance: PENDING. `window2` at the same 25 terms cost 160,421 us, 3.3x the
gather, so if `shiftmatmul` lands near that the torch-level path is dead and the idea
requires NKI, where descriptors are under direct control.

## 8. Reproduce

```bash
kubectl delete job univr-sweep --ignore-not-found
kubectl apply -f sweep-job.yaml
kubectl logs -f job/univr-sweep
```

Results archive to `/var/mdl/univr_neuron/univr_sweep_<ts>.tar.gz`, which is
`s3://621547421844-ap-southeast-4/univr_neuron/`. Re-read any captured profile with:

```bash
python3 profile_roofline.py <dir>/summary.json <label> <pixels> <tap_bytes>
```
