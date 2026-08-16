# Session state

HEAD `c9399d3`, `main`, pushed. Local `~/pave_univr_trn2_repro`.
Origin repo for reference only: `~/pave-unrolling`.

## The goal

Cut 4K frame latency on trn3. Best trn2 number on record is **3673.3 ms** (README,
never reproduced). g6e L40S is 351.1 ms measured, 161 ms with ONNX+TRT.

## Where the time goes — MEASURED, and it is not where the kernel work assumed

Per tile-triplet, device-active, from the microbenchmark:

| | ms | share of device-active |
|---|---|---|
| 14 warps | 614.2 | 63.1% |
| 54 convs | 359.0 | 36.9% |
| **device-active total** | **973.2** | 100% |
| **measured frame wall** | **4125** | device is only **23.6%** |

**~76% of the frame is dispatch and idle.** The resample is 63% of *device* work but
~15% of *wall*, which is why the NKI kernel's 1.16x op-level win vanished end to end.
Eager caches **275 NEFFs** per frame; that call count is the dominant cost.

Caveat: the 973 ms is a sum of single-op profiles, which may undercount concurrency, so
23.6% is an upper bound on device utilisation.

## SOLVED: a fully fusable configuration exists

**4x8 halo 64 with `NEURON_CC_FLAGS="--lnc 1"` compiles ALL FOUR of its padded shapes**
under `torch.compile(backend="neuron", dynamic=False, fullgraph=True)`. Measured, job
`univr-bigtile-noflag-lb62x`:

| tile | padded | px | verdict | compile |
|---|---|---|---|---|
| 0 | 512x576 | 294,912 | FUSES | 117 min |
| 1 | 512x640 | 327,680 | FUSES | 27 min |
| 8 | 576x576 | 331,776 | FUSES | 85 min |
| 9 | 576x640 | 368,640 | FUSES | 24 min |

`ALL SHAPES FUSE = 1`. 4.2 h for the set. 32 tiles, 4 distinct shapes, so ~8 graphs per
frame (4 shapes x 2 triplet timestamps) against **275 NEFFs** eager. Total padded pixel
work is 1.13x the 2x4 halo-128 baseline.

This is the first configuration that can actually run a frame fused. Compile cost is a
one-off the NEFF cache absorbs and is **not** the critical path; steady-state latency is.

### Two independent causes had to be separated to get here

**1. `--model-type unet-inference` caused the DMA rejection.** Perfect separation across
8 arms on one fixed shape (4x8 h64 tile 0, 512x576), job `univr-ccflag-fusion-fc4d6`:

| flags | verdict |
|---|---|
| `--lnc 1` | **FUSES** |
| `--lnc 1 --model-type transformer` | **FUSES** |
| `--lnc 1 --model-type generic` | **FUSES** |
| `--lnc 1 --model-type unet-inference` | REJ `in=23,040` |
| ... `unet-inference --auto-cast=none` | REJ `in=23,040` |
| ... `unet-inference -O2` | REJ `in=23,040` |
| ... `unet-inference --enable-saturate-infinity` | REJ `in=23,040` |
| `--model-type unet-inference` (no `--lnc`) | REJ `in=23,040` |

Every arm carrying `unet-inference` rejected with the identical operand; every arm without
it fused. `--lnc` is irrelevant. The VALUE is what matters, not the flag -- `transformer`
and `generic` both work. That flag came from `~/pave-unrolling`, was never varied there
either, and was carried unquestioned into every run including all the geometry and halo
sweeps. **So those sweeps were varying tile shape while the cause sat in one flag**, which
is why the operand values looked incoherent.

**2. Compiler HOST MEMORY is a separate ceiling, set by tile size, and the flag does not
touch it.**

| shape | px | outcome | compiles |
|---|---|---|---|
| 4x8 h64 512x576 | 294,912 | fuses | serial |
| 4x8 h64 576x640 | 368,640 | fuses | serial |
| 4x8 h128 576x768 | 442,368 | fuses | serial |
| 2x8 h64 928x640 | 593,920 | **OOM at 600Gi, 1000Gi AND 1500Gi** | serial |
| 2x4 h64 928x1088 | 1,009,664 | **OOM at 1500Gi, without the flag, after 147 min** | serial |

Ceiling sits between **~442k and ~594k padded px** and raising memory does not move it --
2x8 died at all three levels. So 2x4 (1.0M px), 2x8 (594k px) and 3x4 (811k px) are all
above it regardless of flags. **4x8 is the only grid entirely under it.**

**Read the `compiles` column: every one of those is a SERIAL, one-graph-at-a-time
measurement** (`--cores 1 --only-tile T`). That is what makes them comparable to each other
and what makes the ceiling a per-compile number. An OOM from a run that compiled several
graphs at once says nothing about it -- see the accuracy-triage OOMs below.

## Still blocking a shippable result

**The baseline does not reproduce.** Same command as the README gives 4125.5 ms and
**81.46 LSB against a bar of 3** -- localised to tile 0, fails on both `nki` and `nki-dyn`,
predates every change this session. It also died outright in this image three times with
`KeyError: torch_mlir.dialects.builtin`. And the eager path has now been REMOVED from the
script entirely, so it cannot be reproduced here at all. Fused numbers are comparable only
to other fused numbers.

## MEASURED: fused latency, and it fails accuracy

`univr-prod-4x8-h64-vggks`, 4x8 halo 64, 8 cores, `--iters 3`, `NEURON_CC_FLAGS="--lnc 1"`:

| arm | median | max_diff | PSNR | NEFFs |
|---|---|---|---|---|
| gather | **3900.5 ms** (3871.4-3946.7) | 92.56 LSB | 48.74 dB | +7 |
| nki-dyn | **3820.4 ms** (3801.4-3836.4) | 92.56 LSB | 48.74 dB | +8 |

Against 3673.3 ms in the README and 4125.5 ms for the same command re-run here. So fusion
RUNS and sits between them -- not a win yet, not a disaster. Both fail the 3-LSB gate.

**The accuracy is bit-identical across two different resample implementations** -- 92.56 LSB
and 48.74 dB to every digit, from plain-torch indirect gather and from an NKI indirect-DMA
kernel. The fault is in code they SHARE, not in either warp.

**And it is not halo starvation**, which is what dropping halo 128 -> 64 would predict. The
spatial breakdown puts the worst error in the INTERIOR:

```
merge seam +/-4 rows   max 28.82
top border 32 rows     max 17.37
bottom border 32 rows  max 43.15
interior (128 cut)     max 92.56   <-- worst
```

Sparse rather than systematic: p50 0.14 LSB, mean 0.356, 1.49% of pixels above 3 LSB,
cosine 0.999968. Isolated pixels going badly wrong throughout the frame.

Note also `d2h` was **91.4%** of the summed per-core time (22,498 ms of 24,627) against `dev`
at 4.4%. The script's own caveat says `dev+d2h` is reliable as a sum and the split only
indicative, but device work is clearly a small fraction of that wall clock.

**`gridsample` cannot compile fused.** OOM-killed at 169 min and again at 94 min at 1500Gi,
on the same geometry where gather and nki-dyn both compile. Consistent with the
microbenchmark: it issues 3-205x more DMA descriptors than gather (3.0 pkt/px at C=3 rising
to 205 at C=128), so its fused graph is far larger. It cannot serve as the reference
implementation under fusion.

Two other host-side caps found along the way, both fixed:
* `FailOnRecompileLimitHit` -- dynamo's `cache_size_limit` defaults to **8**, and this model
  needs exactly 8 graphs (4 padded shapes x 2 triplet timestamps), so it sat on the limit.
  One tile never trips it; 8 cores always does. Now 64 / 256, set at import.
* `backoffLimit` under `spec.template.spec` -- the API server rejects the manifest outright
  in one job and **silently ignores it** in another. It belongs on `Job.spec`.

## The rejection history, ALL of it measured WITH `--model-type unet-inference`

Every verdict below carried the bad flag, so each is a statement about that flag plus the
geometry, not about the geometry alone. Kept because the operand values are still the only
evidence about how the backend lowers the fused DMA.

| config | result |
|---|---|
| 2x4, any halo | `in=17,776,640`. Valid extent alone is 884,736 px |
| 2x8 | OOM at 600Gi, 1000Gi and 1500Gi |
| 3x4 halo 128 | `in=12,615,680` |
| 3x9 halo 128 | `in=8,572,928` |
| 2x16 halo 128 | `in=7,110,656` |
| 2x12 halo 128 | `EBVF030`, 5,983,171 instructions vs a 5,000,000 cap |
| 4x8 halo 128 | 3 of 4 shapes fuse, largest (704x768) rejects `in=7,569,408` |
| 4x8 halo 64 | 3 of 4 fuse, `512x576` rejected `in=23,040` -- **now fuses without the flag** |
| 4x8 halo 48 | 0 of 3 fuse, `in=21,600` and `in=1,167,360` |

A frame needs **every distinct padded shape** to compile. Border tiles get their halo
clipped, so a grid has several shapes. 3-of-4 is worth nothing: tiles of the rejected shape
have no graph, so their pixels are never written and the run dies (`rc=139`).

**The error text, read correctly:**

```
Number of elements in dimension 0 of InstDMACopy's input and dimension 0 of
output AccessPatterns must MATCH, but got in=17776640 and out=128
```

"Match" means **equal**, not divide. I called it a divisibility problem at one point and
that was wrong -- 7 of the 8 rejecting operands divide 128 exactly. `out=128` is the SBUF
partition count and is constant across every rejection (8 shapes, 3 operand forms). The
input side is a flat pattern that was never tiled to partitions.

Three operand forms, showing the backend fusing different numbers of resamples per shape:

| form | shapes |
|---|---|
| `14 x px` | 992x1280, 704x1280, 832x736, 992x512, 704x768 |
| `4 x px` | 480x608 |
| `45 x h` -- no width term, exact | 480x576 (21,600), 512x576 (23,040) |

`45 = 5 x 9`, and the IFBlock `lastconv` emits exactly 5 channels
(`ConvTranspose2d(c, 5, 4, 2, 1)` = flow 4 + mask 1). Plausible origin, unproven.

**Two independent pods reproduced the halo-48 result byte-identically**, so these are
deterministic compiler verdicts rather than flaky runs.

## Dead — do not retry

* **Geometry / halo sweeps at OTHER grids.** Not because geometry is irrelevant -- it was
  retired on the reasoning that the backend's DMA lowering could not be influenced by tile
  shape, and that reasoning was built on rejections caused by `--model-type
  unet-inference`. What actually retires them now is the MEMORY ceiling: 2x4, 2x8 and 3x4
  are all above ~594k padded px and OOM regardless of flags. 4x8 is the only grid under it,
  and it fuses.
* **2x8.** OOM at 1500Gi on its own. Memory does not scale it: 600 -> 1000 -> 1500Gi all
  died. Its non-OOM failure was the instruction cap, 5,988,611 vs 5,000,000.
* **`-O1`** — causes a distinct `in=4` / `I-3276` rejection.
* **`-O3`** — changed the instruction count by **exactly zero** on 2x8 and 2x12.
* **Shrinking the tile to cut instructions** — 2x8 761,856 px gave 5,988,611 and 2x12
  634,880 px gave 5,983,171: a 17% pixel cut bought 0.09% fewer instructions. The count
  is fixed by conv/U-Net structure.
* **bf16** — cut instructions 13.5% (5,180,608), still 3.6% over the cap, and fails the
  quality gate at 23.31 dB.
* **The NKI resample kernel (`shiftwarp`).** 1.16x at the op level with identical
  accuracy (42,439 vs 49,278 us), then **229.72 LSB** in the model. `--record-flow`
  measured real displacement at **29.02 px** across all 112 resample calls; the kernel
  covers R px and silently clamps beyond it. Covering 29 px needs R=30 = 3,721 terms
  against 49 at R=3. The design rested on a 2.33 px figure that was ONE call at ONE tile.
  Code is still in the repo and selectable via `--warp shiftwarp`; do not use it.

## Open

* **Why the fused frame fails accuracy at 92.56 LSB.** The live question.
  `accuracy-triage-job.yaml` runs `gather` at halo 128 -- the only remaining variable, since
  gridsample cannot compile fused and the two warps already agree to every digit. That job
  also produces the system-profile bundle.

  **Two attempts OOMKilled and BOTH were wasted on a self-inflicted confound**:
  `univr-accuracy-triage-rh7rz` at 8 h and `-jcbd4` at 5 h 32 m. Both logs stop at
  `8 replica(s) built` with no compile output, and both entered with the cache empty
  (`find /tmp/neff_cache -name '*.neff' | wc -l` -> `0`). **`torch.compile` is lazy** --
  that line only means eight wrappers exist; neuronx-cc runs on first call, inside
  `run_tiled`, which drives cores from `ThreadPoolExecutor(max_workers=ncore)`. So
  `--cores 8` on a cold cache ran **eight concurrent compiles** of 442k-540k px graphs in one
  container at 1500Gi. Peak host memory was the sum of eight, not one.

  So 540,672 px was never tested against the ceiling. Its only other verdict is a DMA
  rejection (`in=7,569,408`) earned **with** `--model-type unet-inference`. The job now runs
  a serial probe first (8 graphs, largest first, shared cache) and gates the 8-core scored arm
  on it, at 1800Gi to match `prod-4x8-halo64`, the only 8-core run that ever completed.

* **A frame needs 8 graphs and each tile compiles exactly ONE.** `run_tiled` picks the pass
  per tile -- `need_f = (oy+vy) > H/2`, `need_b = oy < H/2` -- and `H/2 = 864` falls exactly on
  the row1/row2 boundary (4 rows x 432), so no tile ever needs both. From
  `plan_tiles(1728, 4096, 4, 8, 128)` the eight (shape x pass) slots and one representative
  tile each, largest first: `9`/`17` 704x768 540,672 px, `8`/`16` 704x640 450,560,
  `1`/`25` 576x768 442,368, `0`/`24` 576x640 368,640. This is why `prod-4x8-halo64`'s serial
  warm-up of tile 9 alone still left **+7 NEFFs** for the 8-core run to compile concurrently:
  one tile seeds one of eight slots. A prewarm must cover all eight or the confound survives.
* **The system-profile bundle for Neuron Explorer.** `accuracy-triage-job.yaml` stages 3-4
  build the directory-upload format: `trace_info.pb` (required), the host `.pb` files,
  `.neff`, `.ntff` with a backfill, plus the source and a PROVENANCE.txt. `trace_info.pb`
  comes from the runtime INSPECT env vars, NOT from a capture command -- capture only ever
  emits `.ntff`. Expect the `.ntff` slot to need the backfill: the forward dispatches async
  and the runtime device-profiler emits nothing for async workloads.
* **Re-measure the microbenchmark at the fused shapes.** Every existing per-op number --
  49,278 us for gather, ~38 ns/descriptor, 2.0 packets/pixel -- was taken at 992x1280 under
  `--model-type unet-inference`. Both are now wrong for the running config: the fused shapes
  are 294k-368k px, and the flag is gone. `microbench-job.yaml` defaults to the four fused
  shapes. **Whether the flag changed single-op numbers is untested** -- it demonstrably
  changes DMA lowering at whole-graph scale. One arm settles it: `gather` at one shape, flag
  on vs flag off, compare descriptor counts. Same means the old numbers stand; different
  means they are void.
* **`--model-type transformer` / `generic` also fuse.** Untested for correctness or speed.
  Worth one check before concluding the flag is purely harmful, since it may affect scheduling.
* **The other grids under the memory ceiling.** Not worth chasing: 2x4 1.0M px, 3x4 811k,
  2x8 594k all OOM regardless of flags, and 2x8 died at 600, 1000 AND 1500Gi.
* **`ip-192-168-192-40` S3 mount is broken** -- a cluster-owner problem, not a kubectl one.
  The Mountpoint pod and the CSI node driver never complete their `/comm/mount.sock`
  handshake: the pod waits exactly 120 s, times out, restarts. The driver DOES request the
  mount and the Mountpoint pod IS created, so it is neither a missing daemon nor IAM. `.40`
  differs from the working `.189` in nodegroup (`trn3-dev1-48xl-efa` vs `cr1-no-efa`), launch
  template, AMI, `CAPACITY_BLOCK` capacity type, and lacks the `alpha.eksctl.io/*` labels.
  Most likely a host filesystem or mount-propagation difference on that AMI. The
  `s3-csi-controller` TLS error in its `--previous` log is 20 h stale and NOT the cause.
  **Every job is pinned to `.189`** -- `s3-mount-test-pod.yaml` checks a node in 85 s.

## Files

| file | what |
|---|---|
| `repro_unrolling_trn2.py` | the model. Added `--warp shiftwarp`, `--record-flow`, `--shiftwarp-radius/-max-c` |
| `TILING_AND_GRAPH.md` | **read first.** Tiling algorithm, halo, quantisation, the full op graph, why the operand is `px x 14` |
| `SINGLE_GRAPH_NCC_EBIR033.md` | the fusion walls, and that fusion is per TILE not per frame |
| `METHOD.md` / `REPRO_README.md` | original bundle docs |
| `microbench.py` | the microbenchmark: 14 resamples + 54 convs, one op per invocation, each scored against a CPU reference |
| `profile_roofline.py` | reads a `summary.json`, prints per-engine time and MFU |
| `microbench-job.yaml` | runs it. `SET=warps|convs|both`, defaults to the four 4x8 halo-64 shapes |
| `ccflag-fusion-job.yaml` | the flag sweep that found `--model-type unet-inference`. Done |
| `prod-4x8-halo64-job.yaml` | the fused timing run that produced 3900.5 / 3820.4 ms |
| `repro-job.yaml` | whole model, original hardcoded `nki` arms |
| `bigtile-noflag-job.yaml` | the job that proved 4x8 halo 64 fuses on all four shapes |
| `fused-config-search-job.yaml` | geometry x warp under fullgraph. Superseded |
| `accuracy-triage-job.yaml` | **the live job.** 4 stages: serial 8-graph compile probe, then the gated 8-core scored arm, then the Neuron Explorer system-profile bundle |
| `s3-mount-test-pod.yaml` | 85-second PVC mount check on a chosen node |
| `nki_shift_warp.py` | the dead kernel, kept for the measured op-level result |

Deleted this session as dead ends: `profile-model-job.yaml`, `compile-modes-job.yaml`,
`fused-4x8-timing-job.yaml`, `halo-shapeset-job.yaml`, `halo64-fusion-job.yaml`,
`fusion-threshold-job.yaml`, `single-graph-retest-job.yaml`, `settle-2x8-job.yaml`,
`AB_RUN.md`, `profile_hotspots.py`.

## Traps that cost runs

* **`kubectl apply` uses the on-disk YAML; the pod clones the script from git.** A pod can
  run new code with a stale command block. Grep the config echo lines, not just the commit.
* **`kubectl set env job/...` cannot patch a Job** — `spec.template` is immutable.
  `--dry-run=client` does NOT catch it; only `--dry-run=server`. Edit the YAML and re-apply.
* **`backoffLimit` belongs on `Job.spec`**, not `spec.template.spec`. Both failure modes
  happened and only one is loud: on `bigtile-noflag` the API server REJECTED the manifest
  (`strict decoding error: unknown field "spec.template.spec.backoffLimit"`) so nothing ran;
  on `halo-shapeset` it was accepted SILENTLY and the retry simply did not exist.
* **`--compile` no longer exists.** The script is always
  `torch.compile(m, backend="neuron", dynamic=False, fullgraph=True)`. There is no eager
  path, no `whole`/`halves`/`stages`. `--per-block` and `--record-flow` now error
  unconditionally because neither is traceable under fullgraph. Consequence: the eager
  figures (3673.3 ms README, 4125.5 ms re-run) can no longer be reproduced here at all, so
  fused numbers are comparable only to other fused numbers.
* **`torch.compile` is LAZY, so `--cores N` on a cold cache means N CONCURRENT compiles.**
  `8 replica(s) built` is printed by `apply_compile` and means only that eight wrappers exist;
  neuronx-cc runs on first call, inside `run_tiled`'s `ThreadPoolExecutor(max_workers=ncore)`.
  Peak host memory is the sum across concurrent compiles, so a cold `--cores 8` run OOMs at
  shapes that compile fine one at a time. This cost **13.5 h across two pods** and produced an
  OOM that was then nearly mis-recorded as a shape-ceiling result. **Always seed the NEFF cache
  serially before going wide**, and check the `NEFFs +N` delta: `+0` proves the cache was warm.
* **Every `FUSES` verdict in this file is a serial `--cores 1 --only-tile T` measurement.**
  Do not compare an OOM from a multi-graph run against them; it is not the same experiment.
* **Dynamo's `cache_size_limit` defaults to 8** and this model needs exactly 8 graphs
  (4 padded shapes x 2 triplet timestamps), so it sat on the limit -- one tile never trips it,
  8 cores always does, and under fullgraph it is a hard error. Set to 64 / 256 at import.
  It is a HOST trace counter: unrelated to cores or device memory, and reducing `--cores`
  would only double wall clock.
* **Names drifted from what the files do**, twice. `warp_op_bench.py` benchmarked warps AND
  convs; `sweep-job.yaml` ran the whole microbenchmark. Now `microbench.py` and
  `microbench-job.yaml`. `conv_op_bench.py` and `conv-bench-job.yaml` are deleted -- they
  differed from the sweep only in which op list they looped over.
* **`NEURON_CC_FLAGS` must NOT contain `--model-type unet-inference`.** It is the single
  cause of the DMA rejection. Use `--lnc 1`.
* **Pin every job to `ip-192-168-192-189`.** `node-type=trn3-dev1` matches more than one
  node and the other one cannot mount the PVC; a job landing there sits Pending on
  `MountVolume.SetUp failed ... DeadlineExceeded` for as long as you let it (74 min in one
  case). Use `s3-mount-test-pod.yaml` to check a node in 85 s instead of hours.
* **`--only-tile 1` is not the largest tile.** Tile 10 for 3x9, tile 9 for 4x8. Testing
  tile 1 produced a false "4x8 FUSES" that stood for several runs.
* **One arm per pod.** Each arm holds its compiled graph, so peak memory is the sum across
  arms. A 2x8 arm killed the pod that would otherwise have timed 4x8.
* **The image tag lies.** `native-pytorch:...sdk2.31.0...` contains `neuronx-cc 2.26.6360.0`.
  Always use `421672808698.dkr.ecr.us-east-1.amazonaws.com/concourse-release-0461d3b:latest`.
* **`--per-block` requires `--compile none`**, so that block table is an EAGER measurement
  and is not comparable to compiled runs.
* **`repro-job.yaml` never passes `--compile`**, so it defaults to eager. The README's
  3673.3 ms is an eager number, and 275 NEFFs is what eager costs.
* **Per-engine `*_active_time_percent` in `summary.json` sum past 100%** — they are
  overlapping busy times, not a partition of time. Do not present them as a decomposition.
* **The saved NEFF+NTFF pairs have never been read** by any code here; `profile_roofline.py`
  reads only the JSON. Every profile conclusion comes from `summary.json` aggregates.
* Never `git add -A`: `.ntff` files are hundreds of MB and GitHub rejects at 100 MB.
