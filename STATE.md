# Session state

HEAD `0e161c2`, `main`, pushed. Local `~/pave_univr_trn2_repro`.
Origin repo for reference only: `~/pave-unrolling`.

## The goal

Cut 4K frame latency on trn3. Best trn2 number on record is **3673.3 ms** (README,
never reproduced). g6e L40S is 351.1 ms measured, 161 ms with ONNX+TRT.

## Where the time goes — MEASURED, and it is not where the kernel work assumed

Per tile-triplet, device-active, from `conv-bench-job.yaml` and the warp sweep:

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

## Two things blocking everything

1. **The baseline does not reproduce.** Same command as the README gives 4125.5 ms and
   **81.46 LSB against a bar of 3** — localised to tile 0, fails on both `nki` and
   `nki-dyn`, predates every change this session. In the last three jobs the eager arm
   also died outright in this image (`KeyError: torch_mlir.dialects.builtin`). **No
   correct baseline exists, so no speed claim can be checked.**
2. **Whole-forward fusion does not compile at any geometry.**

## Fusion: what was tried and what the error actually means

`--compile whole` = `fullgraph=True` over one tile's forward. Rejected everywhere:

| config | result |
|---|---|
| 2x4, any halo | `in=17,776,640`. Valid extent alone is 884,736 px |
| 2x8 | OOM at 600Gi, 1000Gi **and 1500Gi**, on the first arm with nothing else running |
| 4x8 halo 128 | 3 of 4 shapes fuse, largest (704x768) rejects |
| 4x8 halo 64 | 3 of 4 fuse, `512x576` rejects `in=23,040` |
| 4x8 halo 48 | 0 of 3 fuse |

A frame needs **every distinct padded shape** to compile. Border tiles get their halo
clipped, so a grid has several shapes. 3-of-4 is worth nothing: tiles of the rejected
shape have no graph, so their pixels are never written and the run dies (`rc=139`).

**The error, read correctly:**

```
Number of elements in dimension 0 of InstDMACopy's input and dimension 0 of
output AccessPatterns must MATCH, but got in=17776640 and out=128
```

"Match" means **equal**, not divide. I earlier called this a divisibility problem — that
was wrong, 7 of the 8 rejecting operands divide 128 exactly. `out=128` is the SBUF
partition count and is **constant across every rejection** (8 shapes, 3 operand forms).
The input side is a **flat** pattern that was never tiled to partitions.

**So the fault is in how the backend lowered the fused DMA, not in the tile shape.** That
retires geometry as a lever and explains why halo sweeping gave incoherent results.

Three operand forms, showing the backend fusing different numbers of resamples per shape:

| form | shapes |
|---|---|
| `14 x px` | 992x1280, 704x1280, 832x736, 992x512, 704x768 |
| `4 x px` | 480x608 |
| `45 x h` — no width term, exact | 480x576 (21,600), 512x576 (23,040) |

`45 = 5 x 9`, and the IFBlock `lastconv` emits exactly 5 channels
(`ConvTranspose2d(c, 5, 4, 2, 1)` = flow 4 + mask 1). Plausible origin, unproven.

## Dead — do not retry

* **Geometry / halo sweeps.** Reason above. Two independent pods reproduced the halo-48
  result byte-identically, so these are deterministic verdicts, not flaky runs.
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

* **`ccflag-fusion-job.yaml`** — compiler flags, the only axis never varied. Every run in
  this repo and in `~/pave-unrolling` used `--model-type unet-inference` and nothing else.
  Stage 0 dumps `neuronx-cc --help` so real flag names replace guesses; stage 1 sweeps
  against `4x8 halo 64 tile 0` (`512x576`, `in=23,040`) because the other three shapes at
  that halo already fuse, so one working flag completes the set.
* **`prod-4x8-halo64-job.yaml`** — would time and score fused 4x8 halo 64. Currently
  pointless: its `512x576` shape does not compile.
* **Partial fusion (`--compile stages` / `halves`)** — not built. Smaller graphs never
  trigger the 14-resample DMA. Would give ~10-20 graphs instead of 275 eager NEFFs, so
  most of the dispatch win at much better odds of compiling. **This is the most likely
  route to an actual number.**

## Files

| file | what |
|---|---|
| `repro_unrolling_trn2.py` | the model. Added `--warp shiftwarp`, `--record-flow`, `--shiftwarp-radius/-max-c` |
| `TILING_AND_GRAPH.md` | **read first.** Tiling algorithm, halo, quantisation, the full op graph, why the operand is `px x 14` |
| `SINGLE_GRAPH_NCC_EBIR033.md` | the fusion walls, and that fusion is per TILE not per frame |
| `METHOD.md` / `REPRO_README.md` | original bundle docs |
| `warp_op_bench.py` | one warp op at one shape |
| `conv_op_bench.py` | the 54 convs; `--list` prints the checkpoint-verified inventory |
| `profile_roofline.py` | reads a `summary.json`, prints per-engine time and MFU |
| `sweep-job.yaml` | the warp microbench sweep |
| `conv-bench-job.yaml` | profiles all 54 convs |
| `ccflag-fusion-job.yaml` | the open experiment |
| `prod-4x8-halo64-job.yaml` | fused timing, blocked |
| `repro-job.yaml` | whole model, original hardcoded `nki` arms |
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
* **`backoffLimit` belongs on `Job.spec`**, not `spec.template.spec`, where it is silently
  ignored.
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
