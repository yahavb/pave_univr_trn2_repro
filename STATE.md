# Session state

HEAD `7061814`, `main`, pushed. Local `~/pave_univr_trn2_repro`.
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

| shape | px | outcome |
|---|---|---|
| 4x8 h64 512x576 | 294,912 | fuses |
| 4x8 h64 576x640 | 368,640 | fuses |
| 4x8 h128 576x768 | 442,368 | fuses |
| 2x8 h64 928x640 | 593,920 | **OOM at 600Gi, 1000Gi AND 1500Gi** |
| 2x4 h64 928x1088 | 1,009,664 | **OOM at 1500Gi, without the flag, after 147 min** |

Ceiling sits between **~442k and ~594k padded px** and raising memory does not move it --
2x8 died at all three levels. So 2x4 (1.0M px), 2x8 (594k px) and 3x4 (811k px) are all
above it regardless of flags. **4x8 is the only grid entirely under it.**

## Still blocking a shippable result

**The baseline does not reproduce.** Same command as the README gives 4125.5 ms and
**81.46 LSB against a bar of 3** -- localised to tile 0, fails on both `nki` and `nki-dyn`,
predates every change this session. It also died outright in this image three times with
`KeyError: torch_mlir.dialects.builtin`. And the eager path has now been REMOVED from the
script entirely, so it cannot be reproduced here at all. Fused numbers are comparable only
to other fused numbers.

## Next step: the actual number

4x8 halo 64 compiles. What has never been produced is a latency or accuracy figure for it.
Run 8 cores, `--iters 3`, `NEURON_CC_FLAGS="--lnc 1"`, and read the steady-state median
WITH the accuracy gate. `prod-4x8-halo64-job.yaml` is close to this but predates the
--compile removal and the flag finding; check its flags before using it.

Accuracy is not optional: halo 64 is a correctness change (the production halo is 128) and
too little halo fails SILENTLY as edge artefacts that look plausible while disagreeing with
the untiled reference.

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

* **Time and score fused 4x8 halo 64.** The only thing left before there is a result. 8
  cores, `--iters 3`, `NEURON_CC_FLAGS="--lnc 1"`, steady-state median plus the accuracy
  gate. `prod-4x8-halo64-job.yaml` is the closest job but predates both the `--compile`
  removal and the flag finding -- check its flags first.
* **Whether the OTHER grids fuse without the flag.** Not worth chasing: they are above the
  memory ceiling (2x4 1.0M px, 3x4 811k, 2x8 594k) and OOM regardless. Only worth
  revisiting if the compiler's memory behaviour changes.
* **Whether `--model-type transformer` or `generic` is better than dropping it.** All three
  fuse. Untested for output correctness or speed -- `--model-type` may affect scheduling,
  so a fused-and-fast result under `--lnc 1` alone should still be checked against one of
  the others before anyone concludes the flag is purely harmful.
* **`ip-192-168-192-40` S3 mount is broken**, and it is a cluster-owner problem, not a
  kubectl one. The Mountpoint pod and the CSI node driver never complete their
  `/comm/mount.sock` handshake -- the pod waits exactly 120 s, times out, restarts. The
  driver DOES request the mount and the Mountpoint pod IS created, so it is neither a
  missing daemon nor IAM. `.40` differs from the working `.189` in nodegroup
  (`trn3-dev1-48xl-efa` vs `cr1-no-efa`), launch template, AMI, `CAPACITY_BLOCK` capacity
  type, and it lacks the `alpha.eksctl.io/*` labels entirely. Most likely a host
  filesystem or mount-propagation difference on that AMI. The `s3-csi-controller` TLS
  handshake error in its `--previous` log is dated 20 h earlier and is NOT the current
  fault. **Every job is pinned to `.189` as a result** -- see the nodeSelector.

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
| `ccflag-fusion-job.yaml` | the flag sweep that found `--model-type unet-inference`. Done |
| `prod-4x8-halo64-job.yaml` | fused timing. Predates the `--compile` removal and the flag finding -- check its flags |
| `repro-job.yaml` | whole model, original hardcoded `nki` arms |
| `bigtile-noflag-job.yaml` | the job that proved 4x8 halo 64 fuses on all four shapes |
| `fused-config-search-job.yaml` | geometry x warp under fullgraph. Superseded by the above |
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
  unconditionally because neither is traceable under fullgraph.
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
