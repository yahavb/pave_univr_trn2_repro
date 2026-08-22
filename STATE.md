# Session state

HEAD `22b2fe6`, `main`, pushed. Local `~/pave_univr_trn2_repro`.
Origin repo for reference only: `~/pave-unrolling`.

## THE EXPERIMENT MATRIX (2026-08-21) -- three dimensions, measured

Everything below is **tile 9 of 4x8 halo 128 (padded 704x768), 1 core, fp32, real weights,
`--gt ref_cuda_fp32_1728x4096.npy`**. One tile, NOT a frame -- do not multiply by 32.

Three independent switches. They were confounded for most of this project; they are not now.

| dimension | flag | values | what it changes |
|---|---|---|---|
| **resize** | `--resize` | `interpolate` / `precomputed` | step 5. `F.interpolate` derives coordinates IN-GRAPH from `scale_factor`, so the reads lower to SWDGE indirect DMA. `resize_precomputed` computes them once on the host (`_resize_taps`, fp64, cached on `(in_sz, f, dtype, device)`) and bakes them as constant index vectors, so the reads lower to static DMA. |
| **warp** | `--warp` | `gridsample` / `gather` | step 3, the pixel move. Both are BILINEAR -- same rule, two spellings. `gridsample` = one `F.grid_sample(mode="bilinear")`. `gather` = 4x `index_select` + the weighted sum. MEASURED equivalent to **0.003 LSB** in fp32. The label "gather" is a misnomer; the op is `index_select`. |
| **fullgraph** | `--fullgraph` | `0` / `1` | how `torch.compile(backend="neuron")` is invoked. Not a step in the model. |

### The 8 cells

| # | resize | warp | fullgraph | latency | max_diff | PSNR | gate |
|---|---|---|---|---|---|---|---|
| 1 | interpolate | gridsample | 0 | **OOMKilled 9 h** | | | |
| 2 | precomputed | gridsample | 0 | never reached | | | |
| 3 | interpolate | index_select | 0 | 1102.6 ms | 22.38 LSB | 48.45 dB | FAIL |
| 4 | precomputed | index_select | 0 | **393.8 ms** | **0.06 LSB** | **102.27 dB** | **PASS** |
| 5 | interpolate | gridsample | 1 | **OOMKilled 10 h** | | | |
| 6 | precomputed | gridsample | 1 | never reached | | | |
| 7 | interpolate | index_select | 1 | 1090.0 ms | 22.38 LSB | 48.45 dB | FAIL |
| 8 | precomputed | index_select | 1 | **387.0 ms** | **0.06 LSB** | **102.27 dB** | **PASS** |

Provenance: 3/4 `univr-taps-ab-fg0-idxsel-nznz6`; 7/8 `univr-taps-ab-fg1-bfxdr` (384.0 ms) and
`univer-precomputed-index-select-fullgraph-qm26g` (387.0 ms, the profiled re-run -- agree to 0.8%);
5/6 `univr-taps-ab-fg1-gridsample-l4q85`; 1/2 `univer-interpolate-gridsample-xg9r6`. Both gridsample
pods died **inside the FIRST arm's compile at 1800Gi**, never printing `forward complete`, so cells
2 and 6 were never reached.

### MEASURED: gridsample cannot be COMPILED at this tile size, at either fullgraph setting

Cells 1 and 5 are the same verdict at both fullgraph values. `--fullgraph 0` does NOT rescue it:
breaks fall where dynamo puts them, and the segments are still large enough to exhaust 1800Gi.

**This corrects `repro_unrolling_trn2.py`'s own `--fullgraph` help**, which claims `0` "is what
makes `--warp gridsample` usable". It is not. The only configuration in which gridsample ever ran is
`--warp-region eager`, where `torch._dynamo.disable` keeps it OUT of the graph. So the eager region
is not merely a profiling convenience -- **it is load-bearing, and the only way gridsample runs at
all.** Fix that help text.

Consequence: **there is no in-graph gridsample baseline and there never can be one.** Any
before/after story must either hold the warp fixed (cell 7 -> 8) or accept the off-cube eager row.

### The three factor effects

* **resize is the whole story: ~2.8x AND the accuracy fix.** 1102.6 -> 393.8 (2.80x) at
  fullgraph=0, 1090.0 -> 387.0 (2.82x) at fullgraph=1. Replicated, and `max_diff` 22.38 -> 0.06 LSB
  both times. Accuracy is bit-identical across all four measured cells, as expected -- neither the
  warp nor fullgraph touches the resize math.
* **fullgraph is worth 1-3%.** 3 vs 7 = 1.1%; 4 vs 8 = 1.7%. fullgraph=1 wins, but it is not the
  lever. **This retires the earlier impression from `654.8 -> 384.0`, which moved two factors at
  once** (fullgraph AND warp) and made fusion look like a 1.7x win.
* **warp: unmeasurable in-graph at either fullgraph, and that IS the verdict.** gridsample does not
  compile. A config that cannot compile cannot scale to 32 tiles, whatever its per-tile number.

**WINNER: cell 8** -- `--resize precomputed --warp gather --fullgraph 1`, 387.0 ms / 0.06 LSB PASS.

**The defensible improvement number is 2.8x** (cell 7 -> 8: one variable, same warp, same
fullgraph, both gated). 1333.9 -> 387.0 = 3.4x is the end-to-end journey but moves three factors
including the graph structure -- quote it as history, not as a measurement.

### PROFILED at cell 8 (`univer-precomputed-index-select-fullgraph-qm26g`)

NEFF `5b98e62ae7189a`, **306.5 ms** of the 387.0 ms wall, `MFU 0.563%` of a 37.1% ceiling.

**The descriptor-starvation regime is GONE.** `diagnose_neff.py` does NOT fire
`DESCRIPTOR-ISSUE BOUND`, because that test needs the busiest engine at >2x the DMA engine:

| | before `009cee17` | cell 8 `5b98e62a` |
|---|---|---|
| gpsimd active | 67% | 48.8% |
| DMA engine active | **11%** | **48.9%** |
| ratio | **6.0x -- starved** | **1.0x -- not starved** |

`rank_neffs.py` still labels it `SWDGE-heavy`; that label keys off dynamic-vs-static share, NOT the
starvation ratio, so it reads as "unchanged" when the regime has in fact changed. Do not diagnose
from the class name.

What owns the time now:

1. **`[HIGH]` PAYLOAD BELOW DMA SATURATION on `software_dynamic`** -- 26.1% of DMA time,
   **3,774,832 packets at 154 B**, against a 2048 B floor: 13x short. (Before: 5,151,200 at 29 B.
   Packets down 1.4x, payload up 5.3x, still short.) The whole-NEFF average of 43,896 B/transfer
   HIDES this. This is the `index_select` warp, whose addresses come from `flow` at runtime and can
   never be precomputed. **The fix is coalescing, not making addresses static.**
2. **`[MED]` TRANSPOSE-HEAVY 13.5%** (`transpose_flops` 21.09G / `hardware_flops` 156.7G). Suspects
   are the two permutes in `warp_gather`: `.permute(0,2,1)` building `src` and `.permute(0,3,1,2)`
   in `tap()`.
3. **`[MED]` SPILLING 14.6%** of SBUF traffic (5.3 GB). Their notes say spill is LINEAR in area with
   no cliff, so this does not argue for smaller tiles.

Tier-2 source attribution is NOT in these artifacts -- `grep -c 'src=' hotspots_*.txt` is 0, the
stamps live in the ingested parquet. Run `analyze-job.yaml` with
`RUN=exp_univer-precomputed-index_select-fullgraph_t9_20260821_215513`.

### Two harness defects found by these runs

* **`trap ... EXIT` does not survive OOMKilled.** That is SIGKILL; traps do not fire, so no run
  folder and no logs. Both gridsample pods left nothing.
* **Suppressing the compile phase blinded a 9 h run.** `... 2>&1 | tee "$R/compile.log" >/dev/null`
  in the `univer-*` jobs sent everything to a file that died with the container. `kubectl logs`
  returned two lines for nine hours of work. Let the compile phase stream to stdout -- for an
  OOM-prone run it is the ONLY observability, precisely because the trap cannot fire.

### MEASURED 2026-08-22: gridsample DOES compile at 288x320, and the resize fix wins far less there

`univer-gridsample-hotspots-small-scqrt`, `--tiles 1x1 --height 288 --width 320`, gridsample IN the
graph, `fullgraph=0`, real weights, `--gate` (no CUDA golden exists at reduced resolution).

| resize | latency | max_diff | PSNR | gpsimd | DMA engine | ratio |
|---|---|---|---|---|---|---|
| interpolate | 759.0 ms | 52.65 LSB | 44.08 dB | 71.1% | 11.6% | **6.1x** |
| precomputed | **466.0 ms** | **0.00 LSB** | **121.91 dB** | 66.8% | 11.9% | **5.6x** |

**1.63x, not 2.8x -- and the starvation is NOT cleared.** All dominant NEFFs stay `SWDGE-heavy` at
gpsimd 72-75%, and `diagnose_neff.py` fires `DESCRIPTOR-ISSUE BOUND on GPSIMD` on BOTH arms.

Why the same fix pays 1.63x here and 2.82x at the production tile: **the resize is only worth what
the warp leaves on the table.**

| | warp descriptor rate | resize saved | share of total |
|---|---|---|---|
| gridsample @ 288x320 | 3.0-205 pkt/px | 293 ms of 759 | 38.6% |
| index_select @ tile 9 | 1.006 desc/px (the floor) | 706 ms of 1090 | 64.8% |

gridsample's own descriptors swamp the graph, so removing `F.interpolate`'s is a smaller fraction.
**The 3x only manifests once the warp is at the descriptor floor** -- i.e. with `index_select`.

A/B on the dominant NEFF of each arm (`--before rank01_65233172 --after rank01_2bff75b9`):

| | before | after |
|---|---|---|
| `total_time` | 380.7 ms | 235.3 ms (-38.2%) |
| `gpsimd_engine_instruction_count` | 237,197 @ 1209 ns | **141,738 @ 1200 ns** (-40%) |
| `software_dynamic` packets | 7,502,256 @ 21 B | 4,475,408 @ 31 B (-40.3%) |

**`gpsimd_engine_instruction_count` is the cleanest single metric for this fix**: it falls 40% while
ns/instruction is FLAT at ~1200, so the saving is purely count. Time (-38.2%) tracks descriptor
count (-40.3%) essentially 1:1 -- in this graph, time IS descriptor count.

**Tooling note.** `diagnose_neff.py --before/--after` prints the same five verdict labels for both
arms and looks like boilerplate. It is not: the labels collide because both arms land in the same
threshold bands, and the A/B code path COMPUTES the per-verdict evidence and then discards it
(`for sev, label, ev, fix in classify(mb)` prints only `sev` and `label`, dropping `ev`). Tier 1
prints that evidence. Until the skill is patched, re-run tier 1 on each pair separately to see the
numbers, or the verdicts are unverifiable by eye.

### Off-cube: `--warp-region eager` is a FOURTH thing, not fullgraph=0

| resize | warp | latency | max_diff | PSNR |
|---|---|---|---|---|
| interpolate | gridsample + eager region | 1333.9 ms | 22.38 LSB | 48.45 dB |
| precomputed | gridsample + eager region | 654.8 ms | 0.05 LSB | 102.22 dB |

`--warp-region eager` wraps the warp in `torch._dynamo.disable` (`:446`), so dynamo refuses to
trace into it: the warp runs as eager ATen **on device** while the convs around it stay compiled.
That forces `fullgraph=0` (`:1216`) because the region boundary IS a graph break. So these two rows
are NOT cells 1-2 -- plain `--fullgraph 0` leaves the warp compiled and lets dynamo pick the break
points. Pod `univr-taps-ab-h6gd8`.

**The implication only runs one way: eager region => fullgraph=0, never the converse.**

Why the option exists at all, and only for the warp: the warp's addresses come from `flow`, a
runtime tensor, so they can NEVER be precomputed -- and `gather` already measures 1.006 desc/px
against a printed floor of 1.0. When an op cannot be made cheaper, the only remaining variable is
where it executes, hence a region flag with four backends. The resize needed no region because
`scale_factor` is a compile-time constant, so the fix lives inside the graph.

### `--static-resize` is gone

`--static-resize {0,1,2,3}` -> **`--resize {interpolate,precomputed}`** at `83ccc70`.
`1` (pool-down) and `2` (pool+deconv) are deleted along with `resize_down`/`resize_up`: 1 was
measured at 986.9 ms / **22.17 LSB** -- it swaps only the downsample and leaves `F.interpolate` on
the upsample, which IS the accuracy bug, so it can never pass; 2 was never run and existed only to
separate mechanism from coverage. The five legacy job specs keep `STATIC_RESIZE=0|3` as their env
knob and translate it, so their `CFG` labels and PVC cache keys are unchanged.

### Every timed median so far except one contains a compile

`min 392.4 / median 393.8 / max 393412.2`. Three runs in a row. `min ~= median` so the numbers are
real, but the two profiling jobs added at `83ccc70` run `--iters 0` first and print
`NEFFs B -> A during timed run (must be +0)` so it is checkable rather than assumed.

## RESUME HERE

### Jobs in flight (2026-08-20)

| job | what it answers | state when this was written |
|---|---|---|
| `univr-full-taps-1core` (`fg1`) | whole-frame `max_diff` vs **92.56 LSB**, fullgraph=1 | probe done `+0` (adopted the 9-NEFF cache), in `[2/3]` ~5 h, SILENT |
| `univr-full-taps-1core-fg0` | same, fullgraph=0 -- the A/B on whether breaks matter | probe done, all 8 FUSE (8 NEFFs), entering `[2/3]` |
| `univr-2x4-taps-probe` | does 2x4 h128 compile now that taps removed `F.interpolate`? | **YES -- 992x1280 FUSES in 27 and 26 min.** See below |

For each: `grep -aE '^  tile |ALL 8|median=|max_diff=|gate=|NEFFs now' <log>`. Four lines matter.

### MEASURED 2026-08-20: A FRAME NEEDS AT LEAST 15 GRAPHS, NOT 8

`fg1` entered `[2/3]` with a warm 9-NEFF cache and, ~5 h later,
`find /tmp/neff_cache -name '*.neff' | wc -l` returned **15** and was still climbing. So the
frame compiled **6+ graphs the 8-tile probe never built**, at roughly 50 min each.

**Every "a frame needs 8 graphs (4 shapes x 2 passes)" statement elsewhere in this file is WRONG**
and is kept only because the SHAPE inventory is still right. What is wrong is the claim that
(shape x pass) enumerates the graph set. Something else is specialised per tile -- `row0` reaches
dynamo as a python int, and `px0` may too -- and the axes are not known.

Three consequences:

* **This is the definitive cause of the 8-core crash.** The probe builds 8, the frame wants 15+,
  so 7+ compiled concurrently inside `ThreadPoolExecutor(max_workers=8)` and torch raised
  `RuntimeError: Detected that you are using FX to symbolically trace a dynamo-optimized function`.
  Not a subtle race -- a 7-graph pile-up.
* **A prewarm list cannot be derived from tile geometry.** `PREWARM_TILES` is a heuristic and will
  always be incomplete. `[1b]`'s serial full-frame pass (`--cores 1`, all tiles, `--iters 0`) is
  the ONLY sound way to warm the cache, because it compiles exactly what the frame asks for. It is
  skipped when `CORES=1` because `[2/3]` IS that pass.
* **`ALL 8 GRAPHS COMPILE = 1` does not mean the frame will run.** It means the 8 probed shapes
  compile. Do not read it as a green light for `--cores 8`.

Open: WHICH axis produces 15. Worth one cheap run -- `--cores 1` full frame with dynamo's recompile
reasons logged (`TORCH_LOGS=recompiles`) names the guard that fires. Until then the count is
measured and the cause is not.

### MEASURED 2026-08-20: 2x4 halo 128 FUSES -- the "2x4 OOMs" verdict was VOID

```
host mem before tile 1: total 1999G used 106G free 1850G avail 1893G
tile 1  992x1280  FUSES  27 min  NEFFs +1
tile 5  992x1280  FUSES  26 min  NEFFs +1
```

**1,269,760 px per graph -- 2.3x larger than any shape that had ever compiled here -- fuses in 27
min.** The OOM at 600, 1000 AND 1500Gi was measured with `F.interpolate` in the graph and is void,
exactly like every rejection measured with `--model-type unet-inference`. 1850 GB free at the time,
so it was nowhere near a memory limit.

Two shapes remain (`992x1152`, tiles 0 and 4). If they behave the same it is 4 NEFFs in ~1.8 h
against 4x8's 8 in 4.8 h.

| | 4x8 h128 | **2x4 h128** |
|---|---|---|
| total padded px | 15,073,280 | **9,650,176 -- 36% less** |
| tiles / cores | 32 on 8 | **8 on 8, exact** |
| distinct shapes | 4 | **2** |

**And it makes the 704x640 anomaly a 12x inversion, not 3x.** 992x1280 (1.27 M px) compiles in
27 min; 704x640 (0.45 M px) takes 127. **2.8x bigger, 4.7x faster.** Whatever is pathological is
specific to that shape, and 2x4 does not contain it. Still unexplained; one `--only-tile 8` run
with neuronx-cc verbose timing would name the pass that eats it.

Retired by this: "Geometry / halo sweeps" and "2x4 1.0M px OOM" in the Dead list are void for the
same reason. 2x8 and 3x4 have NOT been retested and their OOMs are equally contaminated.

### The agreed working loop

The user names a run, I analyze it with the tooling, **they confirm against Neuron Explorer.**
Disagreements are the point -- see the thresholds below.

```bash
# 1. point the job at the run and apply
#    (edit RUN in analyze-job.yaml -- currently exp_univr-warpeager_gridsample_t9_h128_sr3_20260819_142240)
kubectl delete job univr-analyze --ignore-not-found && kubectl apply -f analyze-job.yaml
# 2. it prints PHASES and VERDICT before the raw tables; also in the timeline tarball
#    as phases.txt / verdict.txt
```

Locally, on anything already downloaded:
```bash
D=scripts/neuron/diagnose_neff.py
python3 $D <pairs-dir>                                   # ranked, one verdict per NEFF
python3 $D --before <pair> --after <pair>                # what a fix changed
python3 $D --instr <analyze.log> --src-root <src-tree>    # time by SOURCE LINE + names the op
python3 scripts/neuron/pq_phases.py <pq-dir> lbl --src-root <src-tree>   # phases, needs parquet
```
`--src-root` must be the tree that BUILT the NEFF: `git show <sha>:file.py > /tmp/src/file.py`.
`analyze-job` resolves that commit itself from `source.tar.gz`/`PROVENANCE.txt` and says so.

### What to check the tooling AGAINST, and what a disagreement means

**Every threshold came from ONE model.** 2 KiB payload floor, 2x issue-vs-move, 15% stall, 5%
spill, 2% transpose. If Explorer shows a bottleneck the classifier missed, or it flags something
Explorer says is fine, that is a threshold to move -- and the change must record WHICH measurement
moved it.

**Known blind spot:** phases merge on the dominant engine only, so a phase where the engine MIX
shifts but the leader does not will not split. If Explorer's timeline shows a transition the phase
table missed, that is the trigger to cluster on the full engine vector instead.

### Two claims of mine that are UNVERIFIED and should be settled by the next analyze run

1. **"pool-down's SWDGE trio (176.6 / 145.7 / 138.7 ms) was the `gridsample` graphs."** I wrote
   that without resolving their source lines and it is probably WRONG: taps never touched the warp,
   yet GpSimd fell 45% -> 1.3%, so those NEFFs were almost certainly `F.interpolate`'s UPSAMPLE.
   Running analyze on either run resolves the `src=` lines and settles it. Fix this file either way.
2. **"the SWDGE cost RELOCATES to the resample after the interpolate fix."** Same root -- it rests
   on claim 1.

### One number already retracted

"5.1M descriptors x 40 ns = 200 ms" matched the measured 199.07 ms to 3% and was **arithmetic
coincidence**. It multiplied `software_dynamic_dma_packet_count` by a per-descriptor cost, but
packets are not descriptors: `dma_transfer_count` is 130,936 at 2,286 B average while packets are
5,151,200 at 29 B. What IS solid: **GpSimd spent 199.07 ms on 160,512 `DMA_INDIRECT` instructions
(1,094 ns each) while the DMA engine sat 89% idle.** The tool now labels every unit for this reason.

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

## MEASURED: halo 128 fuses on all 8 slots, and EVERY TILE STILL FAILS ACCURACY

`univr-accuracy-triage-k2zwh`, 4x8 halo 128, `gather`, serial probe, `--cores 1 --only-tile`,
**neuronx-cc 2.27.2878.0** (not the 2.26.6360.0 every earlier result used):

| tile | padded | px | compile | max_diff | PSNR |
|---|---|---|---|---|---|
| 9 | 704x768 | 540,672 | 73 min | 22.38 LSB | 48.45 dB |
| 17 | 704x768 | 540,672 | 75 min | 31.79 LSB | 49.90 dB |
| 8 | 704x640 | 450,560 | **215 min** | 25.08 LSB | 49.82 dB |
| 16 | 704x640 | 450,560 | **229 min** | 39.01 LSB | 49.49 dB |
| 1 | 576x768 | 442,368 | 33 min | 28.22 LSB | 49.84 dB |
| 25 | 576x768 | 442,368 | 33 min | 40.63 LSB | 49.16 dB |
| 0 | 576x640 | 368,640 | 21 min | 39.26 LSB | 47.66 dB |
| 24 | 576x640 | 368,640 | 22 min | 47.56 LSB | 44.00 dB |

`ALL 8 GRAPHS COMPILE = 1`, 11.7 h for the set, one NEFF per tile exactly as predicted.

**This is the strongest localisation yet: a SINGLE tile, on ONE core, in ONE graph, at the
PRODUCTION halo, already fails — on all four shapes and both passes, 22-48 LSB against a bar
of 3.** Everything the frame adds is therefore exonerated:

* **not the halo** -- 128 is the production value and it fails
* **not the 8-core path** -- one core fails
* **not tile stitching or the merge seam** -- one tile has neither
* **not the replica deep-copy** -- a single replica fails
* **not the resample** -- already known, `gather` and `nki-dyn` agreed to every digit

What is left is the per-tile fused graph itself: a conv, `interpolate`, or `NeuronPReLU`
lowering difference under `fullgraph=True`. The next step is a per-stage numeric diff of one
tile against CPU, not another geometry sweep.

**ANSWERED -- it was `interpolate`, specifically the UPSAMPLE.** Replacing both fixed-factor
resizes with precomputed taps takes tile 9 from 22.17 to **0.05 LSB, PASS**. See
"SOLVED, PROBABLY: `F.interpolate` was BOTH the top NEFF and the accuracy bug". The numbers in
this section were all measured with `F.interpolate` in place and are baseline numbers, not
properties of the tiling.

Caveat: these are per-tile scores over each tile's own region, not whole-frame maxima, so they
are not directly comparable to the 92.56 LSB frame number. Both fail by more than 7x.

**Compile time is NOT monotonic in pixel count.** 704x640 at 450,560 px took 215 and 229 min;
704x768 at 540,672 px took 73 and 75. A 17% SMALLER shape cost 3x MORE. Do not estimate compile
cost from area.

**The memory ceiling was overstated.** 540,672 px fuses in 73 min. The old "~442k-594k px"
bracket came from 2x8 at 593,920 px (928x640) OOMing, so the limit is not area alone -- aspect
ratio or the resulting tile geometry matters. 2x4 (1.0M px) and 2x8 remain measured OOMs; do not
generalise a px threshold from them.

## The 8-core arm produced NO latency: the dynamo fix had silently stopped working

Same job, stage 2, against a fully warm cache (`NEFFs +0`, so the prewarm design worked):

```
torch._dynamo.exc.Unsupported: Dynamo recompile limit exceeded
  ... exceeding the recompile_limit cache size limit (currently set to 8)
```

The script sets `torch._dynamo.config.cache_size_limit = 64`. **On the torch shipped with
neuronx-cc 2.27 the attribute is `recompile_limit`, and assigning the old name neither raises
nor aliases** -- so the limit stayed at 8 and the arm died after the 8 probe compiles had
already spent 11.5 h. Now fixed by setting every name that exists AND verifying the effective
value at import, raising `SystemExit` if it is still below 64.

Also learned: the recompiles are driven by **`row0`**, which dynamo specialises as a python int
at `tau = (t + gamma - gamma * rows / fh ...)` (~line 831) and which varies per tile ROW, on top
of shape and timestamp. The failing frame reached recompile **22**, so "8 graphs" understates the
host-side trace count even though it is exactly right about NEFFs.

## MEASURED: the resample is descriptor-bound, and `gather` is already AT the floor

Job `univr-microbench-c8lm6`, arm `3x704x768` -- the C=3 flow-pyramid site of the largest
4x8 halo-128 tile, which is 38.75% of all resample work:

| | gather | gridsample | ratio |
|---|---|---|---|
| total_active_time | **21.98 ms** | **64.66 ms** | 2.94x |
| sw_dynamic_dma packets | 544,064 | 1,624,192 | **2.99x** |
| **pkt/px** | **1.006** | **3.004** | 2.99x |
| **ns per descriptor** | **40.40** | **39.81** | **1.00x -- flat** |
| hbm_read_bytes | 36.9 MB | 30.4 MB | gridsample reads LESS |
| gpsimd / tensor / MFU | 99.3% / 0.1% / 0.00% | 99.7% / 0.1% / 0.00% | |

**Time = descriptor count x ~40 ns and nothing else.** Not compute (MFU 0.00%, tensor 0.1%),
not bandwidth (gridsample moves FEWER bytes -- same data in 3x more, smaller transfers).
`gather` wins because it is laid out so a pixel's channels are contiguous (`[B*N, C]`, one
descriptor covers the C-run); `F.grid_sample` works on NCHW where channels are strided by H*W.

**The tool prints its own floor -- `min_desc 540672 at 1/px` -- and gather measured 1.006.
gather is AT the theoretical minimum.** So no kernel can beat it by cutting descriptors; the
only remaining axis is the 40.40 ns, i.e. getting off software DGE. `hardware_dynamic_dma` is
0.1% / 0.0% on both -- everything goes through the software path today.

**Frame-level consequence.** 100.45M pixel-warps x 1.006 desc/px x 40.40 ns = 4.08 s
single-core, **~510 ms across 8 cores = ~13% of the 3900 ms frame**. That is the ENTIRE
resample, all 448 calls, and it is already descriptor-optimal. So ~510 ms is the absolute
ceiling on any resample optimisation, and only if the resample became free.

Two per-op numbers on record are now corrected: **`gather` is 1.006 pkt/px, not 2.0** (the old
figure was taken at 992x1280 with `--model-type unet-inference`, so it is void), while
**`gridsample` at 3.0 and ~40 ns/descriptor are CONFIRMED** (3.004 and 40.40 vs the recorded
~38 ns).

## How to assess a resample kernel: `--op sequence`, not per-op arms

`microbench.py` answers "how many descriptors does this op issue" and it answered it. It CANNOT
answer "what does swapping the resample do to the model", for three reasons:

* an isolated op has no pipelining, no layout reuse and no overlap with the convs around it, so
  per-op numbers do not sum to the sequence
* absolute single-io microseconds are noise across runs -- only shares and counts are stable
* a swapped resample changes the LAYOUT its neighbours see. The NKL kernel asserts NHWC while the
  model is NCHW, so it adds two permutes per call that no per-op arm measures at all

**`microbench.py --op sequence`** runs the real sequence at a real padded tile shape as one
compiled graph and reports a warm median, which IS comparable between warps. It reuses the
model's own modules, so it cannot drift from the thing it claims to represent. `SET=sequence` in
`microbench-job.yaml` runs it once per warp, gather first with `--save-out`, the rest `--cmp`.

It is a MODE of the microbenchmark, not a separate script. It briefly was one (`distill_tile.py`,
`distill-job.yaml`) and that was the same naming/divergence mistake this file already records
twice -- `warp_op_bench.py` and `conv_op_bench.py` were merged for exactly this reason. One
script, one job.

Two rules for reading it:
* **the bar is `gather`, not ATen `gridsample`.** gather is what the model uses, and it already
  sits at the 1 descriptor/px floor at 40.40 ns, so a kernel only wins by beating ns/descriptor
  -- i.e. by leaving the software DGE path (`hardware_dynamic_dma` is ~0% today).
* **the ceiling is ~510 ms of a 3900 ms frame (~13%)**, the whole resample, even if it became
  free. Any claimed win larger than that is a measurement error.

## MEASURED: the NKL GridSample kernel WORKS, solves gridsample, and still loses to gather

Job `univr-microbench-snk4g`, `SET=warps TOP=4`, cc 2.27, `nkilib` restored from the PVC,
`max_indices_per_indirect=None` (batching OFF), fp32 so `gather_method="copy"`:

| dim | warp | active | pkt/px | ns/desc | hwDMA% | max_LSB | vs gather |
|---|---|---|---|---|---|---|---|
| 3x704x768 | gather | 21.98 ms | 1.006 | **40.39** | 0.10 | 0.028 | -- |
| | gridsample | 64.66 ms | 3.004 | 39.81 | 0.00 | 0.028 | 2.94x slower |
| | **gridsample-nkl** | **24.06 ms** | **1.006** | **44.22** | **5.80** | 0.028 | **1.09x slower** |
| 3x576x768 | gather | 18.00 ms | 1.006 | 40.44 | 0.10 | 0.0236 | -- |
| | gridsample | 52.95 ms | 3.004 | 39.85 | 0.00 | 0.0234 | 2.94x slower |
| | **gridsample-nkl** | **19.73 ms** | **1.007** | **44.31** | **5.90** | 0.0234 | **1.10x slower** |

**The kernel does exactly what it was built to do.** It collapsed the descriptor blowup
**3.004 -> 1.006 pkt/px**, matching gather exactly -- a **2.7x win over ATen gridsample** -- and
its accuracy is identical to both other warps to three decimals, so it is numerically correct.

**But it does not beat `gather`, which is what the model uses: it is 9-10% slower.** Exactly what
the descriptor-floor argument predicted -- gather already sat at 1 descriptor per output pixel, so
there was no descriptor headroom, and the kernel costs **~9.6% more per descriptor** (44.2 vs
40.4 ns).

One lead remains: `hardware_dynamic_dma` is **5.8%** on the kernel against **0.10%** on gather, so
it does reach hardware DGE -- for ~6% of the work, 94% still software. And this was measured with
**batching DISABLED** (`max_indices_per_indirect=None`), which is the kernel's headline feature.
`SET=nklsweep` sweeps the cap at the 38.75%-of-work dim. **Beating gather means ns/desc below
40.39**; if no cap does that, the CR is a win only over an implementation this model does not use.

## MEASURED: gridsample on CPU is NOT a lever, and the ratio is size-dependent

Job `univr-bench-79h4g`/`univr-bench` at `f26eb21`, `MODE=warpcmp`. The model runs on **neuron**
in every arm; only the RESAMPLE REGION differs (`--warp-region`, its own `torch.compile`, its own
backend). `--warp-region` forces `fullgraph=0` in all arms, so the break structure is identical
and the resample is the only variable. Random weights, so ratios only -- no accuracy claim.

| size | px | aten (neuron) | cpu (inductor) | nki (fallback) | **cpu/aten** |
|---|---|---|---|---|---|
| 128x128 | 16,384 | 98.0 ms | 83.4 ms | 6,908 ms | **0.851x** |
| 192x224 | 43,008 | 187.8 ms | 173.5 ms | 18,048 ms | **0.924x** |
| 288x320 | 92,160 | 392.4 ms | 379.5 ms | 38,490 ms | **0.967x** |

**The CPU advantage is small and VANISHING with size: 0.851 -> 0.924 -> 0.967, converging on
1.0.** The production tile (576x640, 368,640 px) is 4x the largest rung, so on this trend the host
arm is at or past parity there. **Running gridsample on CPU is not a lever.**

**A single small tile would have produced the OPPOSITE conclusion** -- 128x128 alone says "CPU is
15% faster" -- which is exactly why the ladder exists. The work MIX is scale-invariant (90.04% of
pixel-warps at C=3 at every size, because each Contextnet level is a fixed fraction of tile area),
but that does NOT make measured ratios scale-invariant. Prove a ratio at more than one size before
extrapolating it.

**Scaling, which is the diagnostic:**

```
aten   1.00x -> 1.92x -> 4.00x time    for 1.00x -> 2.62x -> 5.62x pixels   sublinear
cpu    1.00x -> 2.08x -> 4.55x                                              sublinear
nki    1.00x -> 2.61x -> 5.57x                                              LINEAR in pixels
```

`aten` and `cpu` are sublinear, so both carry fixed per-call overhead -- the split region's
dispatch, plus 14 host round-trips for `cpu`. **`nki` tracks pixels almost exactly**, which
**refutes the "fixed per-call dispatch overhead" reading of its 6,908 ms**: a fixed cost would
be roughly CONSTANT across the ladder, and it is not. The software-DGE fallback is ~98x slower
**per pixel**, not per call. `nki/aten` runs 70x -> 96x -> 98x.

Absolute scale worth remembering: `aten` at 288x320 is 392 ms for ONE tile, so ~5-6 s a frame once
scaled to the production tile over 32 tiles and 8 cores -- far worse than the 3900 ms fused
number. `fullgraph=0` plus a separately dispatched region is expensive, and **fusion is the thing
that matters**.

## BLOCKED: the NKL GridSample kernel needs unmerged DGE MICROCODE

Every `gridsample-nkl` number measured so far ran on the **software DGE fallback**, so none of
them is a verdict on the kernel. The fast path needs `CR-296821015`
(`NeuronUcode`, branch `ethschan/turbo-cayman-indirect-scatter-add`, tip `496e4aa1`):
*"Enable turbo on Cayman, support batched indirect scatter-add"* -- `turbo_imemcopy*`,
`dge_decode.cpp`, `dge_backend_software.*`, the `cayman/q7/pool/` DSP firmware,
`idma_data_rings_q7.hpp`, `translate_cayman+.hpp`.

The evidence that we were on the fallback was in every profile and we read it as a property of
the kernel instead of a missing dependency:

| | measured |
|---|---|
| `software_dynamic_dma` | ~99% of gpsimd time |
| `hardware_dynamic_dma` | **0.0 - 0.1%** |
| ns/descriptor | 44.2 (kernel) vs 40.4 (gather) |
| kernel config dump | `batched_indirect_gather=False, M_batch=1` |

`batched_indirect_gather=False` was not only our flag: **the ucode in this image cannot do it.**
"turbo" IS the hardware DGE path and "batched indirect scatter-add" IS the ucode side of
`max_indices_per_indirect`.

**It cannot be side-loaded the way `nkilib` was.** It is device firmware loaded by the
runtime/driver, not a `PYTHONPATH` addition; the CR is `[DO NOT MERGE]` with a self-imposed
blocking comment; and **both dry-run builds FAIL** (`cayman-inkling/master` and
`kaena-runtime/ucode`), so there is no artifact to consume.

So the two numbers on record are fallback measurements and must not be quoted as the kernel's
performance: **op-level 24.06 ms** (vs gather 21.98) and **region-split 6908.5 ms** (vs ATen 98.0
at 128x128, i.e. ~493 ms of per-call dispatch once the kernel is not fused).

**Waiting on the ucode reaching the SDK/driver.** The harness is correct and cheap; re-enable with
`WARPCMP_ARMS="nki:gridsample-nkl:neuron aten:gridsample:neuron cpu:gridsample:cpu"` once an image
carries it. Until then the live comparison is `gridsample` on **cpu vs neuron (ATen)**.

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

**`gridsample` cannot compile FUSED.** OOM-killed at 169 min and again at 94 min at 1500Gi,
on the same geometry where gather and nki-dyn both compile. Consistent with the
microbenchmark: it issues 3-205x more DMA descriptors than gather (3.0 pkt/px at C=3 rising
to 205 at C=128), so its fused graph is far larger.

**Read that as a statement about FUSION, not about gridsample.** `fullgraph=True` is the only
thing that ever blocked it, and it is not a requirement -- it was chosen so a dynamo break would
RAISE rather than silently emit a subgraph, because an unguarded break around the
`view(torch.uint32)` index bitcast corrupts which pixels get sampled. **That bitcast is in the
NKI warps only** (`warp_nki`, ~line 480); `gridsample`, `gather` and `window` carry none. So
`--fullgraph 0` is safe for gridsample and is now available, refused for `nki`/`nki-dyn`.

Why this matters: **the resample is the largest device cost** -- 614.2 ms of 973.2 ms
device-active, **63.1%**, against 36.9% for all 54 convs -- and `gridsample` is both the
REFERENCE every other warp is scored against (`microbench.py` scores against
`warp_gridsample`) and its most expensive form. An efficient NKI GridSample therefore attacks
the biggest device-side item in its worst case. The counter-argument previously recorded here
(that a faster resample cannot pay off) rested on the `d2h` phase split and the eager 23.6%
device share, and **both are invalid for that purpose** -- see the traps.

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

* **Why the fused frame fails accuracy at 92.56 LSB.** **LIKELY ANSWERED: `F.interpolate`'s
  upsample.** Tile 9 went 22.17 -> **0.05 LSB (PASS)** with precomputed taps on both resizes.
  Unconfirmed at the FRAME level -- that needs a `MODE=full` taps run, and it is now the single
  highest-value job in the repo. Everything below in this bullet is the pre-answer state.
  `full-taps-1core-job.yaml` and `-fg0-` are the runs that settle it (`accuracy-triage-job.yaml`,
  which used to own this, is deleted -- `univr-bench-job.yaml` supersedes it).

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
* **The system-profile bundle for Neuron Explorer.** `univr-bench-job.yaml` stages [5]-[6]
  build the directory-upload format: `trace_info.pb` (required), the host `.pb` files,
  `.neff`, `.ntff` with a backfill, plus the source and a PROVENANCE.txt. `trace_info.pb`
  comes from the runtime INSPECT env vars, NOT from a capture command -- capture only ever
  emits `.ntff`. Expect the `.ntff` slot to need the backfill: the forward dispatches async
  and the runtime device-profiler emits nothing for async workloads.
* **`--warp gridsample-nkl`: the NKL GridSample kernel, WIRED, never run.**
  `CR-288764575` (`KaenaNeuronKernelLibrary`, author `ethschan`) -- **OPEN at revision 3, NOT
  merged**, so it is absent from every released image and the arm raises `SystemExit` on import
  until the package is on `PYTHONPATH`. Source is `@nki.jit grid_sample(value, grid,
  sampling_mode, coord_mode, input_layout, align_corners, padding_mode,
  max_indices_per_indirect, gather_method)` at
  `src/nkilib_src/nkilib/experimental/indirect/grid_sample.py`.

  Why it is worth running: `gridsample` is the reference every warp is scored against, the
  resample is 63.1% of device-active time, and ATen's `grid_sample` cannot compile fused only
  because its LOWERING explodes into descriptors. A single kernel may fuse where that expansion
  cannot -- in which case `--fullgraph 0` is not even needed.

  Every semantic matches our call exactly: `bilinear`, `border`, `align_corners=True`,
  `minus_one_one`. Integration is the repo's existing `torch_neuronx.nki_hop.wrap_nki`, already
  used for two `@nki.jit` kernels here. Grid math in the new warp is copied verbatim from
  `warp_gridsample`, so an accuracy delta is the kernel and not the coordinates.

  **Two constraints from the kernel's own asserts.** `input_layout` must be **NHWC** -- the NCHW
  option in its docstring is unimplemented -- so the warp permutes in and out, which is a layout
  copy on the hot path at C=3..128; look for it in the NEFF ranking rather than assuming it is
  free. And `gather_method="transpose"` asserts a **2-byte dtype**, so fp32 must use `copy` and
  the faster transpose path is unavailable (bf16 is not an option: 23.31 dB fails the gate).

  **Untested where we need it.** The CR's matrix tops out at 200x200 sampling to 64x64 (~4k
  queries); a 704x768 tile is ~540k, so `max_indices_per_indirect` is the tuning knob and is
  exposed as `--nkl-max-indices`. Our exact combination, **fp32 + bilinear + border +
  align_corners=True, is not in the matrix** (fp32 rows are bilinear/zeros or nearest/border; the
  align_corners=True row is bf16/zeros). C is covered 8-260 **except C=3**, below the smallest
  tested width, and several resample sites here are C=3.

  First test, cheapest first: the microbench arm (`gridsample-nkl` at every C, `NKL=0` disables),
  then one tile -- `--cores 1 --only-tile 9 --halo 128 --warp gridsample-nkl` -- scored against
  `gather`'s **22.38 LSB** on that exact tile.

  Architecturally it is the right shape where `shiftwarp` was wrong: shiftwarp approximated a
  warp as a bounded shift-sum and silently clamped past R, which is why it hit 229.72 LSB against
  a measured 29.02 px displacement. A real grid_sample with explicit OOB handling cannot fail
  that way. `GridSampleBwd` is irrelevant here (inference only).
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
* ~~`ip-192-168-192-40` S3 mount is broken~~ **CLOSED 2026-08-19: no node pinning needed any
  more.** Kept for the diagnosis in case it recurs: the Mountpoint pod and the CSI node driver
  never completed their `/comm/mount.sock` handshake, the pod waited exactly 120 s, timed out and
  restarted. The driver DID request the mount and the Mountpoint pod WAS created, so it was
  neither a missing daemon nor IAM. `.40` differed from `.189` in nodegroup
  (`trn3-dev1-48xl-efa` vs `cr1-no-efa`), launch template, AMI, `CAPACITY_BLOCK` capacity type,
  and lacked the `alpha.eksctl.io/*` labels -- most likely a host filesystem or
  mount-propagation difference on that AMI. The `s3-csi-controller` TLS error in its `--previous`
  log was 20 h stale and NOT the cause.

## Files

| file | what |
|---|---|
| `repro_unrolling_trn2.py` | the model. Added `--warp shiftwarp`, `--record-flow`, `--shiftwarp-radius/-max-c` |
| `TILING_AND_GRAPH.md` | **read first.** Tiling algorithm, halo, quantisation, the full op graph, why the operand is `px x 14` |
| `SINGLE_GRAPH_NCC_EBIR033.md` | the fusion walls, and that fusion is per TILE not per frame |
| `METHOD.md` / `REPRO_README.md` | original bundle docs |
| `microbench.py` | the microbenchmark: 14 resamples + 54 convs, one op per invocation, each scored against a CPU reference |
| `profile_roofline.py` | reads a `summary.json`, prints per-engine time and MFU |
| `microbench-job.yaml` | runs it. `SET=warps|convs|both`, now the four 4x8 **halo-128** shapes and `CHANS=3 16 32 64 128`. **Every earlier warp arm ran at C=3 only** -- the cheapest site -- so gridsample was judged on its best case while its descriptor rate climbs to 205 pkt/px at C=128 (`_C=16`, so the resample runs at C=3/16/32/64/128) |
| `univr-bench-job.yaml` | **the base every job is generated from.** `/neuron-3run-benchmark`: persistent NEFF cache, 3-run flow, consumer-split archive. Currently `MODE=warpeager` |
| `full-taps-1core-job.yaml` | **live.** Full 4K frame, 1 core, taps, `fullgraph=1`. The whole-frame accuracy number |
| `full-taps-1core-fg0-job.yaml` | **live.** Same, `fullgraph=0` -- the A/B on whether graph breaks matter |
| `full-taps-job.yaml` | 8 cores. Crashed once on a 9th graph; carries the `[1b]` serial full-frame warm-up that fixes it. The eventual latency number |
| `tiles2x4-taps-probe-job.yaml` | **queued.** Does 2x4 halo 128 compile now that taps removed `F.interpolate`? 36% less device work if yes |
| `analyze-job.yaml` | per-NEFF attribution to source lines via neuron-explorer + duckdb over the parquet. Now also prints PHASES and VERDICT, and ships `phases.txt` / `verdict.txt` |
| `scripts/neuron/diagnose_neff.py` | **reads a profile and states a verdict.** Discovers engines and DMA modes from key names, classifies by ratios between them, names the source line and the torch op. `--before/--after` diffs two profiles. Also the `neuron-profile-diagnose` skill |
| `scripts/neuron/pq_phases.py` | splits a NEFF into PHASES over time and diagnoses each separately, because a whole-NEFF average of a graph that is compute-bound then issue-bound describes neither |
| `scripts/neuron/` | the skill's helpers, committed so the pod's clone has them: `rank_neffs.py` (fixability + engine-mix gate), `profile_all_neffs.py`, `pick_neffs.py`, `top_neffs.py`, `pq_*.py`, `run_dma_analysis.sh`, `pf_op_summary.py` (chrome-trace op table, resample vs rest by SPAN CONTAINMENT) |
| `s3-mount-test-pod.yaml` | 85-second PVC mount check on a chosen node |
| `nki_shift_warp.py` | the dead kernel, kept for the measured op-level result |

Deleted earlier as dead ends: `profile-model-job.yaml`, `compile-modes-job.yaml`,
`fused-4x8-timing-job.yaml`, `halo-shapeset-job.yaml`, `halo64-fusion-job.yaml`,
`fusion-threshold-job.yaml`, `single-graph-retest-job.yaml`, `settle-2x8-job.yaml`,
`AB_RUN.md`, `profile_hotspots.py`.

Deleted 2026-08-20, each having served its purpose -- `git show <sha>~1` restores any of them:
`accuracy-triage-job.yaml` (superseded by `univr-bench-job.yaml`), `ccflag-fusion-job.yaml`
(found `--model-type unet-inference`, done), `bigtile-noflag-job.yaml` (proved 4x8 h64 fuses),
`prod-4x8-halo64-job.yaml` (produced 3900.5 / 3820.4 ms), `fused-config-search-job.yaml`
(superseded), `repro-job.yaml` (needs the eager path, which no longer exists in the script),
`taps-fg1-probe-job.yaml` (answered: taps DOES trace under `fullgraph=True`).

**Every `#` comment is stripped from the YAML and from `repro_unrolling_trn2.py`.** The reasoning
that used to sit in them lives here now. Two consequences: a manifest no longer explains itself,
and **`repro_unrolling_trn2.py` keeps only the 51 prints some job greps** -- a print added for
human reading will be deleted by the next strip unless a manifest parses it.

## The benchmark harness: `univr-bench-job.yaml`

Adopts `/neuron-3run-benchmark`. Three things no earlier job in this repo had:

**1. The NEFF cache survives the pod.** Every previous job kept it at `/tmp/neff_cache`, the
container's ephemeral layer, so it died with the pod. `univr-accuracy-triage-gmnpq` lost ~3.5 h
of 704x768 compiles that way and every earlier run silently threw its compiles away. The new job
restores from ONE tar object on the PVC and **re-archives after every successful compile**, so a
pod death costs one tile.

**2. The 3-run methodology.** `[1/3]` build/load (the serial probe, not timed), `[2/3]` clean run
with no profiler -- **the only number that is the latency** -- `[3/3]` profiled runs for the
breakdown. Never quote a profiled run's wall clock. Engine **shares** are stable across runs;
absolute single-io µs are noise.

**3. Archive split by CONSUMER, at `/var/mdl/univr/runs/exp_univr-<cfg>-<warp>_<TS>/`:**

```
explorer/upload_bundle.tar.gz    trace_info.pb + .pb + .ntff + PAIRED .neff -> Directory Upload
explorer/systrace_bundle.tar.gz  all ranks, raw
perfetto/trace_rank0.json.gz     -> drag into ui.perfetto.dev
pairs/rankNN_<hash8>__<MB>MB/    ONE .neff + ONE .ntff + summary.json, pre-joined
neff_ranking.txt                 fixability-ranked; START ANALYSIS HERE
frame_<cfg>_<warp>.png/.json     the output artifact -- the correctness gate
univr_results_<TS>.tar.gz        everything else
```

* **Never split a `.neff` from its `.ntff`.** That pair is the atomic unit of per-NEFF analysis
  and rebuilding the join by filename stem is where wrong attribution creeps in. Use only the
  no-suffix `<hash>.ntff`; the profiler also emits `<hash>_rank_0..N.ntff` decoys on the same stem.
* **`/var/mdl` is Mountpoint-for-S3, not posix.** Every file open is a separate S3 GET, so
  `cp -r`, `tar` over a directory, `find` or a glob on the mount stalls for minutes to hours with
  no error. Everything crosses as ONE tar object, extracted locally. Writes go through a
  `publish()` that verifies with `wc -c` and refuses 0 bytes (`stat -c` is GNU-only and its
  absent-fallback reports a successful copy as a mismatch).
* **The archive runs in `trap ... EXIT`,** not as the last block: under `set -euo pipefail` any
  earlier failure would skip a trailing archive and take every artifact with it.
* **`rank_neffs.py` applies the ENGINE-MIX GATE.** A NEFF is a fixable copy only if
  `matmul≈0` AND `tensor% < ~15` AND DMA dominates. `matmul > 0` with `tensor% > ~40` is REAL
  COMPUTE and is not fixable however much its source line looks like a `cat`/`pad`. Never rank
  fixability by wall time -- rank-1 by wall time is usually an honest GEMM.
* Latency needs an **output-correctness gate**: `--save-ref` writes npy + PNG + a sha256
  manifest, and a faster config is often faster because it computes less.

## MODE=warpeager: how to ask "is the hotspot the gridsample ops"

Under `fullgraph=True` that question has no answer in any trace: the tile is ONE graph, so
`torch.profiler` holds `Torch-Compiled Region` and no ops, and the NEFF+NTFF conversion
(`graph.pftrace`, 7.4 GB) shows WHAT ran on the engines with no way back to a Python call.
MEASURED both ways on CPU at 128x128, same model, same warp:

| run | resample spans | `aten::grid_sampler_2d` in the trace |
|---|---|---|
| `--fullgraph 1` | 0 | **absent** -- fused into one inductor kernel |
| `--warp-region eager` | **14** | 14 calls, named, with shapes |

`--warp-region eager` (new) leaves the resample UNCOMPILED and on the device while the convs stay
in their neuron graph. The boundary is the same `torch._dynamo.disable` the `cpu`/`neuron` regions
already use, so it forces `--fullgraph 0`. With `--perfetto` it also wraps every call in a
`resample:C<c>:<h>x<w>` span, which is what makes attribution sound: grid_sample's lowering is
index/clamp/mul arithmetic sharing op names with the convs, so a NAME-based bucket would both miss
its arithmetic and steal the model's. The 14 spans came out at exactly the architectural sites --
6 at C=3 plus 2 each at C=16/32/64/128.

`MODE=warpeager` in `univr-bench-job.yaml` runs it: [A] host trace, [B] `pf_op_summary.py` op
table, [C] runtime INSPECT trace -> device timeline, [D] per-NEFF sweep -- and the eager region
gives the resample its OWN NEFFs, one per site, so `rank_neffs.py` prices it on device instead of
inside a fused tile. Its cache is a SEPARATE tar and `/tmp/neff_cache` is wiped first: [D] ranks
every NEFF in the directory, and leaving the fullgraph graphs there would attribute a fused
graph's device time to a run that never executed it.

**It is not a latency configuration.** `fullgraph=0` with 14 breaks per forward is a different
regime and the shares are of that run's wall. What transfers is WHICH ops the resample is made of
and their relative device cost; the frame number still comes from `MODE=full`.

## SOLVED, PROBABLY: `F.interpolate` was BOTH the top NEFF and the accuracy bug

**Naming.** The flag is `--static-resize N`. Use the plain names, not "srN":

| flag | plain name | downsample | upsample |
|---|---|---|---|
| 0 | **baseline** | `F.interpolate` | `F.interpolate` |
| 1 | **pool-down** | `avg_pool2d` | `F.interpolate` -- UNTOUCHED, and this is the whole story below |
| 2 | **pool+deconv** | `avg_pool2d` | depthwise `conv_transpose2d` |
| 3 | **taps** | precomputed taps | precomputed taps |

"Taps" = the 2 source pixels per axis a bilinear resize blends, plus their 2 weights. Normally the
graph re-derives them every call (`src=(o+0.5)/f-0.5`, clamp, floor, +1, subtract) and THAT address
arithmetic is what lowered to 512 B SWDGE. Precomputed = built once on the host, baked as
constants, graph does only `read*w + read*w`.

Origin is a Slack thread: Liran read the Explorer profile and named the cause before we measured
it -- GpSimdE SWDGE issuing **512 B DMAs** (`src_pattern=[4][128]`, 128 fp32 = one element per
partition) while **TensorE sits at zero**, and 512 B cannot saturate DMA, which wants >= 2 KiB. He
localised it to IFBlock's `F.interpolate` calls. Confirmed: baseline's rank-1 NEFF `009cee17` is
**301.7 ms of a 376.8 ms capture sum, swdyn 97%, tensor 14%, waste 101.0 ms**.

Tile 9 halo 128 (704x768), `gridsample`, `--warp-region eager`, 1 core, warm cache, NO profiler.
`[L] CLEAN LATENCY` is the number:

| arm | clean median | spread | `dev` | vs baseline | max_diff | PSNR | gate |
|---|---|---|---|---|---|---|---|
| baseline | **1313.3 ms** | 1308.0-1323.0 | 1191.1 | -- | 22.45 LSB | 48.46 dB | FAIL |
| pool-down A | **986.9 ms** | 981.7-1022.3 | 869.2 | 1.33x | 22.17 LSB | 48.77 dB | FAIL |
| pool-down B | **1039.2 ms** | 1036.6-1041.9 | 926.5 | 1.26x | 22.17 LSB | 48.77 dB | FAIL |
| **taps** | **647.3 ms** | 643.6-**67203** | 526.3 | **2.03x** | **0.05 LSB** | **102.22 dB** | **PASS** |

**THE 22 LSB FAILURE WAS `F.interpolate`'s UPSAMPLE.** Nothing in this file had ever passed the
3 LSB bar; every arm sat at 22-48 LSB and "why does it fail accuracy" was the live blocking
question. Read the dispatch and it falls out -- `do_up` keeps `F.interpolate` at level 1 and only
level 2/3 replace it:

```
            downsample        upsample            max_diff
baseline    F.interpolate     F.interpolate       22.45
pool-down   avg_pool2d        F.interpolate       22.17   <-- upsample untouched
taps        taps              taps                 0.05
```

Swapping the DOWNsample moved accuracy by 0.28 LSB. Swapping the UPsample took it to 0.05, cosine
**1.000000**, `pixels > 1 LSB = 0.0000%`, worst spatial bucket 0.05. So the fault was never the
halo, the cores, the stitching, the replica copy or the resample -- all of which earlier sections
correctly exonerated -- it was the other fixed-factor resize in the same file.

**taps vs pool-down is CONFOUNDED and is not a mechanism result.** taps replaces BOTH directions;
pool-down replaces one. So the 647.3 vs 986.9 gap mixes "constant indices lower better" with
"one more site got fixed". The honest claim is: **removing `F.interpolate` in both directions is
worth ~2x and fixes correctness.** Separating mechanism from coverage needs **pool+deconv**, which
is implemented and unrun. Do not tell Youval taps beat pooling -- that has not been measured.

**The taps median is NOT yet quotable.** `NEFFs +3 (must be +0, else the median contains a
compile)` and the spread proves it: `[L] max 67203.3 ms`, `[A] max 11285.3 ms` -- a 67-second
iteration is a compile inside the timed loop. It survives the median (5 iters, one high outlier)
and `[A]` independently gives **638.0 ms, min 637.5**, agreeing to 1.5% on tightly clustered fast
iterations, so ~640 ms is very likely real. **Re-run for a +0 median before quoting it.** Also
`[W]` reported `NEW graphs compiled by [W]: 0` while `[L]` compiled 3, so the prewarm does not
cover what the timed loop needs.

Secondary signals: taps produces **55 NEFFs / 13 MB** cache vs pool-down's 63 / 33 MB, and its
host trace is **154.5 MB vs 0.6 MB** -- `index_select` emits far more host-side slices. Different
graph shape, not just different timing.

**The precompute is Liran's formula exactly** (`_resize_taps`, `repro_unrolling_trn2.py:841`):
`out_sz=int(in_sz*f)`, `src=((o+0.5)/f-0.5).clamp(0,in_sz-1)`, `i0=floor`, `i1=min(i0+1,in_sz-1)`,
`wr=src-i0`, `wl=1-wr`, clamp IS the `align_corners=False` replicate. Two deliberate deviations:
torch not numpy, and **float64 then cast** rather than `astype(float32)` -- the weights are the
entire accuracy of the op and computing them in the activation dtype quantises them before use.
Cached on `(in_sz, f, dtype, device)`. Exact vs `F.interpolate`: 2.4e-07 at 1/2 and 1/4, 4.8e-07
at x2/x4/x8, 0.0 at 1/3, 1.5e-05 at x2.5.

**It is NOT a verdict on Liran's NKI kernel.** His H pass has no gather at all -- rows become
literal partition offsets with scalar weights (`nisa.tensor_scalar`, `scalar_tensor_tensor`) and
only the W pass gathers, via `nc_n_gather` on the free axis, channels on partitions in NHWC so
zero transposes. torch `index_select` on NCHW cannot express that. Our taps arm keeps the
indirection deliberately -- it measures whether CONSTANT indices are enough on their own, which is
his open question, not his answer.

### The traces for this run, and how to read them

Run `exp_univr-warpeager_gridsample_t9_h128_sr3_20260819_142240` on `/var/mdl/univr/runs/`.
Stages `[D]`/`[E]`/`[F]` were **still running** when this was written, so the per-NEFF ranking and
the NEFF+NTFF pairs for the taps arm are **not yet recorded here** -- fill them in from
`ranking_..._sr3.txt` and `perneff_..._sr3.txt`.

| artifact | what it answers |
|---|---|
| `pairs/rankNN_<hash8>__<MB>MB/` | `.neff` + `.ntff` + `summary.json`, pre-joined -- Explorer "Individual Files" |
| `ranking_..._sr3.txt` | per-NEFF fixability + engine mix. **START HERE** |
| `perneff_..._sr3.txt` | per-NEFF engine table + MODEL-LEVEL engine totals |
| `perfetto/graph_rankNN_<hash>.pftrace.gz` | what ran INSIDE that graph -> ui.perfetto.dev |
| `perfetto/device_..._sr3.pftrace.gz` | runtime device timeline |
| `perfetto/trace_..._sr3.json.gz` | host trace, resample spans named |
| `explorer/upload_bundle.tar.gz` | Explorer Directory Upload (top level only) |

**Never split a `.neff` from its `.ntff`** -- that pair is the atomic unit of per-NEFF analysis and
rebuilding the join by filename stem is where wrong attribution creeps in. Use only the no-suffix
`<hash>.ntff`; the profiler also emits `<hash>_rank_0..N.ntff` decoys on the same stem.

**The comparison to make on the pairs:** baseline's `009cee17` (301.7 ms, swdyn 97%) has no
counterpart in the taps profile. Confirm that from `ranking_..._sr3.txt` rather than assuming it,
and check whether the remaining SWDGE-heavy entries are the `gridsample` graphs -- in pool-down
they were 176.6 / 145.7 / 138.7 ms, and after the interpolate fix the SWDGE cost RELOCATES to the
resample rather than disappearing. That one is harder: `gather` already measures 1.006 desc/px
against a printed floor of 1.0 at 40.4 ns/desc, and the NKL kernel needs unmerged DGE ucode. Do
not scope an interpolate kernel expecting it to fix the resample.

### What this does NOT license

All of it is `--warp-region eager` with `fullgraph=0`, not the fused config. **No `MODE=full` run
at taps exists**, so the 32-tiles/8-cores arithmetic is unearned. Next, in order:

1. **Re-run taps** for a `+0` median. Warm cache, minutes.
2. **`MODE=full` at taps** -- the prize. If the tile-level 22.17 -> 0.05 collapse carries to the
   frame's **92.56 LSB**, the blocking accuracy failure is solved AND the frame is faster.
3. **pool+deconv**, to separate mechanism from coverage.

## Traps that cost runs

* **`kubectl apply` uses the on-disk YAML; the pod clones the script from git.** A pod can
  run new code with a stale command block. Grep the config echo lines, not just the commit.
* **`kubectl set env job/...` cannot patch a Job** — `spec.template` is immutable.
  `--dry-run=client` does NOT catch it; only `--dry-run=server`. Edit the YAML and re-apply.
* **`backoffLimit` belongs on `Job.spec`**, not `spec.template.spec`. Both failure modes
  happened and only one is loud: on `bigtile-noflag` the API server REJECTED the manifest
  (`strict decoding error: unknown field "spec.template.spec.backoffLimit"`) so nothing ran;
  on `halo-shapeset` it was accepted SILENTLY and the retry simply did not exist.
* **`--compile` no longer exists.** The script always calls
  `torch.compile(m, backend="neuron", dynamic=False, fullgraph=<--fullgraph>)`. There is still no
  eager path and no `whole`/`halves`/`stages`. `--fullgraph 0` allows graph breaks, which
  re-enables `--record-flow`; `--per-block` still errors, and `--fullgraph 0` does NOT rescue it
  (breaks land wherever dynamo puts them, not on module boundaries, so the attribution would be
  arbitrary rather than merely coarse -- that is how a block table once summed to 2.51x the frame). Consequence: the eager
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
* **Dynamo's recompile limit defaults to 8** and this model needs more than that, so it sat on
  the limit -- one tile never trips it, 8 cores always does, and under fullgraph it is a hard
  error. It is a HOST trace counter: unrelated to cores or device memory, and reducing `--cores`
  would only double wall clock. **The config attribute was RENAMED** (`cache_size_limit` ->
  `recompile_limit` on the torch with neuronx-cc 2.27) and assigning the old name is a **silent
  no-op** -- no raise, no alias -- which cost the 8-core arm of `k2zwh` after 11.5 h of compiles
  were already paid for. The script now sets every name that exists and VERIFIES the effective
  value at import. Never set a config knob without reading it back.
* **Names drifted from what the files do**, twice. `warp_op_bench.py` benchmarked warps AND
  convs; `sweep-job.yaml` ran the whole microbenchmark. Now `microbench.py` and
  `microbench-job.yaml`. `conv_op_bench.py` and `conv-bench-job.yaml` are deleted -- they
  differed from the sweep only in which op list they looped over.
* **`NEURON_CC_FLAGS` must NOT contain `--model-type unet-inference`.** It is the single
  cause of the DMA rejection. Use `--lnc 1`.
* **Node pinning is RETIRED (2026-08-19).** Jobs select on `node-type=trn3-dev1` alone and no
  longer name a host. Historical note only: `ip-192-168-192-40` once could not mount the PVC, so
  a job landing there sat Pending on `MountVolume.SetUp failed ... DeadlineExceeded` for as long
  as you let it (74 min in one case). If a Pending-on-mount pod ever reappears,
  `s3-mount-test-pod.yaml` checks a node in 85 s instead of hours.
* **`--only-tile 1` is not the largest tile.** Tile 10 for 3x9, tile 9 for 4x8. Testing
  tile 1 produced a false "4x8 FUSES" that stood for several runs.
* **One arm per pod.** Each arm holds its compiled graph, so peak memory is the sum across
  arms. A 2x8 arm killed the pod that would otherwise have timed 4x8.
* **The image tag lies.** `native-pytorch:...sdk2.31.0...` contains `neuronx-cc 2.26.6360.0`.
  Always use `421672808698.dkr.ecr.us-east-1.amazonaws.com/concourse-release-0461d3b:latest`.
* **`--per-block` requires `--compile none`**, so that block table is an EAGER measurement
  and is not comparable to compiled runs.
* **The README's 3673.3 ms is an EAGER number** and 275 NEFFs is what eager costs. It cannot be
  reproduced: the eager path is gone from the script and `repro-job.yaml`, which relied on it, is
  deleted. Fused numbers are comparable only to other fused numbers.
* **`d2h` is NOT a transfer measurement, and it must never be used to argue device work is
  small.** The forward returns before the device finishes and the completion barrier is the
  `.cpu()` read, so **async device execution is attributed to `d2h`**. The code says so at
  `run_tiled` (~line 979): `prep+dev+d2h` is reliable as a SUM, the `dev`/`d2h` split is
  indicative only. The fused run's `d2h` 91.4% / `dev` 4.4% was read here as "device work is a
  small fraction" and used to argue against optimising the resample. That was wrong -- it is
  mostly device time sitting behind the sync point. Cite the microbenchmark for device shares,
  never the phase split.
* **`23.6% device` is an EAGER number and does not transfer to the fused config.** It came from
  the 275-NEFF eager run, a regime that is dispatch-bound by construction. Do not use it to size
  the payoff of a device-side optimisation under fusion.
* **Per-engine `*_active_time_percent` in `summary.json` sum past 100%** — they are
  overlapping busy times, not a partition of time. Do not present them as a decomposition.
* **The saved NEFF+NTFF pairs have never been read** by any code here; `profile_roofline.py`
  reads only the JSON. Every profile conclusion comes from `summary.json` aggregates.
* Never `git add -A`: `.ntff` files are hundreds of MB and GitHub rejects at 100 MB.
