# NCC_EBIR033: whole-graph `torch.compile` fails on a 14-resample forward, for any resample implementation

`torch.compile(backend="neuron", dynamic=False, fullgraph=True)` over this model's whole
forward fails in the compiler backend. Dynamo traces the graph successfully; `neuronx-cc`
then rejects it.

The failure is **independent of how the resample is written**. Three implementations —
one NKI kernel and two plain PyTorch formulations — produce the identical error code and
the identical operand sizes.

## What "one graph" means here: PER TILE, never the 4K frame

Worth stating up front, because it is easy to read this document as "the 4K frame will not
compile as one graph" and then chase the wrong target.

**The 4K frame is never one graph, at any tile size, by construction.** 1728x4096 is always
split into N tiles that are dispatched across 8 NeuronCores. `build_replicas()` builds one
model replica per core and a `ThreadPoolExecutor(max_workers=ncore)` runs them, so tiles
execute as separate graphs on separate cores. Fusing the frame into a single graph is not
something this design attempts, and shrinking the tile does not move toward it.

What `--compile whole` fuses is **the forward pass of ONE tile** — its 14 resamples plus the
conv trunks, U-Net and Contextnet, into one graph. The script reports the graph inventory
directly:

```
padded tile shapes: 992x1152, 992x1280  -> 2 distinct graph(s) per timestamp
x2 timestamps (triplet)                 -> up to 4 graphs to compile
```

Four graphs, because the tiling produces two distinct padded shapes (edge and interior) and
a triplet runs two timestamps. Two shapes x two timestamps = 4. The count is set by the
number of distinct *shapes*, not by the number of tiles: 8 tiles reuse the same 2 shapes,
so a compiled 2x4 run would still be 4 graphs, and each additional distinct shape adds one.

So the prize is per-tile-shape, and it is a dispatch-count prize:

| | graphs / NEFFs |
|---|---|
| eager (`--compile none`, what every run so far used) | **275 NEFFs cached** |
| whole-graph, if it compiled | **~4** |

The 275 is what an eager run actually caches — every op dispatched separately. The ~4 is the
script's own reported graph structure. Closing that gap is the reason to care about fusion:
`~/pave-unrolling/RESULTS.md` measured 0.82 s of device-active time against a 16.5 s frame
and concluded "95% is dispatch/idle ... the real lever is batching the dispatches."

The practical consequence: because the rejection scales with `px x 14`, buying fusion means
shrinking the tile, which raises tile count and therefore total pixel work (every tile
re-reads its halo). That is the trade `fusion-threshold-job.yaml` exists to price — not a
route to a single 4K graph, which is not on the table.

## Error

```
[ERROR] [NCC_EBIR033] Number of elements in dimension 0 of InstDMACopy's input and
dimension 0 of output AccessPatterns must match, but got in=17776640 and out=128
- Make sure the number of elements is the same in all dimensions of the in/out AccessPatterns
```

Raised as:

```
torch._dynamo.exc.BackendCompilerFailed: backend='neuron' raised:
RuntimeError: Neuron backend NEFF execution setup failed with unexpected error:
Compilation error occurred on Neuron for operation=torch_compile; ...
```

## The operand sizes are structural

```
17,776,640 = 1,269,760 x 14
             |           |
             |           +-- resamples per forward pass
             +-- pixels in the padded tile (992 x 1280)

128        = chosen by the backend, not by the model
```

So under whole-graph fusion the backend forms a single `InstDMACopy` spanning all 14
resamples of the forward, then cannot reconcile it against a 128-element output access
pattern. `out=128` is not a property of the model: it appears even with stock
`F.grid_sample`.

## Three implementations, one error

| `--warp` | resample implementation | result |
|---|---|---|
| `gather` | plain PyTorch `index_select`, 4 taps, no NKI | `NCC_EBIR033`, `in=17776640 and out=128` |
| `gridsample` | stock `F.grid_sample` | `NCC_EBIR033`, `in=17776640 and out=128` |
| `nki-dyn` | NKI indirect-DMA kernel, device-side loop | `NCC_EBIR033`, `in=17776640 and out=128` |

Same code, same operand sizes, all three. This rules out the NKI kernel as the cause and
rules out the gather formulation generally.

The failing DMA instruction is reported as `I-21406-0_VN_0` in every case, identical across
both full runs of the sweep and across all three implementations, so the failure is
deterministic rather than a scheduling or resource artefact.

`NEURON_LAUNCH_BLOCKING=1` does not change the report: the rejection happens during
compilation, not at dispatch, so there is nothing for blocking mode to serialise.

## Reproduce

```bash
for WARP in gather gridsample nki-dyn; do
  NEURON_LAUNCH_BLOCKING=1 python -u repro_unrolling_trn2.py \
    --rs0 rs70.png --rs1 rs71.png --rs2 rs72.png \
    --gt ref_cuda_fp32_1728x4096.npy --weights pre_net_flow.pth \
    --height 1728 --width 4096 --tiles 2x4 --halo 128 --bar 3 \
    --device neuron --cores 1 --only-tile 1 \
    --warp $WARP --dtype fp32 --compile whole --iters 0
done
```

One tile, one core, single forward. `k8s: repro-job.yaml` runs the same sweep.

`--warp gather` and `--warp gridsample` need no NKI toolchain, so the failure reproduces
on a plain torch + torch_neuronx install.

## Environment

```
neuronx-cc     2.26.6360.0+6f180f47
torch          2.11.0+cu130
torch-neuronx  2.11.3.0.1417+1431f083
nki            0.5.0+28631259367.ga768afa6
python         3.12.11
node           trn3-dev1, 8 NeuronCores at LNC=1
NEURON_CC_FLAGS  --target trn2 --lnc 1 --model-type unet-inference
```

## What works

* **Eager (`--compile none`)** compiles and runs. One tile on one core: 2172.6 ms,
  PSNR 92.04 dB, max_diff 1.75 LSB, PASS against the <= 3 LSB bar.
* Per-submodule graph boundaries avoided this error historically, which is consistent
  with the fusion of all 14 resamples being the trigger.

## What we are asking

1. Is a whole-forward graph containing 14 data-dependent resamples expected to be
   expressible on this backend, or is a graph boundary required between them?
2. `out=128` is generated by the backend. What sets it, and can it be made to agree
   with the fused input extent?
3. Is there a supported way to get one graph for this model, short of restoring the
   per-submodule boundaries?

## Caveats, stated plainly

* The error surfaces at a synchronization point. `NEURON_LAUNCH_BLOCKING=1` is set in the
  sweep above so the report is at the failing dispatch, but the traceback's Python frames
  still point at the enclosing forward rather than at a specific fused op.
* We have not established whether the backend or the graph is at fault. The evidence here
  is only that the fault does not depend on the resample implementation.
* Whether accuracy would be correct if this compiled is untested, for the obvious reason.
