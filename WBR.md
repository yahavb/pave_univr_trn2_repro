# WBR — UniVR rolling-shutter correction on Trainium: 7.1x faster, and it now passes accuracy

## Headline

We made the model **7.1x faster on Trainium and simultaneously fixed a correctness failure that
had blocked the port for weeks.** Two changes did it, and neither works without the other.

| | before | after |
|---|---|---|
| latency | 758.9 ms | **106.8 ms** |
| accuracy vs reference | 52.65 LSB — **FAIL** | **0.00 LSB — PASS** |
| image quality | 44.08 dB | **122.79 dB** |
| device time | 761.4 ms | **188.7 ms** |

The bar is 3 LSB. Before this work nothing we ran had ever passed it.

Measured back-to-back in one job at 288x320, one NeuronCore, fp32, production weights.
Independently reproduced in a second job to the same 106.8 ms.

**At full production tile size we measure 2.8x**, not 7.1x — because the larger tile can only run
one of the two changes (see *The compiler constraint*). Both numbers are real; they answer
different questions.

## What the model actually does

Rolling-shutter cameras expose the image one scanline at a time, so a fast-moving subject is
skewed. The model removes that skew by estimating how each pixel moved and warping it back:

1. **Tile** the 4K frame into a grid, because a whole frame does not fit on one core.
2. **Crop** each tile's inputs, with an overlap margin so tile edges stay correct.
3. **Warp** — move every output pixel from a displaced input location.
4. **Stack** the moved pixels into one tensor.
5. **Shrink** that tensor, because large motion is easiest to find at low resolution.
6. **Convolve** the shrunk tensor to predict a motion correction, then repeat at finer scale.

Steps 3 and 5 both read memory at computed locations. That is where the whole story sits.

## What we changed

**Change 1 — where addresses get computed.**

Step 5's shrink used `F.interpolate`, which derives its read addresses *on the device, at runtime*.
The hardware then cannot use its fast bulk-transfer path; it falls back to a mode where a helper
engine must hand-build a descriptor for every small read. That engine became the bottleneck and
the data-mover sat idle waiting for it.

The shrink factor is fixed and known before the model ever runs. So we compute the addresses
**once on the host** and bake them in as constants. Same arithmetic, same result — verified exact
to 2.4e-07 — but the compiler now sees a regular pattern instead of a list of addresses.

Isolating this one change (holding the warp fixed), the profiler shows:

| | before | after |
|---|---|---|
| descriptor-issuing instructions | 237,197 | **141,738** (−40%) |
| cost per instruction | 1209 ns | 1200 ns — **unchanged** |
| small-transfer packets | 7.50 M | **4.48 M** (−40%) |

The cost per operation did not improve at all. We simply stopped issuing 40% of the operations,
and the time fell by the same 40%. That 1:1 relationship is the proof the diagnosis was right.

**Change 2 — how the warp is written.**

Step 3 can be expressed two ways, mathematically identical (we measured them equal to 0.003 LSB):

- `gridsample` — one library call that does the work internally.
- `index_select` — the same four reads and blend, written out explicitly.

`gridsample` issues up to 200x more small transfers than `index_select`, which sits at the
hardware's theoretical minimum. That matters because **the address fix is only worth whatever the
warp leaves on the table:**

| the change we made | in isolation | with the other change also made |
|---|---|---|
| host-computed addresses | 1.6x | **3.9x** |
| explicit warp | 1.8x | **4.4x** |

**They multiply: 1.8 x 3.9 = 7.1, against a measured 7.11x.**

That is the central finding. Neither change is worth much alone, because whichever source of
small transfers you leave behind keeps the data-mover starved. Fix both and the starvation clears:

| share of device time | before | after |
|---|---|---|
| helper engine building addresses | **71.1%** | **33.8%** |
| arithmetic units | **2.4%** | **21.9%** |
| bulk-pattern transfers | 1.0% | **5.5%** |
| total device time | 761.4 ms | **188.7 ms** |

The hardware finally spends its time computing instead of preparing to move data.

## The compiler constraint, stated plainly

We would prefer to compile each tile as a single fused unit — it is ~2% faster and simpler to
reason about. **With `gridsample` we cannot: the compiler runs out of memory trying.** Two attempts
at production tile size were killed after 9 and 10 hours having consumed 1800 GB, without ever
producing a runnable artifact.

The workaround is to deliberately **break the graph** around the warp so it is handled separately.
That works, and it is how every earlier measurement was obtained — but it costs the fusion benefit
and makes performance harder to attribute.

Two things worth flagging to leadership:

- **This is a compiler limitation, not a model or hardware limitation.** The same operation
  compiles without difficulty at 5.9x smaller tiles (288x320). It is a scaling limit in the
  toolchain, and we have the reproduction to file.
- **Choosing `index_select` removes the problem entirely.** It fuses at production tile size and is
  faster. So our recommended configuration does not depend on the compiler improving.

**Recommended configuration:** precomputed addresses + `index_select` + single fused graph.

## Where we stand, and what is next

| | |
|---|---|
| full 4K frame, one core (measured) | 11.3 s |
| projected across 8 cores | ~1.5 s |
| GPU reference (L40S, ONNX+TensorRT) | 0.161 s |

Still meaningfully behind GPU. The next lever is not inside the model — it is **wasted overlap**.
Our current tiling computes 2.13x the pixels in a frame because of edge margins. A larger tile with
a smaller margin brings that to 1.086x, i.e. roughly **half the total work**, for no accuracy cost
we know of. That is a bigger remaining opportunity than anything left in the model code, and the
reference implementation already uses that geometry.

Beyond it, the remaining bottleneck is step 3's warp, whose addresses genuinely depend on runtime
data and therefore cannot be precomputed the way step 5's were. Improving it means larger, fewer
transfers — a kernel-level effort, not a flag.

## Caveats we are carrying

* Both the 7.1x and the 2.8x are **one tile**, not a whole frame, and at different tile sizes.
  The 8-core figure is a projection from a one-core measurement, not a measurement.
* Four of our timed runs had a compilation land inside the measurement window. The medians survive
  it and agree across independent runs to under 1%, but our harness now checks for this explicitly.
* fp32 only. Lower precision was measured at 23 dB — far outside the quality bar — so it is closed.
