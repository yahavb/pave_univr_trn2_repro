#!/usr/bin/env python3
"""Aggregate a torch.profiler chrome trace: is the hotspot the RESAMPLE, and what is it made of?

WHY THIS EXISTS. Under `torch.compile(fullgraph=True)` the whole tile is one graph and a trace of
it holds two "Torch-Compiled Region" slices and no ops -- there is nothing to attribute. Run with
`--warp-region eager --perfetto` and the resample dispatches op by op with a
`resample:C<c>:<h>x<w>` span around every call, so this script can say what fraction of the traced
wall is inside those spans and which ATen ops carry it.

TWO RULES IT ENFORCES, both learned the hard way in this repo:

* ATTRIBUTION IS BY SPAN CONTAINMENT, NOT BY OP NAME. grid_sample's lowering is index / clamp /
  mul / add arithmetic whose op names also appear all over the convs and the flow estimator, so a
  name-based bucket would simultaneously miss the resample's arithmetic and steal ops it never
  issued. Every op here is classified by whether a `resample:` span is one of its ANCESTORS.

* A SYNC IS NOT COMPUTE. The device runs async: the forward returns before it finishes and the
  completion barrier is the host read, so on a CPU-only trace a `.cpu()` / `item` / `_to_copy`
  slice absorbs device time that belongs to whatever produced the tensor. Those ops are reported
  in a separate SYNC line, never folded into a share, and the script says up front whether the
  trace carries device rows at all.

Usage: pf_op_summary.py TRACE.json[.gz] [--top N]
"""
import argparse
import gzip
import json
import os
import sys
from collections import defaultdict

# Host-side slices whose duration is a WAIT on the device, not work. Kept out of every share.
SYNC_HINTS = ("aten::_to_copy", "aten::copy_", "aten::item", "aten::_local_scalar_dense",
              "aten::cpu", "Memcpy", "xla::", "mark_step", "SyncTensors")
# Slices that are the compiled body: real work, but opaque -- no ops inside to attribute.
OPAQUE_HINTS = ("Torch-Compiled Region", "CompiledFunction", "neuron", "NeuronFunction")


def load(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as f:
        d = json.load(f)
    return d["traceEvents"] if isinstance(d, dict) else d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--top", type=int, default=20)
    a = ap.parse_args()
    if not os.path.exists(a.trace):
        print("  pf_op_summary: no trace at %s" % a.trace)
        return 1

    ev = [e for e in load(a.trace) if e.get("ph") == "X" and e.get("dur") is not None]
    if not ev:
        print("  pf_op_summary: trace has no complete ('X') events -- nothing to aggregate. A "
              "0-event trace is the signature of the profiler failing to start, not of an idle run.")
        return 1

    cats = defaultdict(float)
    for e in ev:
        cats[e.get("cat", "?")] += e["dur"]
    dev_rows = [c for c in cats if c not in ("cpu_op", "python_function", "user_annotation",
                                             "cpu_instant", "Trace", "fwdbwd")]
    print("=" * 100)
    print("TRACE %s   %d slices   categories: %s"
          % (os.path.basename(a.trace), len(ev),
             ", ".join("%s(%.0f ms)" % (c, v / 1e3) for c, v in
                       sorted(cats.items(), key=lambda kv: -kv[1]))))
    print("  device rows: %s" % (", ".join(dev_rows) if dev_rows else
                                 "NONE -- every duration below is HOST time (dispatch + waits). "
                                 "Device shares must come from the runtime trace, not from here."))
    print("=" * 100)

    # ── build the containment tree per thread. Chrome traces are flat, so nesting is recovered
    # from (start, end) on the same tid: sort by start, then longest-first so a parent precedes
    # the children it contains, and keep a stack.
    by_tid = defaultdict(list)
    for e in ev:
        by_tid[(e.get("pid"), e.get("tid"))].append(e)

    # ONE THREAD IS THE MEASUREMENT. The trace also holds inductor's compile workers, each with a
    # ~20 ms threading.bootstrap frame, so a max-over-threads denominator or a global op table
    # would mix the model's thread with pool threads that ran no model op. The thread that carries
    # the resample spans IS the model's thread; without spans, fall back to the one with the most
    # ATen ops.
    per = {}
    for tid, evs in by_tid.items():
        st_ = {"roots": 0.0, "resample": 0.0, "sync": 0.0, "opaque": 0.0,
               "self": defaultdict(float), "tot": defaultdict(float), "calls": defaultdict(int),
               "site_us": defaultdict(float), "site_n": defaultdict(int), "n_op": 0, "n_rs": 0}
        per[tid] = st_
        self_us, tot_us, calls = st_["self"], st_["tot"], st_["calls"]
        site_us, site_n = st_["site_us"], st_["site_n"]
        # Longest-first on a tie so a parent is always seen before the children it contains.
        evs.sort(key=lambda e: (e["ts"], -e["dur"]))
        names = [e.get("name", "?") for e in evs]
        durs = [float(e["dur"]) for e in evs]
        child = [0.0] * len(evs)          # direct-children duration, so self = dur - child
        bucket = ["rest"] * len(evs)
        stack = []                        # [(end_ts, index)]
        for i, e in enumerate(evs):
            st, en, nm = e["ts"], e["ts"] + e["dur"], names[i]
            # AN ANCESTOR MUST CONTAIN THE EVENT, and chrome traces are NOT purely nested. The
            # record_function span opens INSIDE the `_record_function_enter_new` frame and outlives
            # it by 580 us, so popping only on `end <= start` left that 3.5 us frame on top and
            # charged the whole resample to it -- the region's self time then read as 2.70 ms of
            # work that was really its child's, and the enter frame took a negative self. So also
            # pop anything that ends before this event does: it cannot be a container.
            while stack and (stack[-1][0] < en or stack[-1][0] <= st):
                stack.pop()
            in_rs = (bucket[stack[-1][1]] == "RESAMPLE" if stack else False) \
                or nm.startswith("resample:")
            bucket[i] = "RESAMPLE" if in_rs else "rest"
            if stack:
                child[stack[-1][1]] += durs[i]
            else:
                st_["roots"] += durs[i]
            # Only an OUTERMOST resample span counts: they do not nest today, and summing nested
            # ones would double count if a future site ever wrapped another.
            if nm.startswith("resample:") and not (stack and bucket[stack[-1][1]] == "RESAMPLE"):
                st_["resample"] += durs[i]
                site_us[nm] += durs[i]
                site_n[nm] += 1
            if e.get("cat") == "cpu_op":
                st_["n_op"] += 1
            tot_us[(nm, bucket[i])] += durs[i]
            calls[(nm, bucket[i])] += 1
            stack.append((en, i))
        for i, nm in enumerate(names):
            s = durs[i] - child[i]
            self_us[(nm, bucket[i])] += s
            if any(h in nm for h in SYNC_HINTS):
                st_["sync"] += s
            if any(h in nm for h in OPAQUE_HINTS):
                st_["opaque"] += s
        st_["n_rs"] = sum(site_n.values())

    main = max(per, key=lambda t: (per[t]["n_rs"], per[t]["n_op"], per[t]["roots"]))
    st_ = per[main]
    self_us, tot_us, calls = st_["self"], st_["tot"], st_["calls"]
    site_us, site_n = st_["site_us"], st_["site_n"]
    resample_us, sync_us, opaque_us = st_["resample"], st_["sync"], st_["opaque"]
    wall = st_["roots"]
    others = sorted((v["roots"] for t, v in per.items() if t != main), reverse=True)
    print()
    print("MODEL THREAD pid=%s tid=%s: %d ATen ops, %d resample spans, %.1f ms of top-level slices"
          % (main[0], main[1], st_["n_op"], st_["n_rs"], wall / 1e3))
    print("  %d other threads excluded (largest %.1f ms) -- inductor compile workers and the "
          "profiler's own rows" % (len(others), others[0] / 1e3 if others else 0.0))
    print("TRACED WALL = this thread's top-level slices: %.1f ms" % (wall / 1e3))
    if wall <= 0:
        return 1
    print("  RESAMPLE spans      : %8.1f ms  %5.1f%% of wall   (%d calls)"
          % (resample_us / 1e3, 100.0 * resample_us / wall, sum(site_n.values())))
    print("  compiled/opaque     : %8.1f ms  %5.1f%% of wall   (self time in %s)"
          % (opaque_us / 1e3, 100.0 * opaque_us / wall, "/".join(OPAQUE_HINTS[:2])))
    print("  sync / host copies  : %8.1f ms  %5.1f%% of wall   NOT compute -- async device time "
          "lands here" % (sync_us / 1e3, 100.0 * sync_us / wall))
    if not site_n:
        print()
        print("  NO resample: spans in this trace. Either the run was not --warp-region eager, or")
        print("  --perfetto was absent so the annotation was never installed. Without them the")
        print("  resample is inside the compiled graph and this question cannot be answered here.")

    if site_n:
        print()
        print("PER SITE (the 14 calls per forward: 6 image warps at C=3, 8 Contextnet at C=16..128)")
        print("  %-28s %6s %10s %10s" % ("span", "calls", "total ms", "mean ms"))
        for nm, us in sorted(site_us.items(), key=lambda kv: -kv[1]):
            print("  %-28s %6d %10.1f %10.3f"
                  % (nm, site_n[nm], us / 1e3, us / 1e3 / max(site_n[nm], 1)))

    print()
    print("TOP OPS BY SELF TIME, split by whether a resample: span contains them")
    print("  %-46s %-9s %7s %10s %10s" % ("op", "bucket", "calls", "self ms", "total ms"))
    for (nm, bk), us in sorted(self_us.items(), key=lambda kv: -kv[1])[:a.top]:
        print("  %-46s %-9s %7d %10.2f %10.2f"
              % (nm[:46], bk, calls.get((nm, bk), 0), us / 1e3, tot_us.get((nm, bk), 0.0) / 1e3))

    rs_self = sum(v for (nm, bk), v in self_us.items() if bk == "RESAMPLE")
    print()
    print("VERDICT INPUT: self time inside resample spans %.1f ms = %.1f%% of the traced wall."
          % (rs_self / 1e3, 100.0 * rs_self / wall))
    print("  Read it against the device-side share, never instead of it: this is host time, and on")
    print("  a CPU-only trace a big number here can be dispatch cost and a small one can hide "
          "device work behind the sync line above.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
