#!/usr/bin/env python3
"""Diagnose a Neuron profile: name the bottleneck, name the source line, no staring at Explorer.

Nothing here knows the word "GpSimd". Engines and DMA modes are DISCOVERED from the summary.json
key names, ranked, and classified by RELATIONSHIPS between them. When tomorrow's hot spot is on
VectorE or on hardware DGE the rules fire the same way and the output names that instead.

Two tiers, because they need different inputs:
  TIER 1  summary.json only. Runs anywhere. Engine ranking, DMA-mode split, payload vs the DMA
          saturation floor, issue-vs-move balance, idle, roofline, and a classification with a
          remedy. Also A/B: --before/--after prints what actually changed.
  TIER 2  needs the instruction table with src= stamps, which comes from neuron-explorer +
          duckdb over the parquet on a Neuron host. Aggregates ALL instruction time BY SOURCE
          LINE, then opens the source and names the op on that line.

UNITS ARE NOT INTERCHANGEABLE and conflating them produced a wrong number once already:
  transfer  a DMA transfer          dma_transfer_count          avg bytes = _average_bytes
  packet    a finer wire unit       <mode>_dma_packet_count     ~40x more numerous
  instr     an engine instruction   <engine>_instruction_count  what the engine actually issues
Every derived figure below states which unit it is in.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

# A DMA transfer below this cannot saturate the engine; it is descriptor-dominated. From the
# Neuron guidance Liran quoted: payloads want >= 2 KiB.
DMA_SATURATION_BYTES = 2048

# Source-line -> "what op is this". Extend freely; unknown lines just print verbatim.
OP_SIGNATURES = [
    (r"F\.interpolate", "F.interpolate -- bilinear resize. If scale_factor and the input shape "
                        "are compile-time constants the tap indices can be PRECOMPUTED on the "
                        "host, which turns indirect DMA into a static access pattern."),
    (r"F\.grid_sample", "F.grid_sample -- flow-guided resample. Its coordinates come from a "
                        "tensor, so they CANNOT be precomputed; this one needs a kernel."),
    (r"index_select|\.gather\(", "indexed gather. Static if the index tensor is a constant, "
                                 "dynamic if it is computed."),
    (r"F\.conv2d|nn\.Conv2d|conv_transpose2d|self\.conv\d*\(|self\.down\d|self\.up\d|conv\(",
     "convolution -- honest compute, usually not fixable."),
    (r"_tb\(|lambda ", "a wrapper/closure, not the op. The src stamp landed on a call site; look "
                      "at what it calls."),
    (r"linspace|arange", "coordinate construction. Constant if the shape is static -- candidate "
                         "for host precompute."),
    (r"nl\.|nisa\.|@nki", "an NKI kernel -- read it directly, the stamp is inside your own code."),
    (r"torch\.cat|F\.pad", "concat/pad -- pure data movement; often fusable into the consumer."),
    (r"\.permute\(|\.transpose\(|\.contiguous\(", "layout change -- a transpose or copy."),
    (r"avg_pool2d|max_pool2d", "pooling -- static access pattern, cheap."),
    (r"F\.relu|sigmoid|prelu", "activation -- elementwise, rarely the bottleneck."),
    (r"\.item\(\)|\.cpu\(\)", "host sync -- forces a device barrier."),
]


def load_summary(path):
    """summary.json is {"n_<hash>": {...}}; some tools emit the inner dict directly."""
    with open(path) as f:
        d = json.load(f)
    if isinstance(d, dict) and len(d) == 1 and isinstance(next(iter(d.values())), dict):
        return next(iter(d.values()))
    return d


def discover(s):
    """Find every engine and DMA mode by KEY SHAPE, so a new engine needs no code change."""
    engines, modes = {}, {}
    for k, v in s.items():
        m = re.fullmatch(r"(\w+)_engine_active_time_percent", k)
        if m and isinstance(v, (int, float)):
            engines[m.group(1)] = v
        m = re.fullmatch(r"(\w+)_dma_active_time_percent", k)
        if m and isinstance(v, (int, float)):
            modes[m.group(1)] = v
    if isinstance(s.get("dma_active_time_percent"), (int, float)):
        modes["dma"] = s["dma_active_time_percent"]
    # cc_cores reports as *_instruction_active_time_percent
    for k, v in s.items():
        m = re.fullmatch(r"(\w+)_instruction_active_time_percent", k)
        if m and m.group(1) not in engines and isinstance(v, (int, float)):
            engines[m.group(1)] = v
    return engines, modes


def num(s, k, default=0.0):
    v = s.get(k, default)
    return v if isinstance(v, (int, float)) else default


def per(a, b):
    return (a / b) if b else float("nan")


def metrics(s):
    engines, modes = discover(s)
    # "dma" is the aggregate; the others are its breakdown. Keep them apart.
    agg = modes.pop("dma", 0.0)
    total = num(s, "total_time")
    idle = 1.0 - num(s, "total_active_time_percent")

    # The issuing side is whichever ENGINE is busiest. Never hardcoded.
    top_eng = max(engines.items(), key=lambda kv: kv[1]) if engines else ("?", 0.0)
    top_mode = max(modes.items(), key=lambda kv: kv[1]) if modes else ("?", 0.0)

    m = {
        "total_s": total,
        "idle_frac": idle,
        "engines": engines,
        "modes": modes,
        "dma_agg_pct": agg,
        "top_engine": top_eng,
        "top_dma_mode": top_mode,
        # issue-vs-move: how much busier is the busiest engine than the DMA engine
        "issue_move_ratio": per(top_eng[1], agg),
        "bytes_per_transfer": num(s, "dma_transfer_average_bytes"),
        "transfers": num(s, "dma_transfer_count"),
        "mfu": num(s, "mfu_estimated_percent"),
        "mfu_ceiling": num(s, "mfu_max_achievable_estimated_percent"),
        "mbu": num(s, "mbu_estimated_percent"),
        "matmuls": num(s, "matmul_instruction_count"),
        "transpose_flop_frac": per(num(s, "transpose_flops"), num(s, "hardware_flops")),
        "spill_bytes": num(s, "spill_reload_bytes") + num(s, "spill_save_bytes"),
        "sbuf_bytes": num(s, "sbuf_read_bytes") + num(s, "sbuf_write_bytes"),
        "hbm_bytes": num(s, "hbm_read_bytes") + num(s, "hbm_write_bytes"),
    }
    # per-mode payload, in BYTES PER PACKET -- labelled, because packets != transfers
    m["mode_payload"] = {}
    for mode in modes:
        n = num(s, "%s_dma_packet_count" % mode)
        sz = num(s, "%s_dma_size" % mode)
        if n:
            m["mode_payload"][mode] = (per(sz, n), int(n), sz)
    # The payload test must run on the mode that OWNS the DMA time, not on the global
    # average: 2286 B/transfer overall hid a 29 B/packet software path underneath it.
    leaf = {k: v for k, v in modes.items() if k in m["mode_payload"]}
    m["dominant_leaf"] = max(leaf.items(), key=lambda kv: kv[1]) if leaf else ("?", 0.0)
    # per-engine ns per instruction -- what an issue-bound engine is really spending
    m["engine_ns_per_instr"] = {}
    for e in engines:
        n = num(s, "%s_engine_instruction_count" % e)
        t = num(s, "%s_engine_active_time" % e)
        if n:
            m["engine_ns_per_instr"][e] = (t / n * 1e9, int(n))
    return m


def classify(m):
    """Ordered rules. Each returns (severity, label, evidence, remedy). Generic by construction:
    the rules test RATIOS between discovered quantities, never a specific engine name."""
    out = []
    te, tv = m["top_engine"]
    tm, tmv = m["top_dma_mode"]
    # Sum only LEAF modes. "dynamic" is a roll-up of hardware+software, so including it
    # double-counted and reported 22% where the true dynamic share is 10.8%.
    leaves = m["mode_payload"]
    dyn = sum(v for k, v in m["modes"].items() if "dynamic" in k and k in leaves)
    sta = sum(v for k, v in m["modes"].items() if k == "static")

    if tv > 0.30 and m["issue_move_ratio"] > 2.0 and dyn > sta:
        ns = m["engine_ns_per_instr"].get(te, (float("nan"), 0))
        out.append((
            "HIGH", "DESCRIPTOR-ISSUE BOUND on %s" % te.upper(),
            "%s is %.0f%% active while the DMA engine is only %.0f%% -- %.1fx. %s issues "
            "%d instructions at %.0f ns each. Dynamic DMA is %.1f%% of NEFF time against "
            "%.1f%% static, so addresses are being supplied at RUNTIME."
            % (te, tv*100, m["dma_agg_pct"]*100, m["issue_move_ratio"], te,
               ns[1], ns[0], dyn*100, sta*100),
            "The mover is starved by the issuer, so more DMA queues will not help. Make the "
            "addresses STATIC: if the index source is a compile-time constant, precompute it on "
            "the host. If it is a tensor, this needs a kernel."))

    dl, dlv = m["dominant_leaf"]
    pay = m["mode_payload"].get(dl, (float("nan"), 0, 0))
    if pay[0] == pay[0] and pay[0] < DMA_SATURATION_BYTES:
        out.append((
            "HIGH", "PAYLOAD BELOW DMA SATURATION on %s" % dl,
            "%s owns %.0f%% of DMA time and moves %.0f B per packet over %d packets; saturation "
            "wants >= %d B, so it is %.0fx short. The whole-NEFF average of %.0f B per transfer "
            "HIDES this -- always read the dominant mode, not the aggregate."
            % (dl, dlv*100, pay[0], pay[1], DMA_SATURATION_BYTES,
               DMA_SATURATION_BYTES / pay[0], m["bytes_per_transfer"]),
            "Time is per-transfer overhead, not bandwidth. Fewer, larger transfers -- coalesce "
            "the access pattern rather than speeding up each one."))

    if m["idle_frac"] > 0.15:
        out.append((
            "MED", "STALLED %.0f%% OF THE TIME" % (m["idle_frac"]*100),
            "total_active_time_percent leaves %.0f%% with no engine busy." % (m["idle_frac"]*100),
            "Something is serialising. Look for the engine that everything waits on -- it is "
            "usually the one ranked first above, not the Sync entries, which are the WAIT."))

    if m["transpose_flop_frac"] > 0.02:
        out.append((
            "MED", "TRANSPOSE-HEAVY",
            "transpose_flops are %.1f%% of hardware_flops." % (m["transpose_flop_frac"]*100),
            "A layout mismatch is being paid for in FLOPs. Check for permute/contiguous on the "
            "hot path, or a kernel asserting a layout the model does not use."))

    if m["sbuf_bytes"] and m["spill_bytes"] / m["sbuf_bytes"] > 0.05:
        out.append((
            "MED", "SPILLING",
            "spill traffic is %.1f%% of SBUF traffic (%.0f MB)."
            % (100*m["spill_bytes"]/m["sbuf_bytes"], m["spill_bytes"]/1e6),
            "The working set does not fit SBUF. Smaller tiles, or fewer live tensors."))

    if m["matmuls"] > 0 and te.startswith("tensor") and m["idle_frac"] < 0.15:
        out.append((
            "INFO", "COMPUTE BOUND -- probably not fixable",
            "TensorE leads at %.0f%% with %d matmuls and only %.0f%% idle."
            % (tv*100, m["matmuls"], m["idle_frac"]*100),
            "This is real work. Do not 'optimise' it by looking at its source line."))

    if not out:
        out.append(("INFO", "NO DOMINANT PATTERN", "No rule fired.",
                    "Compare against another NEFF with --before/--after."))
    return out


def fmt_tier1(name, s, args):
    m = metrics(s)
    W = 92
    print("=" * W)
    print("NEFF %s   total %.2f ms   idle %.0f%%" % (name, m["total_s"]*1e3, m["idle_frac"]*100))
    print("=" * W)

    print("\nENGINES (active % of NEFF time; they OVERLAP, so they do not sum to 100)")
    for e, v in sorted(m["engines"].items(), key=lambda kv: -kv[1]):
        ns, n = m["engine_ns_per_instr"].get(e, (float("nan"), 0))
        bar = "#" * int(round(v * 40))
        print("  %-12s %6.1f%%  %-40s %9d instr  %8.0f ns/instr" % (e, v*100, bar, n, ns))

    print("\nDMA  aggregate %.1f%% active   %.0f B per transfer over %d transfers"
          % (m["dma_agg_pct"]*100, m["bytes_per_transfer"], m["transfers"]))
    for mode, v in sorted(m["modes"].items(), key=lambda kv: -kv[1]):
        if mode == "dma":
            continue
        if mode not in m["mode_payload"]:
            print("  %-22s %6.1f%% active  (roll-up, no packet counter of its own)" % (mode, v*100))
            continue
        pay, n, sz = m["mode_payload"][mode]
        star = "  <-- owns the DMA time" if mode == m["dominant_leaf"][0] else ""
        print("  %-22s %6.1f%% active  %11d packets  %7.0f B/packet  %8.1f MB%s"
              % (mode, v*100, n, pay, sz/1e6, star))

    print("\nROOFLINE  MFU %.3f%% of a %.1f%% ceiling   MBU %.3f%%   HBM %.0f MB"
          % (m["mfu"]*100, m["mfu_ceiling"]*100, m["mbu"]*100, m["hbm_bytes"]/1e6))

    print("\nDIAGNOSIS")
    for sev, label, ev, fix in classify(m):
        print("  [%-4s] %s" % (sev, label))
        for line in wrap(ev, W - 10):
            print("         %s" % line)
        for line in wrap("FIX: " + fix, W - 10):
            print("         %s" % line)
        print()
    return m


def wrap(s, n):
    out, cur = [], ""
    for w in s.split():
        if len(cur) + len(w) + 1 > n:
            out.append(cur); cur = w
        else:
            cur = (cur + " " + w).strip()
    if cur:
        out.append(cur)
    return out


def diff(a, b, na, nb):
    """What actually changed. This is the mode you want when validating a fix."""
    ma, mb = metrics(a), metrics(b)
    print("=" * 92)
    print("A/B   before=%s   after=%s" % (na, nb))
    print("=" * 92)
    rows = [("total_time ms", ma["total_s"]*1e3, mb["total_s"]*1e3, "lower"),
            ("idle %", ma["idle_frac"]*100, mb["idle_frac"]*100, "lower"),
            ("B / transfer", ma["bytes_per_transfer"], mb["bytes_per_transfer"], "higher"),
            ("transfers", ma["transfers"], mb["transfers"], "lower")]
    for e in sorted(set(ma["engines"]) | set(mb["engines"])):
        rows.append(("engine %s %%" % e, ma["engines"].get(e, 0)*100,
                     mb["engines"].get(e, 0)*100, None))
    for mo in sorted(set(ma["modes"]) | set(mb["modes"])):
        rows.append(("dma %s %%" % mo, ma["modes"].get(mo, 0)*100,
                     mb["modes"].get(mo, 0)*100, None))
        pa = ma["mode_payload"].get(mo, (0, 0, 0))
        pb = mb["mode_payload"].get(mo, (0, 0, 0))
        rows.append(("  %s packets" % mo, pa[1], pb[1], None))
    print("  %-26s %14s %14s %10s" % ("metric", "before", "after", "change"))
    for label, x, y, _want in rows:
        if not x and not y:
            continue
        ch = "%+.1f%%" % (100*(y-x)/x) if x else ("new" if y else "-")
        if x and y and (y/x > 3 or x/y > 3):
            ch += "  <<<"
        print("  %-26s %14.4g %14.4g %10s" % (label, x, y, ch))
    print("\nDIAGNOSIS AFTER")
    for sev, label, ev, fix in classify(mb):
        print("  [%-4s] %s" % (sev, label))
    print("\nDIAGNOSIS BEFORE (for reference)")
    for sev, label, ev, fix in classify(ma):
        print("  [%-4s] %s" % (sev, label))


SRC_RE = re.compile(r"src=(?P<file>[^\s:]+):(?P<line>\d+)")
INSTR_RE = re.compile(
    r"^\s*(?P<engine>\w+)\s+(?P<opcode>[A-Z_0-9]+)\s+n=(?P<n>\d+)\s+dur=\s*(?P<dur>[\d.]+)ms")


def tier2(instr_file, src_root, top):
    """Aggregate instruction time BY SOURCE LINE, then open the source and name the op.

    Input is the `top instructions by duration` block the analyze job already prints, or any
    text with `ENGINE OPCODE n=<N> dur=<D>ms ... src=<file>:<line>` lines.
    """
    by_src, by_op, total = {}, {}, 0.0
    for raw in open(instr_file, errors="replace"):
        mi = INSTR_RE.match(raw)
        if not mi:
            continue
        dur = float(mi.group("dur"))
        total += dur
        key = (mi.group("engine"), mi.group("opcode"))
        by_op[key] = by_op.get(key, [0, 0.0])
        by_op[key][0] += int(mi.group("n")); by_op[key][1] += dur
        ms = SRC_RE.search(raw)
        if ms:
            sk = (ms.group("file"), int(ms.group("line")))
            by_src[sk] = by_src.get(sk, [0.0, set()])
            by_src[sk][0] += dur
            by_src[sk][1].add("%s %s" % (mi.group("engine"), mi.group("opcode")))
    if not total:
        print("no instruction lines matched -- expected 'ENGINE OPCODE n=N dur=Dms ... src=f:L'")
        return

    print("=" * 92)
    print("WHERE THE TIME GOES, BY SOURCE LINE   (%.1f ms attributed)" % total)
    print("=" * 92)
    for (f, ln), (dur, ops) in sorted(by_src.items(), key=lambda kv: -kv[1][0])[:top]:
        print("\n  %6.1f ms  %5.1f%%   %s:%d" % (dur, 100*dur/total, f, ln))
        print("            via %s" % ", ".join(sorted(ops)))
        text = read_line(f, ln, src_root)
        if text is None:
            print("            (source not found -- pass --src-root pointing at the tree that "
                  "BUILT this NEFF, at the right commit)")
            continue
        print("            >>> %s" % text.strip())
        hits = [why for rx, why in OP_SIGNATURES if re.search(rx, text)]
        if hits:
            for why in hits:
                for line in wrap("THIS IS: " + why, 76):
                    print("            %s" % line)
        else:
            print("            (no known op signature on this line; widen OP_SIGNATURES)")

    print("\n" + "=" * 92)
    print("BY ENGINE x OPCODE")
    print("=" * 92)
    print("  %-10s %-22s %10s %10s %8s" % ("engine", "opcode", "count", "ms", "share"))
    for (e, op), (n, dur) in sorted(by_op.items(), key=lambda kv: -kv[1][1])[:top]:
        print("  %-10s %-22s %10d %10.2f %7.1f%%" % (e, op, n, dur, 100*dur/total))


def read_line(f, ln, src_root):
    cands = [f]
    if src_root:
        cands.append(os.path.join(src_root, os.path.basename(f)))
        cands.append(os.path.join(src_root, f.lstrip("/")))
    for c in cands:
        try:
            with open(c, errors="replace") as fh:
                lines = fh.readlines()
            if 1 <= ln <= len(lines):
                return lines[ln - 1]
        except OSError:
            continue
    return None


def find_summaries(paths):
    out = []
    for p in paths:
        if os.path.isdir(p):
            for root, _d, files in os.walk(p):
                if "summary.json" in files:
                    out.append((os.path.basename(root), os.path.join(root, "summary.json")))
        elif p.endswith(".json"):
            out.append((os.path.basename(os.path.dirname(p)) or p, p))
    return sorted(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="*", help="pair dirs or summary.json files (tier 1)")
    ap.add_argument("--before", help="summary.json or pair dir (A/B mode)")
    ap.add_argument("--after", help="summary.json or pair dir (A/B mode)")
    ap.add_argument("--instr", help="text with the instruction table incl. src= (tier 2)")
    ap.add_argument("--src-root", default=".", help="tree that BUILT the NEFF, for src= lookup")
    ap.add_argument("--top", type=int, default=12)
    a = ap.parse_args()

    if a.before and a.after:
        (nb, pb), = find_summaries([a.before]) or [(a.before, a.before)]
        (nf, pf), = find_summaries([a.after]) or [(a.after, a.after)]
        diff(load_summary(pb), load_summary(pf), nb, nf)
    elif a.paths:
        found = find_summaries(a.paths)
        if not found:
            sys.exit("no summary.json under %s" % ", ".join(a.paths))
        ranked = []
        for name, p in found:
            s = load_summary(p)
            ranked.append((metrics(s)["total_s"], name, s))
        for _t, name, s in sorted(ranked, reverse=True):
            fmt_tier1(name, s, a)
        if len(ranked) > 1:
            print("=" * 92)
            print("RANKED BY TIME  (fix the top entry whose diagnosis is not COMPUTE BOUND)")
            print("=" * 92)
            for t, name, s in sorted(ranked, reverse=True):
                labels = [l for _s, l, _e, _f in classify(metrics(s))]
                print("  %8.2f ms  %-28s %s" % (t*1e3, name, labels[0] if labels else "-"))

    if a.instr:
        print()
        tier2(a.instr, a.src_root, a.top)

    if not a.paths and not (a.before and a.after) and not a.instr:
        ap.error("give pair dirs, or --before/--after, or --instr")


if __name__ == "__main__":
    main()
