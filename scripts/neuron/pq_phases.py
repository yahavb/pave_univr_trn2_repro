#!/usr/bin/env python3
"""Split a NEFF into PHASES over time and diagnose each one separately.

WHY THIS EXISTS. Whole-NEFF aggregates say "GpSimd 66%, TensorE 14%" -- which is true and
useless, because it averages a graph that is TensorE-bound for one stretch and descriptor-bound
for another into a single number that describes neither. A fix aimed at the average misses.

WHAT IT DOES. Buckets the instruction stream over time, computes per-engine and DMA occupancy per
bucket, merges adjacent buckets that share a dominant engine into a phase, and runs the SAME
issue-vs-move test per phase that diagnose_neff.py runs on the whole NEFF. Then it names the
source line that dominates each phase, so a phase points at code.

GENERIC. Engine names come from the data. No rule mentions an engine. A phase led by VectorE with
starved DMA is reported exactly the way a GpSimd one is.

SCHEMA IS DISCOVERED, NEVER ASSUMED -- same contract as pq_timeline.py. Every column used is
matched against the real parquet header and printed. A missing field is reported, not substituted.

Usage: pq_phases.py <parquet-dir> [label] [--buckets N] [--min-phase-frac F] [--src-root DIR]
"""
import glob
import os
import re
import sys

import duckdb

TIME_HINTS = ("start_ts", "start", "timestamp", "begin", "ts")
DUR_HINTS = ("duration_ns", "duration", "elapsed", "active_time", "busy")
ENG_HINTS = ("engine",)
OP_HINTS = ("opcode", "hlo_name", "name")
SRC_HINTS = ("nki_source_location", "source_location", "nki_source", "debug_info")

# Same floor and ratio the whole-NEFF classifier uses. Kept in sync deliberately: a phase verdict
# and a NEFF verdict that disagree because of different constants would be unreadable.
DMA_SATURATION_BYTES = 2048
ISSUE_MOVE_RATIO = 2.0

OP_SIGNATURES = [
    (r"F\.interpolate", "F.interpolate -- constant scale_factor means the tap indices are "
                        "precomputable on the host"),
    (r"F\.grid_sample", "F.grid_sample -- coordinates come from a tensor, NOT precomputable"),
    (r"index_select|\.gather\(", "indexed gather"),
    (r"conv|self\.down|self\.up", "convolution -- usually honest compute"),
    (r"torch\.cat|F\.pad", "concat/pad -- data movement"),
    (r"permute|transpose|contiguous", "layout change"),
    (r"linspace|arange", "coordinate construction -- constant if the shape is static"),
]


def pick(cols, hints, exclude=()):
    low = {c.lower(): c for c in cols}
    for h in hints:
        for lc, orig in low.items():
            if h in lc and not any(x in lc for x in exclude):
                return orig
    return None


def table_of(con, files, want):
    """Return (view_name, columns) for the parquet whose basename matches `want`."""
    for f in files:
        base = os.path.basename(f).rsplit(".", 1)[0]
        if base.lower() == want.lower():
            v = "t_" + base
            con.execute("CREATE OR REPLACE VIEW %s AS SELECT * FROM read_parquet('%s')" % (v, f))
            cols = [r[0] for r in con.execute("DESCRIBE %s" % v).fetchall()]
            return v, cols
    return None, []


def read_src_line(path, ln, src_root):
    for c in (path, os.path.join(src_root or ".", os.path.basename(path)),
              os.path.join(src_root or ".", path.lstrip("/"))):
        try:
            with open(c, errors="replace") as fh:
                lines = fh.readlines()
            if 1 <= ln <= len(lines):
                return lines[ln - 1].strip()
        except OSError:
            continue
    return None


def name_op(text):
    if not text:
        return None
    for rx, why in OP_SIGNATURES:
        if re.search(rx, text):
            return why
    return None


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    pqdir = sys.argv[1]
    label = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith("--") else ""
    buckets, min_frac, src_root = 200, 0.02, None
    for i, a in enumerate(sys.argv):
        if a == "--buckets" and i + 1 < len(sys.argv):
            buckets = int(sys.argv[i + 1])
        if a == "--min-phase-frac" and i + 1 < len(sys.argv):
            min_frac = float(sys.argv[i + 1])
        if a == "--src-root" and i + 1 < len(sys.argv):
            src_root = sys.argv[i + 1]

    files = sorted(glob.glob(os.path.join(pqdir, "*.parquet")))
    if not files:
        print("no parquet in %s" % pqdir)
        return 1
    con = duckdb.connect()

    ins, icols = table_of(con, files, "Instruction")
    if not ins:
        print("PHASES: no Instruction table in %s -- cannot build a time series" % pqdir)
        return 1
    c_t = pick(icols, TIME_HINTS, exclude=("end",))
    c_d = pick(icols, DUR_HINTS)
    c_e = pick(icols, ENG_HINTS)
    c_o = pick(icols, OP_HINTS)
    c_s = pick(icols, SRC_HINTS)
    print("=" * 100)
    print("PHASES  %s   buckets=%d" % (label or pqdir, buckets))
    print("=" * 100)
    print("  columns matched from the real header: time=%s dur=%s engine=%s op=%s src=%s"
          % (c_t, c_d, c_e, c_o, c_s))
    if not (c_t and c_d and c_e):
        print("  MISSING a required field -- refusing to guess. No phase table.")
        return 1

    lo, hi = con.execute("SELECT min(%s), max(%s+%s) FROM %s" % (c_t, c_t, c_d, ins)).fetchone()
    span = (hi or 0) - (lo or 0)
    if not span:
        print("  zero span")
        return 1
    w = span / float(buckets)

    # DMA occupancy per bucket, from whichever DMA table carries aggregated packets.
    dma, dcols = table_of(con, files, "DmaPacketAggregated")
    if not dma:
        dma, dcols = table_of(con, files, "DmaPacket")
    d_t = pick(dcols, TIME_HINTS, exclude=("end",)) if dma else None
    d_d = pick(dcols, DUR_HINTS) if dma else None
    d_sz = pick(dcols, ("size", "bytes", "length")) if dma else None

    rows = con.execute("""
        SELECT CAST((%s - %d) / %f AS INTEGER) AS b, %s AS eng, SUM(%s) AS busy, COUNT(*) AS n
        FROM %s GROUP BY 1, 2
    """ % (c_t, lo, w, c_e, c_d, ins)).fetchall()
    per_b = {}
    for b, eng, busy, n in rows:
        b = max(0, min(buckets - 1, int(b)))
        per_b.setdefault(b, {})[str(eng)] = per_b.setdefault(b, {}).get(str(eng), 0) + (busy or 0)

    dma_b, dma_bytes_b = {}, {}
    if dma and d_t and d_d:
        # GROUP BY 1 only. An earlier version put the size aggregate in the GROUP BY, which
        # duckdb rejects outright -- caught by the synthetic two-phase fixture, not on the cluster.
        for b, busy, sz in con.execute("""
            SELECT CAST((%s - %d) / %f AS INTEGER) AS b, SUM(%s), %s
            FROM %s GROUP BY 1
        """ % (d_t, lo, w, d_d, ("SUM(%s)" % d_sz) if d_sz else "0", dma)).fetchall():
            b = max(0, min(buckets - 1, int(b)))
            dma_b[b] = dma_b.get(b, 0) + (busy or 0)
            dma_bytes_b[b] = dma_bytes_b.get(b, 0) + (sz or 0)

    # dominant engine per bucket, then merge adjacent buckets that agree
    dom = []
    for b in range(buckets):
        e = per_b.get(b, {})
        dom.append(max(e.items(), key=lambda kv: kv[1])[0] if e else None)
    phases, start = [], 0
    for b in range(1, buckets + 1):
        if b == buckets or dom[b] != dom[start]:
            phases.append((start, b - 1, dom[start]))
            start = b
    merged = [p for p in phases if (p[1] - p[0] + 1) / float(buckets) >= min_frac]
    if not merged:
        merged = phases

    print("\n  %-13s %9s %6s  %-11s %7s %7s %8s  %s"
          % ("bucket range", "ms", "share", "leads", "its %", "dma %", "ratio", "verdict"))
    for b0, b1, eng in merged:
        nb = b1 - b0 + 1
        ph_ns = nb * w
        busy = {}
        for b in range(b0, b1 + 1):
            for e, v in per_b.get(b, {}).items():
                busy[e] = busy.get(e, 0) + v
        dbusy = sum(dma_b.get(b, 0) for b in range(b0, b1 + 1))
        dbytes = sum(dma_bytes_b.get(b, 0) for b in range(b0, b1 + 1))
        lead = max(busy.items(), key=lambda kv: kv[1]) if busy else ("-", 0)
        lead_pct = lead[1] / ph_ns
        dma_pct = dbusy / ph_ns
        ratio = (lead_pct / dma_pct) if dma_pct else float("inf")
        verdict = "-"
        if lead_pct > 0.30 and ratio > ISSUE_MOVE_RATIO:
            verdict = "ISSUE-BOUND on %s" % lead[0]
        elif dma_pct > lead_pct:
            pay = (dbytes / dbusy) if dbusy else 0
            verdict = "MOVE-BOUND (DMA leads)" if not dbytes else (
                "MOVE-BOUND, payload OK" if pay >= DMA_SATURATION_BYTES
                else "MOVE-BOUND, payload %.0f B below the %d B floor" % (pay, DMA_SATURATION_BYTES))
        elif lead_pct > 0.60:
            verdict = "SATURATED on %s" % lead[0]
        elif lead_pct < 0.20:
            verdict = "IDLE / serialised"
        print("  %-13s %9.2f %5.1f%%  %-11s %6.1f%% %6.1f%% %7s  %s"
              % ("%d-%d" % (b0, b1), ph_ns / 1e6, 100.0 * nb / buckets, lead[0],
                 100 * lead_pct, 100 * dma_pct,
                 ("%.1fx" % ratio) if ratio != float("inf") else "inf", verdict))

    # what code owns each phase
    if c_s:
        print("\n  SOURCE LINE THAT OWNS EACH PHASE")
        for b0, b1, _eng in merged:
            t0 = lo + b0 * w
            t1 = lo + (b1 + 1) * w
            got = con.execute("""
                SELECT %s AS src, %s AS op, SUM(%s) AS busy
                FROM %s WHERE %s >= %f AND %s < %f AND %s IS NOT NULL
                GROUP BY 1, 2 ORDER BY busy DESC LIMIT 1
            """ % (c_s, c_o or "'-'", c_d, ins, c_t, t0, c_t, t1, c_s)).fetchall()
            if not got:
                print("    %-13s (no src stamps in this phase)" % ("%d-%d" % (b0, b1)))
                continue
            src, op, busy = got[0]
            m = re.match(r"^(?P<f>[^\s:]+):(?P<l>\d+)", str(src))
            txt = read_src_line(m.group("f"), int(m.group("l")), src_root) if m else None
            print("    %-13s %8.2f ms  %-18s %s" % ("%d-%d" % (b0, b1), (busy or 0) / 1e6,
                                                    str(op)[:18], src))
            if txt:
                print("                  >>> %s" % txt[:110])
                why = name_op(txt)
                if why:
                    print("                  THIS IS: %s" % why)

    print("\n  READ IT THIS WAY: a phase whose verdict is ISSUE-BOUND is the one to fix, and the")
    print("  source line under it is where. A NEFF with two different verdicts in two phases")
    print("  cannot be fixed by one change -- the whole-NEFF average describes neither.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
