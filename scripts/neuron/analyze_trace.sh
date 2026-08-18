#!/usr/bin/env bash
# Decode preserved NEFF+NTFF pairs into a TIME SERIES. Runs on ANY Linux host with the Neuron SDK.
#
# NO NEURON DEVICE IS NEEDED: the profiler only parses a recorded trace. So this runs on a trn
# instance, a plain Linux box with the SDK installed, or in a pod with no device claim -- and it
# does NOT queue behind or steal cores from a benchmark.
#
# It will NOT run on macOS: the SDK tools package has no build for Darwin (verified by resolving
# it against the vendor index). The query side is only duckdb and runs anywhere, which is why the
# output is plain text you can read off the host.
#
# Usage:
#   analyze_trace.sh <pairs-dir-or-results-tar> [outdir]
#   ONLY_FIRST=0 analyze_trace.sh ./results            # every pair, not just the heaviest
#   HEAD=200 BUCKETS=200 analyze_trace.sh ./results/pairs
#
# The single implementation: analyze-job.yaml calls this same script rather than repeating it.
set -uo pipefail

SRC="${1:?usage: analyze_trace.sh <pairs-dir-or-results-tar> [outdir]}"
OUT="${2:-./trace_analysis}"
ONLY_FIRST="${ONLY_FIRST:-1}"
HEAD="${HEAD:-60}"
BUCKETS="${BUCKETS:-80}"
INGEST_TIMEOUT="${INGEST_TIMEOUT:-5400}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "$OUT"
REPORT="$OUT/timeline_report.txt"; : > "$REPORT"

say() { echo "$@" | tee -a "$REPORT"; }

# ── the profiler CLI. neuron-profile is deprecated on newer SDKs and its capture fails with a
# banner pointing at neuron-explorer, so prefer that and fall back. This is the CAPTURE/VIEW
# subcommand, NOT the view SERVER (--ingest-only, :3001/:3002), which hangs on big DMA graphs.
PROF="$(command -v neuron-explorer || command -v neuron-profile \
        || echo /opt/aws/neuron/bin/neuron-explorer)"
if [ ! -x "$PROF" ] && ! command -v "$PROF" >/dev/null 2>&1; then
  echo "NO PROFILER CLI FOUND. This needs the Neuron SDK tools (neuron-explorer or"
  echo "neuron-profile) on PATH or at /opt/aws/neuron/bin. It is Linux-only; on macOS there is"
  echo "no build and the decode cannot be done locally at all."
  exit 1
fi
say "profiler CLI : $PROF"
say "version      : $("$PROF" --version 2>&1 | head -1)"

python3 -c "import duckdb" 2>/dev/null || {
  echo "installing duckdb (query side)"
  pip install --quiet duckdb 2>/dev/null || pip3 install --quiet duckdb 2>/dev/null || {
    python3 -m venv "$OUT/venv" && "$OUT/venv/bin/pip" install --quiet duckdb
    PY="$OUT/venv/bin/python"; }
}
PY="${PY:-python3}"
say "python       : $PY  (duckdb $("$PY" -c 'import duckdb;print(duckdb.__version__)' 2>&1))"

# ── locate the pairs. Accept a directory, or a results tar (what the benchmark leaves behind).
WORK="$OUT/work"; rm -rf "$WORK"; mkdir -p "$WORK"
if [ -f "$SRC" ] && case "$SRC" in *.tar.gz) true;; *) false;; esac; then
  say "restoring from tar: $SRC ($(du -h "$SRC" | cut -f1))"
  tar -xzf "$SRC" -C "$WORK"
  ROOT="$WORK"
else
  ROOT="$SRC"
fi
mapfile -t PAIRS < <(find "$ROOT" -type d -name 'rank*' | sort)
if [ "${#PAIRS[@]}" -eq 0 ]; then
  say "no rank*/ pair directories under $ROOT"
  exit 1
fi
say ""
say "pairs found:"
for p in "${PAIRS[@]}"; do say "  $(basename "$p")  [$(ls "$p" | tr '\n' ' ')]"; done

# ── ingest one pair to parquet.
# The CLI writes its tables and then IDLES on a server it never needs, so it must be backgrounded
# and killed once the output stops growing. Kill on SIZE-STABILISE, never on file existence: the
# summary table is created early and filled last, so "the file appeared" means nothing.
ingest() {
  local neff="$1" ntff="$2" out="$3"
  rm -rf "$out"; mkdir -p "$out"
  timeout "$INGEST_TIMEOUT" "$PROF" view -n "$neff" -s "$ntff" \
      --output-format parquet --output-file "$out" \
      --ignore-instruction-hierarchy --ignore-event-trace >"$out/.ingest.log" 2>&1 &
  local pid=$! prev=-1 cur stable=0 i
  for i in $(seq 1 1080); do            # up to 90 min at 5 s
    kill -0 "$pid" 2>/dev/null || break
    cur=$(du -sb "$out" 2>/dev/null | cut -f1)
    if [ "$cur" = "$prev" ] && [ "$cur" != "0" ]; then
      stable=$((stable+1)); [ "$stable" -ge 6 ] && break
    else stable=0; fi
    prev="$cur"; sleep 5
  done
  kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null
  local n; n=$(ls "$out"/*.parquet 2>/dev/null | wc -l)
  say "    parquet tables: $n   size $(du -sh "$out" 2>/dev/null | cut -f1)"
  [ "$n" -gt 0 ]
}

# LARGEST FIRST: the heavy graph is the point, and the small ones add little once it is decoded.
# If the big one cannot be ingested the log says so and the rest still run.
for p in $(for d in "${PAIRS[@]}"; do echo "$(du -sb "$d" | cut -f1) $d"; done | sort -rn | cut -d' ' -f2-); do
  neff=$(ls "$p"/*.neff 2>/dev/null | head -1)
  ntff=$(ls "$p"/*.ntff 2>/dev/null | head -1)
  say ""
  say "########## $(basename "$p") ##########"
  if [ -z "$neff" ] || [ -z "$ntff" ]; then
    say "  incomplete pair (need one .neff AND one .ntff) -- skip"
    continue
  fi
  say "  neff $(du -h "$neff" | cut -f1)   ntff $(du -h "$ntff" | cut -f1)"
  h=$(basename "$neff" .neff); pq="$OUT/pq_$h"
  if ingest "$neff" "$ntff" "$pq"; then
    "$PY" "$HERE/pq_timeline.py" "$pq" "$(basename "$p")" \
        --head "$HEAD" --buckets "$BUCKETS" 2>&1 | tee -a "$REPORT"
    [ -f "$HERE/pq_dma_report.py" ] && "$PY" "$HERE/pq_dma_report.py" "$pq" "$h" 2>&1 | tee -a "$REPORT"
    [ -f "$HERE/pq_pad_shapes.py" ] && "$PY" "$HERE/pq_pad_shapes.py" "$pq" "$h" 2>&1 | tee -a "$REPORT"
  else
    say "  INGEST PRODUCED NO TABLES -- tail of the converter's own log:"
    tail -12 "$pq/.ingest.log" 2>/dev/null | sed 's/^/    /' | tee -a "$REPORT"
  fi
  [ "$ONLY_FIRST" = "1" ] && { say "  (ONLY_FIRST=1 -- stopping after the heaviest pair)"; break; }
done

say ""
say "report: $REPORT"
