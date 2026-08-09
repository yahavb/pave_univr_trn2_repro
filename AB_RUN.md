# A/B: shiftwarp vs gather, whole model

Question: how does the op-level **1.16x** at the C=3 resample site manifest end to end.

## Status

| arm | pod | state | stage [3/3] median |
|---|---|---|---|
| `shiftwarp` (kernel) | `univr-trn-repro-gkxm6` | RUNNING, started 2026-08-09 ~15:20 local | pending |
| `gather` (baseline) | — | not started | — |

## Arm 1: shiftwarp — RUNNING

Launched from `repro-job.yaml` at commit `00a9beb`, which ships
`MEASURE_WARP=shiftwarp` and `FUNCTIONAL_WARP=shiftwarp`.

```
kubectl logs univr-trn-repro-gkxm6 > /tmp/univr-trn-repro-gkxm6
```

Check these, in order:

1. `MEASURE_WARP: shiftwarp   FUNCTIONAL_WARP: shiftwarp` — knobs took effect.
2. `shiftwarp     R=3 -> 49 terms, covers |disp| <= 3 px; sites with C > 3 fall back to gather`
   — if this line is ABSENT the flag did not take and the run is void.
3. `=== [3/3] MEASUREMENT — 8 tiles / 8 cores, warp=shiftwarp, 3 iters ===` then its median.
4. The accuracy line: must be PASS at `--bar 3`. `shiftwarp` and `gather` scored an
   identical 0.0039 LSB against `gridsample` off-device, so a divergence here points at
   the ctx-site fallback or something device-side, not the C=3 math.

## Arm 2: gather — run after arm 1 finishes

Edit BOTH env values in `repro-job.yaml` (~line 126-133) from `shiftwarp` to `gather`,
then:

```
kubectl delete job univr-trn-repro --ignore-not-found
kubectl apply -f ~/pave_univr_trn2_repro/repro-job.yaml
```

Do NOT use `kubectl set env job/...` for this — I gave that command earlier and it is
WRONG. A Job's `spec.template` is immutable once created. Verified against the API
server with `--dry-run=server`, which rejects it:

```
error: failed to patch env update to pod template:
Job.batch "univr-trn-repro" is invalid: spec.template: ... field is immutable
```

Note `--dry-run=client` does NOT catch this: it renders the patched object locally and
looks like it worked. Editing the YAML and re-applying is the only reliable path.

## Comparing

`2_measurement_shiftwarp.log` vs `2_measurement_gather.log`, stage [3/3] median.
Reference points already printed in the log header: bundle 3673.3 ms / 98.68 dB /
0.64 LSB; CUDA L40S same frame eager 351.1 ms, inductor 237.9 ms, TRT 161 ms.

**Expect a small delta.** Only the C=3 image-warp site is accelerated — the four ctx
sites (C=16/32/64/128) fall back to `gather` because the kernel is verified only at
C=3. The resample is ~62% of a forward. So 1.16x on part of 62% is a low-single-digit
end-to-end gain; 1.01-1.05x is consistent with the op-level number, not a failure.

## Op-level numbers this is testing (measured, `univr_sweep_20260809_150451`)

| arm | active_us | gate | max_LSB |
|---|---|---|---|
| gather (baseline) | 49,278 | PASS | 0.0417 |
| **shiftwarp R=3** | **42,439** | **PASS** | **0.0417** |
| shiftwarp R=2 | 39,479 | FAIL | 76.1907 |

R=2 is faster but WRONG on the real 2.33 px flow — the accuracy cliff is hard at
exactly 2.0 px. R=3 is the smallest radius that passes.
