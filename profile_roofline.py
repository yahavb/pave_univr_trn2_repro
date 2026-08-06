#!/usr/bin/env python3
import json
import sys


def g(d, *keys):
    for k in keys:
        if k in d:
            try:
                return float(d[k])
            except (TypeError, ValueError):
                pass
    return 0.0


def main():
    path = sys.argv[1]
    label = sys.argv[2] if len(sys.argv) > 2 else path
    expect_px = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    expect_bytes = int(sys.argv[4]) if len(sys.argv) > 4 else 0

    d = json.load(open(path))
    s = d.get("summary", d)
    if isinstance(s, list):
        s = s[0] if s else {}

    total = g(s, "total_active_time")
    sw = g(s, "software_dynamic_dma_packet_count")
    hbm_r = g(s, "hbm_read_bytes")
    hbm_w = g(s, "hbm_write_bytes")
    dma_b = g(s, "dma_transfer_total_bytes")
    mfu = g(s, "mfu_hlo_max_achievable_estimated_percent", "mfu_inst_max_achievable_estimated_percent")
    mbu = g(s, "mbu_estimated_percent")

    eng = {}
    for k, v in s.items():
        if k.endswith("_active_time") and not k.startswith("total"):
            try:
                eng[k[:-len("_active_time")]] = float(v)
            except (TypeError, ValueError):
                pass

    print("=" * 78)
    print("ROOFLINE FROM PROFILE: %s" % label)
    print("=" * 78)
    print("  total_active_time      %.6f s" % total)
    print()
    print("  %-22s %12s %8s" % ("engine", "active_s", "% total"))
    for k, v in sorted(eng.items(), key=lambda kv: -kv[1]):
        print("  %-22s %12.6f %7.1f%%" % (k, v, 100 * v / total if total else 0))
    print()
    print("  sw_dynamic_dma_packets %d" % sw)
    if expect_px:
        print("    per output pixel     %.4f   (px=%d)" % (sw / expect_px if expect_px else 0, expect_px))
    print("  dma_transfer_bytes     %d" % dma_b)
    if sw:
        print("    bytes per packet     %.1f" % (dma_b / sw))
    print("  hbm_read_bytes         %d" % hbm_r)
    print("  hbm_write_bytes        %d" % hbm_w)
    if expect_bytes:
        print("    hbm_read / tap_bytes %.2fx   (tap_bytes=%d)" % (
            hbm_r / expect_bytes if expect_bytes else 0, expect_bytes))
    print()
    print("  achieved DMA GB/s      %.3f" % (dma_b / total / 1e9 if total else 0))
    print("  achieved HBM GB/s      %.3f" % ((hbm_r + hbm_w) / total / 1e9 if total else 0))
    if sw and total:
        print("  descriptors / s        %.3f M" % (sw / total / 1e6))
        print("  ns per descriptor      %.2f" % (total / sw * 1e9))
    print("  mfu_max_achievable     %.4f %%" % mfu)
    print("  mbu_estimated          %.4f %%" % mbu)
    print()
    print("  BOUND: descriptor-rate if ns/desc is flat across shapes;")
    print("         bandwidth if achieved GB/s approaches the HBM ceiling;")
    print("         neither if both are far below -- then it is latency/serialisation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
