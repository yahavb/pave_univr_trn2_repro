"""Print the key swdge/GpSimd warp-hotspot metrics from a neuron-profile
summary-json file. Usage: python parse_summary_json.py <summary.json>"""
import sys
import json


def main():
    try:
        d = json.load(open(sys.argv[1]))
    except Exception as e:
        print("  parse fail:", e)
        return
    nodes = [v for v in d.values() if isinstance(v, dict)] if isinstance(d, dict) else d
    if not nodes:
        nodes = [d]
    n = max(nodes, key=lambda x: x.get("total_time", 0) or 0)
    g = lambda k: n.get(k, 0) or 0
    print(f"  total_time_us      = {g('total_time')*1e6:.1f}")
    print(f"  gpsimd_active_pct  = {g('gpsimd_engine_active_time_percent')*100:.1f}")
    print(f"  sw_dyn_dma_pct     = {g('software_dynamic_dma_active_time_percent')*100:.1f}")
    print(f"  hw_dyn_dma_pct     = {g('hardware_dynamic_dma_active_time_percent')*100:.1f}")
    print(f"  sw_dyn_dma_packets = {g('software_dynamic_dma_packet_count')}")
    print(f"  dma_transfer_avg_B = {g('dma_transfer_average_bytes'):.1f}")
    print(f"  vector_active_pct  = {g('vector_engine_active_time_percent')*100:.1f}")
    print(f"  hbm_read_MB        = {g('hbm_read_bytes')/1048576:.2f}")


if __name__ == "__main__":
    main()
