"""ForestTraceQA reference tables (paper Section 4)."""

from __future__ import annotations

from typing import Any

TABLE1_MAIN = [
    {"method": "Vanilla LLM", "BP": 12.59, "SA": 7.98, "IA": 10.29, "MT": 9.48, "PO": 7.61, "overall": 9.59},
    {"method": "Text-to-SQL", "BP": 31.41, "SA": 21.71, "IA": 1.64, "MT": 13.12, "PO": 9.09, "overall": 15.39},
    {"method": "Vanilla RAG", "BP": 27.18, "SA": 17.32, "IA": 10.93, "MT": 20.27, "PO": 13.94, "overall": 17.93},
    {"method": "HG-Summary LLM", "BP": 44.16, "SA": 18.33, "IA": 10.32, "MT": 25.70, "PO": 20.18, "overall": 23.74},
    {"method": "Tool-Agent", "BP": 32.99, "SA": 21.40, "IA": 20.29, "MT": 13.07, "PO": 10.56, "overall": 19.66},
    {"method": "Scene-Graph Agent", "BP": 40.69, "SA": 36.18, "IA": 44.98, "MT": 14.25, "PO": 18.97, "overall": 31.01},
    {"method": "ForestHG-Trace", "BP": 75.59, "SA": 69.63, "IA": 53.62, "MT": 46.35, "PO": 27.04, "overall": 54.45},
]

TABLE3_ABLATION = [
    {"variant": "V1 BEH", "BEH": True, "REH": False, "CEH": False, "overall": 39.76},
    {"variant": "V2 REH", "BEH": False, "REH": True, "CEH": False, "overall": 13.78},
    {"variant": "V3 CEH", "BEH": False, "REH": False, "CEH": True, "overall": 17.44},
    {"variant": "V4 BEH+REH", "BEH": True, "REH": True, "CEH": False, "overall": 45.92},
    {"variant": "V5 REH+CEH", "BEH": False, "REH": True, "CEH": True, "overall": 21.15},
    {"variant": "V6 BEH+CEH", "BEH": True, "REH": False, "CEH": True, "overall": 49.54},
    {"variant": "V7 Full", "BEH": True, "REH": True, "CEH": True, "overall": 54.45},
]

GT_TRACE_LENGTH = {"BP": 2.75, "SA": 7.09, "IA": 13.95, "MT": 17.02, "PO": 22.27}

RELIABILITY_BACKBONES = [
    {"model": "GPT-5-nano", "AF1": 90.51, "HAR": 17.33, "CDF1": 99.01},
    {"model": "Qwen3.5-397B", "AF1": 93.99, "HAR": 11.33, "CDF1": 93.05},
    {"model": "Qwen3.5-9B", "AF1": 99.33, "HAR": 1.33, "CDF1": 55.07},
]


def benchmarks_bundle() -> dict[str, Any]:
    return {
        "table1_main_accuracy": TABLE1_MAIN,
        "table3_hyperedge_ablation": TABLE3_ABLATION,
        "gt_trace_length_mean": GT_TRACE_LENGTH,
        "reliability_track": RELIABILITY_BACKBONES,
        "benchmark_scale": {
            "instances": 2000,
            "scenes": 100,
            "neon_sites": 20,
            "task_groups": ["BP", "SA", "IA", "MT", "PO"],
            "levels": ["H0", "H1", "H2", "H3", "H4"],
        },
    }
