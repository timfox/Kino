"""VarRate paper anchors (Tables 1–3) — arXiv:2607.15498."""

from __future__ import annotations

from typing import Any

# Table 1: two-model LongBench-style means (illustrative paper anchors).
TABLE_1_TWO_MODEL = [
    {"model": "Llama-3.1-8B", "method": "Full", "avg": 48.2, "kv_frac": 1.00},
    {"model": "Llama-3.1-8B", "method": "SnapKV", "avg": 46.1, "kv_frac": 0.20},
    {"model": "Llama-3.1-8B", "method": "VarRate", "avg": 47.4, "kv_frac": 0.20},
    {"model": "Mistral-7B", "method": "Full", "avg": 44.8, "kv_frac": 1.00},
    {"model": "Mistral-7B", "method": "SnapKV", "avg": 42.5, "kv_frac": 0.20},
    {"model": "Mistral-7B", "method": "VarRate", "avg": 43.9, "kv_frac": 0.20},
]

# Table 2: reuse collapse — eviction methods lose quality under prefix reuse;
# VarRate keeps ranks (no eviction) so quality holds.
TABLE_2_REUSE = [
    {"method": "SnapKV", "fresh": 46.1, "reuse": 41.0, "delta": -5.1},
    {"method": "KVzip", "fresh": 45.8, "reuse": 40.2, "delta": -5.6},
    {"method": "VarRate", "fresh": 47.4, "reuse": 47.1, "delta": -0.3},
]

# Table 3: prefill overhead vs KVzip (relative units; lower better).
TABLE_3_PREFILL = [
    {"method": "KVzip", "prefill_overhead_x": 1.85},
    {"method": "SnapKV", "prefill_overhead_x": 1.12},
    {"method": "VarRate", "prefill_overhead_x": 1.08},
]

PAPER_ANCHORS: dict[str, Any] = {
    "kappa": 0.20,
    "rmin": 16,
    "r_max": 1024,
    "stride": 16,
    "window": 64,
    "llama_varrate_avg": 47.4,
    "reuse_delta_varrate": -0.3,
    "prefill_overhead_x": 1.08,
}


def benchmarks_bundle() -> dict[str, Any]:
    return {
        "paper": "arXiv:2607.15498",
        "table_1": TABLE_1_TWO_MODEL,
        "table_2_reuse": TABLE_2_REUSE,
        "table_3_prefill": TABLE_3_PREFILL,
        "anchors": PAPER_ANCHORS,
    }
