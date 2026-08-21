"""Paper anchors and table stubs for Yi (arXiv:2607.15576)."""

from __future__ import annotations

from typing import Any

PAPER_ANCHORS: dict[str, float | int | str] = {
    "system": "Yi",
    "sift800m_update_speedup": 1.75,
    "sift800m_search_speedup": 1.8,
    "sift800m_peak_mem_frac": 0.73,
    "deep100m_update_qps": 1600,
    "deep100m_search_qps": 4800,
    "deep100m_vs_odinann_update": 2.8,
    "deep100m_vs_odinann_search": 1.7,
    "tasklet_speedup": 4.02,
    "buffer_speedup": 1.33,
    "layout_speedup": 1.04,
    "vs_ip_diskann_update": 3.0,
    "default_R": 96,
    "buffer_gb": 4.0,
    "delete_lru_frac": 0.04,
}

TABLE_1_UPDATE_SUPPORT: list[dict[str, Any]] = [
    {"solution": "DiskANN", "insert": "out-of-place", "delete": "out-of-place"},
    {"solution": "OdinANN", "insert": "in-place", "delete": "out-of-place"},
    {"solution": "IP-DiskANN", "insert": "—", "delete": "in-place"},
    {"solution": "Yi", "insert": "in-place", "delete": "in-place"},
]

# Abstract / §5.2 SIFT800M vs OdinANN
TABLE_SIFT800M: list[dict[str, Any]] = [
    {
        "metric": "update_throughput_vs_odinann",
        "value": 1.75,
        "note": "abstract; batch-dependent 2.3×@8M / 1.5×@20M in §5.2.2",
    },
    {"metric": "search_throughput_vs_odinann", "value": 1.8},
    {"metric": "peak_memory_frac_of_odinann", "value": 0.73},
]

TABLE_DEEP100M: list[dict[str, Any]] = [
    {"metric": "yi_update_qps", "value": 1600},
    {"metric": "yi_search_qps", "value": 4800},
    {"metric": "update_vs_odinann", "value": 2.8},
    {"metric": "search_vs_odinann", "value": 1.7},
    {"metric": "search_vs_spfresh", "value": 3.9},
    {"metric": "memory_gb_approx", "value": 15.4},
]

TABLE_BREAKDOWN: list[dict[str, Any]] = [
    {"component": "+Tasklet", "rel_throughput": 4.02},
    {"component": "+Buffer", "rel_throughput": 1.33},
    {"component": "+Layout", "rel_throughput": 1.04},
]


def benchmarks_bundle() -> dict[str, Any]:
    return {
        "anchors": dict(PAPER_ANCHORS),
        "table_1_update_support": list(TABLE_1_UPDATE_SUPPORT),
        "table_sift800m": list(TABLE_SIFT800M),
        "table_deep100m": list(TABLE_DEEP100M),
        "table_breakdown": list(TABLE_BREAKDOWN),
    }
