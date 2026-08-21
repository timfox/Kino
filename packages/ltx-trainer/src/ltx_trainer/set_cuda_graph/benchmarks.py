"""Table 1–2 and summary anchors (§5.2–5.3)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.set_cuda_graph.constants import (
    FIG6_BATCH4096_OVERHEAD_PCT,
    FIG6_BATCH4_OVERHEAD_PCT,
    OVERHEAD_REDUCTION_VS,
    TABLE1_AVERAGE_SPEEDUP,
    TABLE1_SPEEDUPS,
    TABLE2_OVERHEAD_RATIO,
    THROUGHPUT_SPEEDUP_RANGE,
)


def table_1_speedups() -> list[dict[str, Any]]:
    return [dict(row) for row in TABLE1_SPEEDUPS]


def table_1_averages() -> dict[str, Any]:
    return dict(TABLE1_AVERAGE_SPEEDUP)


def table_2_overhead() -> dict[str, Any]:
    return dict(TABLE2_OVERHEAD_RATIO)


def fig6_overhead_anchors() -> dict[str, Any]:
    return {
        "batch_size_4_pct": dict(FIG6_BATCH4_OVERHEAD_PCT),
        "batch_size_4096_pct": dict(FIG6_BATCH4096_OVERHEAD_PCT),
    }


def summary_anchors() -> dict[str, Any]:
    avg = TABLE1_AVERAGE_SPEEDUP
    return {
        "throughput_speedup_range": THROUGHPUT_SPEEDUP_RANGE,
        "avg_speedup_vs_sync": avg["Synchronous"]["combined"],
        "avg_speedup_vs_graph": avg["Graph"]["combined"],
        "avg_speedup_vs_batching": avg["Batching"]["combined"],
        "avg_speedup_vs_queue": avg["Queue"]["combined"],
        "overhead_reduction_vs_batching_pct": OVERHEAD_REDUCTION_VS["batching_pct"],
        "overhead_reduction_vs_queue_pct": OVERHEAD_REDUCTION_VS["queue_pct"],
        "set_overhead_rtx3090": TABLE2_OVERHEAD_RATIO["rtx_3090"]["SET"],
        "knn_queue_speedup_blackwell": 2.09,
    }
