"""Throughput-oriented workflow metrics (Tip 10)."""

from __future__ import annotations

from typing import Any


def throughput_metrics(
    *,
    useful_tasks_completed: int,
    wall_hours: float,
    queue_wait_hours: float,
    compute_hours: float,
) -> dict[str, Any]:
    effective = max(wall_hours - queue_wait_hours, 1e-6)
    return {
        "useful_tasks_per_wall_hour": round(useful_tasks_completed / wall_hours, 4),
        "useful_tasks_per_effective_hour": round(useful_tasks_completed / effective, 4),
        "queue_fraction": round(queue_wait_hours / max(wall_hours, 1e-6), 4),
        "compute_utilisation": round(compute_hours / max(wall_hours, 1e-6), 4),
        "focus": "workflow throughput over single-job peak FLOPS",
    }


def compare_strategies(
    exhaustive_runs: int,
    selective_runs: int,
    *,
    exhaustive_informative: float,
    selective_informative: float,
) -> dict[str, Any]:
    """Example: fewer high-value simulations may beat many uninformative ones."""
    ex_score = exhaustive_runs * exhaustive_informative
    sel_score = selective_runs * selective_informative
    return {
        "exhaustive": {"runs": exhaustive_runs, "informative_rate": exhaustive_informative, "score": ex_score},
        "selective": {"runs": selective_runs, "informative_rate": selective_informative, "score": sel_score},
        "better": "selective" if sel_score >= ex_score else "exhaustive",
    }
