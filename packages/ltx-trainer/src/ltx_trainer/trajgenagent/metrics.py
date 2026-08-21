"""Aggregation-level spatiotemporal metrics (Sec. IV-C)."""

from __future__ import annotations

import math
from typing import Sequence

from ltx_trainer.trajgenagent.benchmarks import TABLE_II_MOBILITYSYN, TABLE_II_NUMOSIM
from ltx_trainer.trajgenagent.trajectory import DailyTrajectory, daily_distance_km, radius_of_gyration_km


def jensen_shannon_divergence(p: Sequence[float], q: Sequence[float]) -> float:
    eps = 1e-12
    p = [max(eps, x) for x in p]
    q = [max(eps, x) for x in q]
    sp, sq = sum(p), sum(q)
    p = [x / sp for x in p]
    q = [x / sq for x in q]
    m = [(a + b) / 2 for a, b in zip(p, q, strict=True)]

    def entropy(xs: Sequence[float]) -> float:
        return -sum(x * math.log(x) for x in xs if x > 0)

    return entropy(m) - (entropy(p) + entropy(q)) / 2


def trajectory_level_metrics(traj: DailyTrajectory) -> dict[str, float]:
    visits = traj.visits
    durations = [v.duration_minutes for v in visits]
    return {
        "distance_km": daily_distance_km(visits),
        "g_radius_km": radius_of_gyration_km(visits),
        "mean_duration_min": sum(durations) / max(1, len(durations)),
        "daily_loc": float(len(visits)),
    }


def paper_table_ii_row(dataset: str, model: str = "TrajGenAgent") -> dict[str, float]:
    table = TABLE_II_NUMOSIM if dataset == "NumoSim" else TABLE_II_MOBILITYSYN
    row = next(r for r in table if r["model"] == model)
    return {
        "distance": float(row["distance"]),
        "g_radius": float(row["g_radius"]),
        "duration": float(row["duration"]),
        "daily_loc": float(row["daily_loc"]),
        "i_rank": float(row["i_rank"]),
        "g_rank": float(row["g_rank"]),
        "transition": float(row["transition"]),
    }


def compare_to_paper_stub(
    traj: DailyTrajectory,
    *,
    dataset: str = "NumoSim",
) -> dict[str, object]:
    local = trajectory_level_metrics(traj)
    paper = paper_table_ii_row(dataset)
    return {"local": local, "paper_jsd_anchors": paper}


def compare_to_reference_live(
    generated: Sequence[DailyTrajectory],
    reference: Sequence[DailyTrajectory],
    *,
    dataset: str = "NumoSim",
) -> dict[str, object]:
    from ltx_trainer.trajgenagent.aggregate_metrics import compare_to_reference

    return compare_to_reference(generated, reference, dataset=dataset)
