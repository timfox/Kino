"""Paper reference tables + computed IKD diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ltx_trainer.livebrowsecomp.ikd import (
    ModelScoreRow,
    compare_blocked_vs_closed,
    ikd_from_trajectories,
)
from ltx_trainer.livebrowsecomp.trajectory import SearchTrajectory

# Closed-book pass@4 from paper Fig 2 (selected cells)
CLOSED_BOOK_PASS4: list[dict[str, Any]] = [
    {"model": "MiniMax M2.5", "BrowseComp": 44.5, "BrowseComp-ZH": 58.5, "HLE": 30.0, "GAIA": 32.0},
    {"model": "Kimi K2.6", "BrowseComp": 25.5, "BrowseComp-ZH": 62.0, "HLE": 34.0, "GAIA": 38.0},
    {"model": "Seed 2.0", "BrowseComp": 38.2, "BrowseComp-ZH": 46.8, "HLE": 50.2, "GAIA": 48.2},
    {"model": "DeepSeek-V4-Pro", "BrowseComp": 20.4, "BrowseComp-ZH": 59.1, "HLE": 48.5, "GAIA": 36.0},
    {"model": "GLM 5.1", "BrowseComp": 23.3, "BrowseComp-ZH": 52.5, "HLE": 26.7, "GAIA": 28.4},
]

EVIDENCE_BLOCKED: list[dict[str, Any]] = [
    {"model": "GLM 5.0", "closed": 21.3, "blocked": 7.4, "delta": -13.9},
    {"model": "GLM 5.1", "closed": 23.3, "blocked": 9.4, "delta": -13.9},
    {"model": "MiniMax M2.5", "closed": 44.5, "blocked": 8.0, "delta": -36.5},
    {"model": "Kimi-K2.5", "closed": 19.7, "blocked": 2.8, "delta": -16.9},
    {"model": "Kimi-K2.6", "closed": 25.5, "blocked": 2.3, "delta": -23.2},
    {"model": "DeepSeek-V4-Pro", "closed": 22.5, "blocked": 7.0, "delta": -15.5},
]

TRAJECTORY_STATS: dict[str, float] = {
    "DeepSeek-V3.2_evidence_use_pct": 32.2,
    "GLM-5.1_evidence_use_pct": 24.7,
    "MiniMax-M2.5_evidence_use_pct": 30.8,
    "Kimi-K2.5_evidence_use_pct": 31.5,
}


@dataclass
class IKDSummary:
    closed_book_avg: float
    evidence_blocked_avg_closed: float
    evidence_blocked_avg_blocked: float
    blocked_worse_than_closed: bool
    model_originated_query_fraction: float


def ikd_summary() -> IKDSummary:
    rows = [
        ModelScoreRow(r["model"], closed_book=r["closed"], blocked=r["blocked"])
        for r in EVIDENCE_BLOCKED
    ]
    cmp = compare_blocked_vs_closed(rows)
    return IKDSummary(
        closed_book_avg=38.9,
        evidence_blocked_avg_closed=cmp["avg_closed"],  # type: ignore[arg-type]
        evidence_blocked_avg_blocked=cmp["avg_blocked"],  # type: ignore[arg-type]
        blocked_worse_than_closed=bool(cmp["blocked_worse_than_closed"]),
        model_originated_query_fraction=0.55,
    )


def closed_book_on_livebenchmark() -> list[dict[str, Any]]:
    models = [
        "DeepSeek-V3.2",
        "GLM-5.0",
        "GLM-5.1",
        "MiniMax M2.5",
        "Kimi K2.5",
        "Kimi K2.6",
        "Seed2.0",
        "DeepSeek-V4-Pro",
        "GPT-5.4",
    ]
    live_scores = [0.0, 0.7, 2.0, 0.5, 0.7, 1.3, 2.0, 1.0, 0.2]
    browse = [11.0, 21.3, 23.3, 44.5, 19.7, 25.5, 20.4, 22.5, 37.5]
    return [
        {"model": m, "browsecomp_plus_closed_pct": bc, "livebrowsecomp_closed_pct": lv}
        for m, bc, lv in zip(models, browse, live_scores, strict=True)
    ]


def compute_ikd_from_trajectories(trajectories: list[SearchTrajectory]) -> dict[str, Any]:
    """Live diagnostic from agent logs (not paper constants)."""
    return ikd_from_trajectories(trajectories)
