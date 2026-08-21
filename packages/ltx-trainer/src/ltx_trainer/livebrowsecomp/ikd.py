"""Compute IKD diagnostics from submission logs (not only paper constants)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from ltx_trainer.livebrowsecomp.metrics import avg_at_k, mean, pass_at_k_from_bools
from ltx_trainer.livebrowsecomp.trajectory import SearchTrajectory, analyze_trajectories


@dataclass
class ModelScoreRow:
    model: str
    closed_book: float | None = None
    with_tools: float | None = None
    blocked: float | None = None

    @property
    def search_contribution(self) -> float | None:
        if self.closed_book is None or self.with_tools is None:
            return None
        return self.with_tools - self.closed_book

    @property
    def blocked_vs_closed_delta(self) -> float | None:
        if self.closed_book is None or self.blocked is None:
            return None
        return self.blocked - self.closed_book


def summarize_closed_book(
    per_question_correct: dict[str, list[bool]],
    *,
    k: int = 4,
) -> float:
    """pass@k across questions from boolean sample lists."""
    if not per_question_correct:
        return 0.0
    scores = [pass_at_k_from_bools(flags, k) for flags in per_question_correct.values()]
    return 100.0 * mean(scores)


def summarize_avg_at(
    per_question_correct: dict[str, list[bool]],
    *,
    k: int = 4,
) -> float:
    if not per_question_correct:
        return 0.0
    scores = [avg_at_k(flags, k) for flags in per_question_correct.values()]
    return 100.0 * mean(scores)


def ikd_from_trajectories(trajectories: list[SearchTrajectory]) -> dict[str, Any]:
    agg = analyze_trajectories(trajectories)["aggregate"]
    return {
        "model_originated_query_rate": agg["model_originated_rate"],
        "evidence_use_rate": agg["evidence_use_rate"],
        "interpretation": (
            "High model-originated rate with low evidence use suggests IKD "
            "(search used to confirm parametric hypotheses)."
        ),
    }


def compare_blocked_vs_closed(rows: Sequence[ModelScoreRow]) -> dict[str, Any]:
    """Table 1 style aggregation from live measurements."""
    paired = [r for r in rows if r.closed_book is not None and r.blocked is not None]
    if not paired:
        return {"blocked_worse_than_closed": None, "avg_closed": None, "avg_blocked": None}
    avg_c = mean([r.closed_book for r in paired])  # type: ignore[arg-type]
    avg_b = mean([r.blocked for r in paired])  # type: ignore[arg-type]
    return {
        "blocked_worse_than_closed": avg_b < avg_c,
        "avg_closed": round(avg_c, 2),
        "avg_blocked": round(avg_b, 2),
        "per_model": [
            {
                "model": r.model,
                "closed": r.closed_book,
                "blocked": r.blocked,
                "delta": r.blocked_vs_closed_delta,
            }
            for r in paired
        ],
    }
