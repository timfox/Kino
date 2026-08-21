"""Metric helpers and Semantic Coverage Factor (SCF) stub (Sec. 5.4)."""

from __future__ import annotations

from typing import Iterable


def lower_is_better(value: float, *, best: float, worst: float) -> float:
    """Min-max normalize so 1.0 = best."""
    if worst <= best:
        return 1.0 if value <= best else 0.0
    return float((worst - value) / (worst - best))


def higher_is_better(value: float, *, best: float, worst: float) -> float:
    if worst <= best:
        return 1.0 if value >= best else 0.0
    return float((value - worst) / (best - worst))


def normalized_scenario_score(
    metrics: dict[str, float],
    *,
    lower_keys: Iterable[str] = ("FAD", "KL", "WER", "FD"),
    higher_keys: Iterable[str] = ("IS", "CLAP", "UTMOS", "SCF"),
    reference: dict[str, dict[str, float]] | None = None,
) -> float:
    """Fig. 4 style normalized score across metrics in one scenario."""
    ref = reference or {}
    scores: list[float] = []
    for key, val in metrics.items():
        bounds = ref.get(key, {"best": val, "worst": val})
        if key in lower_keys or key.startswith("FAD") or key.startswith("KL") or key == "WER":
            scores.append(lower_is_better(val, best=bounds.get("best", val), worst=bounds.get("worst", val)))
        elif key in higher_keys or key in ("IS", "CLAP", "UTMOS", "SCF"):
            scores.append(higher_is_better(val, best=bounds.get("best", val), worst=bounds.get("worst", val)))
    return sum(scores) / len(scores) if scores else float("nan")


def semantic_coverage_factor(
    event_similarities: list[float],
    *,
    threshold: float = 0.5,
) -> float:
    """SCF stub (Sec. 5.4): mean of similarities above threshold, normalized by |events|."""
    if not event_similarities:
        return 0.0
    kept = [s for s in event_similarities if s >= threshold]
    return len(kept) / len(event_similarities)
