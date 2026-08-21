"""Temporal and fusion feature helpers (toy / reference)."""

from __future__ import annotations

from typing import Any


def temporal_difference(day2: list[float], day1: list[float]) -> list[float]:
    """Δ embedding = Day2 − Day1 (paper primary temporal representation)."""
    if len(day2) != len(day1):
        raise ValueError("day1 and day2 embeddings must have equal length")
    return [b - a for a, b in zip(day1, day2, strict=True)]


def temporal_concatenate(day1: list[float], day2: list[float]) -> list[float]:
    """Preserve both time points (weaker than difference in paper ablations)."""
    return list(day1) + list(day2)


def fuse_concatenate(view_vectors: list[list[float]]) -> list[float]:
    """Multi-view feature concatenation (best fusion in paper)."""
    out: list[float] = []
    for v in view_vectors:
        out.extend(v)
    return out


def fuse_average_features(view_vectors: list[list[float]]) -> list[float]:
    n = len(view_vectors)
    if n == 0:
        return []
    dim = len(view_vectors[0])
    return [sum(v[i] for v in view_vectors) / n for i in range(dim)]


def patient_representation(
    view_deltas: dict[str, list[float]],
    *,
    fusion: str = "concatenate",
) -> list[float]:
    """Build patient-level vector from per-view temporal-difference embeddings."""
    ordered = [view_deltas[k] for k in sorted(view_deltas)]
    if fusion == "concatenate":
        return fuse_concatenate(ordered)
    if fusion == "average":
        return fuse_average_features(ordered)
    raise ValueError(f"Unknown fusion {fusion!r}")
