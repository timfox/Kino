"""Multi-slot heatmap generation from attention maps (Sec. 3.1)."""

from __future__ import annotations


def attention_to_heatmap(scores: list[float], *, resolution: int = 64) -> list[float]:
    """Flatten attention row into spatial heatmap grid."""
    if not scores:
        return [0.0] * resolution
    step = max(1, len(scores) // resolution)
    return [max(scores[i : i + step]) for i in range(0, min(len(scores), resolution * step), step)][:resolution]


def multi_slot_heatmaps(
    attention_scores: list[list[float]],
    *,
    resolution: int = 64,
) -> list[list[float]]:
    """One heatmap per slot/query from per-head attention peaks."""
    heatmaps: list[list[float]] = []
    for head_rows in attention_scores:
        if not head_rows:
            continue
        peak_row = max(head_rows)
        heatmaps.append(attention_to_heatmap([peak_row] * resolution, resolution=resolution))
    if not heatmaps and attention_scores:
        heatmaps.append([0.0] * resolution)
    return heatmaps
