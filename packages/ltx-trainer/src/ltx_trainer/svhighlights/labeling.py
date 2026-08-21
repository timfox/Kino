"""2-second clip labels from aligned highlight timestamps."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from ltx_trainer.svhighlights.config import SvHighlightsConfig


def highlight_intervals_from_indices(
    frame_indices: Sequence[int],
    *,
    fps: float = 30.0,
    window_s: float = 1.0,
) -> list[tuple[float, float]]:
    """Convert aligned frame indices to 1 s windows centered on each match."""
    half = window_s / 2.0
    intervals: list[tuple[float, float]] = []
    for idx in frame_indices:
        t = idx / fps
        intervals.append((max(0.0, t - half), t + half))
    return merge_overlapping_intervals(intervals)


def merge_overlapping_intervals(intervals: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    if not intervals:
        return []
    sorted_iv = sorted(intervals, key=lambda x: x[0])
    merged: list[tuple[float, float]] = [sorted_iv[0]]
    for start, end in sorted_iv[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def clip_labels(
    duration_s: float,
    highlight_intervals: Sequence[tuple[float, float]],
    *,
    clip_duration_s: float = 2.0,
    overlap_threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Non-overlapping 2 s clips; label 1 if >= overlap_threshold with any highlight interval."""
    n_clips = max(int(np.ceil(duration_s / clip_duration_s)), 1)
    labels = np.zeros(n_clips, dtype=np.int8)
    starts = np.arange(n_clips, dtype=np.float64) * clip_duration_s
    ends = np.minimum(starts + clip_duration_s, duration_s)
    for i, (cs, ce) in enumerate(zip(starts, ends, strict=True)):
        clip_len = ce - cs
        if clip_len <= 0:
            continue
        for hs, he in highlight_intervals:
            overlap = max(0.0, min(ce, he) - max(cs, hs))
            if overlap / clip_len >= overlap_threshold:
                labels[i] = 1
                break
    return starts, labels


def labels_from_alignment(
    aligned: Sequence[tuple[int | None, float]],
    *,
    fps: float = 30.0,
    duration_s: float | None = None,
    cfg: SvHighlightsConfig | None = None,
) -> dict[str, object]:
    from ltx_trainer.svhighlights.config import SvHighlightsConfig

    c = cfg or SvHighlightsConfig()
    indices = [i for i, _ in aligned if i is not None]
    if duration_s is None and indices:
        duration_s = (max(indices) + 1) / fps
    duration_s = duration_s or 60.0
    intervals = highlight_intervals_from_indices(indices, fps=fps, window_s=c.highlight_window_s)
    starts, labels = clip_labels(
        duration_s,
        intervals,
        clip_duration_s=c.clip_duration_s,
        overlap_threshold=c.label_overlap_threshold,
    )
    return {
        "clip_starts_s": starts.tolist(),
        "clip_labels": labels.tolist(),
        "n_positive": int(labels.sum()),
        "n_clips": int(len(labels)),
        "highlight_intervals": intervals,
    }
