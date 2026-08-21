"""Temporal alignment metrics: Hit Coverage, Timing Error, Perfect Align."""

from __future__ import annotations

import numpy as np

from ltx_trainer.flatsounds.onset import adaptive_tolerance_s, detect_onsets


def hit_coverage(
    audio: np.ndarray,
    annotated_times_s: np.ndarray,
    *,
    sr: int = 16000,
) -> float:
    """Fraction of annotated events with a detected onset in tolerance window."""
    ann = np.asarray(annotated_times_s, dtype=np.float64).ravel()
    if ann.size == 0:
        return 1.0
    detected = detect_onsets(audio, sr=sr)
    tol = adaptive_tolerance_s(len(ann))
    covered = 0
    for t in ann:
        if detected.size == 0:
            break
        if np.any(np.abs(detected - t) <= tol):
            covered += 1
    return covered / ann.size


def timing_error_ms(
    audio: np.ndarray,
    annotated_times_s: np.ndarray,
    *,
    sr: int = 16000,
) -> float:
    """Mean |detected - gt| in ms for covered hits only."""
    ann = np.asarray(annotated_times_s, dtype=np.float64).ravel()
    if ann.size == 0:
        return 0.0
    detected = detect_onsets(audio, sr=sr)
    tol = adaptive_tolerance_s(len(ann))
    errs: list[float] = []
    for t in ann:
        if detected.size == 0:
            break
        d = np.abs(detected - t)
        idx = int(np.argmin(d))
        if d[idx] <= tol:
            errs.append(1000.0 * d[idx])
    return float(np.mean(errs)) if errs else float("nan")


def perfect_align(
    coverages: list[float],
) -> float:
    """Percentage of generations with 100% hit coverage."""
    if not coverages:
        return 0.0
    return 100.0 * sum(1 for c in coverages if c >= 1.0 - 1e-9) / len(coverages)
