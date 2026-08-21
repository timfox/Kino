"""Objective metrics (Sec. 4.1–4.4)."""

from __future__ import annotations


def lsd_reduction_pct(baseline_lsd: float, improved_lsd: float) -> float:
    if baseline_lsd <= 0.0:
        return 0.0
    return 100.0 * (baseline_lsd - improved_lsd) / baseline_lsd


def formant_peak_error_hz(target_hz: float, synth_hz: float) -> float:
    return abs(target_hz - synth_hz)
