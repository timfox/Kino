"""Landweber iCQT inverse stub (Sec. 4)."""

from __future__ import annotations

import math


def estimate_frame_bound(probe_energy: float, *, power_iters: int = 20) -> float:
    """Upper frame bound B from power iteration stub."""
    b = probe_energy
    for _ in range(power_iters):
        b = min(b * 1.05, probe_energy * 2.0)
    return max(b, 1e-6)


def landweber_step_size(frame_bound: float, step_fraction: float = 1.8) -> float:
    b = max(frame_bound, 1e-6)
    return step_fraction / b


def landweber_contraction(step_size: float, frame_bound: float) -> float:
    return abs(1.0 - step_size * frame_bound)


def reconstruction_snr_db(original: list[float], reconstructed: list[float]) -> float:
    if not original:
        return 0.0
    signal_power = sum(x * x for x in original) / len(original)
    noise_power = sum((a - b) ** 2 for a, b in zip(original, reconstructed, strict=True)) / len(original)
    if noise_power <= 0:
        return float("inf")
    return 10.0 * math.log10(signal_power / noise_power)


def landweber_reconstruct(
    analysis: list[float],
    adjoint: list[float],
    *,
    iterations: int = 32,
    step_size: float = 0.01,
) -> list[float]:
    """Single-vector Landweber: x <- x + alpha * A*(y - A x)."""
    x = [0.0] * len(adjoint)
    for _ in range(iterations):
        residual = [a - b for a, b in zip(analysis, x, strict=True)]
        update = [step_size * r for r in residual]
        x = [xi + ui for xi, ui in zip(x, update, strict=True)]
    return x


def icqt_snr_meets_target(
    original: list[float],
    reconstructed: list[float],
    *,
    min_snr_db: float = 30.0,
) -> bool:
    return reconstruction_snr_db(original, reconstructed) >= min_snr_db
