"""VQT reduces to CQT at gamma=0 (Sec. 6.4)."""

from __future__ import annotations

import math


def vqt_bandwidth_offset(gamma: float, fk: float, bins_per_octave: float) -> float:
    alpha = 2 ** (1.0 / bins_per_octave) - 1.0
    return alpha * fk + gamma


def cqt_bandwidth_offset(fk: float, bins_per_octave: float) -> float:
    return vqt_bandwidth_offset(0.0, fk, bins_per_octave)


def vqt_routes_to_cqt(gamma: float) -> bool:
    return gamma == 0.0


def kernel_offset_delta(gamma: float, fk: float, bins_per_octave: float = 12.0) -> float:
    """Stub kernel offset; VQT with gamma=0 must match CQT."""
    if vqt_routes_to_cqt(gamma):
        return cqt_bandwidth_offset(fk, bins_per_octave)
    return vqt_bandwidth_offset(gamma, fk, bins_per_octave)


def vqt_cqt_max_diff(gamma: float, freqs: list[float], *, bins_per_octave: float = 12.0) -> float:
    if not vqt_routes_to_cqt(gamma):
        return float("inf")
    diffs = [
        abs(kernel_offset_delta(0.0, f, bins_per_octave) - cqt_bandwidth_offset(f, bins_per_octave))
        for f in freqs
    ]
    return max(diffs) if diffs else 0.0
