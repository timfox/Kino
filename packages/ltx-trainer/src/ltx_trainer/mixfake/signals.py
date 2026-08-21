"""Signal-level priors: HHT/IF and TKEO (Sec. III-C, III-D)."""

from __future__ import annotations

import math
from typing import Sequence


def teager_kaiser_energy(h_prev: float, h: float, h_next: float) -> float:
    r"""Ψ[h(n)] = h(n)^2 - h(n-1)·h(n+1) (Eq. 6), scalar."""
    return h * h - h_prev * h_next


def teager_kaiser_sequence(values: Sequence[float]) -> list[float]:
    if len(values) < 3:
        return []
    out: list[float] = []
    for i in range(1, len(values) - 1):
        out.append(abs(teager_kaiser_energy(values[i - 1], values[i], values[i + 1])))
    return out


def feature_flux(values: Sequence[float]) -> float:
    r"""Flux = σ(|h(n) - h(n-1)|) (Eq. 7), scalar sequence."""
    if len(values) < 2:
        return 0.0
    diffs = [abs(values[i] - values[i - 1]) for i in range(1, len(values))]
    mean = sum(diffs) / len(diffs)
    var = sum((d - mean) ** 2 for d in diffs) / len(diffs)
    return math.sqrt(var)


def instantaneous_frequency_from_phase(theta_prev: float, theta: float) -> float:
    r"""f(n) = |θ(n) - θ(n-1)| (Eq. 4)."""
    return abs(theta - theta_prev)


def frequency_prompt_fusion(f_high: float, f_all: float, f_low: float) -> float:
    r"""Toy scalar for Linear([f_high; f_all; f_low]) (Eq. 5)."""
    return (f_high + f_all + f_low) / 3.0
