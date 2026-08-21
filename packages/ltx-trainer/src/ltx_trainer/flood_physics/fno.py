"""Fourier Neural Operator stub for basin-scale patterns."""

from __future__ import annotations

import math
from typing import Any


def _dct_energy(values: list[float]) -> list[float]:
    """Toy low-frequency energy extraction (no FFT dep)."""
    n = len(values)
    if n == 0:
        return []
    mean = sum(values) / n
    out = [mean]
    for k in range(1, min(4, n)):
        acc = sum(values[i] * math.cos(math.pi * k * i / n) for i in range(n))
        out.append(acc / n)
    return out


def fno_forward(flat_features: list[float], *, modes: int = 16) -> list[float]:
    """Global operator path: amplify low-frequency basin context."""
    energy = _dct_energy(flat_features)
    scale = sum(abs(e) for e in energy[:modes]) / max(len(energy[:modes]), 1)
    return [v + 0.1 * scale for v in flat_features]


def fno_card() -> dict[str, Any]:
    return {"operator": "FNO", "domain": "frequency / basin-scale", "modes": 16}
