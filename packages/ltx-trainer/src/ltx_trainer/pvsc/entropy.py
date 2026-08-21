"""Entropy-based symbol-length matching (Eq. 14, rate map)."""

from __future__ import annotations

import math

import numpy as np

DEFAULT_RATE_SET: tuple[int, ...] = (
    0, 4, 8, 12, 16, 20, 24, 28, 36, 44, 52, 60, 68, 84, 100, 128,
)


def gaussian_entropy_bits(sigma: float) -> float:
    """Differential entropy proxy for quantized Gaussian block (bits)."""
    sigma = max(float(sigma), 1e-6)
    return 0.5 * math.log2(2 * math.pi * math.e * sigma * sigma)


def symbol_length_factor(
    sigma: float,
    *,
    eta: float = 0.2,
    channels: int = 128,
) -> int:
    """Eq. (14) simplified: k = Q(eta * sum -log2 P)."""
    bits = -eta * channels * gaussian_entropy_bits(sigma)
    return max(0, int(round(bits)))


def quantize_rate(k: int, rate_set: tuple[int, ...] | None = None) -> int:
    """Map estimated k to nearest discrete rate level."""
    rates = rate_set or DEFAULT_RATE_SET
    return min(rates, key=lambda r: abs(r - k))
