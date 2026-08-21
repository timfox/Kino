"""Feature-space helpers for LiFT (toy / reference)."""

from __future__ import annotations

import math
from typing import Any


def fourier_depth_encoding(depth: int, num_frequencies: int = 4) -> list[float]:
    """Simplified Fourier features γ(d) for depth coordinate (paper [40])."""
    out: list[float] = [float(depth)]
    for k in range(num_frequencies):
        freq = 2.0**k * math.pi
        out.extend([math.sin(freq * depth), math.cos(freq * depth)])
    return out


def temporal_difference_volume(day2: list[float], day1: list[float]) -> list[float]:
    """Adjacent conceptual Δ along depth for flattened slice descriptors."""
    if len(day2) != len(day1):
        raise ValueError("day1 and day2 must match length")
    return [b - a for a, b in zip(day1, day2, strict=True)]


def drift_target_mean(
    generated_features: list[list[float]],
    real_features: list[list[float]],
    bandwidth: float = 1.0,
) -> dict[str, Any]:
    """Toy Laplacian-kernel drift: pull batch mean of gen toward weighted real mean."""
    if not generated_features or not real_features:
        return {"drift_l2": 0.0, "n_gen": 0, "n_real": 0}

    def _mean(vecs: list[list[float]]) -> list[float]:
        dim = len(vecs[0])
        return [sum(v[i] for v in vecs) / len(vecs) for i in range(dim)]

    g = _mean(generated_features)
    r = _mean(real_features)
    # scalar weights from L1 distance to r (toy kernel)
    weights = [math.exp(-sum(abs(gi - ri) for gi, ri in zip(g, r)) / bandwidth) for _ in real_features]
    wsum = sum(weights) or 1.0
    mu = [sum(w * ri for w, ri in zip(weights, r)) / wsum for ri in range(len(r))]
    drift_l2 = sum((gi - mi) ** 2 for gi, mi in zip(g, mu))
    return {"drift_l2": drift_l2, "n_gen": len(generated_features), "n_real": len(real_features)}
