"""Keyframe-to-sketch conversion (Sec. 3.3.4, Eq. 3)."""

from __future__ import annotations

from typing import Sequence


def color_dodge_sketch(
    grayscale: Sequence[int],
    inverted_eroded: Sequence[int],
    *,
    epsilon: float = 1e-6,
) -> list[int]:
    r"""S(x) = min(255, G(x)·255 / (255 − E(x) + ε)) per pixel (Eq. 3)."""
    out: list[int] = []
    for g, e in zip(grayscale, inverted_eroded, strict=True):
        denom = max(epsilon, 255.0 - float(e))
        s = min(255.0, float(g) * 255.0 / denom)
        out.append(int(round(s)))
    return out


def toy_grayscale_line() -> tuple[list[int], list[int]]:
    """Simple 1D edge-like profile for smoke tests."""
    g = [200, 180, 120, 80, 120, 180, 200]
    inv = [255 - x for x in g]
    eroded = [max(0, x - 10) for x in inv]
    return g, eroded


def shot_boundary(content_diff: float, *, threshold: float = 25.0) -> int:
    """bt = 1 if Dt > τ else 0 (Eq. 1)."""
    return 1 if content_diff > threshold else 0
