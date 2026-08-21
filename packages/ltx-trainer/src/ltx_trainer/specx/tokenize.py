"""Structured spectral tokenization (Appendix B toy implementation)."""

from __future__ import annotations

import numpy as np


def discretize_ir_spectrum(
    intensities: np.ndarray,
    *,
    num_tokens: int = 400,
    value_range: tuple[float, float] = (0.0, 100.0),
) -> np.ndarray:
    """Downsample IR vector to fixed tokens and quantize to integers in [0, 100]."""
    x = np.asarray(intensities, dtype=np.float64).ravel()
    if x.size == 0:
        return np.zeros(num_tokens, dtype=np.int32)
    xi = np.linspace(0, x.size - 1, num=num_tokens)
    sampled = np.interp(xi, np.arange(x.size), x)
    lo, hi = value_range
    scaled = (sampled - lo) / max(hi - lo, 1e-8) * 100.0
    return np.clip(np.round(scaled), 0, 100).astype(np.int32)


def format_1h_peak(
    start: float,
    end: float,
    multiplicity: str,
    integration: float,
) -> str:
    """Format one 1H-NMR peak per paper convention."""
    return f"{start:.2f} {end:.2f} {multiplicity} {integration:.0f}H"


def format_13c_peaks(centroids_ppm: list[float]) -> str:
    parts = [f"{c:.1f}" for c in centroids_ppm]
    return "13CNMR " + " ".join(parts)
