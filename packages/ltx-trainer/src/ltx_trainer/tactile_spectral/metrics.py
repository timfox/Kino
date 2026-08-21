"""Spectral correlation and critical-band regression (Fig. 3)."""

from __future__ import annotations

import torch
from torch import Tensor

from ltx_trainer.tactile_spectral.spectrum import (
    band_mask,
    log_spaced_band_edges,
    magnitude_spectrum,
)


def spectral_correlation(
    original: Tensor,
    rendered: Tensor,
    *,
    sample_rate: float,
    f_low: float = 20.0,
    f_high: float = 1000.0,
) -> float:
    """Pearson correlation of magnitude spectra in tactile band."""
    m0, f0 = magnitude_spectrum(original, sample_rate=sample_rate)
    m1, f1 = magnitude_spectrum(rendered, sample_rate=sample_rate)
    n = min(m0.numel(), m1.numel())
    m0, m1 = m0[:n], m1[:n]
    f = f0[:n]
    mask = band_mask(f, f_low, f_high)
    if mask.sum() < 2:
        return 0.0
    a = m0[mask].float()
    b = m1[mask].float()
    a = a - a.mean()
    b = b - b.mean()
    denom = a.norm() * b.norm()
    if denom < 1e-12:
        return 0.0
    return float((a * b).sum() / denom)


def critical_band_energies(
    mag: Tensor,
    freqs: Tensor,
    *,
    n_bands: int,
    f_low: float = 20.0,
    f_high: float = 1000.0,
) -> Tensor:
    """Energy in n_bands log-spaced triangular filters."""
    edges = log_spaced_band_edges(f_low, f_high, n_bands, device=mag.device)
    energies = []
    for i in range(n_bands):
        lo, hi = float(edges[i].item()), float(edges[i + 1].item())
        m = band_mask(freqs, lo, hi)
        energies.append(mag[m].pow(2).sum() if m.any() else torch.tensor(0.0, device=mag.device))
    return torch.stack(energies)


def band_energy_difference(
    original: Tensor,
    rendered: Tensor,
    *,
    sample_rate: float,
    n_bands: int = 9,
    f_low: float = 20.0,
    f_high: float = 1000.0,
) -> Tensor:
    """Absolute spectral energy difference per critical band."""
    m0, f0 = magnitude_spectrum(original, sample_rate=sample_rate)
    m1, _ = magnitude_spectrum(rendered, sample_rate=sample_rate)
    n = min(m0.numel(), m1.numel())
    e0 = critical_band_energies(m0[:n], f0[:n], n_bands=n_bands, f_low=f_low, f_high=f_high)
    e1 = critical_band_energies(m1[:n], f0[:n], n_bands=n_bands, f_low=f_low, f_high=f_high)
    return (e0 - e1).abs()


def predict_similarity_from_bands(
    energy_diff: Tensor,
    *,
    weights: Tensor | None = None,
) -> float:
    """Linear proxy: lower band error → higher similarity (Fig. 3C)."""
    z = energy_diff.float()
    if weights is not None:
        z = z * weights
    err = float(z.mean().item())
    return max(0.0, min(6.0, 6.0 - err * 6.0))
