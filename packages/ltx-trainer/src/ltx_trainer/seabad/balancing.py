"""Diversity-aware species balancing helpers (Eqs. 1–4)."""

from __future__ import annotations

import numpy as np


def gini_coefficient(species_counts: np.ndarray | list[int]) -> float:
    """Eq. (4): Gini coefficient for species sample-count inequality."""
    n = np.sort(np.asarray(species_counts, dtype=np.float64))
    s = int(n.size)
    if s == 0:
        return 0.0
    total = float(n.sum())
    if total <= 0.0:
        return 0.0
    index = np.arange(1, s + 1, dtype=np.float64)
    return float(2.0 * np.sum(index * n) / (s * total) - (s + 1) / s)


def gini_reduction_pct(gini_before: float, gini_after: float) -> float:
    """Relative inequality reduction reported in paper (13.7% for 0.601 → 0.519)."""
    if gini_before <= 0.0:
        return 0.0
    return float(100.0 * (gini_before - gini_after) / gini_before)


def salience_score(
    mean_contrast: float,
    mean_centroid_hz: float,
    *,
    fs_hz: float = 16_000.0,
    contrast_scale: float = 40.0,
) -> float:
    """Eq. (1): acoustic salience for clip ranking during balancing."""
    return 0.7 * (mean_contrast / contrast_scale) + 0.3 * (mean_centroid_hz / fs_hz)


def per_species_base_allocation(n_target: int, n_species: int) -> int:
    """Eq. (2): floor(N_target / S) base clips per species."""
    if n_species <= 0:
        return 0
    return n_target // n_species
