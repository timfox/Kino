"""Stimulus reconstruction AAD metrics — Eqs. (3)–(5)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson correlation coefficient ρ(x, y)."""
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if x.size != y.size or x.size < 2:
        return 0.0
    x = x - x.mean()
    y = y - y.mean()
    denom = float(np.linalg.norm(x) * np.linalg.norm(y))
    if denom < 1e-12:
        return 0.0
    return float(np.dot(x, y) / denom)


def decoding_accuracy(
    rho_attended: Sequence[float],
    rho_unattended: Sequence[float],
) -> float:
    """Eq. (3): fraction of test trials where ρ_att > ρ_unatt."""
    if len(rho_attended) != len(rho_unattended) or not rho_attended:
        return 0.0
    hits = sum(1 for a, u in zip(rho_attended, rho_unattended, strict=True) if a > u)
    return hits / len(rho_attended)


def pcc_loss(rho_attended: float) -> float:
    """Eq. (4): minimize negative PCC with attended envelope."""
    return -float(rho_attended)


def contrastive_pcc_loss(
    rho_attended: float,
    rho_unattended_speakers: Sequence[float],
) -> float:
    """Eq. (5): contrastive PCC across competing unattended speakers."""
    if not rho_unattended_speakers:
        return pcc_loss(rho_attended)
    n = len(rho_unattended_speakers)
    return (-float(rho_attended) + sum(float(r) for r in rho_unattended_speakers)) / n


def rho_delta(rho_attended: float, rho_unattended: float) -> float:
    """ρ_Δ = ρ_a − ρ_u (Table 2 column)."""
    return float(rho_attended) - float(rho_unattended)
