"""Copy-synthesis loss terms (Sec. 3.4)."""

from __future__ import annotations

import math


def overtone_salience(
    band_h_energy: float,
    band_adjacent_energy: float,
    epsilon: float = 1e-10,
) -> float:
    """S_ot in dB (Eq. 4)."""
    return 10.0 * math.log10(band_h_energy / (band_adjacent_energy + epsilon) + epsilon)


def overtone_salience_loss(target_sot: float, synth_sot: float) -> float:
    return (target_sot - synth_sot) ** 2


def combined_loss(
    lstft: float,
    lmel: float,
    lharm: float,
    lenergy: float,
    lot: float,
    *,
    lam_mel: float = 1.0,
    lam_harm: float = 1.0,
    lam_energy: float = 1.0,
    lam_ot: float = 1.0,
) -> float:
    return lstft + lam_mel * lmel + lam_harm * lharm + lam_energy * lenergy + lam_ot * lot
