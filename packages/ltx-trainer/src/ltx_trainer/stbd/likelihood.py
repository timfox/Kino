"""Subspace Bingham vs deterministic-contribution likelihoods — §II-B, §IV-A."""

from __future__ import annotations

import numpy as np

from ltx_trainer.stbd.steering import normalize_observation, subspace_projector


def bingham_log_likelihood(
    z: np.ndarray,
    projector: np.ndarray,
    kappa: float = 10.0,
) -> float:
    """log p(z) ∝ κ ||P z||^2 for normalized z (normalizer omitted)."""
    z = np.asarray(z, dtype=np.complex128).ravel()
    p = np.asarray(projector, dtype=np.complex128)
    energy = float(np.linalg.norm(p @ z) ** 2)
    return kappa * energy


def aggregate_bingham_log_likelihood(
    z_bins: list[np.ndarray],
    projectors: list[np.ndarray],
    kappa: float = 10.0,
) -> float:
    """Product over frequency bins — §IV-A proposed likelihood."""
    return sum(bingham_log_likelihood(z, p, kappa) for z, p in zip(z_bins, projectors, strict=True))


def deterministic_contribution_log_likelihood(
    ez: np.ndarray,
    predicted: np.ndarray,
    sigma_v_sq: float,
) -> float:
    """Conventional superpositional TBD baseline — Eq. (4) style."""
    r = np.asarray(ez, dtype=np.complex128).ravel() - np.asarray(predicted, dtype=np.complex128).ravel()
    return -float(np.linalg.norm(r) ** 2) / sigma_v_sq


def aggregate_deterministic_log_likelihood(
    ez_bins: list[np.ndarray],
    predicted_bins: list[np.ndarray],
    sigma_v_sq: float,
) -> float:
    return sum(
        deterministic_contribution_log_likelihood(ez, pred, sigma_v_sq)
        for ez, pred in zip(ez_bins, predicted_bins, strict=True)
    )


def snr_to_noise_variance(signal_power: float, snr_db: float) -> float:
    """σ_v^2 from SNR in dB."""
    snr_lin = 10.0 ** (snr_db / 10.0)
    return signal_power / max(snr_lin, 1e-12)
