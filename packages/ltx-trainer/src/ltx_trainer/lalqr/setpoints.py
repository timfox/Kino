"""Latent linear feature setpoints (LLFS) — Sec. 4.3."""

from __future__ import annotations

import numpy as np


def latent_feature_direction(basis: np.ndarray, contrastive_rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """e_z and unit v_z from Eq. (17)."""
    ez = project_rows_mean(basis, contrastive_rows)
    norm = float(np.linalg.norm(ez))
    if norm <= 1e-12:
        vz = np.zeros_like(ez)
    else:
        vz = ez / norm
    return ez, vz


def project_rows_mean(basis: np.ndarray, rows: np.ndarray) -> np.ndarray:
    z_rows = rows @ basis
    return z_rows.mean(axis=0)


def latent_feature_strength(z: np.ndarray, vz: np.ndarray) -> float:
    """β^z = v_z^T z — Eq. (18)."""
    return float(np.dot(vz, z))


def llfs_setpoint(ez: np.ndarray, *, lambda_setpoint: float) -> float:
    """β^{z,*} = λ ||e_z||_2 — Eq. (18)."""
    return float(lambda_setpoint * np.linalg.norm(ez))


def tracking_error(z: np.ndarray, vz: np.ndarray, beta_star: float) -> float:
    """α = β* − v_z^T z — Eq. (19)."""
    return float(beta_star - np.dot(vz, z))


def minimum_norm_latent_shift(z: np.ndarray, vz: np.ndarray, alpha: float) -> np.ndarray:
    """z' = z + α v_z — Eq. (19)."""
    return z + alpha * vz
