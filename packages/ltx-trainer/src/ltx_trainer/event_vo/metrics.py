"""Trajectory evaluation (Sim(3) Umeyama, ATE — paper uses evo [14])."""

from __future__ import annotations

import numpy as np


def align_sim3_umeyama(
    est: np.ndarray,
    gt: np.ndarray,
) -> tuple[np.ndarray, float]:
    """
    Align estimated trajectory to ground truth (7-DoF similarity).

    est, gt: (N, 3)
    Returns aligned est and scale.
    """
    est = np.asarray(est, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    n = min(len(est), len(gt))
    if n < 2:
        return est, 1.0
    est = est[:n]
    gt = gt[:n]
    mu_e = est.mean(axis=0)
    mu_g = gt.mean(axis=0)
    est_c = est - mu_e
    gt_c = gt - mu_g
    cov = gt_c.T @ est_c / n
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    var_e = (est_c**2).sum() / n
    scale = float(np.trace(np.diag(D) @ S) / max(var_e, 1e-12))
    t = mu_g - scale * R @ mu_e
    aligned = (scale * (R @ est.T)).T + t
    return aligned, scale


def absolute_trajectory_error(aligned: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    n = min(len(aligned), len(gt))
    err = np.linalg.norm(aligned[:n] - gt[:n], axis=1)
    return {
        "ate_mean_m": float(err.mean()),
        "ate_rmse_m": float(np.sqrt((err**2).mean())),
        "ate_max_m": float(err.max()),
    }
