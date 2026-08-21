"""Proxy HAD scoring and ArmCBA metric."""

from __future__ import annotations

import numpy as np

from ltx_trainer.ahad.hsi_ops import affine_subspace_basis, mean_shift


def rx_anomaly_scores(y: np.ndarray, *, subspace_dim: int = 8) -> np.ndarray:
    """RX-style global anomaly scores in [0, 1] (proxy for benchmark HAD)."""
    y_s, mu = mean_shift(y)
    h, w, c = y.shape
    flat = y_s.reshape(-1, c)
    cov = flat.T @ flat / max(flat.shape[0] - 1, 1)
    cov += np.eye(c) * 1e-4
    inv = np.linalg.inv(cov)
    scores = np.sum(flat @ inv * flat, axis=1)
    u = affine_subspace_basis(y_s, subspace_dim)
    proj = flat @ u @ u.T
    residual = flat - proj
    res_energy = np.sum(residual**2, axis=1)
    combined = 0.55 * scores + 0.45 * res_energy
    combined = (combined - combined.min()) / (combined.max() - combined.min() + 1e-8)
    return combined.reshape(h, w)


def auc_pd_pf(scores: np.ndarray, mask: np.ndarray) -> float:
    """Trapezoidal AUC(PD, PF) for binary anomaly mask."""
    y = scores.ravel().astype(np.float64)
    t = mask.ravel().astype(bool)
    if t.sum() == 0 or (~t).sum() == 0:
        return 0.5
    order = np.argsort(-y)
    y_sorted = y[order]
    t_sorted = t[order]
    tps = np.cumsum(t_sorted)
    fps = np.cumsum(~t_sorted)
    tpr = tps / t.sum()
    fpr = fps / (~t).sum()
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(tpr, fpr))
    return float(np.trapz(tpr, fpr))


def armcba(auc_up: float, auc_ap: float) -> float:
    """ArmCBA = (1 - AUC_ap/AUC_up) × 100% (Section IV-B)."""
    if auc_up <= 1e-8:
        return 0.0
    return max(0.0, (1.0 - auc_ap / auc_up) * 100.0)


def glf_denoise_proxy(y: np.ndarray, *, rank: int = 8) -> np.ndarray:
    """Low-rank smoothing proxy for GLF restoration (Section IV-D)."""
    h, w, c = y.shape
    flat = y.reshape(-1, c)
    u, s, vt = np.linalg.svd(flat, full_matrices=False)
    k = min(rank, len(s))
    recon = (u[:, :k] * s[:k]) @ vt[:k]
    recon = recon.reshape(h, w, c)
    return 0.65 * recon + 0.35 * y
