"""Bootstrapped cross-validation metric simulation on toy 4D volumes."""

from __future__ import annotations

import numpy as np

from ltx_trainer.sparse4d_bcv.nyquist import interlaced_time_indices


def normalized_cross_correlation(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a - a.mean()
    b = b - b.mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom < 1e-12:
        return 0.0
    return float(np.dot(a.ravel(), b.ravel()) / denom)


def mse(a: np.ndarray, b: np.ndarray) -> float:
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    return float(np.mean(d * d))


def interlaced_cv_metric(
    recon_a: np.ndarray,
    recon_b: np.ndarray,
) -> float:
    """Eq. (5) toy: correlate even frames of a with odd frames of b."""
    even, odd = interlaced_time_indices(recon_a.shape[0])
    return normalized_cross_correlation(recon_a[even], recon_b[odd])


def bootstrap_cv_summary(
    ground_truth: np.ndarray,
    *,
    num_subsets: int = 10,
    noise: float = 0.15,
    seed: int = 0,
) -> dict[str, float]:
    """Simulate subset reconstructions and cross-validation statistics."""
    rng = np.random.default_rng(seed)
    gt = np.asarray(ground_truth, dtype=np.float64)
    subset_to_gt: list[float] = []
    cv_scores: list[float] = []
    for _ in range(num_subsets):
        recon_i = gt + noise * rng.standard_normal(gt.shape)
        recon_j = gt + noise * rng.standard_normal(gt.shape)
        subset_to_gt.append(normalized_cross_correlation(recon_i, gt))
        cv_scores.append(interlaced_cv_metric(recon_i, recon_j))
    return {
        "subset_to_gt_ncc_mean": float(np.mean(subset_to_gt)),
        "cv_interlaced_ncc_mean": float(np.mean(cv_scores)),
        "subset_to_gt_ncc_std": float(np.std(subset_to_gt)),
    }
