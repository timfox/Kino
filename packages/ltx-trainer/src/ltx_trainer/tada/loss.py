"""TADA training loss: covariance + distribution + realism (Sec. 3.4)."""

from __future__ import annotations

import numpy as np

from ltx_trainer.tada.residuals import frobenius_cov_distance, patch_residuals, residual_covariance


def wasserstein1_stub(samples_a: np.ndarray, samples_b: np.ndarray) -> float:
    """
    Toy Earth Mover (W1) proxy: mean L2 between sorted per-dimension marginals.

    Paper uses differentiable W1; full optimal transport is external.
    """
    a = np.sort(samples_a.reshape(samples_a.shape[0], -1), axis=0)
    b = np.sort(samples_b.reshape(samples_b.shape[0], -1), axis=0)
    n = min(a.shape[0], b.shape[0])
    return float(np.mean(np.abs(a[:n] - b[:n])))


def realism_l2(tif_before: np.ndarray, tif_after: np.ndarray) -> float:
    """ℓ2 between normalized TIF and emulated developed TIF (paper realism term)."""
    a = _normalize01(tif_before)
    b = _normalize01(tif_after)
    return float(np.mean((a - b) ** 2))


def _normalize01(img: np.ndarray) -> np.ndarray:
    x = np.asarray(img, dtype=np.float64)
    lo, hi = x.min(), x.max()
    if hi - lo < 1e-12:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def tada_loss(
    source_image: np.ndarray,
    target_image: np.ndarray,
    *,
    emulated_source: np.ndarray | None = None,
    patch_shape: tuple[int, int] = (8, 16),
    lam: float = 1.0,
    mu: float = 1.0,
    gamma: float = 1.0,
) -> dict[str, float]:
    """
    L = λ‖Cov(E(S))−Cov(E(T))‖²_F + μ·d(E(S),E(T)) + γ·ℓ2(S_TIF, S_TADA).
    """
    s_img = emulated_source if emulated_source is not None else source_image
    patches_s = patch_residuals(s_img, patch_shape)
    patches_t = patch_residuals(target_image, patch_shape)
    cov_s = residual_covariance(patches_s)
    cov_t = residual_covariance(patches_t)
    geom = frobenius_cov_distance(cov_s, cov_t)
    dist = wasserstein1_stub(patches_s, patches_t)
    real = realism_l2(source_image, s_img)
    total = lam * geom + mu * dist + gamma * real
    return {
        "total": total,
        "geometric_alignment": geom,
        "distribution_alignment": dist,
        "realism_l2": real,
    }


def regret_stub(pe_source_on_target: float, intrinsic_difficulty_target: float) -> float:
    """R(S,T) = P_E(train S, test T) − intrinsic difficulty(T); stub returns non-negative regret."""
    return max(0.0, pe_source_on_target - intrinsic_difficulty_target)
