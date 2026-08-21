"""Feature learning and subspace distance (Proposition 4.1, Figure 4)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ltx_trainer.dopr.config import SyntheticFeatureConfig
from ltx_trainer.dopr.optimizer import DoPrState, baseline_gp_step, dopr_step
from ltx_trainer.dopr.config import DoPrConfig


def subspace_distance(g: np.ndarray, g_star: np.ndarray) -> float:
    """dist(G, G*) = ||G P_{G*⊥}||_op for row-orthonormal G*."""
    p_perp = np.eye(g_star.shape[1]) - g_star.T @ g_star
    return float(np.linalg.norm(g @ p_perp, ord=2))


def _make_anisotropic_sigma(d: int, log_span: float, rng: np.random.RandomState) -> np.ndarray:
    o = rng.randn(d, d)
    q, _ = np.linalg.qr(o)
    diag = np.logspace(0.0, log_span, d)
    return q @ np.diag(diag) @ q.T


@dataclass(frozen=True)
class FeatureLearningCurve:
    method: str
    validation_loss: list[float]
    subspace_distance: list[float]


def _synthetic_batch(cfg: SyntheticFeatureConfig, rng: np.random.RandomState) -> tuple[np.ndarray, np.ndarray]:
    sigma_x = _make_anisotropic_sigma(cfg.d_x, cfg.anisotropy_log_span, rng)
    f0 = rng.randn(cfg.d_y, cfg.k)
    b = rng.randn(cfg.d_y, cfg.d_y)
    skew = 0.005 * (b - b.T)
    f_star = (np.eye(cfg.d_y) + skew + 0.5 * skew @ skew) @ f0
    g_star_raw = rng.randn(cfg.k, cfg.d_x)
    q, _ = np.linalg.qr(g_star_raw.T, mode="reduced")
    g_star = q.T

    x = rng.choice([-1.0, 1.0], size=(cfg.batch_size, cfg.d_x))
    x = x @ np.linalg.cholesky(sigma_x).T
    x = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8) * np.sqrt(cfg.d_x)
    y = (f_star @ g_star @ x.T).T + 0.01 * rng.randn(cfg.batch_size, cfg.d_y)
    return x, y, f_star, g_star


def run_feature_learning_synthetic(
    method: str,
    cfg: SyntheticFeatureConfig | None = None,
) -> FeatureLearningCurve:
    cfg = cfg or SyntheticFeatureConfig()
    rng = np.random.RandomState(cfg.seed)
    x, y, f_star, g_star = _synthetic_batch(cfg, rng)

    f = f_star.copy()
    g = rng.randn(cfg.k, cfg.d_x) * 0.01

    val_loss: list[float] = []
    dist_hist: list[float] = []

    dopr_cfg = DoPrConfig(gp="sgd", learning_rate=2e-4, damping=0.1)
    state = DoPrState()
    lr_sgd = 0.002
    lr_adam = 0.005

    for _ in range(cfg.steps):
        pred = (f @ g @ x.T).T
        residual = pred - y
        grad_f = (residual.T @ (g @ x.T).T) / cfg.batch_size
        grad_g = (f.T @ residual.T) @ x / cfg.batch_size
        grad_f = np.clip(grad_f, -1.0, 1.0)
        grad_g = np.clip(grad_g, -1.0, 1.0)

        if method == "sgd":
            g -= lr_sgd * grad_g
        elif method == "dopr_sgd":
            g, state = dopr_step(g, grad_g, x, cfg=dopr_cfg, state=state)
        elif method == "adam":
            g, state = baseline_gp_step(g, grad_g, gp="adam", learning_rate=lr_adam, state=state)
        else:
            raise ValueError(method)

        mse = float(np.mean(residual**2))
        val_loss.append(mse)
        dist_hist.append(subspace_distance(g, g_star))

    return FeatureLearningCurve(method=method, validation_loss=val_loss, subspace_distance=dist_hist)


def ap_vs_gd_feature_step(
    f: np.ndarray,
    g: np.ndarray,
    g_star: np.ndarray,
    sigma_s: np.ndarray,
    eta: float,
) -> tuple[np.ndarray, np.ndarray]:
    """One-step Prop 4.1 update on G with fixed F (simplified)."""
    p_perp = np.eye(g_star.shape[1]) - g_star.T @ np.linalg.pinv(g_star @ g_star.T) @ g_star
    # GD on L = ||F G - F* G*||^2
    g_next_gd = g - eta * (f.T @ f @ g @ sigma_s - f.T @ f @ g_star @ sigma_s)
    g_next_ap = g - eta * (f.T @ f @ g)
    g_next_gd_perp = g_next_gd @ p_perp
    g_next_ap_perp = g_next_ap @ p_perp
    return g_next_gd_perp, g_next_ap_perp
