"""SOAR self-corrective alignment stub (§2.6.3, §3.2.3)."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.dotstts.config import DotsttsConfig


def cfg_velocity(
    v_cond: np.ndarray,
    v_uncond: np.ndarray,
    gamma: float,
) -> np.ndarray:
    """Classifier-free guidance extrapolation (Eq. 6)."""
    return v_cond + gamma * (v_cond - v_uncond)


def soar_on_trajectory_loss(
    pred: np.ndarray,
    x1: np.ndarray,
    x0: np.ndarray,
    omega: float = 1.0,
) -> float:
    target = x1 - x0
    return float(omega * np.mean((pred - target) ** 2))


def soar_auxiliary_target(x1: np.ndarray, x_aux: np.ndarray, tau_aux: float, eps: float = 1e-4) -> np.ndarray:
    """Endpoint-consistent velocity (Eq. 10)."""
    denom = max(1.0 - tau_aux, eps)
    return (x1 - x_aux) / denom


def soar_loss(
    on_losses: list[float],
    aux_losses: list[float],
    lambda_aux: float = 1.0,
) -> float:
    b = len(on_losses)
    ma = len(aux_losses)
    num = sum(on_losses) + lambda_aux * sum(aux_losses)
    den = b + lambda_aux * ma
    return num / den if den else 0.0


def soar_demo(*, seed: int = 0, cfg: DotsttsConfig | None = None) -> dict[str, Any]:
    cfg = cfg or DotsttsConfig()
    rng = np.random.default_rng(seed)
    x0 = rng.normal(size=(4, cfg.latent_dim))
    x1 = rng.normal(size=(4, cfg.latent_dim))
    v_cond = x1 - x0
    v_uncond = rng.normal(size=v_cond.shape, scale=0.5)
    v_cfg = cfg_velocity(v_cond, v_uncond, cfg.soar_gamma)
    on = soar_on_trajectory_loss(v_cfg, x1, x0)
    x_aux = x0 + 0.1 * v_cfg
    u_aux = soar_auxiliary_target(x1, x_aux, tau_aux=0.7)
    aux = float(np.mean((v_cond - u_aux) ** 2))
    total = soar_loss([on], [aux] * cfg.soar_aux_samples, lambda_aux=cfg.soar_lambda_aux)
    return {
        "reward_free": True,
        "lambda_aux": cfg.soar_lambda_aux,
        "gamma_soar": cfg.soar_gamma,
        "aux_samples_per_batch": cfg.soar_aux_samples,
        "combined_loss": round(total, 6),
    }
