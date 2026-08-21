"""Domain-Shift Feature Augmentation (DSFA) stub (§3.2)."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

from ltx_trainer.dsfa.config import DsfaConfig


def channel_stats(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Channel-wise mean and std over time: x shape (B, C, T)."""
    mu = np.mean(x, axis=2)
    sigma = np.std(x, axis=2) + 1e-8
    return mu, sigma


def batch_variance(stats: np.ndarray) -> np.ndarray:
    """Per-channel variance across batch."""
    return np.var(stats, axis=0)


def sample_epsilon(
    shape: tuple[int, ...],
    *,
    distribution: Literal["uniform", "gaussian"],
    rng: np.random.Generator,
) -> np.ndarray:
    if distribution == "uniform":
        return rng.uniform(-1.0, 1.0, size=shape)
    return rng.normal(0.0, 1.0, size=shape)


def dsfa_transform(
    x: np.ndarray,
    *,
    distribution: Literal["uniform", "gaussian"] = "uniform",
    seed: int = 0,
) -> np.ndarray:
    """AdaIN-style DSFA on feature map x (B, C, T)."""
    rng = np.random.default_rng(seed)
    mu, sigma = channel_stats(x)
    sigma_mu = np.sqrt(batch_variance(mu) + 1e-8)
    sigma_sig = np.sqrt(batch_variance(sigma) + 1e-8)
    eps_mu = sample_epsilon(mu.shape, distribution=distribution, rng=rng)
    eps_sig = sample_epsilon(sigma.shape, distribution=distribution, rng=rng)
    beta = mu + eps_mu * sigma_mu
    gamma = sigma + eps_sig * sigma_sig
    normed = (x - mu[:, :, None]) / sigma[:, :, None]
    return gamma[:, :, None] * normed + beta[:, :, None]


def maybe_apply_dsfa(
    x: np.ndarray,
    *,
    prob: float,
    seed: int = 0,
    cfg: DsfaConfig | None = None,
) -> tuple[np.ndarray, bool]:
    cfg = cfg or DsfaConfig()
    rng = np.random.default_rng(seed)
    if rng.uniform() >= prob:
        return x, False
    dist: Literal["uniform", "gaussian"] = "uniform" if cfg.noise_uniform else "gaussian"
    return dsfa_transform(x, distribution=dist, seed=seed + 1), True


def dsfa_demo(*, seed: int = 0, cfg: DsfaConfig | None = None) -> dict[str, Any]:
    cfg = cfg or DsfaConfig()
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(4, 32, 50))
    out, applied = maybe_apply_dsfa(x, prob=cfg.dsfa_prob, seed=seed, cfg=cfg)
    return {
        "dsfa_prob": cfg.dsfa_prob,
        "applied": applied,
        "input_shape": list(x.shape),
        "output_shape": list(out.shape),
        "distribution": "uniform" if cfg.noise_uniform else "gaussian",
    }
