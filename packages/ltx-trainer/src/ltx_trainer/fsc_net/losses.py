"""Multi-resolution STFT, LSD, and LSGAN loss stubs (§II-D)."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.fsc_net.config import FscNetConfig


def spectral_convergence(y_hat: np.ndarray, y: np.ndarray) -> float:
    num = np.linalg.norm(np.abs(y) - np.abs(y_hat))
    den = np.linalg.norm(np.abs(y)) + 1e-8
    return float(num / den)


def log_magnitude_loss(y_hat: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(np.abs(np.log(np.abs(y) + 1e-8) - np.log(np.abs(y_hat) + 1e-8))))


def mr_stft_loss(y_hat: np.ndarray, y: np.ndarray) -> float:
    return spectral_convergence(y_hat, y) + log_magnitude_loss(y_hat, y)


def log_spectral_distance(y_hat: np.ndarray, y: np.ndarray, *, eps: float = 1e-8) -> float:
    ratio = (np.abs(y) ** 2 + eps) / (np.abs(y_hat) ** 2 + eps)
    return float(np.sqrt(np.mean(np.log10(ratio) ** 2)))


def stage_loss(
    y_hat: np.ndarray,
    y: np.ndarray,
    *,
    cfg: FscNetConfig | None = None,
) -> float:
    cfg = cfg or FscNetConfig()
    return mr_stft_loss(y_hat, y) + cfg.lambda_lsd * log_spectral_distance(y_hat, y)


def losses_demo(*, seed: int = 0, cfg: FscNetConfig | None = None) -> dict[str, Any]:
    cfg = cfg or FscNetConfig()
    rng = np.random.default_rng(seed)
    y = rng.uniform(0.2, 1.0, (32, 128))
    y_hat = y + 0.05 * rng.standard_normal(y.shape)
    return {
        "mr_stft": mr_stft_loss(y_hat, y),
        "lsd": log_spectral_distance(y_hat, y),
        "stage_loss": stage_loss(y_hat, y, cfg=cfg),
        "lambda_lsd": cfg.lambda_lsd,
        "lambda_adv": cfg.lambda_adv,
    }
