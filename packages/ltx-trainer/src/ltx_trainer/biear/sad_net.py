"""Sector-wise SAD-Net multi-task heads (§2.1, §2.4)."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.biear.config import BiearConfig


def sector_for_azimuth(azimuth_deg: float, cfg: BiearConfig | None = None) -> int:
    cfg = cfg or BiearConfig()
    return int(azimuth_deg // cfg.sector_deg) % cfg.n_sectors


def sad_net_forward(
    embedding: np.ndarray,
    *,
    seed: int = 0,
    cfg: BiearConfig | None = None,
) -> list[dict[str, Any]]:
    cfg = cfg or BiearConfig()
    rng = np.random.default_rng(seed)
    bottleneck = embedding
    if bottleneck.shape[0] != 300:
        bottleneck = np.pad(bottleneck, (0, max(0, 300 - bottleneck.shape[0])))[:300]
    w = rng.normal(0, 0.01, size=(300, 100))
    shared = np.maximum(0, bottleneck @ w)
    sectors: list[dict[str, Any]] = []
    for s in range(cfg.n_sectors):
        detect_logit = float(shared[s % 100] - 0.5)
        azimuth = s * cfg.sector_deg + cfg.sector_deg / 2
        dist_class = int(rng.integers(0, 5))
        sectors.append(
            {
                "sector": s,
                "detect_logit": detect_logit,
                "azimuth_deg": azimuth,
                "distance_class": dist_class,
            }
        )
    return sectors


def joint_loss_weights(cfg: BiearConfig | None = None) -> dict[str, float]:
    cfg = cfg or BiearConfig()
    return {
        "lambda_detect": cfg.lambda_detect,
        "lambda_azimuth": cfg.lambda_azimuth,
        "lambda_distance": cfg.lambda_distance,
    }


def sad_net_demo(*, seed: int = 0, cfg: BiearConfig | None = None) -> dict[str, Any]:
    cfg = cfg or BiearConfig()
    emb = np.random.default_rng(seed).normal(size=300)
    sectors = sad_net_forward(emb, seed=seed, cfg=cfg)
    return {
        "n_sectors": cfg.n_sectors,
        "sector_deg": cfg.sector_deg,
        "outputs_per_sector": ["detect", "azimuth", "distance"],
        "active_sectors": sum(1 for s in sectors if s["detect_logit"] > 0),
    }
