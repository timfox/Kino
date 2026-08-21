"""DistilHuBERT + mean-pool + MLP classifier stub (§3.3)."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.seam.config import SeamConfig


def mean_pool_encoder(
    frame_features: np.ndarray,
) -> np.ndarray:
    return np.mean(frame_features, axis=0)


def mlp_logits(
    z: np.ndarray,
    *,
    seed: int = 0,
) -> float:
    rng = np.random.default_rng(seed)
    w1 = rng.normal(0, 0.05, size=(z.shape[0], 64))
    w2 = rng.normal(0, 0.05, size=(64, 1))
    h = np.maximum(0, z @ w1)
    return float((h @ w2).squeeze())


def classify_window(
    wave: np.ndarray,
    *,
    seed: int = 0,
    cfg: SeamConfig | None = None,
) -> dict[str, Any]:
    cfg = cfg or SeamConfig()
    # Stub frame features from waveform energy bands
    n_frames = max(1, len(wave) // 320)
    bands = np.array_split(wave[: n_frames * 320], n_frames)
    feats = np.array([np.sqrt(np.mean(b**2) + 1e-8) for b in bands])
    z = mean_pool_encoder(feats.reshape(-1, 1))
    logit = mlp_logits(z, seed=seed)
    prob = 1.0 / (1.0 + np.exp(-logit))
    return {
        "backbone": cfg.backbone,
        "params_m": cfg.backbone_params_m,
        "unfreeze_layers": cfg.unfreeze_layers,
        "scripted_prob": float(prob),
        "window_s": cfg.window_s,
    }


def model_demo(*, seed: int = 0, cfg: SeamConfig | None = None) -> dict[str, Any]:
    cfg = cfg or SeamConfig()
    rng = np.random.default_rng(seed)
    wave = rng.normal(0, 0.08, size=int(cfg.window_s * cfg.sample_rate_hz))
    return classify_window(wave, seed=seed, cfg=cfg)
