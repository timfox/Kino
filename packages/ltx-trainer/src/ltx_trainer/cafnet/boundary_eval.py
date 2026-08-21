"""Boundary MAE evaluation for CAFNet splice detection."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.cafnet.config import CafNetConfig
from ltx_trainer.cafnet.features import extract_feature_triplet
from ltx_trainer.cafnet.model import cafnet_forward


def boundary_mae(predicted_s: np.ndarray, gold_s: np.ndarray) -> float:
    pred = np.asarray(predicted_s, dtype=np.float64).ravel()
    gold = np.asarray(gold_s, dtype=np.float64).ravel()
    n = min(pred.size, gold.size)
    if n == 0:
        return 0.0
    return float(np.mean(np.abs(pred[:n] - gold[:n])))


def eval_synthetic_clips(
    *,
    n_clips: int = 16,
    seed: int = 42,
    cfg: CafNetConfig | None = None,
) -> dict[str, Any]:
    """Run CAFNet forward on synthetic audio; score boundary head vs known splice times."""
    cfg = cfg or CafNetConfig()
    rng = np.random.default_rng(seed)
    sr = cfg.sample_rate_hz
    preds: list[float] = []
    golds: list[float] = []
    for i in range(n_clips):
        dur = 3.0 + 0.2 * rng.random()
        t = np.arange(int(sr * dur)) / sr
        wave = np.sin(2 * np.pi * (220 + 20 * i) * t)
        splice_s = dur * (0.35 + 0.1 * rng.random())
        idx = int(splice_s * sr)
        wave[idx:] *= 0.6
        feats = extract_feature_triplet(wave, cfg)
        out = cafnet_forward(feats, cfg=cfg, seed=seed + i)
        pred_boundary = float(out["boundary_seconds"]["start"])
        preds.append(pred_boundary)
        golds.append(splice_s)
    mae = boundary_mae(np.array(preds), np.array(golds))
    return {"boundary_mae_s": mae, "n_clips": n_clips, "mean_gold_s": float(np.mean(golds))}


def boundary_eval_smoke(cfg: CafNetConfig | None = None, *, seed: int = 0) -> dict[str, Any]:
    cfg = cfg or CafNetConfig()
    out = eval_synthetic_clips(n_clips=8, seed=seed, cfg=cfg)
    return {"boundary_mae_finite": np.isfinite(out["boundary_mae_s"]), **out}
