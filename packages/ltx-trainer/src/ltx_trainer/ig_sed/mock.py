"""IG-SED temporal attribution smoke (arXiv:2605.23293)."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.ig_sed.config import IgSedConfig
from ltx_trainer.ig_sed.ig import aggregate_to_frames, binarize_percentile, integrated_gradients_1d
from ltx_trainer.ig_sed.metrics import frame_f1, pointing_game, temporal_iou


def evaluation_smoke(cfg: IgSedConfig | None = None) -> dict[str, Any]:
    c = cfg or IgSedConfig()
    x = np.linspace(0.0, 1.0, 64, dtype=np.float64)

    def score_fn(arr: np.ndarray) -> float:
        return float(np.sum(arr**2))

    attr = integrated_gradients_1d(x, np.zeros_like(x), score_fn, steps=8)
    frames = aggregate_to_frames(attr, sample_rate_hz=32000, frame_ms=100)
    gt = np.zeros(frames.size, dtype=bool)
    gt[min(2, gt.size - 1)] = True
    mask = binarize_percentile(frames, 56.0)
    iou = temporal_iou(mask, gt)
    _, _, f1 = frame_f1(mask, gt)
    return {
        "paper": c.paper_arxiv,
        "n_frames": int(frames.size),
        "temporal_iou": round(iou, 3),
        "toy_iou": round(iou, 3),
        "frame_f1": round(f1, 3),
        "pointing_hit": pointing_game(frames, gt),
    }
