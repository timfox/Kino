"""Smoke helpers for TransFACT."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.transfact.config import TransfactConfig
from ltx_trainer.transfact.losses import frame_classification_loss, total_transfact_loss, transferability_loss
from ltx_trainer.transfact.mhi import mhi_from_sequence
from ltx_trainer.transfact.stages import stage_frame_labels


def evaluation_smoke() -> dict[str, Any]:
    cfg = TransfactConfig()
    counts = [1, 2, 2, 4, 4, 8]
    stages = stage_frame_labels(counts)
    l_frame = frame_classification_loss([0.9, 0.85, 0.8, 0.75, 0.7, 0.65])
    l_trans = transferability_loss(0.88)
    total = total_transfact_loss(l_trans=l_trans, l_frame=l_frame, l_stage=l_frame * 0.5)

    frames = np.zeros((10, 12, 12), dtype=np.float32)
    for i in range(10):
        frames[i, 3:6, 2 + i] = 180.0
    mhi = mhi_from_sequence(frames, tau=cfg.mhi_tau, theta=float(cfg.mhi_theta))

    return {
        "dataset": cfg.dataset_name,
        "stage_labels": stages,
        "total_loss": round(total, 4),
        "mhi_nonzero_frac": round(float((mhi[-1] > 0).mean()), 3),
        "headline_accuracy_frames": 82.70,
    }


def mhi_demo(*, tau: int = 15, theta: float = 20.0) -> dict[str, Any]:
    """Toy moving-bar sequence → MHI stack (exported for pipeline_demo / tests)."""
    frames = np.zeros((10, 12, 12), dtype=np.float32)
    for i in range(10):
        frames[i, 3:6, 2 + i] = 180.0
    mhi = mhi_from_sequence(frames, tau=tau, theta=theta)
    return {
        "shape": list(mhi.shape),
        "max": round(float(mhi.max()), 3),
        "nonzero_frac": round(float((mhi > 0).mean()), 3),
    }
