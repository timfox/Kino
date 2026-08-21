"""CFG-aware MeanFlow distillation stub (§2.6.3, §3.2.4)."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.dotstts.config import DotsttsConfig
from ltx_trainer.dotstts.soar import cfg_velocity


def mean_velocity_target(x_ta: np.ndarray, x_tb: np.ndarray, delta_t: float) -> np.ndarray:
    """Teacher mean velocity finite difference (Eq. 13)."""
    return (x_tb - x_ta) / max(delta_t, 1e-6)


def meanflow_loss(
    student_pred: np.ndarray,
    teacher_mean: np.ndarray,
    eps: float = 1e-4,
) -> tuple[float, float]:
    """Adaptive weighted MSE (Eq. 14–15)."""
    per_sample = float(np.mean((student_pred - teacher_mean) ** 2))
    weight = (per_sample + eps) ** (-0.5)
    return per_sample, weight * per_sample


def meanflow_demo(*, seed: int = 0, cfg: DotsttsConfig | None = None) -> dict[str, Any]:
    cfg = cfg or DotsttsConfig()
    rng = np.random.default_rng(seed)
    x0 = rng.normal(size=(cfg.patch_frames, cfg.latent_dim))
    x1 = rng.normal(size=(cfg.patch_frames, cfg.latent_dim))
    ta, tb = 0.2, 0.6
    x_ta = (1 - ta) * x0 + ta * x1
    v_teacher = cfg_velocity(x1 - x0, rng.normal(size=x1.shape, scale=0.3), cfg.cfg_gamma)
    x_tb = x_ta + (tb - ta) * v_teacher
    target = mean_velocity_target(x_ta, x_tb, tb - ta)
    student = target + rng.normal(scale=0.08, size=target.shape)
    loss, weighted = meanflow_loss(student, target)
    return {
        "nfe_inference": cfg.mf_nfe,
        "cfg_fused_in_teacher": True,
        "single_conditional_pass": True,
        "mean_velocity_loss": round(loss, 6),
        "adaptive_weighted_loss": round(weighted, 6),
    }
