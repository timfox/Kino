"""Flow-matching objective and curriculum stages (Sec. 4.4–4.5)."""

from __future__ import annotations

from enum import Enum
from typing import Any

import numpy as np

from ltx_trainer.foley_omni.conditioning import apply_sync_injection
from ltx_trainer.foley_omni.config import FoleyOmniConfig


class CurriculumStage(str, Enum):
    TEXT_AUDIO = "text_audio_pretraining"  # TTA, TTS, TTM
    VIDEO_EXPAND = "video_conditioned"  # V2A, VisualTTS
    V2ST_FINETUNE = "complete_soundtrack"  # mixed V2ST + replay


def interpolate_path(x0: np.ndarray, x1: np.ndarray, t: float) -> np.ndarray:
    """x_t = (1 - t) x0 + t x1 (Eq. 5)."""
    return (1.0 - t) * x0 + t * x1


def target_velocity(x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
    """v* = x1 - x0 (Eq. 6)."""
    return x1 - x0


def flow_matching_loss(
    v_pred: np.ndarray,
    x0: np.ndarray,
    x1: np.ndarray,
) -> float:
    """L = ||v_theta - (x1 - x0)||^2_2 (Eq. 7)."""
    target = target_velocity(x0, x1)
    diff = v_pred - target
    return float(np.mean(diff ** 2))


def curriculum_schedule(cfg: FoleyOmniConfig | None = None) -> list[dict[str, Any]]:
    """Table 10 progressive training stages."""
    cfg = cfg or FoleyOmniConfig()
    return [
        {
            "stage": CurriculumStage.TEXT_AUDIO.value,
            "task_groups": ["TTA", "TTS", "TTM"],
            "epochs": cfg.stage1_epochs,
            "learning_rate": cfg.learning_rate_stage12,
        },
        {
            "stage": CurriculumStage.VIDEO_EXPAND.value,
            "task_groups": ["V2A", "VisualTTS"],
            "epochs": cfg.stage2_epochs,
            "learning_rate": cfg.learning_rate_stage12,
        },
        {
            "stage": CurriculumStage.V2ST_FINETUNE.value,
            "task_groups": ["V2ST"],
            "epochs": cfg.stage3_epochs,
            "learning_rate": cfg.learning_rate_stage3,
            "replay_hours_per_task": 100,
        },
    ]


def predict_velocity_stub(
    x_tilde: np.ndarray,
    t: float,
    c_uni: np.ndarray,
    *,
    seed: int = 0,
) -> np.ndarray:
    """Toy v_theta for smoke — linear in x_tilde and context."""
    rng = np.random.default_rng(seed + int(t * 1000))
    w = rng.standard_normal(x_tilde.shape)
    ctx = float(np.mean(c_uni)) if c_uni.size else 0.0
    return w * 0.1 + ctx * 0.01


def training_step_smoke(
    *,
    dim: int = 16,
    t: float = 0.5,
    seed: int = 0,
) -> dict[str, float]:
    """One forward-style smoke: inject sync, predict velocity, compute FM loss."""
    rng = np.random.default_rng(seed)
    x0 = rng.standard_normal(dim)
    x1 = rng.standard_normal(dim)
    z_sync = rng.standard_normal(dim) * 0.05
    x_t = interpolate_path(x0, x1, t)
    x_tilde = apply_sync_injection(x_t, z_sync)
    c_uni = rng.standard_normal(dim + 4)
    v_pred = predict_velocity_stub(x_tilde, t, c_uni, seed=seed)
    loss = flow_matching_loss(v_pred, x0, x1)
    return {"t": t, "loss": loss, "x_tilde_norm": float(np.linalg.norm(x_tilde))}
