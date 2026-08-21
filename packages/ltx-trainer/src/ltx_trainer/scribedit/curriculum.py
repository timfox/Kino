"""Coverage-then-Realism two-stage training schedule (Sec. 4.2, 5.1)."""

from __future__ import annotations

import random
from enum import Enum

from ltx_trainer.scribedit.config import ScribEditConfig


class TrainingStage(Enum):
    """Stage 1: synthetic coverage; Stage 2: real-world realism."""

    SYNTHETIC_COVERAGE = 1
    REAL_REALISM = 2


def stage_training_flags(
    stage: TrainingStage,
    *,
    cfg: ScribEditConfig | None = None,
) -> dict[str, bool | float]:
    """Return loss/mosaic flags for the given curriculum stage."""
    cfg = cfg or ScribEditConfig()
    if stage == TrainingStage.SYNTHETIC_COVERAGE:
        return {
            "use_edit_focused_loss": True,
            "use_multi_task_mosaic": True,
            "edit_lambda": cfg.edit_loss_lambda,
        }
    return {
        "use_edit_focused_loss": False,
        "use_multi_task_mosaic": False,
        "edit_lambda": 0.0,
    }


def sample_is_mosaic(rng: random.Random | None = None, *, cfg: ScribEditConfig | None = None) -> bool:
    """Sample multi-task mosaic vs single-task per 4:1 ratio (Sec. 5.1)."""
    cfg = cfg or ScribEditConfig()
    r = rng or random
    single, multi = cfg.single_to_multi_ratio
    total = single + multi
    return r.random() < multi / total
