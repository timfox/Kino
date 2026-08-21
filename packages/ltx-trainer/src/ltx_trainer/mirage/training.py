"""Two-stage flow-matching training stub (Sec. 4.5, Appendix C)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ltx_trainer.mirage.config import MirageConfig


@dataclass
class TrainingStage:
    name: str
    train_side_branch: bool
    train_lora: bool
    lr: float


def training_schedule(cfg: MirageConfig | None = None) -> list[TrainingStage]:
    cfg = cfg or MirageConfig()
    return [
        TrainingStage("stage1_side_branch", True, False, cfg.stage1_lr),
        TrainingStage("stage2_lora_joint", True, True, cfg.stage2_lr),
    ]


def flow_matching_loss(pred: np.ndarray, target: np.ndarray) -> float:
    diff = np.asarray(pred, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    return float(np.mean(diff * diff))


def run_toy_training(cfg: MirageConfig | None = None, *, seed: int = 0) -> dict[str, Any]:
    cfg = cfg or MirageConfig()
    rng = np.random.default_rng(seed)
    target = rng.standard_normal((cfg.latent_channels, cfg.latent_height, cfg.latent_width))
    stages = training_schedule(cfg)
    losses: list[float] = []
    pred = rng.standard_normal(target.shape)
    for stage in stages:
        lr = stage.lr
        for _ in range(3):
            loss = flow_matching_loss(pred, target)
            pred = pred - lr * (pred - target)
            losses.append(loss)
    return {
        "stages": [s.name for s in stages],
        "final_loss": float(flow_matching_loss(pred, target)),
        "loss_monotone": losses[-1] <= losses[0],
        "lora_rank": cfg.lora_rank,
        "lora_alpha": cfg.lora_alpha,
        "lora_dropout": cfg.lora_dropout,
        "text_dropout": cfg.text_dropout,
        "optimizer": {"beta": (cfg.adamw_beta1, cfg.adamw_beta2), "weight_decay": cfg.weight_decay},
        "control_layers": list(cfg.control_layers),
        "flow_steps": cfg.flow_steps,
        "scheduler": cfg.scheduler,
    }
