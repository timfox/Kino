"""MUST four-stage temporal optimization recipe (Sec. 4, Fig. 4)."""

from __future__ import annotations

from enum import Enum
from typing import Any

import numpy as np

from ltx_trainer.mustbench.config import MustBenchConfig


class MustStage(str, Enum):
    ENCODER_PRETRAIN = "transition_aware_encoder"
    CAPTION_PRETRAIN = "timestamped_caption"
    QA_FINETUNE = "qa_finetune"
    GRPO = "grpo"


def stage_pipeline(cfg: MustBenchConfig | None = None) -> list[dict[str, Any]]:
    cfg = cfg or MustBenchConfig()
    return [
        {
            "stage": MustStage.ENCODER_PRETRAIN.value,
            "order": 1,
            "objectives": ["transition probability (BCE-Dice)", "mood-change (CCC loss)"],
            "backbone": "MERT + LoRA",
        },
        {
            "stage": MustStage.CAPTION_PRETRAIN.value,
            "order": 2,
            "objectives": ["timestamped music caption generation"],
            "features": ["MUST tokens + Qwen2.5 AE", "dynamic transition-aware sampling"],
            "token_rate": cfg.must_token_rate,
        },
        {
            "stage": MustStage.QA_FINETUNE.value,
            "order": 3,
            "objectives": ["5 MUSTBENCH QA tasks"],
            "loss": "answer-only, sample-normalized, task-balanced SFT",
            "lora_rank": cfg.lora_rank,
        },
        {
            "stage": MustStage.GRPO.value,
            "order": 4,
            "objectives": ["TSG exponential timestamp reward", "MTR Gaussian soft-F1"],
            "tasks": ["TSG", "MTR"],
        },
    ]


def bce_dice_loss(pred: np.ndarray, target: np.ndarray, smooth: float = 1e-6) -> float:
    """Stage 1 transition probability BCE-Dice proxy."""
    pred = np.clip(pred, 1e-6, 1 - 1e-6)
    target = np.clip(target, 0.0, 1.0)
    bce = -np.mean(target * np.log(pred) + (1 - target) * np.log(1 - pred))
    inter = np.sum(pred * target)
    dice = 1.0 - (2 * inter + smooth) / (np.sum(pred) + np.sum(target) + smooth)
    return float(bce + dice)


def ccc_loss(pred: np.ndarray, gold: np.ndarray, eps: float = 1e-8) -> float:
    """Stage 1 mood-change concordance correlation coefficient loss."""
    mu_p, mu_g = float(np.mean(pred)), float(np.mean(gold))
    var_p, var_g = float(np.var(pred)), float(np.var(gold))
    cov = float(np.mean((pred - mu_p) * (gold - mu_g)))
    ccc = 2 * cov / (var_p + var_g + (mu_p - mu_g) ** 2 + eps)
    return float(1.0 - ccc)


def dynamic_sampling_weights(transition_prob: np.ndarray, num_tokens: int) -> np.ndarray:
    """Transition-aware dynamic sampling — more tokens at high transition probability."""
    prob = np.asarray(transition_prob, dtype=np.float64)
    prob = prob / (prob.sum() + 1e-8)
    idx = np.linspace(0, len(prob) - 1, num_tokens).astype(int)
    return prob[idx] / (prob[idx].sum() + 1e-8)


def sft_loss_balanced(task_losses: dict[str, float]) -> float:
    """Stage 3 — equal weight per QA type present in minibatch."""
    if not task_losses:
        return 0.0
    return float(np.mean(list(task_losses.values())))


def must_train_smoke(*, seed: int = 42) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    frames = 100
    pred = rng.uniform(0, 1, frames)
    target = np.zeros(frames)
    target[50] = 1.0
    target = np.convolve(target, np.exp(-0.5 * (np.arange(-5, 6) / 2) ** 2), mode="same")
    target = target / (target.max() + 1e-8)
    trans_loss = bce_dice_loss(pred, target)
    mood_loss = ccc_loss(rng.standard_normal(20), rng.standard_normal(20))
    weights = dynamic_sampling_weights(pred, 32)
    sft = sft_loss_balanced({"TSG": 0.4, "LTR": 0.3, "TAD": 0.8, "GTO": 0.35, "MTR": 0.5})
    return {
        "transition_loss": trans_loss,
        "mood_loss": mood_loss,
        "sampling_weights_sum": float(weights.sum()),
        "sft_balanced": sft,
        "num_stages": len(stage_pipeline()),
    }
