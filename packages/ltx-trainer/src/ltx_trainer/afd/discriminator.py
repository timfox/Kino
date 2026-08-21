"""Prompt-paired Bradley–Terry discriminator (Sec. 3.1, Eq. 1–2, 10)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ltx_trainer.afd.config import AFDConfig


def bradley_terry_loss(
    d_teacher: Tensor,
    d_student: Tensor,
) -> Tensor:
    """Eq. (1): −log σ(D(x^T) − D(ŷ))."""
    return (-F.logsigmoid(d_teacher - d_student)).mean()


def batch_baseline(scores: Tensor) -> Tensor:
    """Batch mean baseline b(y) for advantage (Sec. 3.1)."""
    return scores.mean()


def advantage_score(
    d_student: Tensor,
    baseline: Tensor | None = None,
) -> Tensor:
    """Eq. (2): r = sg(D(ŷ) − b)."""
    b = batch_baseline(d_student) if baseline is None else baseline
    return (d_student - b).detach()


def density_ratio_logit(d_score: Tensor, eps: float = 1e-6) -> Tensor:
    """Eq. (10): log D/(1−D) ≈ log π_T/π_θ at optimum."""
    d = d_score.clamp(eps, 1.0 - eps)
    return torch.log(d / (1.0 - d))


def normalize_advantages_to_weights(
    advantages: Tensor,
    *,
    clip_max: float = 5.0,
) -> Tensor:
    """Map advantages to w ∈ [0, 1] within minibatch (Sec. 3.2)."""
    adv = advantages.clamp(-clip_max, clip_max)
    lo, hi = adv.min(), adv.max()
    if (hi - lo).abs() < 1e-6:
        return torch.ones_like(adv) * 0.5
    return (adv - lo) / (hi - lo)


class PromptPairedDiscriminator(nn.Module):
    """LoRA-adapted video scorer stub (VideoAlign-style)."""

    def __init__(self, cfg: AFDConfig, *, embed_dim: int = 64) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1),
        )

    def forward(self, video_feat: Tensor, prompt_feat: Tensor) -> Tensor:
        """Return scalar logit per sample; features are pre-pooled stubs."""
        x = video_feat + prompt_feat
        return self.encoder(x).squeeze(-1)
