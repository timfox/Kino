"""Group Relative Policy Optimization stub (Eq. 3–4)."""

from __future__ import annotations

import torch
from torch import Tensor


def group_relative_advantage(rewards: Tensor) -> Tensor:
    """A(s, a_k) = R_k - mean(R) over group (Eq. 3)."""
    if rewards.dim() == 1:
        baseline = rewards.mean()
        return rewards - baseline
    baseline = rewards.mean(dim=-1, keepdim=True)
    return rewards - baseline


def clipped_policy_loss(
    log_probs: Tensor,
    old_log_probs: Tensor,
    advantages: Tensor,
    *,
    clip_eps: float = 0.2,
) -> Tensor:
    """PPO-style clipped surrogate (GRPO uses same update as PPO)."""
    ratio = torch.exp(log_probs - old_log_probs)
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
    return -torch.min(unclipped, clipped).mean()


def grpo_total_loss(
    policy_loss: Tensor,
    kl: Tensor,
    *,
    beta: float = 0.01,
) -> Tensor:
    """L_total = L_GRPO - β * D_KL (Eq. 4)."""
    return policy_loss - beta * kl
