"""GRPO training losses for smoke."""

from __future__ import annotations

import torch
from torch import Tensor

from ltx_trainer.panoenv.grpo import clipped_policy_loss, grpo_total_loss, group_relative_advantage


def grpo_step_loss(
    log_probs: Tensor,
    old_log_probs: Tensor,
    rewards: Tensor,
    kl: Tensor,
    *,
    beta: float = 0.01,
) -> Tensor:
    adv = group_relative_advantage(rewards)
    l_grpo = clipped_policy_loss(log_probs, old_log_probs, adv)
    return grpo_total_loss(l_grpo, kl, beta=beta)
