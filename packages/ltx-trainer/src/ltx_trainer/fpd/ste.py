"""Straight-through estimator for discrete bottleneck (Eq. 8)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def soft_codebook_embedding(logits: Tensor, codebook: Tensor) -> Tensor:
    """Probability-weighted codebook mixture ẽ_i = Σ_v p_θ,iv E_v."""
    probs = F.softmax(logits, dim=-1)
    return probs @ codebook


def straight_through_embedding(
    logits: Tensor,
    hard_indices: Tensor,
    codebook: Tensor,
) -> Tensor:
    """Eq. (8): e_STE = E[ẑ] + ẽ − sg(ẽ)."""
    hard = codebook[hard_indices]
    soft = soft_codebook_embedding(logits, codebook)
    return hard + soft - soft.detach()


def sample_hard_indices(logits: Tensor) -> Tensor:
    """Hard categorical sample from per-position logits [..., K]."""
    return torch.argmax(logits, dim=-1)
