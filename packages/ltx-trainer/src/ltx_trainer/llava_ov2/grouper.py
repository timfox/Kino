"""Group-visible attention masks for codec / frame / image inputs (Sec. 2.2)."""

from __future__ import annotations

import torch
from torch import Tensor


def group_visible_mask(group_ids: Tensor) -> Tensor:
    """Token i attends to j iff group_ids[i] == group_ids[j]."""
    g = group_ids.view(-1, 1)
    return (g == g.T).float()


def fixed_four_slot_groups(num_tokens: int, slots_per_group: int = 4) -> Tensor:
    """Sampled-frame / IPPP fixed grouping."""
    return torch.arange(num_tokens) // slots_per_group


def codec_gop_groups(num_tokens: int, gop_ids: Tensor) -> Tensor:
    """Map per-token GOP id κ_u to group-visible assignment."""
    if gop_ids.numel() != num_tokens:
        raise ValueError("gop_ids length must match num_tokens")
    return gop_ids


def image_single_group(num_tokens: int) -> Tensor:
    """Static image: degenerate single temporal group."""
    return torch.zeros(num_tokens, dtype=torch.long)
