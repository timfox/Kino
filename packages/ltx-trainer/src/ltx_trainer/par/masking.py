"""Masked autoregressive token layout (Eq. 4–5)."""

from __future__ import annotations

import torch
from torch import Tensor


def random_mask(num_tokens: int, ratio: float, *, generator: torch.Generator | None = None) -> Tensor:
    """Binary mask M: 1 = predict (masked), 0 = known context."""
    probs = torch.full((num_tokens,), ratio)
    return torch.bernoulli(probs, generator=generator)


def outpainting_known_mask(num_tokens: int, known_fraction: float) -> Tensor:
    """Sk positions: 0 = known (condition), 1 = generate (Su)."""
    k = int(num_tokens * (1.0 - known_fraction))
    mask = torch.ones(num_tokens)
    mask[:k] = 0.0
    return mask
