"""Fixed-rate token selection (energy global, uniform, random)."""

from __future__ import annotations

import torch
from torch import Tensor


def token_energy(values: Tensor) -> Tensor:
    """Mean squared coefficient per token: (B, N)."""
    return values.pow(2).mean(dim=-1)


def select_topk_energy(values: Tensor, keep_ratio: float) -> Tensor:
    """Boolean mask (B, N) via global energy top-k."""
    b, n, _ = values.shape
    k = max(1, int(n * keep_ratio))
    e = token_energy(values)
    _, idx = torch.topk(e, k, dim=-1)
    mask = torch.zeros(b, n, dtype=torch.bool, device=values.device)
    mask.scatter_(1, idx, True)
    return mask


def select_uniform_stride(n: int, keep_ratio: float, device: torch.device) -> Tensor:
    """1D stride mask pattern, expanded for batch in caller."""
    k = max(1, int(n * keep_ratio))
    step = max(1, n // k)
    mask = torch.zeros(n, dtype=torch.bool, device=device)
    mask[::step][:k] = True
    return mask


def apply_mask(values: Tensor, mask: Tensor) -> Tensor:
    """Zero dropped tokens."""
    return values * mask.unsqueeze(-1).to(values.dtype)


def psnr_from_mse(mse: float, data_range: float = 1.0) -> float:
    if mse <= 0:
        return 99.0
    import math

    return 10.0 * math.log10((data_range**2) / mse)
