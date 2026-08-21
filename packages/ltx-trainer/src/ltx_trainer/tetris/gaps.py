"""Maximum tile sampling gaps (Sec. 6.2)."""

from __future__ import annotations

import torch
from torch import Tensor


def laplace_miss_rate(missed: Tensor, total: Tensor) -> Tensor:
    """MissRate = (MissedA + 1) / (TotalA + 2)."""
    if total.dim() == 2:
        total = total.unsqueeze(-1).expand_as(missed)
    return (missed + 1.0) / (total + 2.0)


def max_gap_matrix(
    miss_rates: Tensor,
    tolerance: float,
    gammas: tuple[int, ...],
) -> Tensor:
    """¯g^M_{i,j} = max{γ ∈ Γ | MissRate(γ)_{i,j} ≤ M̄}, default 1."""
    h, w, g_len = miss_rates.shape
    out = torch.ones(h, w, dtype=torch.long)
    for gi, gamma in enumerate(gammas):
        eligible = miss_rates[:, :, gi] <= tolerance
        out[eligible] = torch.maximum(out[eligible], torch.tensor(gamma, dtype=torch.long))
    return out
