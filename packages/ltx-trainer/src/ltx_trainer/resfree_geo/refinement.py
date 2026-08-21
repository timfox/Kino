"""Post-hoc α stretching refinement (Sec. 4.3)."""

from __future__ import annotations

import torch
from torch import Tensor


def refined_mapping(
    coords: Tensor,
    displacement: Tensor,
    boundary_mask: Tensor,
    alpha: Tensor,
) -> Tensor:
    """u_α = x + B ⊙ α ⊙ φ (Eq. 19)."""
    return coords + boundary_mask * alpha.view(-1, 1, 1) * displacement


def line_search_alpha(
    displacement: Tensor,
    boundary_mask: Tensor,
    coords: Tensor,
    objective,
    *,
    dim: int = 2,
    steps: int = 5,
) -> Tensor:
    """Coordinate-wise golden-section style search on α ∈ [0.5, 1.5]."""
    alpha = torch.ones(dim)
    for i in range(dim):
        best_a, best_j = 1.0, float("inf")
        for a in torch.linspace(0.5, 1.5, steps):
            trial = alpha.clone()
            trial[i] = float(a)
            u = refined_mapping(coords, displacement, boundary_mask, trial)
            val = float(objective(u).detach())
            if val < best_j:
                best_j, best_a = val, float(a)
        alpha[i] = best_a
    return alpha
