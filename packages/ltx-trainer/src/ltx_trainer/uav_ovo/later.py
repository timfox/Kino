"""LATER: LoRA-anchored test-time re-centering (Sec. 3.2)."""

from __future__ import annotations

from collections import deque

import torch
from torch import Tensor


def build_lora_subspace(
    b_matrices: list[Tensor],
    feature_dim: int,
    keep_ratio: float = 1e-4,
) -> Tensor:
    """Stack LoRA B matrices, SVD → U ∈ R^{d×k} for semantic anchor directions."""
    rows: list[Tensor] = []
    for b in b_matrices:
        if b.shape[0] != feature_dim:
            continue
        rows.append(b.T)
    if not rows:
        return torch.zeros(feature_dim, 0)
    m = torch.cat(rows, dim=0)
    _, s, vh = torch.linalg.svd(m, full_matrices=False)
    if s.numel() == 0:
        return torch.zeros(feature_dim, 0)
    thresh = float(s[0]) * keep_ratio
    keep = (s > thresh).sum().item()
    keep = max(int(keep), 1)
    return vh[:keep].T


def orthogonal_projector(u: Tensor, feature_dim: int) -> Tensor:
    """P⊥ = I − U U^T."""
    if u.numel() == 0 or u.shape[1] == 0:
        return torch.eye(feature_dim, device=u.device, dtype=u.dtype)
    proj = u @ u.T
    eye = torch.eye(feature_dim, device=u.device, dtype=u.dtype)
    return eye - proj


def later_recenter(
    h: Tensor,
    mu_s: Tensor,
    mu_t: Tensor,
    p_perp: Tensor,
    alpha: float = 1.0,
) -> Tensor:
    """Δ = α P⊥(μt − μs), h̃ = h − Δ."""
    delta = alpha * (p_perp @ (mu_t - mu_s))
    return h - delta


class OnlineTargetCenter:
    """Online queue Q for target feature center μt."""

    def __init__(self, max_size: int) -> None:
        self._queue: deque[Tensor] = deque(maxlen=max_size)

    def push(self, h: Tensor) -> None:
        self._queue.append(h.detach())

    def mean(self) -> Tensor:
        if not self._queue:
            raise RuntimeError("empty target queue")
        return torch.stack(list(self._queue)).mean(dim=0)

    def __len__(self) -> int:
        return len(self._queue)
