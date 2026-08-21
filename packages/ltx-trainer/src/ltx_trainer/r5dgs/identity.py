"""Identity encoding and alpha compositing (Sec. II-A, Eq. 1)."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from ltx_trainer.r5dgs.config import R5DGSConfig


def composite_identity_along_ray(
    identity: Tensor,
    alpha_prime: Tensor,
) -> Tensor:
    """Front-to-back compositing of identity vectors.

    ``identity`` (R, D), ``alpha_prime`` (R,) opacities along a ray (nearest to camera first).
    Eq. (1): ``E_id = Σ_i e_i α'_i Π_{j=1}^{i-1}(1 - α'_j)``.
    """
    r, d = identity.shape
    if alpha_prime.shape[0] != r:
        raise ValueError("alpha_prime must match ray sample count")
    t = torch.ones((), device=identity.device, dtype=identity.dtype)
    acc = torch.zeros(d, device=identity.device, dtype=identity.dtype)
    for i in range(r):
        a = alpha_prime[i].clamp(0.0, 1.0)
        acc = acc + t * a * identity[i]
        t = t * (1.0 - a)
    return acc


def composite_identity_batched(
    identity: Tensor,
    alpha_prime: Tensor,
) -> Tensor:
    """Vectorized (B, R, D) and (B, R) → (B, D); front-to-back ray order."""
    a = alpha_prime.clamp(0.0, 1.0)
    m = 1.0 - a
    t = torch.cumprod(torch.cat([torch.ones_like(m[:, :1]), m[:, :-1]], dim=1), dim=1)
    w = t * a
    return (w.unsqueeze(-1) * identity).sum(dim=1)


class IdentityClassifier(nn.Module):
    """Maps composited identity features to logits over object classes (2D identity loss)."""

    def __init__(self, cfg: R5DGSConfig) -> None:
        super().__init__()
        self.head = nn.Linear(cfg.identity_dim, cfg.num_classes)

    def forward(self, e_id: Tensor) -> Tensor:
        """``e_id`` (B, D) → logits (B, K)."""
        return self.head(e_id)
