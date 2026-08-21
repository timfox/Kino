"""SAE feature decomposition and inference-time erasure (Sec. 3.3, Eq. 2–4, 7)."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class SparseAutoencoder(nn.Module):
    """Minimal SAE: f = ReLU(W_enc h + b_enc), h_hat = W_dec f + b_dec."""

    def __init__(self, d_model: int, d_sae: int) -> None:
        super().__init__()
        self.encoder = nn.Linear(d_model, d_sae)
        self.decoder = nn.Linear(d_sae, d_model)

    def encode(self, h: Tensor) -> Tensor:
        return torch.relu(self.encoder(h))

    def decode(self, f: Tensor) -> Tensor:
        return self.decoder(f)

    def forward(self, h: Tensor) -> tuple[Tensor, Tensor]:
        f = self.encode(h)
        return f, self.decode(f)


def shared_mask(f_neg: Tensor, eps: float = 1e-8) -> Tensor:
    """Eq. (3): m_shared from negative-set activations."""
    if f_neg.dim() == 1:
        denom = f_neg.max() + eps
        return f_neg / denom
    denom = f_neg.amax(dim=0, keepdim=True) + eps
    return f_neg / denom


def specificity_mask(f_neg: Tensor, eps: float = 1e-8) -> Tensor:
    """Eq. (4): m_spec = 1 - m_shared."""
    return 1.0 - shared_mask(f_neg, eps=eps)


def concept_erasure(
    h: Tensor,
    sae: SparseAutoencoder,
    m_spec: Tensor,
    *,
    gamma: float = 10.0,
) -> Tensor:
    """Eq. (7): h' = h - gamma * W_dec(f ⊙ m_spec)."""
    f = sae.encode(h)
    if m_spec.shape != f.shape:
        raise ValueError("m_spec must match sparse feature shape")
    v_tar = sae.decode(f * m_spec)
    return h - gamma * v_tar


def aggregate_activation(f: Tensor, mask: Tensor) -> Tensor:
    """S = sum(f ⊙ m) used in L_con."""
    return (f * mask).sum()
