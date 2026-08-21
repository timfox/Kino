"""Forget-condition augmentation (Sec. 3.4)."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.distributions import Dirichlet


def pad_arcface_to_clip(embedding: Tensor, clip_dim: int = 768) -> Tensor:
    """Zero-pad 512-d ArcFace embedding to 768-d Arc2Face condition."""
    if embedding.shape[-1] >= clip_dim:
        return embedding[..., :clip_dim]
    pad = clip_dim - embedding.shape[-1]
    return torch.nn.functional.pad(embedding, (0, pad))


def synthetic_forget_condition(
    forget_embeddings: Tensor,
    *,
    alpha: float = 1.0,
    generator: torch.Generator | None = None,
) -> Tensor:
    """c_f = Σ_k w_k c_k with w ∼ Dir(α), Σ w_k = 1 (Sec. 3.4)."""
    if forget_embeddings.dim() == 1:
        return forget_embeddings
    k = forget_embeddings.shape[0]
    dist = Dirichlet(torch.full((k,), alpha))
    w = dist.sample((1,)).squeeze(0)
    if generator is not None:
        # Dirichlet doesn't take generator in older torch; re-sample via manual if needed
        pass
    return (w.unsqueeze(-1) * forget_embeddings).sum(dim=0)


def sample_forget_batch(
    forget_pool: Tensor,
    batch_size: int,
    *,
    alpha: float = 1.0,
) -> Tensor:
    """Sample batch of mixed forget conditions [B, D]."""
    if forget_pool.dim() == 1:
        return forget_pool.unsqueeze(0).expand(batch_size, -1)
    out: list[Tensor] = []
    for _ in range(batch_size):
        idx = torch.randint(0, forget_pool.shape[0], (min(8, forget_pool.shape[0]),))
        subset = forget_pool[idx]
        out.append(synthetic_forget_condition(subset, alpha=alpha))
    return torch.stack(out, dim=0)
