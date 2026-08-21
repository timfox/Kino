"""S.E.E.D. — Semantic Energy–Entropy Density scale selection (Sec. 2.2, Eq. 3–4)."""

from __future__ import annotations

import torch
from torch import Tensor


def semantic_similarity(image_emb: Tensor, text_emb: Tensor) -> Tensor:
    """Eq. (1): clamped cosine similarity per crop, shape (K,)."""
    if image_emb.dim() == 1:
        image_emb = image_emb.unsqueeze(0)
    img = image_emb / image_emb.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    txt = text_emb / text_emb.norm().clamp(min=1e-8)
    return torch.clamp((img @ txt).squeeze(-1), min=0.0)


def normalized_entropy(attribution: Tensor, eps: float = 1e-8) -> Tensor:
    """Normalized entropy term in Eq. (3) for one crop map (H×W) or flat (Np,)."""
    flat = attribution.reshape(-1).clamp(min=0.0)
    total = flat.sum() + eps
    if total <= eps:
        return torch.tensor(0.0, device=attribution.device, dtype=attribution.dtype)
    p = flat / total
    ent = -(p * torch.log(p + eps)).sum()
    n_p = flat.numel()
    return ent / torch.log(torch.tensor(float(n_p), device=attribution.device))


def spatial_seed_score(attribution: Tensor, eps: float = 1e-8) -> Tensor:
    """Eq. (3): energy density × normalized entropy."""
    flat = attribution.reshape(-1).clamp(min=0.0)
    energy = flat.mean()
    ent_norm = normalized_entropy(flat, eps=eps)
    return energy * ent_norm


def minmax_normalize(scores: Tensor, eps: float = 1e-8) -> Tensor:
    lo = scores.min()
    hi = scores.max()
    if float(hi - lo) < eps:
        return torch.ones_like(scores)
    return (scores - lo) / (hi - lo + eps)


def select_scale_seed(
    s_sem: Tensor,
    s_spa: Tensor,
    *,
    eps: float = 1e-8,
) -> tuple[int, Tensor]:
    """Eq. (4): k* = argmax_k hat(S_sem) * hat(S_spa)."""
    if s_sem.shape != s_spa.shape:
        raise ValueError("s_sem and s_spa must have the same length")
    joint = minmax_normalize(s_sem, eps=eps) * minmax_normalize(s_spa, eps=eps)
    k_star = int(torch.argmax(joint).item())
    return k_star, joint
