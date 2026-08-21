"""Text-space task routing (Sec. 3.2, Eq. 2–3)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def cosine_sim(a: Tensor, b: Tensor, eps: float = 1e-8) -> Tensor:
    """Cosine similarity; supports broadcast batching."""
    a = F.normalize(a, dim=-1, eps=eps)
    b = F.normalize(b, dim=-1, eps=eps)
    return (a * b).sum(dim=-1)


def task_text_prototypes(class_text_embeds: list[Tensor]) -> Tensor:
    """Eq. (2): ``τ^t = (1/|Y_t|) Σ_{c∈Y_t} e_c``.

    ``class_text_embeds[t]`` is (|Y_t|, d) frozen class embeddings for task ``t``.
    Returns ``tau`` (T, d).
    """
    means = [e.mean(dim=0) for e in class_text_embeds]
    return torch.stack(means, dim=0)


def route_task(v: Tensor, tau: Tensor) -> Tensor:
    """Eq. (3): ``t* = argmax_t sim(v, τ^t)``.

    ``v`` (B, d) or (d,), ``tau`` (T, d) → indices (B,) long.
    """
    if v.dim() == 1:
        v = v.unsqueeze(0)
    sims = cosine_sim(v.unsqueeze(1), tau.unsqueeze(0))  # (B, T)
    return sims.argmax(dim=-1)
