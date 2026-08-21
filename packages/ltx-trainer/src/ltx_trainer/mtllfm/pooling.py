"""Temporal softmax pooling (Sec. 3.3, Eq. 2–4)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def temporal_softmax_pool(
    features: Tensor,
    *,
    w: Tensor | None = None,
    b: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """
    Pool sequence ``F_m`` → ``f_m`` with attention ``α_m``.

    Args:
        features: (T, d) projected modality features.
        w, b: optional learned score parameters (d,) and scalar.

    Returns:
        pooled: (d,) aggregated vector.
        alpha: (T,) attention distribution.
    """
    if features.dim() != 2:
        raise ValueError("features must be (T, d)")
    t_steps, dim = features.shape
    if w is None:
        w = torch.randn(dim, device=features.device, dtype=features.dtype) * 0.02
    if b is None:
        b = torch.zeros(1, device=features.device, dtype=features.dtype)

    scores = torch.tanh(features @ w + b)  # (T,)
    alpha = F.softmax(scores, dim=0)
    pooled = (alpha.unsqueeze(-1) * features).sum(dim=0)
    return pooled, alpha
