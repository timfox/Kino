"""Lightweight K-means for visual prototype fitting (Sec. 3.3)."""

from __future__ import annotations

import torch
from torch import Tensor


def kmeans_torch(
    x: Tensor,
    k: int,
    *,
    max_iter: int = 20,
    seed: int | None = None,
) -> tuple[Tensor, Tensor]:
    """Lloyd on ``x`` (N, d) → centroids (k, d), assignment (N,) long.

    Initializes with first k points if N >= k else repeats mean.
    """
    n, d = x.shape
    if k < 1:
        raise ValueError("k must be positive")
    if n == 0:
        raise ValueError("empty x")
    rng = torch.Generator(device=x.device)
    if seed is not None:
        rng.manual_seed(seed)
    if n >= k:
        perm = torch.randperm(n, generator=rng, device=x.device)[:k]
        centers = x[perm].clone()
    else:
        centers = x.mean(dim=0, keepdim=True).expand(k, -1).clone()

    assign = torch.zeros(n, dtype=torch.long, device=x.device)
    for _ in range(max_iter):
        dist = torch.cdist(x, centers)  # (N, k)
        new_assign = dist.argmin(dim=-1)
        if torch.equal(new_assign, assign):
            break
        assign = new_assign
        for j in range(k):
            m = assign == j
            if m.any():
                centers[j] = x[m].mean(dim=0)
            else:
                centers[j] = x[torch.randint(0, n, (1,), device=x.device)]
    return centers, assign
