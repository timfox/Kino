"""Entity-Binding Distillation via DINOv2 cluster maps (Sec. 4.3, Eq. 5)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def kmeans_cluster_maps(features: Tensor, k: int, *, steps: int = 10) -> Tensor:
    """``features`` (T, N, D) → binary cluster map C (T, K, N)."""
    t, n, d = features.shape
    out = torch.zeros(t, k, n, device=features.device)
    for ti in range(t):
        x = features[ti]
        perm = torch.randperm(n, device=x.device)[:k]
        centers = x[perm].clone()
        for _ in range(steps):
            dist = torch.cdist(x, centers)
            assign = dist.argmin(dim=1)
            for ki in range(k):
                mask = assign == ki
                if mask.any():
                    centers[ki] = x[mask].mean(dim=0)
        dist = torch.cdist(x, centers)
        assign = dist.argmin(dim=1)
        for ki in range(k):
            out[ti, ki] = (assign == ki).float()
    return out


def hungarian_match_cost(attn: Tensor, clusters: Tensor) -> Tensor:
    """Per-frame BCE cost matrix (K, K) for matching. ``attn`` (1, K, N), ``clusters`` (1, K, N)."""
    if attn.shape[-1] != clusters.shape[-1]:
        n = min(attn.shape[-1], clusters.shape[-1])
        attn = attn[..., :n]
        clusters = clusters[..., :n]
    k = attn.shape[1]
    cost = torch.zeros(k, k, device=attn.device)
    for i in range(k):
        for j in range(k):
            a = attn[0, i]
            c = clusters[0, j]
            bce = F.binary_cross_entropy(a, c, reduction="none").mean()
            cost[i, j] = bce
    return cost


def greedy_hungarian_match(cost: Tensor) -> list[int]:
    """Greedy assignment when scipy unavailable (smoke tests)."""
    k = cost.shape[0]
    used_r, used_c, perm = set(), set(), []
    flat = cost.flatten()
    for _ in range(k):
        idx = flat.argmin()
        i, j = idx // k, idx % k
        if i in used_r or j in used_c:
            flat[idx] = 1e6
            continue
        used_r.add(i)
        used_c.add(j)
        perm.append((i, j))
        flat[idx] = 1e6
    mapping = {i: j for i, j in perm}
    return [mapping.get(i, i) for i in range(k)]


def entity_binding_distillation_loss(
    attn_hat: Tensor,
    cluster_maps: Tensor,
) -> Tensor:
    """Eq. (5): BCE between matched slot–cluster pairs, averaged over T,K,N."""
    t, k, n = attn_hat.shape
    total = attn_hat.new_tensor(0.0)
    count = 0
    for ti in range(t):
        cost = hungarian_match_cost(attn_hat[ti : ti + 1], cluster_maps[ti : ti + 1])
        perm = greedy_hungarian_match(cost)
        for slot_i, cluster_j in enumerate(perm):
            a = attn_hat[ti, slot_i].clamp(1e-6, 1 - 1e-6)
            c = cluster_maps[ti, cluster_j]
            bce = -(c * a.log() + (1 - c) * (1 - a).log())
            total = total + bce.mean()
            count += 1
    return total / max(count, 1)
