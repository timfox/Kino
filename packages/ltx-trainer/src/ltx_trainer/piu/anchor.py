"""Proximity-based anchor selection (Sec. 3.3, Eq. 5–6)."""

from __future__ import annotations

import random
from typing import Sequence

import torch
from torch import Tensor

from ltx_trainer.piu.config import PIUConfig


def identity_centroid(embeddings: Tensor) -> Tensor:
    """μ_i = (1/|D_i|) Σ w(x) (Eq. 5)."""
    if embeddings.dim() == 1:
        return embeddings
    return embeddings.mean(dim=0)


def cosine_similarity(a: Tensor, b: Tensor) -> float:
    """s_j = ⟨μ_f, μ_j⟩ / (‖μ_f‖‖μ_j‖)."""
    a_n = torch.nn.functional.normalize(a.float(), dim=-1)
    b_n = torch.nn.functional.normalize(b.float(), dim=-1)
    return float((a_n * b_n).sum().item())


def candidate_anchor_set(
    forget_centroid: Tensor,
    other_centroids: dict[int, Tensor],
    *,
    tau: float = 0.2,
    epsilon: float = 1e-2,
    forget_id: int | None = None,
) -> list[int]:
    """A_τ = {j ≠ f : |s_j − τ| < ε} (Eq. 6)."""
    candidates: list[int] = []
    for jid, mu_j in other_centroids.items():
        if forget_id is not None and jid == forget_id:
            continue
        s_j = cosine_similarity(forget_centroid, mu_j)
        if abs(s_j - tau) < epsilon:
            candidates.append(jid)
    return candidates


def select_anchor_identity(
    forget_centroid: Tensor,
    other_centroids: dict[int, Tensor],
    *,
    cfg: PIUConfig | None = None,
    forget_id: int | None = None,
    rng: random.Random | None = None,
) -> int:
    """Uniformly sample a ∼ A_τ; fallback to nearest-τ if empty."""
    cfg = cfg or PIUConfig()
    pool = candidate_anchor_set(
        forget_centroid,
        other_centroids,
        tau=cfg.anchor_tau,
        epsilon=cfg.anchor_epsilon,
        forget_id=forget_id,
    )
    if pool:
        rng = rng or random.Random()
        return rng.choice(pool)
    # fallback: identity whose similarity is closest to τ
    best_id = -1
    best_dist = float("inf")
    for jid, mu_j in other_centroids.items():
        if forget_id is not None and jid == forget_id:
            continue
        dist = abs(cosine_similarity(forget_centroid, mu_j) - cfg.anchor_tau)
        if dist < best_dist:
            best_dist = dist
            best_id = jid
    if best_id < 0:
        raise ValueError("no anchor candidates available")
    return best_id


def build_centroids_from_clusters(
    cluster_embeddings: dict[int, Tensor],
) -> dict[int, Tensor]:
    """Map identity id -> centroid μ_i."""
    return {i: identity_centroid(e) for i, e in cluster_embeddings.items()}
