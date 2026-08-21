"""Feature collapse / manifold alignment metrics — Sec. 5, Table 5, Fig. 4."""

from __future__ import annotations

import torch
from torch import Tensor


def slot_diversity(slots: Tensor) -> float:
    """Mean pairwise cosine distance between slots within a frame — Table 5."""
    s = slots[0]
    n = s.shape[0]
    if n < 2:
        return 0.0
    s_norm = torch.nn.functional.normalize(s, dim=-1)
    sim = s_norm @ s_norm.T
    mask = ~torch.eye(n, dtype=torch.bool, device=s.device)
    return float((1.0 - sim[mask]).mean().item())


def slot_temporal_variance(slot_trajectory: list[Tensor], slot_idx: int = 0) -> float:
    """Variance of a slot across time — Table 5."""
    traj = torch.stack([s[0, slot_idx] for s in slot_trajectory], dim=0)
    return float(traj.var(dim=0).mean().item())


def latent_distance(fw_slots: list[Tensor], bw_slots: list[Tensor]) -> float:
    """Mean L2 between forward/backward slots at matched timesteps."""
    dists = [torch.mean((f - b) ** 2).sqrt() for f, b in zip(fw_slots, bw_slots, strict=True)]
    return float(torch.stack(dists).mean().item())


def reconstruction_consensus(fw_recons: list[Tensor], bw_recons: list[Tensor], targets: list[Tensor]) -> float:
    """Mean reconstruction error vs ground-truth features for both streams."""
    fw_err = sum(torch.mean((r - t) ** 2) for r, t in zip(fw_recons, targets, strict=True)) / len(targets)
    bw_err = sum(torch.mean((r - t) ** 2) for r, t in zip(bw_recons, targets, strict=True)) / len(targets)
    return float(((fw_err + bw_err) / 2).item())


def manifold_scatter_point(
    fw_slots: list[Tensor],
    bw_slots: list[Tensor],
    fw_recons: list[Tensor],
    bw_recons: list[Tensor],
    targets: list[Tensor],
) -> dict[str, float]:
    """Single-frame summary for Fig. 4 style analysis."""
    return {
        "latent_distance": latent_distance(fw_slots, bw_slots),
        "reconstruction_consensus": reconstruction_consensus(fw_recons, bw_recons, targets),
    }
