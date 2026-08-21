"""Adaptive densification / pruning (Algorithm 1)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class DensifyConfig:
    grad_threshold: float = 0.0002
    scene_extent: float = 1.0
    epsilon_extent: float = 0.01
    split_children: int = 2
    min_opacity: float = 0.005
    max_screen_radius: float = 20.0
    opacity_reset_cap: float = 0.01


def should_densify(grad_norm: float, cfg: DensifyConfig) -> bool:
    return grad_norm >= cfg.grad_threshold


def clone_mask(max_scale: Tensor, cfg: DensifyConfig) -> Tensor:
    return max_scale <= (cfg.epsilon_extent * cfg.scene_extent)


def prune_mask(opacity: Tensor, screen_radius: Tensor, cfg: DensifyConfig) -> Tensor:
    return (opacity < cfg.min_opacity) | (screen_radius > cfg.max_screen_radius)


def split_scales(scale: Tensor, n_children: int = 2) -> Tensor:
    """Σ_i / (0.8 N) scaling rule from Algorithm 1."""
    return scale / (0.8 * n_children)


def reset_opacity(opacity: Tensor, cap: float = 0.01) -> Tensor:
    return torch.minimum(opacity, torch.full_like(opacity, cap))


def densify_step_report(
    grad_norms: Tensor,
    opacities: Tensor,
    max_scales: Tensor,
    *,
    cfg: DensifyConfig | None = None,
) -> dict[str, int]:
    """Count clone/split/prune candidates (stub planner)."""
    cfg = cfg or DensifyConfig()
    densify = grad_norms >= cfg.grad_threshold
    clone = densify & clone_mask(max_scales, cfg)
    split = densify & ~clone
    prune = prune_mask(opacities, torch.zeros_like(opacities), cfg)
    return {
        "clone": int(clone.sum()),
        "split": int(split.sum()),
        "prune": int(prune.sum()),
    }
