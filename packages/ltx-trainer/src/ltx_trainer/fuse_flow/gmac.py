"""GMAC — geometry-aligned metric-scale extrinsic lifting (stub)."""

from __future__ import annotations

from typing import Any

import torch


def closed_form_scale(
    numerators: torch.Tensor,
    denominators: torch.Tensor,
    weights: torch.Tensor | None = None,
    *,
    eps: float = 1e-8,
) -> float:
    """Eq. (12): weighted least-squares scale s* from depth consistency."""
    w = weights if weights is not None else torch.ones_like(numerators)
    num = (w * numerators).sum()
    den = (w * denominators).sum().clamp_min(eps)
    return float((num / den).item())


def metric_lift_translations(
    translations: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Apply uniform metric scale to backbone-relative translations."""
    return translations * scale


def cycle_consistency_prune(
    pair_scores: torch.Tensor,
    *,
    keep_fraction: float = 0.85,
) -> torch.Tensor:
    """CC gate: keep top fraction of correspondence pairs by consistency score."""
    k = max(1, int(pair_scores.numel() * keep_fraction))
    thresh = torch.topk(pair_scores, k).values.min()
    return pair_scores >= thresh


def gmac_refine_stub(
    *,
    n_cameras: int = 4,
    n_pairs: int = 32,
    seed: int = 0,
) -> dict[str, Any]:
    """Synthetic RC correspondences → closed-form scale → lifted extrinsics."""
    g = torch.Generator().manual_seed(seed)
    z_obs = torch.rand(n_pairs, generator=g) * 2.0 + 0.5
    z_proj_unit = torch.rand(n_pairs, generator=g) * 0.5 + 0.8
    w = torch.rand(n_pairs, generator=g)
    cc = torch.rand(n_pairs, generator=g)
    kept = cycle_consistency_prune(cc, keep_fraction=0.85)
    w = w * kept.float()
    true_s = 1.35
    z_proj = z_proj_unit * true_s
    numerators = w * z_obs * z_proj
    denominators = w * z_proj * z_proj
    s_hat = closed_form_scale(numerators, denominators, w)
    trans = torch.randn(n_cameras, 3, generator=g) * 0.3
    trans_metric = metric_lift_translations(trans, s_hat)
    return {
        "n_cameras": n_cameras,
        "n_correspondences": n_pairs,
        "n_pairs_after_cc": int(kept.sum().item()),
        "scale_estimate": s_hat,
        "scale_target": true_s,
        "translation_metric_norm": float(trans_metric.norm(dim=-1).mean()),
    }
