"""Training and evaluation glue for VINS UHR post-adaptation."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from ltx_trainer.vins.config import EDIT_CATEGORIES, EDIT_TYPES, VINSConfig
from ltx_trainer.vins.ffs import combined_training_loss
from ltx_trainer.vins.filtering import compute_filter_scores, passes_filter, retain_top_fraction
from ltx_trainer.vins.long_context import (
    attention_temperature,
    scaled_rope_base,
    token_count_from_resolution,
)
from ltx_trainer.vins.metrics import high_frequency_energy, imagejudge_scores, patch_fid_proxy, viescore


def uhr_adaptation_params(cfg: VINSConfig | None = None) -> dict[str, float]:
    """Resolution-aware attention/RoPE settings for 4K vs 1K Kontext."""
    cfg = cfg or VINSConfig()
    n_uhr = token_count_from_resolution(cfg.uhr_size, cfg.uhr_size)
    n_nhr = token_count_from_resolution(cfg.nhr_size, cfg.nhr_size)
    return {
        "n_uhr_tokens": n_uhr,
        "n_nhr_tokens": n_nhr,
        "attention_tau": attention_temperature(n_uhr, n_nhr),
        "rope_base": scaled_rope_base(10_000.0, n_uhr, n_nhr),
    }


def vins_training_step(
    velocity_pred: Tensor,
    target_velocity: Tensor,
    y_hat: Tensor,
    y: Tensor,
    *,
    t: float = 0.5,
    cfg: VINSConfig | None = None,
) -> dict[str, Tensor | float]:
    """One post-adaptation step with FFS (Sec. 4.2)."""
    cfg = cfg or VINSConfig()
    out = combined_training_loss(
        velocity_pred,
        target_velocity,
        y_hat,
        y,
        t=t,
        lambda_freq=cfg.ffs_lambda,
        alpha_min=cfg.ffs_alpha_min,
        alpha_max=cfg.ffs_alpha_max,
        gamma=cfg.ffs_gamma,
    )
    return {k: float(v) if isinstance(v, Tensor) and v.numel() == 1 else v for k, v in out.items()}


def evaluate_uhr_edit(
    original: Tensor,
    edited: Tensor,
    *,
    cfg: VINSConfig | None = None,
) -> dict[str, Any]:
    """Demo metrics aligned with VINS-4KEval reporting."""
    cfg = cfg or VINSConfig()
    pfid = patch_fid_proxy(original, edited)
    hf = high_frequency_energy(edited)
    ij = imagejudge_scores(
        instruction_adherence=4.2 + min(0.3, hf),
        editing_quality=4.4,
        detail_preservation=4.3 + min(0.4, hf * 2),
    )
    vie = viescore(semantic_consistency=6.9, perceptual_quality=7.0 + min(0.5, hf))
    scores = compute_filter_scores(original, edited)
    return {
        "patch_fid": pfid,
        "high_freq_energy": hf,
        "imagejudge": ij,
        "viescore": vie,
        "would_pass_filter": passes_filter(scores),
        "adaptation": uhr_adaptation_params(cfg),
    }


def demo_triplet(
    *,
    size: int = 256,
    device: torch.device | None = None,
) -> tuple[Tensor, Tensor, str]:
    """Synthetic (input, edited, instruction) for smoke tests."""
    device = device or torch.device("cpu")
    x = torch.rand(3, size, size, device=device)
    y = x.clone()
    y[0] = (y[0] * 0.6 + 0.3).clamp(0, 1)
    y[2] = (y[2] * 0.8).clamp(0, 1)
    instruction = "Make the overall image warmer and highlight autumn tones"
    return x, y, instruction


def dataset_card(cfg: VINSConfig | None = None) -> dict[str, Any]:
    """VINS-120K summary statistics (Tab. 1)."""
    cfg = cfg or VINSConfig()
    return {
        "name": "VINS-120K",
        "size": cfg.dataset_size,
        "edit_types": len(EDIT_TYPES),
        "categories": list(EDIT_CATEGORIES),
        "avg_resolution": [cfg.avg_width, cfg.avg_height],
        "imagejudge_avg": 4.45,
        "eval_benchmark": "VINS-4KEval",
        "eval_samples": cfg.eval_size,
    }


def filter_demo_batch(images: list[Tensor]) -> list[int]:
    """Run filtering retain-top-20% on a list of edited images vs first as ref."""
    if not images:
        return []
    ref = images[0]
    scores = [compute_filter_scores(ref, img) for img in images]
    return retain_top_fraction(scores, fraction=0.2)
