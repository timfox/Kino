"""Training-step glue for ControlLight (Fig. 5)."""

from __future__ import annotations

import torch
from torch import Tensor

from ltx_trainer.controllight.config import ControlLightConfig
from ltx_trainer.controllight.edges import misalignment_weight_map, resize_weight_to_latent
from ltx_trainer.controllight.losses import (
    interpolate_latent,
    velocity_target,
    weighted_flow_matching_loss,
)
from ltx_trainer.controllight.retinex import build_light100k_group


def select_pseudo_target(
    group: dict[float, Tensor],
    strength: float,
) -> Tensor:
    """Pick I_s from training group; linearly blend nearest keys if s not exact."""
    if strength in group:
        return group[strength]
    keys = sorted(group.keys())
    if strength <= keys[0]:
        return group[keys[0]]
    if strength >= keys[-1]:
        return group[keys[-1]]
    for i in range(len(keys) - 1):
        if keys[i] <= strength <= keys[i + 1]:
            t = (strength - keys[i]) / (keys[i + 1] - keys[i])
            return (1 - t) * group[keys[i]] + t * group[keys[i + 1]]
    return group[keys[-1]]


def controllight_training_loss(
    v_pred: Tensor,
    z0: Tensor,
    z1: Tensor,
    i0: Tensor,
    is_image: Tensor,
    *,
    t: float = 0.5,
    cfg: ControlLightConfig | None = None,
    use_weighted: bool = True,
) -> Tensor:
    """Single FM step with optional L_wFM (Sec. 3.2)."""
    cfg = cfg or ControlLightConfig()
    v_star = velocity_target(z1, z0)
    if not use_weighted:
        return ((v_pred - v_star).pow(2)).mean()
    w_img = misalignment_weight_map(
        i0,
        is_image,
        dist_threshold=cfg.dist_threshold_px,
        alpha=cfg.mask_alpha,
        wmin=cfg.weight_min,
    )
    f_w = resize_weight_to_latent(w_img, v_pred.shape)
    while f_w.dim() < v_pred.dim():
        f_w = f_w.unsqueeze(-1)
    return weighted_flow_matching_loss(v_pred, v_star, f_w)


def prepare_training_batch(
    i0: Tensor,
    i1: Tensor,
    strength: float,
    *,
    cfg: ControlLightConfig | None = None,
) -> tuple[Tensor, dict[float, Tensor]]:
    """Build pseudo target and full group for one pair."""
    cfg = cfg or ControlLightConfig()
    group = build_light100k_group(
        i0,
        i1,
        strengths=cfg.enhancement_strengths,
        use_retinex=True,
        beta_scale=cfg.reflectance_beta_scale,
    )
    target = select_pseudo_target(group, strength)
    return target, group
