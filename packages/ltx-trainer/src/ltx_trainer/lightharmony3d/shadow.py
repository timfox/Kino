"""PBR-guided shadow ratio and 3DGS compositing (Sec. 3.6, Eq. 7–9)."""

from __future__ import annotations

import torch
from torch import Tensor

from ltx_trainer.lightharmony3d.color import linear_to_srgb, srgb_to_linear
from ltx_trainer.lightharmony3d.config import SHADOW_EPS, SHADOW_GAMMA, SHADOW_LAMBDA, SHADOW_MIN_R0, SHADOW_SMIN


def raw_shadow_ratio(r0: Tensor, r1: Tensor, *, eps: float = SHADOW_EPS, min_r0: float = SHADOW_MIN_R0) -> Tensor:
    """Per-channel shadow ratio in linear space (Eq. 7)."""
    r0_lin = srgb_to_linear(r0)
    r1_lin = srgb_to_linear(r1)
    ratio = torch.clamp(r1_lin / (r0_lin + eps), 0.0, 1.0)
    valid = r0_lin > min_r0
    return torch.where(valid, ratio, torch.ones_like(ratio))


def shape_shadow_ratio(
    s: Tensor,
    *,
    gamma: float = SHADOW_GAMMA,
    smin: float = SHADOW_SMIN,
    lam: float = SHADOW_LAMBDA,
) -> Tensor:
    """Eq. (8): contrast + floor + global intensity."""
    s_tilde = torch.clamp(s.pow(gamma), min=smin)
    return 1.0 - lam * (1.0 - s_tilde)


def composite_with_3dgs(
    background: Tensor,
    object_rgb: Tensor,
    object_mask: Tensor,
    shadow_hat: Tensor,
) -> Tensor:
    """Eq. (9): linear composite then sRGB."""
    b_lin = srgb_to_linear(background)
    o_lin = srgb_to_linear(object_rgb)
    m = object_mask.clamp(0, 1)
    if m.dim() == 3:
        m = m.unsqueeze(0)
    if shadow_hat.dim() == 3:
        shadow_hat = shadow_hat.unsqueeze(0)
    shaded_bg = b_lin * shadow_hat
    comp_lin = shaded_bg * (1.0 - m) + o_lin * m
    return linear_to_srgb(comp_lin)
