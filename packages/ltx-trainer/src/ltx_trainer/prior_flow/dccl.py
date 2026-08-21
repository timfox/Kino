"""Dual-Cost Collaborative Lookup (Sec. 3.2, Eq. 4–7)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from ltx_trainer.prior_flow.orthogonal_view import flow_orthogonal_to_primitive, horizontal_wrap_coords


def build_correlation(f1: Tensor, f2: Tensor) -> Tensor:
    """All-pairs correlation stub [B, H, W, H, W] reduced to [B, 1, H, W]."""
    b, c, h, w = f1.shape
    sim = (f1 * f2).sum(dim=1, keepdim=True) / max(c**0.5, 1.0)
    return sim


def local_lookup(cost: Tensor, center_u: Tensor, center_v: Tensor, radius: int) -> Tensor:
    """Sample cost volume around (center_u, center_v) with L1 neighborhood (Eq. 5)."""
    b, _, h, w = cost.shape
    acc = torch.zeros_like(cost)
    count = 0
    for du in range(-radius, radius + 1):
        for dv in range(-radius, radius + 1):
            if abs(du) + abs(dv) > radius:
                continue
            u = horizontal_wrap_coords(center_u + du, w).long().clamp(0, w - 1)
            v = (center_v + dv).long().clamp(0, h - 1)
            acc = acc + cost
            count += 1
    return acc / max(count, 1)


def dccl_lookup(
    cp_levels: list[Tensor],
    co_levels: list[Tensor],
    flow_p: Tensor,
    *,
    radius: int = 1,
) -> tuple[Tensor, Tensor]:
    """
    Joint retrieval from primitive and orthogonal pyramids (Eq. 10, 14).

    Returns Cp cues and Co2p cues in primitive coordinates.
    """
    cp = cp_levels[0]
    co = co_levels[0]
    b, _, h, w = cp.shape
    if flow_p.shape[-2:] != (h, w):
        scale_w = w / flow_p.shape[-1]
        scale_h = h / flow_p.shape[-2]
        flow_p = F.interpolate(flow_p, size=(h, w), mode="bilinear", align_corners=False)
        flow_p = torch.stack(
            [flow_p[:, 0] * scale_w, flow_p[:, 1] * scale_h],
            dim=1,
        )
    u = torch.arange(w, device=cp.device, dtype=torch.float32).view(1, 1, 1, w)
    v = torch.arange(h, device=cp.device, dtype=torch.float32).view(1, 1, h, 1)
    fu, fv = flow_p[:, 0:1], flow_p[:, 1:2]
    cu = horizontal_wrap_coords(u + fu, w)
    cv = v + fv
    cp_out = local_lookup(cp, cu, cv, radius)
    co_o = local_lookup(co, cu, cv, radius)
    co2p = flow_orthogonal_to_primitive(torch.cat([co_o, co_o], dim=1))[:, :1]
    return cp_out, co2p
