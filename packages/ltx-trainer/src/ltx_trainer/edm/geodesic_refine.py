"""Geodesic flow refinement with π / π⁻¹ (Sec. 4.2, Eq. 11–12)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ltx_trainer.edm.spherical import cartesian_to_spherical, spherical_to_cartesian


class GeodesicRefiner(nn.Module):
    def __init__(self, ch: int) -> None:
        super().__init__()
        self.refine = nn.Sequential(
            nn.Conv2d(ch * 2 + 1 + 2, ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, 2, 1),
        )

    def forward(
        self,
        feat_a: Tensor,
        feat_b: Tensor,
        s_match: Tensor,
        u_erp: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Refine in ERP space then map back to sphere."""
        u_pred = cartesian_to_spherical(s_match).permute(0, 3, 1, 2)
        u_erp_ch = u_erp.permute(0, 3, 1, 2) if u_erp.shape[-1] == 2 else u_erp
        grid = torch.stack(
            [
                (u_pred[:, 0] / 3.14159).clamp(-1, 1),
                (u_pred[:, 1] / 1.5708).clamp(-1, 1),
            ],
            dim=-1,
        )
        fb_warp = F.grid_sample(feat_b, grid, align_corners=True, mode="bilinear")
        corr = (feat_a * fb_warp).sum(dim=1, keepdim=True)
        delta_u = self.refine(torch.cat([feat_a, fb_warp, corr, u_erp_ch], dim=1))
        u_new = u_erp_ch + 0.1 * delta_u
        s_new = spherical_to_cartesian(u_new.permute(0, 2, 3, 1))
        s_new = s_new / s_new.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        return s_new, u_new
