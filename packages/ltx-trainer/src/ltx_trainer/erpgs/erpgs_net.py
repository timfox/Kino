"""ErpGS omnidirectional Gaussian splatting stub (Sec. 3)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ltx_trainer.erpgs.config import ErpGSConfig
from ltx_trainer.erpgs.depth_normal import depth_to_normal_erp
from ltx_trainer.erpgs.erp_projection import distortion_weight_map
from ltx_trainer.erpgs.losses import total_loss


class ErpGSStub(nn.Module):
    """Learnable Gaussian parameters + lightweight ERP rasterizer surrogate."""

    def __init__(self, cfg: ErpGSConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or ErpGSConfig()
        ng = self.cfg.num_gaussians
        d = self.cfg.feature_dim
        self.positions = nn.Parameter(torch.randn(ng, 3) * 0.1)
        self.scales = nn.Parameter(torch.ones(ng, 3) * 0.05)
        self.opacity = nn.Parameter(torch.zeros(ng))
        self.features = nn.Parameter(torch.randn(ng, d) * 0.01)
        self.decoder = nn.Sequential(
            nn.Linear(d, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 3),
            nn.Sigmoid(),
        )
        self.depth_head = nn.Conv2d(3, 1, 3, padding=1)

    def rasterize_rgb(self, h: int, w: int) -> Tensor:
        """Alpha-compositing surrogate: splat features to ERP grid (Eq. 5–6 stub)."""
        device = self.positions.device
        grid_y = torch.linspace(-1, 1, h, device=device)
        grid_x = torch.linspace(-1, 1, w, device=device)
        gy, gx = torch.meshgrid(grid_y, grid_x, indexing="ij")
        pts = torch.stack([gx, gy], dim=-1).view(-1, 2)

        colors = []
        for i in range(self.cfg.num_gaussians):
            c = self.decoder(self.features[i])
            dist = (pts - torch.tanh(self.positions[i, :2])).pow(2).sum(-1)
            alpha = torch.sigmoid(self.opacity[i]) * torch.exp(-dist / self.scales[i, 0].clamp_min(0.01))
            colors.append(alpha.unsqueeze(-1) * c)
        comp = torch.stack(colors, dim=0).sum(0)
        rgb = comp.view(h, w, 3).permute(2, 0, 1).unsqueeze(0)
        return rgb.clamp(0, 1)

    def forward(
        self,
        gt_rgb: Tensor,
        *,
        mask: Tensor | None = None,
        iteration: int = 15_000,
    ) -> dict[str, Tensor]:
        h, w = gt_rgb.shape[-2:]
        pred = self.rasterize_rgb(h, w)
        if pred.shape[-2:] != gt_rgb.shape[-2:]:
            pred = F.interpolate(pred, size=gt_rgb.shape[-2:], mode="bilinear", align_corners=False)
        depth = F.softplus(self.depth_head(pred)) + 0.1
        n_depth = depth_to_normal_erp(depth)
        n_render = depth_to_normal_erp(depth.detach() + 0.02 * torch.randn_like(depth))
        weight = distortion_weight_map(h, w, device=gt_rgb.device)
        losses = total_loss(
            pred,
            gt_rgb,
            n_render,
            n_depth,
            self.scales,
            weight,
            mask,
            self.cfg,
            iteration=iteration,
        )
        return {"pred_rgb": pred, "depth": depth, "weight": weight, **losses}
