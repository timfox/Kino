"""Spherical Spatial Alignment Module (Sec. 4.1, Eq. 3–10)."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from ltx_trainer.edm.config import EdmConfig
from ltx_trainer.edm.spherical import pi_inv_erp_grid


class GPKernelMatcher(nn.Module):
    """Exponential cosine kernel GP stub (Eq. 3–4)."""

    def __init__(self, ch: int, tau: float = 5.0, sigma_n: float = 0.1) -> None:
        super().__init__()
        self.tau = tau
        self.sigma_n = sigma_n
        self.pe_conv = nn.Conv2d(3, ch, 1)

    def forward(self, feat_a: Tensor, feat_b: Tensor, grid_cart_b: Tensor) -> Tensor:
        b, c, h, w = feat_b.shape
        chi = torch.cos(self.pe_conv(grid_cart_b.permute(0, 3, 1, 2)))
        fb = feat_b.flatten(2).transpose(1, 2)  # B×HW×C
        fa = feat_a.flatten(2).transpose(1, 2)
        sim = torch.bmm(fa, fb.transpose(1, 2)) / (c**0.5)
        k = torch.exp(self.tau * (sim - 1.0))
        kbb = torch.bmm(k, k.transpose(1, 2)) + self.sigma_n * torch.eye(
            k.shape[-1], device=k.device
        ).unsqueeze(0)
        mu = torch.linalg.solve(kbb, torch.bmm(k, chi.flatten(2).transpose(1, 2)))
        return mu.view(b, h, w, -1).permute(0, 3, 1, 2)


class SSAModule(nn.Module):
    def __init__(self, cfg: EdmConfig) -> None:
        super().__init__()
        self.cfg = cfg
        ch = cfg.feature_ch
        self.gp = GPKernelMatcher(ch, cfg.gp_tau, cfg.gp_sigma_n)
        self.decoder = nn.Sequential(
            nn.Conv2d(ch * 2, ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, 4, 1),
        )

    def forward(self, feat_a: Tensor, feat_b: Tensor) -> tuple[Tensor, Tensor]:
        _, _, h, w = feat_b.shape
        if self.cfg.use_spherical_pe:
            grid = pi_inv_erp_grid(h, w, feat_b.device).unsqueeze(0).expand(feat_b.shape[0], -1, -1, -1)
        else:
            grid = torch.zeros(feat_b.shape[0], h, w, 3, device=feat_b.device)
        mu = self.gp(feat_a, feat_b, grid)
        dec = self.decoder(torch.cat([mu, feat_a], dim=1))
        s_raw = dec[:, :3].permute(0, 2, 3, 1)
        s_match = s_raw / s_raw.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        certainty = torch.sigmoid(dec[:, 3:4])
        return s_match, certainty
