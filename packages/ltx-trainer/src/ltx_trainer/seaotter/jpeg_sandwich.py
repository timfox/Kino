"""End-to-end learned JPEG sandwich training step (Eq. 5)."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from ltx_trainer.seaotter.color_transform import LearnedColorTransform
from ltx_trainer.seaotter.config import SeaotterConfig
from ltx_trainer.seaotter.quantization import LearnedQuantizationBank
from ltx_trainer.seaotter.rate_proxy import rate_proxy_bpp


class JpegSandwich(nn.Module):
    """F → quantize proxy → F^{-1} with shared (F, F^{-1}) and per-rate Q(k)."""

    def __init__(self, cfg: SeaotterConfig | None = None) -> None:
        super().__init__()
        cfg = cfg or SeaotterConfig()
        self.cfg = cfg
        self.color = LearnedColorTransform(cfg)
        self.qbank = LearnedQuantizationBank(cfg)
        self.alpha_calib = nn.Parameter(torch.ones(cfg.num_rate_points))

    def forward_rate(self, rgb: Tensor, rate_idx: int) -> tuple[Tensor, Tensor]:
        """Returns reconstruction and bpp proxy at rate k."""
        y = self.color(rgb, training_noise=self.training)
        q = self.qbank(rate_idx)
        # Pseudo-DCT: downsample; fall back to RGB if decoder output is tiny
        pool_src = y if min(y.shape[-2], y.shape[-1]) >= 8 else rgb
        latent = torch.nn.functional.avg_pool2d(pool_src, 8)
        bpp = rate_proxy_bpp(
            latent,
            q,
            alpha_calib=float(self.alpha_calib[rate_idx].detach()),
            pixels=rgb.shape[-2] * rgb.shape[-1],
        )
        recon = self.color.inverse(y)
        return recon, bpp


def multi_rate_loss(
    rgb: Tensor,
    target: Tensor,
    sandwich: JpegSandwich,
) -> Tensor:
    """Eq. (5): sum_k w_k * (log10 MSE_k + λ_k * bpp_k)."""
    cfg = sandwich.cfg
    total = torch.tensor(0.0, device=rgb.device)
    for k in range(cfg.num_rate_points):
        recon, bpp = sandwich.forward_rate(rgb, k)
        mse = (recon - target).pow(2).mean()
        term = torch.log10(mse.clamp(min=1e-10)) + cfg.lagrange_multipliers[k] * bpp
        total = total + cfg.rate_loss_weights[k] * term
    return total


def sandwich_training_smoke(cfg: SeaotterConfig | None = None) -> dict[str, float]:
    cfg = cfg or SeaotterConfig()
    torch.manual_seed(3940)
    model = JpegSandwich(cfg)
    x = torch.randn(2, 3, 64, 64).clamp(-1, 1)
    loss = multi_rate_loss(x, x, model)
    loss.backward()
    return {
        "multi_rate_loss": float(loss.detach()),
        "q_shape": float(model.qbank.raw.shape[0]),
    }
