"""Lightweight learnable MEF stub (15-ch in → 3-ch out)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

from ltx_trainer.raim_mef.fusion import select_three_frames, weighted_fusion
from ltx_trainer.raim_mef.metrics import leaderboard_score, lpips_proxy, psnr_from_mse, ssim_proxy


@dataclass
class RaimMefConfig:
    in_frames: int = 5
    use_three_frame: bool = True
    base_ch: int = 32


class RaimMefFusion(nn.Module):
    """CNN fusion stub inspired by WHU-VIP / nunucccb pipelines."""

    def __init__(self, cfg: RaimMefConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or RaimMefConfig()
        in_ch = 9 if self.cfg.use_three_frame else self.cfg.in_frames * 3
        ch = self.cfg.base_ch
        self.encoder = nn.Sequential(
            nn.Conv2d(in_ch, ch, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(ch, ch * 2, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(ch * 2, ch * 2, 3, padding=1),
            nn.GELU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(ch * 2, ch, 4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(ch, 3, 3, padding=1),
        )
        self.refine = nn.Conv2d(6, 3, 1)

    def forward(self, stack: Tensor, ev_stops: tuple[float, ...] | list[float]) -> Tensor:
        if self.cfg.use_three_frame and stack.shape[0] > 3:
            stack_in = select_three_frames(stack, ev_stops)
        else:
            stack_in = stack
        x = torch.cat([stack_in[i] for i in range(stack_in.shape[0])], dim=0).unsqueeze(0)
        h, w = x.shape[-2:]
        feat = self.encoder(x)
        if feat.shape[-2:] != (h, w):
            feat = torch.nn.functional.interpolate(feat, size=(h, w), mode="bilinear", align_corners=False)
        pred = self.decoder(feat).squeeze(0)
        baseline = weighted_fusion(stack, ev_stops)
        if pred.shape != baseline.shape:
            pred = torch.nn.functional.interpolate(
                pred.unsqueeze(0), size=baseline.shape[-2:], mode="bilinear", align_corners=False
            ).squeeze(0)
        return (pred + self.refine(torch.cat([pred, baseline], dim=0))).clamp(0, 1)

    def training_step(self, stack: Tensor, gt: Tensor, ev_stops: tuple[float, ...] | list[float]) -> tuple[Tensor, dict[str, float]]:
        pred = self.forward(stack, ev_stops)
        l1 = torch.nn.functional.l1_loss(pred, gt)
        with torch.no_grad():
            mse = float((pred - gt).pow(2).mean())
        psnr = psnr_from_mse(mse)
        ssim = ssim_proxy(pred, gt)
        lp = lpips_proxy(pred, gt)
        score = leaderboard_score(psnr, ssim, lp)
        loss = l1 + 0.1 * (1.0 - score / 80.0)
        return loss, {
            "loss_l1": float(l1.detach()),
            "psnr": psnr,
            "ssim": ssim,
            "lpips": lp,
            "leaderboard_score": score,
        }
