"""LF-Diff model — LPR-conditioned diffusion HDR reconstruction."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

from ltx_trainer.lf_diff.lpenet_stub import DHRStub, LPENetStub, encode_bracket_stack
from ltx_trainer.lf_diff.tonemap import lpr_l1, tonemap_l1


@dataclass
class LfDiffConfig:
    bracket_count: int = 3
    latent_ch: int = 32
    denoise_steps: int = 10


class LfDiff(nn.Module):
    def __init__(self, cfg: LfDiffConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or LfDiffConfig()
        in_ch = 3 * self.cfg.bracket_count
        self.lpenet = LPENetStub(in_ch=in_ch, latent_ch=self.cfg.latent_ch)
        self.dhr = DHRStub(latent_ch=self.cfg.latent_ch)
        self.noise_proj = nn.Conv2d(self.cfg.latent_ch, self.cfg.latent_ch, 1)

    def encode_lpr(self, brackets: list[Tensor]) -> Tensor:
        return encode_bracket_stack(self.lpenet, brackets)

    def decode_hdr(self, lpr: Tensor, z: Tensor | None = None) -> Tensor:
        if z is None:
            z = torch.zeros_like(lpr)
        return self.dhr(lpr, self.noise_proj(z))

    @torch.no_grad()
    def infer(self, brackets: list[Tensor], *, steps: int | None = None) -> Tensor:
        self.eval()
        lpr = self.encode_lpr(brackets)
        n = steps or self.cfg.denoise_steps
        z = torch.randn_like(lpr) * 0.05
        for _ in range(n):
            z = 0.92 * z + 0.08 * torch.randn_like(lpr)
        out = self.decode_hdr(lpr, z)
        return out.squeeze(0) if out.dim() == 4 and brackets and brackets[0].dim() == 3 else out

    def training_step(self, brackets: list[Tensor], hdr_gt: Tensor) -> tuple[Tensor, dict[str, float]]:
        lpr = self.encode_lpr(brackets)
        z = torch.randn_like(lpr)
        pred = self.decode_hdr(lpr, z)
        if pred.dim() == 4 and hdr_gt.dim() == 3:
            pred = pred.squeeze(0)
        loss_tm = tonemap_l1(pred, hdr_gt)
        with torch.no_grad():
            lpr_tgt = self.encode_lpr([hdr_gt.clamp(0, 1)] * len(brackets))
        loss_lpr = lpr_l1(lpr, lpr_tgt)
        loss = loss_tm + 0.5 * loss_lpr + 0.01 * pred.abs().mean()
        return loss, {
            "loss_tonemap": float(loss_tm.detach()),
            "loss_lpr": float(loss_lpr.detach()),
        }
