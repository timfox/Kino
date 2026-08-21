"""MultiPanFusion dual-branch + EPPA stub (Sec. 3.2)."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from ltx_trainer.sdt2i.config import Sdt2iConfig
from ltx_trainer.sdt2i.multidiffusion import merge_latents


class EPPAStub(nn.Module):
    """Equirectangular–perspective projection attention placeholder."""

    def __init__(self, ch: int) -> None:
        super().__init__()
        self.attn = nn.Conv2d(ch * 2, ch, 1)

    def forward(self, pano: Tensor, persp: Tensor, *, fg_mask: Tensor | None = None) -> Tensor:
        if fg_mask is not None and fg_mask.sum() > 0:
            return pano  # disable EPPA on foreground during bootstrap
        fused = self.attn(torch.cat([pano, persp], dim=1))
        return pano + 0.1 * fused


class MultiPanFusionStub(nn.Module):
    def __init__(self, cfg: Sdt2iConfig) -> None:
        super().__init__()
        self.cfg = cfg
        ch = 16
        self.pano_enc = nn.Conv2d(cfg.latent_ch, ch, 3, padding=1)
        self.pers_enc = nn.Conv2d(cfg.latent_ch, ch, 3, padding=1)
        self.eppa = EPPAStub(ch)
        self.dec = nn.Conv2d(ch, cfg.latent_ch, 3, padding=1)
        self.bg_color = nn.Parameter(torch.zeros(3))

    def _md_paths(
        self,
        latent: Tensor,
        masks: list[Tensor],
        branch: str,
    ) -> Tensor:
        paths = []
        for m in masks:
            noise = torch.randn_like(latent) * 0.01
            paths.append(latent * m + noise * (1 - m))
        if branch in ("pano", "both"):
            return merge_latents(paths, masks)
        return paths[0] if paths else latent

    def forward(
        self,
        latent: Tensor,
        masks: list[Tensor],
        *,
        bootstrap_step: int = 0,
    ) -> Tensor:
        pano = self.pano_enc(latent)
        pers = self.pers_enc(latent)
        fg = masks[0] if masks else None
        use_fg_eppa = self.cfg.fg_eppa and bootstrap_step >= self.cfg.bootstrap_steps
        pano = self.eppa(pano, pers, fg_mask=None if use_fg_eppa else fg)
        if self.cfg.md_branch == "pers":
            out_lat = self._md_paths(latent, masks, "pers")
        elif self.cfg.md_branch == "both":
            lp = self._md_paths(latent, masks, "pano")
            out_lat = lp
        else:
            out_lat = self._md_paths(latent, masks, "pano")
        return self.dec(pano) + 0.5 * out_lat
