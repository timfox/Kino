"""MultiStitchDiffusion stub (Sec. 3.1)."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from ltx_trainer.sdt2i.config import Sdt2iConfig, PANORAMA_TRIGGER
from ltx_trainer.sdt2i.multidiffusion import bootstrap_mask_focus, cyclic_pad_mask, merge_latents
from ltx_trainer.sdt2i.stitch import stitch_latents


class MultiStitchDiffusionStub(nn.Module):
    def __init__(self, cfg: Sdt2iConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or Sdt2iConfig()
        self.unet = nn.Sequential(
            nn.Conv2d(4, 16, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(16, 4, 3, padding=1),
        )
        self.trigger = PANORAMA_TRIGGER

    def denoise_path(self, latent: Tensor, mask: Tensor, local_prompt_emb: float) -> Tensor:
        m = cyclic_pad_mask(mask) if mask.shape[-1] == latent.shape[-1] else mask
        if m.shape[-2:] != latent.shape[-2:]:
            m = torch.nn.functional.interpolate(m, size=latent.shape[-2:], mode="nearest")
        return self.unet(latent) * m + local_prompt_emb * 0.01

    def forward(
        self,
        latent: Tensor,
        masks: list[Tensor],
        *,
        step: int = 0,
    ) -> Tensor:
        paths = []
        for i, m in enumerate(masks):
            focus = bootstrap_mask_focus(step, self.cfg.bootstrap_steps, m)
            paths.append(self.denoise_path(latent, focus, float(i + 1)))
        merged = merge_latents(paths, masks)
        if self.cfg.use_stitch:
            merged = stitch_latents(merged)
        return merged
