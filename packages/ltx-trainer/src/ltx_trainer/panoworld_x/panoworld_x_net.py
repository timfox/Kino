"""PanoWorld-X DiT stack stub (CogVideoX-5B-I2V fine-tune surrogate)."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from ltx_trainer.panoworld_x.config import PanoWorldXConfig
from ltx_trainer.panoworld_x.dit_block import ExplorableSphereAwareDiTBlock


class PanoWorldXStub(nn.Module):
    def __init__(self, cfg: PanoWorldXConfig | None = None, num_blocks: int = 2) -> None:
        super().__init__()
        self.cfg = cfg or PanoWorldXConfig()
        gh = self.cfg.height // self.cfg.patch_size
        gw = self.cfg.width // self.cfg.patch_size
        self.num_tokens = gh * gw
        self.patch_embed = nn.Conv2d(3, self.cfg.hidden_dim, kernel_size=self.cfg.patch_size, stride=self.cfg.patch_size)
        self.blocks = nn.ModuleList(
            [ExplorableSphereAwareDiTBlock(self.cfg) for _ in range(num_blocks)]
        )
        self.head = nn.Linear(self.cfg.hidden_dim, 3 * self.cfg.patch_size**2)

    def patchify(self, frame: Tensor) -> Tensor:
        """Single ERP frame [B, 3, H, W] → tokens [B, N, D]."""
        feat = self.patch_embed(frame)
        return feat.flatten(2).transpose(1, 2)

    def unpatchify(self, tokens: Tensor) -> Tensor:
        b, n, _ = tokens.shape
        gh = self.cfg.height // self.cfg.patch_size
        gw = self.cfg.width // self.cfg.patch_size
        pix = self.head(tokens)
        return pix.transpose(1, 2).reshape(b, 3, gh, gw, self.cfg.patch_size, self.cfg.patch_size).permute(
            0, 1, 2, 4, 3, 5
        ).reshape(b, 3, self.cfg.height, self.cfg.width)

    def forward(
        self,
        frame: Tensor,
        route: Tensor | None = None,
    ) -> dict[str, Tensor]:
        tokens = self.patchify(frame)
        for block in self.blocks:
            tokens = block(tokens, route)
        recon = self.unpatchify(tokens)
        return {"reconstruction": recon, "tokens": tokens}
