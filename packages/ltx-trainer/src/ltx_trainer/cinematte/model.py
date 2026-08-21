"""CineMatte full model: Siamese frozen ViT + FBAM + JAFAR + DPT decoder."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

from ltx_trainer.cinematte.backbone import VitBackbone, build_backbone
from ltx_trainer.cinematte.decoder import DPTDecoder
from ltx_trainer.cinematte.fbam import FBAM
from ltx_trainer.cinematte.upsampler import JafarUpsampler


@dataclass
class CineMatteConfig:
    backbone: str = "stub"
    embed_dim: int = 256
    fbam_layers: int = 2
    fbam_heads: int = 8
    upsampler_dim: int = 128
    upsampler_window: int = 0  # 0 = global attention
    freeze_backbone: bool = True


class CineMatte(nn.Module):
    """Cross-attention-conditioned background matting (arXiv:2605.18328)."""

    def __init__(self, cfg: CineMatteConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or CineMatteConfig()
        self.backbone: VitBackbone = build_backbone(self.cfg.backbone, embed_dim=self.cfg.embed_dim)  # type: ignore[assignment]
        dim = int(self.backbone.embed_dim)
        self.fbam = FBAM(dim, num_layers=self.cfg.fbam_layers, num_heads=self.cfg.fbam_heads)
        self.upsampler = JafarUpsampler(dim, self.cfg.upsampler_dim, window_size=self.cfg.upsampler_window)
        self.decoder = DPTDecoder(dim)
        if self.cfg.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    @property
    def embed_dim(self) -> int:
        return int(self.backbone.embed_dim)

    def encode_pair(self, image: Tensor, background: Tensor) -> tuple[Tensor, Tensor]:
        """Siamese frozen encoder on input frame and captured background."""
        fi = self.backbone.forward_tokens(image).to_map()
        fb = self.backbone.forward_tokens(background).to_map()
        return fi, fb

    def forward(self, image: Tensor, background: Tensor) -> Tensor:
        """Predict alpha matte ``[B,1,H,W]`` in ``[0,1]``."""
        h, w = image.shape[-2:]
        img_map, bg_map = self.encode_pair(image, background)
        aligned = self.fbam(img_map, bg_map)
        out_h, out_w = h // 2, w // 2
        hr = self.upsampler(image, img_map, out_h=out_h, out_w=out_w)
        logits = self.decoder(aligned, img_map, hr)
        if logits.shape[-2:] != (h, w):
            logits = torch.nn.functional.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)
        return logits.sigmoid()

    def forward_logits(self, image: Tensor, background: Tensor) -> Tensor:
        """Raw logits before sigmoid (for training losses)."""
        h, w = image.shape[-2:]
        img_map, bg_map = self.encode_pair(image, background)
        aligned = self.fbam(img_map, bg_map)
        out_h, out_w = h // 2, w // 2
        hr = self.upsampler(image, img_map, out_h=out_h, out_w=out_w)
        logits = self.decoder(aligned, img_map, hr)
        if logits.shape[-2:] != (h, w):
            logits = torch.nn.functional.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)
        return logits
