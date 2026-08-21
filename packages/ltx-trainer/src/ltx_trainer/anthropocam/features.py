"""Proxy multi-scale feature extractor for CPU smoke (VGG-16 layer names preserved)."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from ltx_trainer.anthropocam.config import CONTENT_LAYER, DEFAULT_STYLE_LAYERS


class ProxyVGGFeatures(nn.Module):
    """Lightweight conv tower; each block tagged with a VGG-style layer name."""

    def __init__(
        self,
        layer_names: tuple[str, ...] = DEFAULT_STYLE_LAYERS,
        *,
        content_layer: str = CONTENT_LAYER,
    ) -> None:
        super().__init__()
        ordered = list(dict.fromkeys((*layer_names, content_layer)))
        self.layer_names = tuple(ordered)
        channels = [3] + [16 * (2**min(i, 2)) for i in range(len(ordered))]
        blocks: list[nn.Module] = []
        for i in range(len(ordered)):
            blocks.append(nn.Conv2d(channels[i], channels[i + 1], 3, padding=1))
            blocks.append(nn.ReLU(inplace=True))
        self.tower = nn.Sequential(*blocks)

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        feats: dict[str, Tensor] = {}
        h = x
        for i, name in enumerate(self.layer_names):
            h = self.tower[i * 2 + 1](self.tower[i * 2](h))
            b, c, ht, wd = h.shape
            feats[name] = h.reshape(b, c, ht * wd)
        return feats
