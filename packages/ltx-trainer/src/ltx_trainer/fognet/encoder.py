"""CLIP-style stub encoders for FogNet."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class ClipVisualStub(nn.Module):
    """Maps video frames to token embeddings (ViT-B/16 proxy)."""

    def __init__(self, embed_dim: int = 256, patch: int = 16) -> None:
        super().__init__()
        self.patch = patch
        self.proj = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=patch, stride=patch),
            nn.GELU(),
            nn.Conv2d(64, embed_dim, kernel_size=1),
        )
        self.pos = nn.Parameter(torch.randn(1, 64, embed_dim) * 0.02)

    def encode_frames(self, video: Tensor) -> Tensor:
        """``video``: ``(B,T,C,H,W)`` → ``(B,T,D)`` pooled per frame."""
        b, t, c, h, w = video.shape
        flat = video.reshape(b * t, c, h, w)
        feat = self.proj(flat).flatten(2).transpose(1, 2)
        feat = feat + self.pos[:, : feat.shape[1]]
        pooled = feat.mean(dim=1).view(b, t, -1)
        return pooled

    def forward(self, video: Tensor) -> Tensor:
        if video.dim() == 4:
            video = video.unsqueeze(0)
        return self.encode_frames(video)


class ClipTextStub(nn.Module):
    """Class prompt embeddings without external tokenizer."""

    def __init__(self, num_classes: int, embed_dim: int = 256) -> None:
        super().__init__()
        self.embed = nn.Embedding(num_classes, embed_dim)
        nn.init.normal_(self.embed.weight, std=0.02)

    def forward(self, labels: Tensor) -> Tensor:
        return self.embed(labels)
