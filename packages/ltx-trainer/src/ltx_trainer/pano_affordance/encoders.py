"""Dual-encoder stubs with LoRA (Sec. III-B)."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from ltx_trainer.pano_affordance.config import PanoAffordanceConfig


class LoRALinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, rank: int) -> None:
        super().__init__()
        self.base = nn.Linear(in_features, out_features)
        self.lora_a = nn.Linear(in_features, rank, bias=False)
        self.lora_b = nn.Linear(rank, out_features, bias=False)
        nn.init.zeros_(self.lora_b.weight)

    def forward(self, x: Tensor) -> Tensor:
        return self.base(x) + self.lora_b(self.lora_a(x))


class VisionEncoderStub(nn.Module):
    """DINOv2-style patch encoder stub."""

    def __init__(self, cfg: PanoAffordanceConfig) -> None:
        super().__init__()
        d = cfg.embed_dim
        self.patch = nn.Conv2d(3, d, kernel_size=14, stride=14)
        self.lora_attn = LoRALinear(d, d, cfg.lora_rank)
        self.norm = nn.LayerNorm(d)

    def forward(self, image: Tensor) -> Tensor:
        x = self.patch(image)
        b, d, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        tokens = self.norm(tokens + self.lora_attn(tokens))
        return tokens


class TextEncoderStub(nn.Module):
    """CLIP + CoOp-style prompt learner stub."""

    def __init__(self, cfg: PanoAffordanceConfig) -> None:
        super().__init__()
        d = cfg.embed_dim
        self.prompt = nn.Parameter(torch.randn(1, 4, d) * 0.02)
        self.class_embed = nn.Embedding(cfg.num_classes, d)

    def forward(self, class_ids: Tensor | None = None, *, num_classes: int | None = None) -> Tensor:
        c = num_classes or self.class_embed.num_embeddings
        idx = torch.arange(c, device=self.class_embed.weight.device)
        base = self.class_embed(idx).unsqueeze(0)
        prompt = self.prompt.mean(dim=1, keepdim=True)
        return (base + prompt).expand(1, -1, -1)
