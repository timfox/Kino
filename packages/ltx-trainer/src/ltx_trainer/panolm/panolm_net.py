"""PanoLM vision tower stub with PHA blocks (Fig. 4)."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from ltx_trainer.panolm.config import PanoLMConfig
from ltx_trainer.panolm.pha import PanoramicHybridBlock
from ltx_trainer.panolm.psa import SimplifiedSparseAttention


class LoRAAttentionStub(nn.Module):
    def __init__(self, dim: int, rank: int) -> None:
        super().__init__()
        self.lora_a = nn.Linear(dim, rank, bias=False)
        self.lora_b = nn.Linear(rank, dim, bias=False)
        nn.init.zeros_(self.lora_b.weight)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.lora_b(self.lora_a(x))


class PanoLMVisionStub(nn.Module):
    """ERP patch encoder + PHA replacement blocks."""

    def __init__(self, cfg: PanoLMConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or PanoLMConfig()
        d = self.cfg.embed_dim
        self.patch = nn.Conv2d(3, d, kernel_size=14, stride=14)
        self.lora = LoRAAttentionStub(d, self.cfg.lora_rank)
        top_k = min(self.cfg.psa_top_k, 64)
        self.pha = PanoramicHybridBlock(d, window_size=self.cfg.window_size, top_k=top_k)
        self.ssa = SimplifiedSparseAttention(d, top_k=top_k)
        self.proj = nn.Linear(d, d)

    def forward(self, image: Tensor) -> Tensor:
        x = self.patch(image)
        b, d, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        tokens = self.lora(tokens)
        tokens = self.pha(tokens)
        return self.proj(tokens)


class PanoLMStub(nn.Module):
    """Vision + text pooling stub for VQA heatmap / answer head."""

    def __init__(self, cfg: PanoLMConfig | None = None, vocab_size: int = 32000) -> None:
        super().__init__()
        self.cfg = cfg or PanoLMConfig()
        d = self.cfg.embed_dim
        self.vision = PanoLMVisionStub(self.cfg)
        self.text_embed = nn.Embedding(vocab_size, d)
        self.answer_head = nn.Linear(d, vocab_size)

    def forward(self, image: Tensor, input_ids: Tensor) -> dict[str, Tensor]:
        vision_tokens = self.vision(image)
        v = vision_tokens.mean(dim=1)
        t = self.text_embed(input_ids).mean(dim=1)
        fused = v + t
        logits = self.answer_head(fused)
        return {"vision_tokens": vision_tokens, "logits": logits}
