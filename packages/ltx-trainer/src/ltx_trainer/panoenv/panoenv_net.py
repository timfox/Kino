"""PanoEnv-RL policy stub (Qwen2.5-VL + LoRA decoder)."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from ltx_trainer.panoenv.config import PanoEnvConfig


class LoRADecoderStub(nn.Module):
    def __init__(self, dim: int, rank: int) -> None:
        super().__init__()
        self.down = nn.Linear(dim, rank, bias=False)
        self.up = nn.Linear(rank, dim, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.up(self.down(x))


class PanoEnvVisionStub(nn.Module):
    def __init__(self, cfg: PanoEnvConfig) -> None:
        super().__init__()
        d = cfg.embed_dim
        self.patch = nn.Conv2d(3, d, kernel_size=8, stride=8)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, image: Tensor) -> Tensor:
        x = self.patch(image).flatten(2).transpose(1, 2)
        return self.pool(x.transpose(1, 2)).squeeze(-1)


class PanoEnvRLStub(nn.Module):
    """Frozen-vision + LoRA language stub for GRPO smoke."""

    def __init__(self, cfg: PanoEnvConfig | None = None, vocab_size: int = 4096) -> None:
        super().__init__()
        self.cfg = cfg or PanoEnvConfig()
        d = self.cfg.embed_dim
        self.vision = PanoEnvVisionStub(self.cfg)
        self.prompt_embed = nn.Embedding(vocab_size, d)
        self.lora = LoRADecoderStub(d, self.cfg.lora_rank)
        self.logit_head = nn.Linear(d, vocab_size)

    def forward(self, image: Tensor, prompt_ids: Tensor) -> dict[str, Tensor]:
        v = self.vision(image)
        t = self.prompt_embed(prompt_ids).mean(dim=1)
        h = self.lora(v + t)
        logits = self.logit_head(h)
        log_probs = torch.log_softmax(logits, dim=-1)
        return {"logits": logits, "log_probs": log_probs, "hidden": h}
