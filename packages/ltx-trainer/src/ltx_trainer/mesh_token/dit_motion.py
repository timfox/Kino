"""Per-latent-frame motion cross-attention on frozen Wan DiT (Eq. 5)."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class MotionCrossAttention(nn.Module):
    """Zv + CA(Zv, Zm) with zero-init output projection (§3.3)."""

    def __init__(self, dim: int, num_heads: int = 8) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.out_proj = nn.Linear(dim, dim)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, video_tokens: Tensor, motion_tokens: Tensor) -> Tensor:
        # video_tokens: B, M, D; motion_tokens: B, N, D
        out, _ = self.attn(video_tokens, motion_tokens, motion_tokens)
        return video_tokens + self.out_proj(out)


class MotionConditionedDiTBlock(nn.Module):
    """Text CA stub + motion CA per latent frame."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.text_ca = MotionCrossAttention(dim)
        self.motion_ca = MotionCrossAttention(dim)

    def forward(
        self,
        zv: Tensor,
        zm: Tensor,
        text_tokens: Tensor,
    ) -> Tensor:
        zv = self.text_ca(zv, text_tokens)
        return self.motion_ca(zv, zm)


def dit_motion_smoke(dim: int = 512, tokens: int = 64, motion_n: int = 54) -> dict[str, list[int]]:
    block = MotionConditionedDiTBlock(dim)
    zv = torch.randn(1, tokens, dim)
    zm = torch.randn(1, motion_n, dim)
    text = torch.randn(1, 32, dim)
    out = block(zv, zm, text)
    return {"in": list(zv.shape), "out": list(out.shape)}
