"""Local-to-global quality aggregation: HPA + patch attention (Sec. 3.4)."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from ltx_trainer.vuboiqa.pdff import DeformConvStub


class HPAStub(nn.Module):
    """Hierarchical pooling attention (Eq. 10–15)."""

    def __init__(self, ch: int, groups: int = 4) -> None:
        super().__init__()
        self.pre = nn.Sequential(DeformConvStub(ch), nn.Conv2d(ch, ch, 3, padding=1))
        self.local = nn.Conv2d(ch, ch, 3, padding=1, groups=groups)
        self.groups = groups

    def forward(self, x: Tensor) -> Tensor:
        a = self.pre(x)
        w1 = torch.softmax(a.mean(dim=(2, 3)), dim=-1)
        w2 = torch.softmax(self.local(a).mean(dim=(2, 3)), dim=-1)
        return a * (1.0 + 0.1 * w1.unsqueeze(-1).unsqueeze(-1) + 0.1 * w2.unsqueeze(-1).unsqueeze(-1))


class PatchAttention(nn.Module):
    """Multi-head self-attention over K patch tokens (Eq. 17–19)."""

    def __init__(self, dim: int, num_heads: int = 8) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.ff = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )
        self.norm = nn.LayerNorm(dim)
        self.pos = nn.Parameter(torch.randn(1, 32, dim) * 0.02)

    def forward(self, tokens: Tensor) -> Tensor:
        # tokens: B×K×D
        k = tokens.shape[1]
        x = tokens + self.pos[:, :k, :]
        y, _ = self.attn(x, x, x)
        x = self.norm(x + y)
        return self.norm(x + self.ff(x))


class LGQAModule(nn.Module):
    def __init__(self, ch: int, embed_dim: int, num_heads: int) -> None:
        super().__init__()
        self.hpa = HPAStub(ch)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.embed = nn.Linear(ch, embed_dim)
        self.pa = PatchAttention(embed_dim, num_heads)
        self.score = nn.Linear(embed_dim, 1)

    def forward(self, patch_feats: Tensor, global_feat: Tensor) -> Tensor:
        # patch_feats: B×K×C×h×w
        b, k, c, h, w = patch_feats.shape
        vecs = []
        for i in range(k):
            f = self.hpa(patch_feats[:, i])
            v = self.pool(f).flatten(1) + self.pool(global_feat).flatten(1)
            vecs.append(self.embed(v))
        tokens = torch.stack(vecs, dim=1)
        out = self.pa(tokens)
        # Eq. 21: mean over K then FC
        return self.score(out.mean(dim=1)).squeeze(-1)
