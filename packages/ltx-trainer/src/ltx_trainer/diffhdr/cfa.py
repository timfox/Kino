"""Context-focused cross-attention (Sec. 3.4, Eq. 7–8)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ContextFocusedAttention(nn.Module):
    """
    Mask-guided routing of cross-attention outputs at inference.

    r = r_base + α_over M_over ⊙ (r_over - r_base) + α_under M_under ⊙ (r_under - r_base)
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.null = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.over = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.under = nn.Parameter(torch.randn(1, 1, dim) * 0.02)

    def _ca(self, x: Tensor, ctx: Tensor) -> Tensor:
        q = self.q(x)
        k = self.k(ctx)
        v = self.v(ctx)
        attn = torch.softmax(q @ k.transpose(-2, -1) / (x.shape[-1] ** 0.5), dim=-1)
        return attn @ v

    def forward(
        self,
        x: Tensor,
        mask_over: Tensor,
        mask_under: Tensor,
        *,
        alpha_over: float = 1.0,
        alpha_under: float = 1.0,
    ) -> Tensor:
        b, n, d = x.shape
        r_base = self._ca(x, self.null.expand(b, -1, -1))
        r_over = self._ca(x, self.over.expand(b, -1, -1))
        r_under = self._ca(x, self.under.expand(b, -1, -1))
        m_over = self._resize_mask(mask_over, n)
        m_under = self._resize_mask(mask_under, n)
        return (
            r_base
            + alpha_over * m_over * (r_over - r_base)
            + alpha_under * m_under * (r_under - r_base)
        )

    @staticmethod
    def _resize_mask(mask: Tensor, n: int) -> Tensor:
        if mask.dim() == 3:
            mask = mask.unsqueeze(0)
        side = int(n**0.5)
        if side * side == n:
            m = F.interpolate(mask, size=(side, side), mode="bilinear", align_corners=False)
            return m.flatten(2).transpose(1, 2)
        return F.adaptive_avg_pool1d(mask.flatten(2), n).transpose(1, 2)
