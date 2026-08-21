"""FAS, ME, and CSA modules (Sec. 4.1–4.3)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def _cosine_sim(a: Tensor, b: Tensor) -> Tensor:
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)
    return (a * b).sum(dim=-1)


class FogAwareSelection(nn.Module):
    """Global self-attention over concatenated clean/foggy tokens (Eq. 1–2)."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.query = nn.Linear(dim, dim, bias=False)

    def forward(self, vc: Tensor, vf: Tensor) -> tuple[Tensor, Tensor]:
        """``vc``, ``vf``: ``(B,T,D)`` → selected ``(B,T,D)`` each."""
        vcat = torch.cat([vc, vf], dim=1)
        global_q = self.query(vcat.mean(dim=1, keepdim=True))
        sim = _cosine_sim(global_q.expand_as(vcat), vcat)
        weights = F.softmax(sim, dim=1).unsqueeze(-1)
        vatt = (weights * vcat).sum(dim=1, keepdim=True).expand_as(vcat)
        t = vc.shape[1]
        return vatt[:, :t], vatt[:, t:]


class MutualEnhancement(nn.Module):
    """Bidirectional cross-attention (Eq. 3–5)."""

    def __init__(self, dim: int, heads: int = 4) -> None:
        super().__init__()
        self.cross_cf = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.cross_fc = nn.MultiheadAttention(dim, heads, batch_first=True)

    def forward(self, va_c: Tensor, va_f: Tensor) -> tuple[Tensor, Tensor]:
        vd_f, _ = self.cross_cf(va_c, va_f, va_f)
        vd_f = vd_f + va_f
        vd_c, _ = self.cross_fc(va_f, va_c, va_c)
        vd_c = vd_c + va_c
        return vd_c, vd_f


class CrossStreamAlignment(nn.Module):
    """Frame-level similarity matrix for temporal contrastive loss (Eq. 6–7)."""

    @staticmethod
    def similarity(vd_f: Tensor, vd_c: Tensor) -> Tensor:
        """Return ``(B,T,T)`` cosine similarity."""
        f = F.normalize(vd_f, dim=-1)
        c = F.normalize(vd_c, dim=-1)
        return torch.matmul(f, c.transpose(1, 2))

    @staticmethod
    def temporal_loss(sc: Tensor) -> Tensor:
        """Diagonal contrastive loss LTemp."""
        b, t, _ = sc.shape
        losses = []
        for i in range(t):
            row = sc[:, i]
            pos = row[:, i]
            denom = torch.logsumexp(row, dim=1)
            losses.append(-(pos - denom))
        return torch.stack(losses, dim=1).mean()
