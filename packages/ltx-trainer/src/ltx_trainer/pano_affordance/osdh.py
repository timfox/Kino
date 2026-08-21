"""Omni-Spherical Densification Head (Sec. III-D, Fig. 3, Eq. 5–7)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class OSDH(nn.Module):
    def __init__(self, top_k: int = 10, temperature: float = 1.0) -> None:
        super().__init__()
        self.top_k = top_k
        self.temperature = temperature
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, f_v: Tensor, a_init: Tensor) -> Tensor:
        """
        f_v: [B, L, D] refined visual tokens
        a_init: [B, C, L] initial affordance maps
        Returns a_refined [B, C, L]
        """
        b, c, l = a_init.shape
        if f_v.shape[1] != l:
            l = min(f_v.shape[1], l)
            f_v = f_v[:, :l]
            a_init = a_init[..., :l]
        f_norm = F.normalize(f_v, dim=-1)
        s = torch.bmm(f_norm, f_norm.transpose(1, 2))  # [B, L, L]
        mu = a_init.mean(dim=(1, 2), keepdim=True)
        sigma = a_init.std(dim=(1, 2), keepdim=True).clamp_min(1e-6)
        conf = torch.sigmoid((a_init - mu) / (sigma * self.temperature))
        a_refined = a_init.clone()
        for bi in range(b):
            for ci in range(c):
                scores = a_init[bi, ci]
                k = min(self.top_k, l)
                seeds = torch.topk(scores, k=k).indices
                prop = torch.zeros(l, device=a_init.device, dtype=a_init.dtype)
                for j in seeds:
                    prop = torch.maximum(prop, s[bi, :, j] * conf[bi, ci, j])
                a_refined[bi, ci] = a_init[bi, ci] + self.alpha * prop
        return a_refined
