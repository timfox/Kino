"""RSimVQ quantizer stub (Sec. 3.1.2)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class SimVQLayer(nn.Module):
    """SimVQ: frozen codebook + linear projection (Sec. 3.1.2)."""

    def __init__(self, dim: int, codebook_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(dim, dim, bias=False)
        codebook = torch.randn(codebook_size, dim)
        codebook = F.normalize(codebook, dim=-1)
        self.register_buffer("codebook", codebook)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        # x: (B, T, D)
        z = self.proj(x)
        z_n = F.normalize(z, dim=-1)
        cb = F.normalize(self.codebook, dim=-1)
        indices = torch.argmax(z_n @ cb.t(), dim=-1)
        quantized = cb[indices]
        return quantized, indices


class RSimVQ(nn.Module):
    """Residual SimVQ stack (Sec. 3.1.2)."""

    def __init__(self, dim: int, codebook_size: int, num_codebooks: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(SimVQLayer(dim, codebook_size) for _ in range(num_codebooks))

    def forward(self, x: Tensor) -> tuple[Tensor, list[Tensor]]:
        residual = x
        codes: list[Tensor] = []
        out = torch.zeros_like(x)
        for layer in self.layers:
            q, idx = layer(residual)
            codes.append(idx)
            out = out + q
            residual = residual - q.detach()
        return out, codes
