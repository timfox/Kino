"""GateFuse module — GRU-style fusion of Fsp and Feq (Sec. 3.3.2)."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class GateFuse(nn.Module):
    """Reset gate r on Fsp, forget gate z on Feq; fused = r*Fsp + z*Feq."""

    def __init__(self, ch: int) -> None:
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(ch * 2, ch * 2),
            nn.SiLU(),
            nn.Linear(ch * 2, ch * 2),
        )

    def forward(self, fsp: Tensor, feq: Tensor) -> Tensor:
        if fsp.shape != feq.shape:
            feq = torch.nn.functional.interpolate(
                feq.transpose(1, 2), size=fsp.shape[1], mode="linear", align_corners=False
            ).transpose(1, 2)
        cat = torch.cat([fsp, feq], dim=-1)
        gates = torch.sigmoid(self.gate(cat))
        r, z = gates.chunk(2, dim=-1)
        return r * fsp + z * feq


class BiFuseStub(nn.Module):
    def forward(self, fsp: Tensor, feq: Tensor) -> Tensor:
        m = torch.sigmoid(feq)
        return fsp + m * feq


class UniFuseStub(nn.Module):
    def forward(self, fsp: Tensor, feq: Tensor) -> Tensor:
        return fsp + feq
