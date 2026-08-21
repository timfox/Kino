"""Learned DCT-domain quantization matrices Q(k) (Eq. 4)."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from ltx_trainer.seaotter.color_transform import softsign
from ltx_trainer.seaotter.config import SeaotterConfig


def quantize_matrix(raw: Tensor) -> Tensor:
    """Q = 128.5 + 127.5 * softsign(Q_raw) ∈ (1, 256)."""
    return 128.5 + 127.5 * softsign(raw)


class LearnedQuantizationBank(nn.Module):
    """K independent 3×8×8 quantization tensors."""

    def __init__(self, cfg: SeaotterConfig | None = None) -> None:
        super().__init__()
        cfg = cfg or SeaotterConfig()
        self.cfg = cfg
        k, b = cfg.num_rate_points, cfg.dct_block_size
        self.raw = nn.Parameter(torch.randn(k, 3, b, b) * 0.01)

    def forward(self, rate_idx: int | None = None) -> Tensor:
        q = quantize_matrix(self.raw)
        if rate_idx is None:
            return q
        return q[rate_idx]

    def deployment_matrices(self) -> list[Tensor]:
        """Rounded integer Q for JPEG metadata (deployment)."""
        return [q.round().clamp(1, 255).to(torch.int32) for q in self.forward()]
