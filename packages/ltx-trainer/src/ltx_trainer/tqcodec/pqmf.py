"""PQMF subband modeling (Sec. 3.2, Fig. 1)."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from ltx_trainer.tqcodec.config import CORE_SUBBANDS, HIGH_SUBBAND_LATENT_DIM, PQMF_SUBBANDS


class PQMFAnalysis(nn.Module):
    """Decompose waveform into subbands (stub filterbank)."""

    def __init__(self, subbands: int = PQMF_SUBBANDS) -> None:
        super().__init__()
        self.subbands = subbands
        self.filters = nn.Conv1d(1, subbands, kernel_size=63, stride=subbands, padding=31, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.filters(x)


class PQMFSynthesis(nn.Module):
    def __init__(self, subbands: int = PQMF_SUBBANDS) -> None:
        super().__init__()
        self.subbands = subbands
        self.filters = nn.ConvTranspose1d(subbands, 1, kernel_size=63, stride=subbands, padding=31, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.filters(x)


class SubbandEncoder(nn.Module):
    """Core vs high-band imbalanced encoders (Sec. 3.2)."""

    def __init__(self, core_dim: int = 128, high_dim: int = 6) -> None:
        super().__init__()
        self.core_dim = core_dim
        self.high_dim = high_dim

    def split_bands(self, subbands: Tensor) -> tuple[Tensor, Tensor]:
        # subbands: (B, 16, T')
        core = subbands[:, :CORE_SUBBANDS]
        high = subbands[:, CORE_SUBBANDS:]
        return core, high

    def concat_latents(self, core_z: Tensor, high_z: Tensor) -> Tensor:
        return torch.cat([core_z, high_z], dim=1)
