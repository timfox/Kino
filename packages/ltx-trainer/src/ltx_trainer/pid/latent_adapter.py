"""Sigma-aware latent adapter (Eq. 4–6, Sec. 3.2)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ltx_trainer.pid.config import PiDConfig


class _ResBlock512(nn.Module):
    """GN4 → SiLU → Conv → GN4 → SiLU → Conv + skip (Appendix 4.2)."""

    def __init__(self, channels: int = 512) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(4, channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(4, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        h = F.silu(self.norm1(x))
        h = self.conv1(h)
        h = F.silu(self.norm2(h))
        h = self.conv2(h)
        return x + h


class LatentProjectionAdapter(nn.Module):
    """Resize latent → ResBlocks → per-block token features (Eq. 4)."""

    def __init__(
        self,
        latent_channels: int = 16,
        out_channels: int = 512,
        *,
        cfg: PiDConfig | None = None,
    ) -> None:
        super().__init__()
        cfg = cfg or PiDConfig()
        self.cfg = cfg
        c = cfg.adapter_channels
        self.stem = nn.Sequential(
            nn.Conv2d(latent_channels, c, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(c, c, 3, padding=1),
        )
        self.blocks = nn.ModuleList([_ResBlock512(c) for _ in range(4)])
        self.out_heads = nn.ModuleDict()

    def _ensure_head(self, block_id: str, hidden: int) -> nn.Linear:
        if block_id not in self.out_heads:
            lin = nn.Linear(self.cfg.adapter_channels, hidden)
            nn.init.zeros_(lin.weight)
            nn.init.zeros_(lin.bias)
            self.out_heads[block_id] = lin
        return self.out_heads[block_id]

    def forward(
        self,
        z_noisy: Tensor,
        *,
        target_hw: tuple[int, int],
        block_id: str = "0",
        hidden_dim: int | None = None,
    ) -> Tensor:
        """Return token features [B, N, d] aligned to patch grid."""
        hidden_dim = hidden_dim or self.cfg.hidden_dim
        z = F.interpolate(z_noisy, size=target_hw, mode="nearest")
        h = self.stem(z)
        for blk in self.blocks:
            h = blk(h)
        tokens = h.flatten(2).transpose(1, 2)  # B, N, C
        return self._ensure_head(block_id, hidden_dim)(tokens)


class SigmaAwareGate(nn.Module):
    """Eq. (6): g = sigmoid(Linear([h,l]) - α·σ)."""

    def __init__(self, hidden_dim: int, *, cfg: PiDConfig | None = None) -> None:
        super().__init__()
        cfg = cfg or PiDConfig()
        self.alpha = nn.Parameter(torch.tensor(float(cfg.gate_alpha)))
        self.proj = nn.Linear(hidden_dim * 2, 1)
        nn.init.zeros_(self.proj.weight)
        nn.init.constant_(self.proj.bias, float(cfg.gate_bias))

    def forward(self, h: Tensor, l: Tensor, sigma: float | Tensor) -> Tensor:
        if isinstance(sigma, (int, float)):
            sigma_t = torch.full(
                (h.shape[0], h.shape[1], 1),
                float(sigma),
                device=h.device,
                dtype=h.dtype,
            )
        else:
            sigma_t = sigma.view(-1, 1, 1).expand(h.shape[0], h.shape[1], 1)
        x = torch.cat([h, l], dim=-1)
        return torch.sigmoid(self.proj(x) - self.alpha * sigma_t)


def inject_latent(
    h: Tensor,
    l: Tensor,
    sigma: float,
    gate: SigmaAwareGate,
) -> Tensor:
    """Eq. (5): h ← h + g(h,l,σ) ⊙ l."""
    g = gate(h, l, sigma)
    return h + g * l
