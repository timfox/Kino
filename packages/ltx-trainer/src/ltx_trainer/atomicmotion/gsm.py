"""Global Synchronized Modulation (Sec. 3.3, Eq. 3–5)."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from ltx_trainer.atomicmotion.config import AtomicMotionConfig
from ltx_trainer.atomicmotion.partitions import PARTITION_NAMES, PartitionName, atomic_intent_partition


class GlobalSynchronizedModulation(nn.Module):
    """GSM: E'_p = E_p ⊙ (1 + α_p) + β_p."""

    def __init__(self, cfg: AtomicMotionConfig | None = None):
        super().__init__()
        cfg = cfg or AtomicMotionConfig()
        self.cfg = cfg
        self.embed = nn.ModuleDict(
            {
                name: nn.Linear(cfg.feature_dim, cfg.embed_dim)
                for name in PARTITION_NAMES
            }
        )
        self.mod_mlp = nn.Sequential(
            nn.Linear(cfg.embed_dim * 5, cfg.embed_dim * 2),
            nn.GELU(),
            nn.Linear(cfg.embed_dim * 2, cfg.embed_dim * 10),
        )

    def forward(self, x: Tensor) -> dict[PartitionName, Tensor]:
        parts = atomic_intent_partition(x)
        embedded: list[Tensor] = []
        for name in PARTITION_NAMES:
            p = parts[name]
            if p.dim() == 3:
                # [T, Jp, C] -> mean pool joints then embed
                feat = self.embed[name](p.mean(dim=1))
            else:
                feat = self.embed[name](p.mean(dim=2))
            embedded.append(feat)
        concat = torch.cat(embedded, dim=-1)
        mod = self.mod_mlp(concat)
        mod = mod.view(*mod.shape[:-1], 5, 2, self.cfg.embed_dim)
        alpha, beta = mod[..., 0, :], mod[..., 1, :]
        out: dict[PartitionName, Tensor] = {}
        for i, name in enumerate(PARTITION_NAMES):
            e = embedded[i]
            out[name] = e * (1.0 + alpha[..., i, :]) + beta[..., i, :]
        return out
