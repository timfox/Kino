"""Multi-head attention trait → reflectance stub (Fig. 4)."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from ltx_trainer.leaf_mhsa.config import LeafMHSAConfig


class TraitToSpectraMHSA(nn.Module):
    """Per-trait embedding + dual attention + 1D conv head → reflectance."""

    def __init__(self, cfg: LeafMHSAConfig | None = None, *, n_bands: int | None = None) -> None:
        super().__init__()
        self.cfg = cfg or LeafMHSAConfig()
        self.n_bands = n_bands if n_bands is not None else self.cfg.demo_n_bands
        d = self.cfg.embed_dim
        self.trait_embed = nn.Linear(1, d)
        self.attn1 = nn.MultiheadAttention(d, self.cfg.n_heads, batch_first=True)
        self.attn2 = nn.MultiheadAttention(d, self.cfg.n_heads, batch_first=True)
        self.norm = nn.LayerNorm(d)
        tok = max(self.cfg.conv_kernel, 8)
        self.to_spec = nn.Linear(self.cfg.n_traits * d, self.cfg.conv_filters[0] * tok)
        self.conv = nn.Sequential(
            nn.Conv1d(
                self.cfg.conv_filters[0],
                self.cfg.conv_filters[0],
                self.cfg.conv_kernel,
                padding="same",
            ),
            nn.ReLU(),
            nn.Conv1d(
                self.cfg.conv_filters[0],
                self.cfg.conv_filters[1],
                self.cfg.conv_kernel,
                padding="same",
            ),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.cfg.conv_filters[1] * tok, self.cfg.fc_hidden),
            nn.BatchNorm1d(self.cfg.fc_hidden),
            nn.ReLU(),
            nn.Dropout(self.cfg.dropout),
            nn.Linear(self.cfg.fc_hidden, self.n_bands),
        )
        self._spec_tokens = tok

    def forward(self, traits: Tensor) -> Tensor:
        # traits: (B, 16)
        b = traits.shape[0]
        x = self.trait_embed(traits.unsqueeze(-1))
        x, _ = self.attn1(x, x, x)
        x, _ = self.attn2(x, x, x)
        x = self.norm(x)
        h = self.to_spec(x.reshape(b, -1)).view(b, self.cfg.conv_filters[0], self._spec_tokens)
        h = self.conv(h)
        return self.head(h)
