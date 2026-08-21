"""BiLT-Autoencoder full model."""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor

from ltx_trainer.bilt.config import BiLTConfig
from ltx_trainer.bilt.decoder import PhysicsConstrainedDecoder
from ltx_trainer.bilt.scanner import BiLTScanner


class BiLTAutoencoder(nn.Module):
    """Mixed autoencoder with BiLT scanner encoder and physics-constrained decoder."""

    def __init__(self, cfg: BiLTConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or BiLTConfig()
        self.scanner = BiLTScanner(self.cfg)
        self.decoder = PhysicsConstrainedDecoder(self.cfg.latent_dim)

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        z, probes = self.scanner(x)
        op = self.decoder(z)
        mu_a, mu_s = PhysicsConstrainedDecoder.split_channels(op)
        return {
            "decode_output": z,
            "encode_output": op,
            "mu_a": mu_a,
            "mu_s": mu_s,
            "probes": probes,
            "attention": probes,  # alias for interpretability hooks
        }

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
