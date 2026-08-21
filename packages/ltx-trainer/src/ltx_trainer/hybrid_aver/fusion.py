"""Hybrid fusion classifier stub."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from ltx_trainer.hybrid_aver.attention import BidirectionalCrossFusion
from ltx_trainer.hybrid_aver.config import HybridAverConfig
from ltx_trainer.hybrid_aver.film import FiLMAudioConditioner
from ltx_trainer.hybrid_aver.maf import ModalityTemporalAttention


class HybridAverFusionStub(nn.Module):
    """FiLM → cross-attn → multimodal Transformer → MAF → classifier."""

    def __init__(self, cfg: HybridAverConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or HybridAverConfig()
        d = self.cfg.embed_dim
        self.film = FiLMAudioConditioner(d)
        self.cross = BidirectionalCrossFusion(d, self.cfg.num_heads)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=self.cfg.num_heads,
            dim_feedforward=d * 4,
            batch_first=True,
            dropout=0.1,
        )
        self.mm_encoder = nn.TransformerEncoder(enc_layer, num_layers=self.cfg.num_fusion_layers)
        self.maf = ModalityTemporalAttention(d)
        self.classifier = nn.Sequential(
            nn.Dropout(self.cfg.classifier_dropout),
            nn.Linear(d, self.cfg.num_classes),
        )

    def forward(self, visual: Tensor, audio: Tensor) -> dict[str, Tensor]:
        # visual, audio: (B, T, D) cached VideoMAE / AST embeddings
        a_cond = self.film(audio, visual)
        ha, hv = self.cross(visual, a_cond)
        mm = torch.cat([ha, hv], dim=1)
        encoded = self.mm_encoder(mm)
        fused = self.maf(encoded)
        logits = self.classifier(fused)
        return {"logits": logits, "fused": fused, "a_cond": a_cond}
