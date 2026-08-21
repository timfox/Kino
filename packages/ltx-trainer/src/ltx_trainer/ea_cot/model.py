"""Entity–claim binding scorer stub."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ltx_trainer.ea_cot.config import EACoTConfig


class EntityBindingStub(nn.Module):
    """Maps pooled speech features + entity ids to claim consistency logits."""

    def __init__(self, cfg: EACoTConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or EACoTConfig()
        d = self.cfg.demo_entity_dim
        self.entity_emb = nn.Embedding(64, d)
        self.speech_proj = nn.Linear(self.cfg.demo_claim_len, d)
        self.claim_proj = nn.Linear(self.cfg.demo_claim_len, d)
        self.score = nn.Linear(d * 2, 1)

    def forward(
        self,
        speech_feat: Tensor,
        entity_ids: Tensor,
        claim_feat: Tensor,
    ) -> Tensor:
        # speech_feat, claim_feat: (B, L); entity_ids: (B, E)
        d = self.cfg.demo_entity_dim
        sp = self.speech_proj(speech_feat)
        ent = self.entity_emb(entity_ids).mean(dim=1)
        cl = self.claim_proj(claim_feat)
        bound = (sp + ent).reshape(sp.shape[0], d)
        h = torch.cat([bound, cl], dim=-1)
        return self.score(h).squeeze(-1)


def binding_bce(logits: Tensor, labels: Tensor) -> Tensor:
    return F.binary_cross_entropy_with_logits(logits, labels.float())
