"""SSM-MoE transformer block and stacked model."""

from __future__ import annotations

import torch
import torch.nn as nn

from ltx_trainer.ssm_moe.config import SSMMoEConfig
from ltx_trainer.ssm_moe.moe import MoEFeedForward
from ltx_trainer.ssm_moe.routing_mamba import RoutingMambaLayer
from ltx_trainer.ssm_moe.ssm_core import SelectiveSSM
from ltx_trainer.ssm_moe.swimba import SwimbaLayer


class SSMMoEBlock(nn.Module):
    """Pre-norm block: selective SSM token-mixer + MoE channel-mixer (mode-dependent)."""

    def __init__(self, cfg: SSMMoEConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.norm2 = nn.LayerNorm(cfg.d_model)

        if cfg.mode == "swimba":
            self.ssm = SwimbaLayer(
                cfg.d_model,
                cfg.d_state,
                cfg.num_experts,
                top_k=cfg.top_k,
                d_inner=cfg.d_inner,
            )
            self.moe: MoEFeedForward | None = None
        elif cfg.mode == "routing_mamba":
            self.ssm = RoutingMambaLayer(
                cfg.d_model,
                cfg.d_state,
                cfg.num_experts,
                top_k=cfg.top_k,
                d_inner=cfg.d_inner,
            )
            self.moe = None
        else:
            self.ssm = SelectiveSSM(cfg.d_model, cfg.d_state, cfg.d_inner)
            self.moe = MoEFeedForward(
                cfg.d_model,
                num_experts=cfg.num_experts,
                top_k=cfg.top_k,
                d_ff=cfg.d_inner,
            )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        aux_logits: list[torch.Tensor] = []
        h = self.norm1(x)
        if isinstance(self.ssm, (SwimbaLayer, RoutingMambaLayer)):
            ssm_out, logits = self.ssm(h)
            aux_logits.append(logits)
        else:
            ssm_out = self.ssm(h)
        x = x + ssm_out

        h = self.norm2(x)
        if self.moe is not None:
            moe_out, logits = self.moe(h)
            aux_logits.append(logits)
            x = x + moe_out
        return x, aux_logits


class SSMMoEModel(nn.Module):
    """Stacked SSM-MoE language-model backbone (embedding + blocks + LM head)."""

    def __init__(self, cfg: SSMMoEConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or SSMMoEConfig()
        c = self.cfg
        self.embed = nn.Embedding(c.vocab_size, c.d_model)
        self.blocks = nn.ModuleList(SSMMoEBlock(c) for _ in range(c.n_layers))
        self.norm_f = nn.LayerNorm(c.d_model)
        self.lm_head = nn.Linear(c.d_model, c.vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        x = self.embed(input_ids)
        all_logits: list[torch.Tensor] = []
        for block in self.blocks:
            x, aux = block(x)
            all_logits.extend(aux)
        x = self.norm_f(x)
        logits = self.lm_head(x)
        return logits, all_logits

    def num_parameters(self) -> dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        active = total  # stub: full count; real MoE reports active experts only
        return {"total": total, "active_per_token_stub": active // max(self.cfg.num_experts // self.cfg.top_k, 1)}
