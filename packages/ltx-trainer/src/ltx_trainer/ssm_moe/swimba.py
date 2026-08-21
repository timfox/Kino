"""Swimba: MoE-parameterized SSM with a single state trajectory (arXiv:2603.06938)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ltx_trainer.ssm_moe.moe import MoERouter
from ltx_trainer.ssm_moe.ssm_core import mix_ssm_streams, scan_with_params


class SwimbaLayer(nn.Module):
    """
    Each expert proposes selective SSM streams (Δ, B, C); router mixes in parameter
    space, then one recurrence updates hidden state (no per-expert state duplication).
    """

    def __init__(
        self,
        d_model: int,
        d_state: int,
        num_experts: int,
        top_k: int = 2,
        d_inner: int | None = None,
    ) -> None:
        super().__init__()
        d_inner = d_inner or d_model * 2
        self.d_state = d_state
        self.d_inner = d_inner
        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=False)
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)
        self.router = MoERouter(d_model, num_experts, top_k=top_k)

        self.expert_dt = nn.ModuleList(nn.Linear(d_inner, d_state, bias=True) for _ in range(num_experts))
        self.expert_B = nn.ModuleList(nn.Linear(d_inner, d_state, bias=False) for _ in range(num_experts))
        self.expert_C = nn.ModuleList(nn.Linear(d_inner, d_state, bias=False) for _ in range(num_experts))

        a = torch.arange(1, d_state + 1, dtype=torch.float32)
        self.A_log = nn.Parameter(torch.log(a))
        self.D = nn.Parameter(torch.ones(d_inner))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        xz = self.in_proj(x)
        x_inner, gate = xz.chunk(2, dim=-1)
        x_inner = F.silu(x_inner)

        weights, indices, logits = self.router(x)
        # Build full expert weight tensor (B, L, E) from top-k for mixing all streams.
        bsz, seqlen, _ = x.shape
        num_experts = len(self.expert_dt)
        full_w = torch.zeros(bsz, seqlen, num_experts, device=x.device, dtype=x.dtype)
        for k in range(indices.shape[-1]):
            full_w.scatter_add_(-1, indices[..., k : k + 1], weights[..., k : k + 1])

        streams: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        for e in range(num_experts):
            delta = F.softplus(self.expert_dt[e](x_inner))
            b = self.expert_B[e](x_inner)
            c = self.expert_C[e](x_inner)
            streams.append((delta, b, c))

        delta_m, b_m, c_m = mix_ssm_streams(streams, full_w)
        y = scan_with_params(
            x_inner,
            gate,
            delta_m,
            b_m,
            c_m,
            a_log=self.A_log,
            d=self.D,
            out_proj=self.out_proj,
        )
        return y, logits
