"""Routing Mamba: shared-router MoE over linear projections (arXiv:2506.18145)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ltx_trainer.ssm_moe.moe import MoERouter
from ltx_trainer.ssm_moe.ssm_core import SelectiveSSM


class ProjectionExpertBank(nn.Module):
    """Bank of linear projection experts for a single projection role."""

    def __init__(self, in_features: int, out_features: int, num_experts: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_experts, out_features, in_features) * 0.02)
        if out_features > 0:
            self.bias = nn.Parameter(torch.zeros(num_experts, out_features))
        else:
            self.bias = None

    def forward(self, x: torch.Tensor, expert_weights: torch.Tensor) -> torch.Tensor:
        """
        ``x``: (B, L, in)
        ``expert_weights``: (B, L, E) mixture over experts
        """
        bsz, seqlen, _ = x.shape
        num_experts = self.weight.shape[0]
        out_dim = self.weight.shape[1]
        out = torch.zeros(bsz, seqlen, out_dim, device=x.device, dtype=x.dtype)
        for e in range(num_experts):
            w_e = expert_weights[..., e].unsqueeze(-1)
            y_e = F.linear(x, self.weight[e], self.bias[e] if self.bias is not None else None)
            out = out + w_e * y_e
        return out


class RoutingMambaLayer(nn.Module):
    """
    RoM-style layer: one shared router gates multiple projection expert banks,
    then a selective SSM core consumes the mixed projections.
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
        self.d_inner = d_inner
        self.d_state = d_state
        self.router = MoERouter(d_model, num_experts, top_k=top_k)

        self.in_experts = ProjectionExpertBank(d_model, d_inner * 2, num_experts)
        self.dt_experts = ProjectionExpertBank(d_inner, d_state, num_experts)
        self.B_experts = ProjectionExpertBank(d_inner, d_state, num_experts)
        self.C_experts = ProjectionExpertBank(d_inner, d_state, num_experts)
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

        a = torch.arange(1, d_state + 1, dtype=torch.float32)
        self.A_log = nn.Parameter(torch.log(a))
        self.D = nn.Parameter(torch.ones(d_inner))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        weights, indices, logits = self.router(x)
        bsz, seqlen, _ = x.shape
        num_experts = self.in_experts.weight.shape[0]
        full_w = torch.zeros(bsz, seqlen, num_experts, device=x.device, dtype=x.dtype)
        for k in range(indices.shape[-1]):
            full_w.scatter_add_(-1, indices[..., k : k + 1], weights[..., k : k + 1])

        xz = self.in_experts(x, full_w)
        x_inner, gate = xz.chunk(2, dim=-1)
        x_inner = F.silu(x_inner)

        delta = F.softplus(self.dt_experts(x_inner, full_w))
        b_coef = self.B_experts(x_inner, full_w)
        c_coef = self.C_experts(x_inner, full_w)

        from ltx_trainer.ssm_moe.ssm_core import scan_with_params

        y = scan_with_params(
            x_inner,
            gate,
            delta,
            b_coef,
            c_coef,
            a_log=self.A_log,
            d=self.D,
            out_proj=self.out_proj,
        )
        return y, logits
