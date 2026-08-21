"""Selective state space (Mamba-style) core with sequential scan."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelectiveSSM(nn.Module):
    """
    Input-dependent selective SSM recurrence.

    For each token: ``h_t = exp(Δ ⊙ A) h_{t-1} + (Δ ⊙ B) x_t``, ``y_t = C^T h_t + D x_t``.
    Uses a diagonal stable ``A`` (negative) and low-rank input projections for ``Δ, B, C``.
    """

    def __init__(self, d_model: int, d_state: int, d_inner: int | None = None) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        d_inner = d_inner or d_model * 2
        self.d_inner = d_inner

        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=False)
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)
        self.dt_proj = nn.Linear(d_inner, d_state, bias=True)
        self.B_proj = nn.Linear(d_inner, d_state, bias=False)
        self.C_proj = nn.Linear(d_inner, d_state, bias=False)
        self.D = nn.Parameter(torch.ones(d_inner))

        # Diagonal A with negative real parts (stable).
        a = torch.arange(1, d_state + 1, dtype=torch.float32)
        self.A_log = nn.Parameter(torch.log(a))

    def _discretize(self, x_inner: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return per-token Δ, B, C with shapes (B, L, d_state)."""
        delta = F.softplus(self.dt_proj(x_inner))
        b = self.B_proj(x_inner)
        c = self.C_proj(x_inner)
        return delta, b, c

    def scan(self, x: torch.Tensor) -> torch.Tensor:
        """Sequential selective scan. ``x``: (B, L, d_model) → (B, L, d_model)."""
        bsz, seqlen, _ = x.shape
        xz = self.in_proj(x)
        x_inner, gate = xz.chunk(2, dim=-1)
        x_inner = F.silu(x_inner)

        delta, b_coef, c_coef = self._discretize(x_inner)
        a = -torch.exp(self.A_log)  # (d_state,)

        h = torch.zeros(bsz, self.d_state, device=x.device, dtype=x.dtype)
        ys: list[torch.Tensor] = []
        for t in range(seqlen):
            dt = delta[:, t]  # (B, d_state)
            b_t = b_coef[:, t]
            c_t = c_coef[:, t]
            x_t = x_inner[:, t]  # (B, d_inner)
            # Broadcast x_t into state update via mean projection for stub simplicity.
            x_scalar = x_t.mean(dim=-1, keepdim=True)  # (B, 1)
            decay = torch.exp(dt * a.unsqueeze(0))
            h = decay * h + (dt * b_t) * x_scalar
            y_t = (h * c_t).sum(dim=-1, keepdim=True).expand_as(x_t)
            y_t = y_t + self.D * x_t
            ys.append(y_t)

        y = torch.stack(ys, dim=1)
        y = y * F.silu(gate)
        return self.out_proj(y)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scan(x)


def mix_ssm_streams(
    streams: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Swimba-style parameter-space mixing of expert SSM streams.

    ``streams``: list of (delta, B, C) each (B, L, d_state)
    ``weights``: (B, L, K) router softmax over experts K
    """
    delta = torch.zeros_like(streams[0][0])
    b = torch.zeros_like(streams[0][1])
    c = torch.zeros_like(streams[0][2])
    for k, (dk, bk, ck) in enumerate(streams):
        w = weights[..., k].unsqueeze(-1)
        delta = delta + w * dk
        b = b + w * bk
        c = c + w * ck
    return delta, b, c


def scan_with_params(
    x_inner: torch.Tensor,
    gate: torch.Tensor,
    delta: torch.Tensor,
    b_coef: torch.Tensor,
    c_coef: torch.Tensor,
    *,
    a_log: torch.Tensor,
    d: torch.Tensor,
    out_proj: nn.Linear,
) -> torch.Tensor:
    """Run recurrence with pre-mixed SSM parameters (Swimba single trajectory)."""
    bsz, seqlen, d_inner = x_inner.shape
    d_state = delta.shape[-1]
    a = -torch.exp(a_log)
    h = torch.zeros(bsz, d_state, device=x_inner.device, dtype=x_inner.dtype)
    ys: list[torch.Tensor] = []
    for t in range(seqlen):
        dt = delta[:, t]
        b_t = b_coef[:, t]
        c_t = c_coef[:, t]
        x_t = x_inner[:, t]
        x_scalar = x_t.mean(dim=-1, keepdim=True)
        decay = torch.exp(dt * a.unsqueeze(0))
        h = decay * h + (dt * b_t) * x_scalar
        y_t = (h * c_t).sum(dim=-1, keepdim=True).expand_as(x_t)
        y_t = y_t + d * x_t
        ys.append(y_t)
    y = torch.stack(ys, dim=1)
    y = y * F.silu(gate)
    return out_proj(y)
