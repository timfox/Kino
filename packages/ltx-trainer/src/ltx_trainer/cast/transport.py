"""Bounded local stochastic transport on ordered supports (Sec. 3)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ltx_trainer.cast.config import CASTConfig, TRANSPORT_RADIUS
from ltx_trainer.cast.simplex import normalize_simplex, support_mean


def apply_local_transport(anchor: Tensor, kernel_logits: Tensor, *, radius: int = TRANSPORT_RADIUS) -> Tensor:
    """Apply row-stochastic radius-r transport; kernel_logits shape (B, D, 2r+1)."""
    b, d, _ = kernel_logits.shape
    probs = F.softmax(kernel_logits, dim=-1)
    out = torch.zeros_like(anchor)
    for o_idx, offset in enumerate(range(-radius, radius + 1)):
        target = (torch.arange(d, device=anchor.device) + offset).clamp(0, d - 1)
        gathered = anchor.gather(-1, target.unsqueeze(0).expand(b, -1))
        out = out + gathered * probs[..., o_idx]
    return normalize_simplex(out)


class TransportHead(nn.Module):
    """Predict radius-1 row-stochastic kernel and bounded transport strength."""

    def __init__(self, cfg: CASTConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.kernel_size = 2 * TRANSPORT_RADIUS + 1
        self.kernel_proj = nn.Linear(cfg.hidden_dim, cfg.support_dim * self.kernel_size)
        self.rho_proj = nn.Linear(cfg.hidden_dim, 1)

    def forward(self, hidden: Tensor, anchor: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        b, d = anchor.shape
        logits = self.kernel_proj(hidden).view(b, d, self.kernel_size)
        transported = apply_local_transport(anchor, logits)
        rho_raw = torch.sigmoid(self.rho_proj(hidden).squeeze(-1)) * self.cfg.rho_max

        if self.cfg.ordered_support:
            delta_mu = support_mean(transported) - support_mean(anchor)
            support_span = anchor.argmax(dim=-1).float() - anchor.argmin(dim=-1).float()
            budget = self.cfg.delta_mu + self.cfg.delta_sigma * support_span.abs()
            gate = torch.clamp(budget / (delta_mu.abs() + 1e-6), max=1.0)
            rho = rho_raw * gate
        else:
            rho = torch.zeros_like(rho_raw)

        return transported, rho, logits
