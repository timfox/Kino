"""Neuronal Stochastic Attention Circuit (NSAC): OU logits + logistic-normal attention (Razzaq & Zhao, arXiv:2605.26061)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ltx_trainer.nsac.config import NSACConfig
from ltx_trainer.nsac.gates import OUGateHead
from ltx_trainer.nsac.ou import ou_mean_variance


class NSACStochasticAttention(nn.Module):
    """Multi-head NSAC: OU moments on logits, Gaussian draw, softmax → stochastic attention (paper §3.1)."""

    def __init__(self, cfg: NSACConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d, h = cfg.d_model, cfg.n_heads
        if d % h != 0:
            raise ValueError(f"d_model ({d}) must be divisible by n_heads ({h})")
        self.d_head = d // h
        self.Wq = nn.Linear(d, d, bias=True)
        self.Wk = nn.Linear(d, d, bias=True)
        self.Wv = nn.Linear(d, d, bias=True)
        self.gate = OUGateHead(2 * self.d_head)
        self.ta = nn.Parameter(torch.zeros(h))
        self.tb = nn.Parameter(torch.zeros(h))
        self.out_proj = nn.Linear(d, d, bias=True)
        self.attn_drop = nn.Dropout(cfg.dropout)
        self.dropout = nn.Dropout(cfg.dropout)

    def _time_query(self, x: Tensor) -> Tensor:
        """Per-head normalized time t ∈ (0,1) (paper §3.1, Hasani et al. style)."""
        # tsample: (B, L); ta, tb: (H,)
        tsample = x.mean(dim=-1)
        return torch.sigmoid(-self.ta.view(1, self.cfg.n_heads, 1) * tsample.unsqueeze(1) + self.tb.view(1, self.cfg.n_heads, 1))

    def forward(self, x: Tensor) -> Tensor:
        """``x`` (B, L, D) → context (B, L, D)."""
        cfg = self.cfg
        b, l, d = x.shape
        h, dh = cfg.n_heads, self.d_head

        q = self.Wq(x).view(b, l, h, dh).transpose(1, 2)
        k = self.Wk(x).view(b, l, h, dh).transpose(1, 2)
        v = self.Wv(x).view(b, l, h, dh).transpose(1, 2)

        qh = q.unsqueeze(3).expand(b, h, l, l, dh)
        kh = k.unsqueeze(2).expand(b, h, l, l, dh)
        u = torch.cat([qh, kh], dim=-1)
        kappa, phi, psi = self.gate(u)

        t_q = self._time_query(x)
        t_expand = t_q.unsqueeze(-1).expand(b, h, l, l)

        a0 = torch.zeros((), device=x.device, dtype=x.dtype)
        mu, var = ou_mean_variance(a0, kappa, phi, psi, t_expand, kappa_floor=cfg.kappa_floor)
        std = var.sqrt()
        if self.training:
            eps = torch.randn_like(mu)
        else:
            eps = torch.zeros_like(mu)
        logits = mu + std * eps

        if cfg.use_sparse_curation and l > cfg.top_k:
            prescore = (q.unsqueeze(-2) * k.unsqueeze(-3)).sum(dim=-1)
            _, top_idx = prescore.topk(min(cfg.top_k, l), dim=-1)
            mask = torch.zeros_like(logits, dtype=torch.bool)
            mask.scatter_(-1, top_idx, True)
            logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)

        attn = F.softmax(logits, dim=-1)
        attn = self.attn_drop(attn)
        ctx = torch.matmul(attn, v)
        ctx = ctx.transpose(1, 2).contiguous().view(b, l, d)
        ctx = self.dropout(self.out_proj(ctx))
        return ctx


class NSACRegressor(nn.Module):
    """Sequence in → pooled Gaussian predictive head (μ, log σ) for vector targets."""

    def __init__(self, in_dim: int, d_out: int, cfg: NSACConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.input_proj = nn.Linear(in_dim, d) if in_dim != d else nn.Identity()
        self.ln1 = nn.LayerNorm(d)
        self.nsac = NSACStochasticAttention(cfg)
        self.ln2 = nn.LayerNorm(d)
        self.head = nn.Linear(d, 2 * d_out)
        nn.init.zeros_(self.head.bias)
        nn.init.xavier_uniform_(self.head.weight)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """``x`` (B, L, in_dim) → ``mu``, ``log_std`` each (B, d_out)."""
        h = self.input_proj(x)
        h = h + self.nsac(self.ln1(h))
        h = self.ln2(h)
        pooled = h.mean(dim=1)
        out = self.head(pooled)
        mu, log_std = out.chunk(2, dim=-1)
        log_std = log_std.clamp(min=-10.0, max=4.0)
        return mu, log_std
