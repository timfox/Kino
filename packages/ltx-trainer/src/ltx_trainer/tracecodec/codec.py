"""Context-conditioned packet-action VAE stub — § 3.4."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ltx_trainer.tracecodec.actions import (
    COARSE_FIELDS,
    PacketAction,
    TimedAction,
    action_to_vector,
    vector_to_action,
)
from ltx_trainer.tracecodec.config import TraceCodecConfig


def _embed_action(action: PacketAction, *, dim: int) -> Tensor:
    """Compact coarse embedding for toy codec."""
    ctrl_ids = {"syn": 0, "syn_ack": 1, "ack": 2, "data": 3, "fin": 4, "rst": 5, "other": 6}
    feats = [
        float(action.flow_token % 32) / 32.0,
        float(action.direction),
        float(ctrl_ids.get(action.tcp_ctrl.value, 6)) / 7.0,
        float(min(action.l4_payload_len, 1500)) / 1500.0,
        float(action.context.gap_bucket) / 8.0,
    ]
    base = torch.tensor(feats, dtype=torch.float32)
    pad = torch.zeros(dim - base.numel())
    return torch.cat([base, pad])[:dim]


@dataclass
class CodecOutput:
    z: Tensor
    mu: Tensor
    logvar: Tensor
    recon: PacketAction
    delta_t_ms: float


class TraceCodecStub(nn.Module):
    """Staged decoder: coarse fields then detail + timing head."""

    def __init__(self, cfg: TraceCodecConfig | None = None) -> None:
        super().__init__()
        cfg = cfg or TraceCodecConfig()
        self.cfg = cfg
        d = cfg.embed_dim
        self.enc = nn.Sequential(nn.Linear(d + 1, d), nn.ReLU(), nn.Linear(d, d))
        self.mu = nn.Linear(d, cfg.latent_dim)
        self.logvar = nn.Linear(d, cfg.latent_dim)
        self.dec = nn.Sequential(nn.Linear(cfg.latent_dim, d), nn.ReLU(), nn.Linear(d, d))
        self.dt_head = nn.Linear(d, 1)

    def encode(self, action: PacketAction, delta_t_ms: float) -> tuple[Tensor, Tensor]:
        x = _embed_action(action, dim=self.cfg.embed_dim)
        dt = torch.tensor([delta_t_ms / 1000.0], dtype=torch.float32)
        h = self.enc(torch.cat([x, dt]))
        return self.mu(h), self.logvar(h)

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: Tensor, *, template: PacketAction) -> tuple[PacketAction, float]:
        h = self.dec(z)
        dt = float(F.relu(self.dt_head(h)).item() * 1000.0)
        vec = action_to_vector(template)
        # Residual-style detail decode (identity coarse for stub stability)
        recon = vector_to_action(vec, context=template.context)
        return recon, dt

    def forward(self, timed: TimedAction) -> CodecOutput:
        mu, logvar = self.encode(timed.action, timed.delta_t_ms)
        z = self.reparameterize(mu, logvar)
        recon, dt = self.decode(z, template=timed.action)
        return CodecOutput(z=z, mu=mu, logvar=logvar, recon=recon, delta_t_ms=dt)


def coarse_field_accuracy(original: PacketAction, recon: PacketAction) -> float:
    hits = 0
    for name in COARSE_FIELDS:
        if getattr(original, name) == getattr(recon, name):
            hits += 1
    return hits / len(COARSE_FIELDS)
