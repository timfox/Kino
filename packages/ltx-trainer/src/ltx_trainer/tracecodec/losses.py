"""Training objective L = L_recon + λ_Δt L_Δt + β L_KL — § 3.4."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from ltx_trainer.tracecodec.actions import PacketAction
from ltx_trainer.tracecodec.codec import TraceCodecStub, coarse_field_accuracy
from ltx_trainer.tracecodec.config import TraceCodecConfig


def kl_divergence(mu: Tensor, logvar: Tensor) -> Tensor:
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())


def trace_codec_loss(
    *,
    original: PacketAction,
    recon: PacketAction,
    delta_t_true: float,
    delta_t_pred: float,
    mu: Tensor,
    logvar: Tensor,
    cfg: TraceCodecConfig | None = None,
) -> dict[str, float]:
    cfg = cfg or TraceCodecConfig()
    l_recon = 1.0 - coarse_field_accuracy(original, recon)
    l_dt = abs(delta_t_true - delta_t_pred) / max(delta_t_true, 1e-3)
    l_kl = float(kl_divergence(mu, logvar).item())
    total = l_recon + cfg.lambda_dt * l_dt + cfg.beta_kl * l_kl
    return {
        "loss": total,
        "l_recon": l_recon,
        "l_dt": l_dt,
        "l_kl": l_kl,
        "action_accuracy": coarse_field_accuracy(original, recon),
    }


def training_step(model: TraceCodecStub, timed, *, cfg: TraceCodecConfig | None = None) -> dict[str, float]:
    out = model(timed)
    losses = trace_codec_loss(
        original=timed.action,
        recon=out.recon,
        delta_t_true=timed.delta_t_ms,
        delta_t_pred=out.delta_t_ms,
        mu=out.mu,
        logvar=out.logvar,
        cfg=cfg,
    )
    return losses
