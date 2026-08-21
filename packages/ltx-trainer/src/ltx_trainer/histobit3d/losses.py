"""Multiscale content consistency and style losses (Sec. 2.3–2.4, Eq. 1–4)."""

from __future__ import annotations

from typing import Callable

import torch
from torch import Tensor

from ltx_trainer.histobit3d.config import HistoBIT3DConfig, MSC_SCALES


def _l1_features(a: Tensor, b: Tensor) -> Tensor:
    return torch.mean(torch.abs(a - b))


def multiscale_content_loss(
    phi_forward: dict[int, Tensor],
    phi_backward: dict[int, Tensor],
    *,
    scales: tuple[int, ...] = MSC_SCALES,
    channel_sample: int | None = None,
) -> Tensor:
    """L_BIT or L_H&E term (Eq. 1 or 2) over encoder scales k ∈ K."""
    losses: list[Tensor] = []
    for k in scales:
        if k not in phi_forward or k not in phi_backward:
            continue
        f = phi_forward[k]
        b = phi_backward[k].detach()
        if channel_sample is not None and f.shape[1] > channel_sample:
            idx = torch.linspace(0, f.shape[1] - 1, channel_sample).long()
            f, b = f[:, idx], b[:, idx]
        losses.append(_l1_features(f, b))
    if not losses:
        return torch.tensor(0.0)
    return sum(losses) / len(losses)


def bidirectional_msc_loss(
    bit_feats_fwd: dict[int, Tensor],
    bit_feats_bwd: dict[int, Tensor],
    he_feats_fwd: dict[int, Tensor],
    he_feats_bwd: dict[int, Tensor],
) -> Tensor:
    """L_msc = L_BIT + L_H&E (Eq. 1–2)."""
    l_bit = multiscale_content_loss(bit_feats_fwd, bit_feats_bwd)
    l_he = multiscale_content_loss(he_feats_fwd, he_feats_bwd)
    return l_bit + l_he


def adain_token(
    s_source: Tensor,
    s_target: Tensor,
    eps: float = 1e-5,
) -> Tensor:
    """AdaIN in token space (Eq. 3)."""
    mu_s = s_source.mean(dim=-1, keepdim=True)
    sig_s = s_source.std(dim=-1, keepdim=True) + eps
    mu_t = s_target.mean(dim=-1, keepdim=True)
    sig_t = s_target.std(dim=-1, keepdim=True) + eps
    normalized = (s_source - mu_s) / sig_s
    return sig_t * normalized + mu_t


def style_statistics_loss(s_fake: Tensor, s_real: Tensor) -> Tensor:
    """L_style — match μ, σ of ViT bottleneck tokens (Eq. 4)."""
    mu_f = s_fake.mean(dim=-1)
    mu_r = s_real.mean(dim=-1).detach()
    sig_f = s_fake.std(dim=-1)
    sig_r = s_real.std(dim=-1).detach()
    return torch.mean((mu_f - mu_r) ** 2) + torch.mean((sig_f - sig_r) ** 2)


def update_style_prototype(
    prototype: Tensor | None,
    s_he: Tensor,
    *,
    alpha: float = 0.99,
) -> Tensor:
    """EMA update of s̄_H&E (Sec. 2.4)."""
    if prototype is None:
        return s_he.detach().clone()
    return alpha * prototype + (1.0 - alpha) * s_he.detach()


def total_histobit_loss(
    *,
    cycle_loss: Tensor,
    identity_loss: Tensor,
    msc_loss: Tensor,
    style_loss: Tensor,
    cfg: HistoBIT3DConfig | None = None,
) -> dict[str, Tensor]:
    cfg = cfg or HistoBIT3DConfig()
    total = (
        cfg.lambda_cycle * cycle_loss
        + cfg.lambda_idt * identity_loss
        + cfg.lambda_msc * msc_loss
        + cfg.lambda_style * style_loss
    )
    return {
        "l_cycle": cycle_loss,
        "l_idt": identity_loss,
        "l_msc": msc_loss,
        "l_style": style_loss,
        "loss": total,
    }


def demo_feature_maps(batch: int = 2, channels: int = 64, h: int = 32, w: int = 32) -> dict[int, Tensor]:
    """Synthetic encoder features for smoke tests."""
    out: dict[int, Tensor] = {}
    for k in MSC_SCALES:
        ch = max(4, channels // k)
        hh, ww = max(2, h // k), max(2, w // k)
        out[k] = torch.randn(batch, ch, hh, ww)
    return out
