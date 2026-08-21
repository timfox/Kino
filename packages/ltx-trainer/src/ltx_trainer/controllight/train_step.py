"""ControlLight training step: sample strength s, pseudo target I_s, L_wFM (Fig. 5, Sec. 3.3)."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from ltx_trainer.controllight.config import ControlLightConfig
from ltx_trainer.controllight.losses import interpolate_latent, velocity_target
from ltx_trainer.controllight.pipeline import controllight_training_loss, prepare_training_batch


@dataclass
class ControlLightBatch:
    strength: float
    i0: Tensor
    i1: Tensor
    target: Tensor
    group: dict[float, Tensor]
    t: float
    z0: Tensor
    z1: Tensor
    zt: Tensor


def sample_training_strength(cfg: ControlLightConfig, rng: random.Random | None = None) -> float:
    """Uniform over G strengths used in paper training."""
    r = rng or random
    return r.choice(list(cfg.enhancement_strengths))


def encode_latent_stub(image: Tensor, *, channels: int = 8) -> Tensor:
    """Placeholder VAE encode until FLUX.2-klein hook is wired."""
    if image.dim() == 3:
        image = image.unsqueeze(0)
    c, h, w = image.shape[-3], image.shape[-2], image.shape[-1]
    lh, lw = max(h // 8, 4), max(w // 8, 4)
    return torch.randn(image.shape[0], channels, lh, lw, device=image.device, dtype=image.dtype) * 0.1 + image.mean()


def build_training_batch(
    i0: Tensor,
    i1: Tensor,
    *,
    cfg: ControlLightConfig | None = None,
    strength: float | None = None,
    t: float | None = None,
    rng: random.Random | None = None,
) -> ControlLightBatch:
    """One Light100K training sample: pick s, I_s, flow time t, latents."""
    cfg = cfg or ControlLightConfig()
    s = strength if strength is not None else sample_training_strength(cfg, rng)
    t_val = t if t is not None else (rng or random).random()
    target, group = prepare_training_batch(i0, i1, s, cfg=cfg)
    z1 = encode_latent_stub(target)
    z0 = torch.randn_like(z1)
    zt = interpolate_latent(z0, z1, t_val)
    return ControlLightBatch(
        strength=s,
        i0=i0,
        i1=i1,
        target=target,
        group=group,
        t=t_val,
        z0=z0,
        z1=z1,
        zt=zt,
    )


def training_step(
    v_pred: Tensor,
    batch: ControlLightBatch,
    *,
    cfg: ControlLightConfig | None = None,
    use_weighted: bool = True,
) -> Tensor:
    """LwFM on predicted velocity vs v* = z1 - z0."""
    return controllight_training_loss(
        v_pred,
        batch.z0,
        batch.z1,
        batch.i0,
        batch.target,
        t=batch.t,
        cfg=cfg,
        use_weighted=use_weighted,
    )


def training_step_demo(*, seed: int = 0, size: int = 64) -> dict[str, Any]:
    """CPU smoke: full batch + L_FM vs L_wFM."""
    torch.manual_seed(seed)
    cfg = ControlLightConfig()
    i0 = torch.rand(3, size, size) * 0.12
    i1 = torch.rand(3, size, size) * 0.55 + 0.35
    batch = build_training_batch(i0, i1, cfg=cfg, strength=0.6, t=0.5)
    v_pred = velocity_target(batch.z1, batch.z0) + 0.01 * torch.randn_like(batch.z1)
    loss_w = training_step(v_pred, batch, cfg=cfg, use_weighted=True)
    loss_f = training_step(v_pred, batch, cfg=cfg, use_weighted=False)
    return {
        "strength": batch.strength,
        "t": batch.t,
        "target_shape": list(batch.target.shape),
        "loss_LwFM": float(loss_w.item()),
        "loss_LFM": float(loss_f.item()),
        "LwFM_le_LFM": bool(loss_w.item() <= loss_f.item() + 1e-3),
    }
