"""Inference: integrate flow from t=0→1 (Eq. 4)."""

from __future__ import annotations

import torch
from torch import Tensor

from ltx_trainer.instructav2av.backbone import DualStreamEditor
from ltx_trainer.instructav2av.config import InstructAV2AVConfig
from ltx_trainer.instructav2av.flow import sample_noise_like


@torch.no_grad()
def instructav2av_edit(
    zs_v: Tensor,
    zs_a: Tensor,
    inst_emb: Tensor,
    *,
    steps: int = 28,
    cfg: InstructAV2AVConfig | None = None,
) -> tuple[Tensor, Tensor]:
    """Euler solve d/dt(z_v, z_a) = v_θ from noise to edited latents."""
    cfg = cfg or InstructAV2AVConfig()
    if zs_v.dim() == 3:
        zs_v = zs_v.unsqueeze(0)
        zs_a = zs_a.unsqueeze(0)
        inst_emb = inst_emb.unsqueeze(0) if inst_emb.dim() == 1 else inst_emb

    model = DualStreamEditor(cfg=cfg, latent_channels=zs_v.shape[1])
    zt_v = sample_noise_like(zs_v)
    zt_a = sample_noise_like(zs_a)

    dt = 1.0 / steps
    for i in range(steps):
        t = i * dt
        uv, ua = model(zt_v, zt_a, zs_v, zs_a, inst_emb, t=t, cross_modal=True)
        zt_v = zt_v + uv * dt
        zt_a = zt_a + ua * dt

    return zt_v.clamp(-4, 4), zt_a.clamp(-4, 4)
