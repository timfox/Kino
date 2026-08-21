"""Shift-equivariant Euler sampler (Appendix A.2)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor

from ltx_trainer.gimbal360.circular_vae import CircularVAEStub
from ltx_trainer.gimbal360.config import CFG_SCALE, DENOISE_STEPS, Gimbal360Config
from ltx_trainer.gimbal360.fill_conditioning import concat_fill_channels
from ltx_trainer.gimbal360.gimbal360_net import Gimbal360CompletionStub
from ltx_trainer.gimbal360.topology import roll_azimuth


@dataclass
class SamplerConfig:
    num_steps: int = DENOISE_STEPS
    cfg_scale: float = CFG_SCALE
    step_size: float = 0.15  # stub Euler step (not production Flux schedule)


def predict_eps_fill(
    model: Gimbal360CompletionStub,
    perspective: Tensor,
    mask: Tensor,
    z_t: Tensor,
    *,
    z_ref: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Denoiser input: DAL warp then z_t ⊕ M ⊕ z_canonical."""
    _flow, _rot, z_canonical = model.dal(perspective)
    if z_canonical.shape[-2:] != mask.shape[-2:]:
        z_canonical = torch.nn.functional.interpolate(
            z_canonical, size=mask.shape[-2:], mode="bilinear", align_corners=False
        )
    z_cond = z_ref if z_ref is not None else z_canonical
    x = concat_fill_channels(z_t, mask, z_cond)
    return model.denoise(x), z_canonical


def shift_equivariant_sample(
    model: Gimbal360CompletionStub,
    perspective: Tensor,
    mask: Tensor,
    z_init: Tensor,
    *,
    sampler: SamplerConfig | None = None,
    shift_deltas: Sequence[int] | None = None,
    generator: torch.Generator | None = None,
) -> dict[str, Tensor]:
    """
    Per-step random azimuth roll on latents; compensate total δ after denoising.

    Mirrors Appendix A.2: cumulative translation on the S¹ manifold during
    denoising, then a single inverse roll to re-align with canonical conditioning.
    """
    cfg = sampler or SamplerConfig(num_steps=min(DENOISE_STEPS, 8), cfg_scale=1.0)
    width = z_init.shape[-1]
    z = z_init
    total_delta = 0
    if shift_deltas is None:
        shift_deltas = [
            int(torch.randint(0, max(width, 1), (1,), generator=generator).item())
            for _ in range(cfg.num_steps)
        ]

    model.eval()
    with torch.no_grad():
        for delta_t in shift_deltas:
            total_delta = (total_delta + int(delta_t)) % max(width, 1)
            z_step = roll_azimuth(z, int(delta_t))
            m_step = roll_azimuth(mask, int(delta_t))
            eps, _ = predict_eps_fill(model, perspective, m_step, z_step, z_ref=z_init)
            z = z_step - cfg.step_size * eps
        z = roll_azimuth(z, -total_delta)

    return {
        "latent": z,
        "total_azimuth_shift": torch.tensor(total_delta, device=z.device),
        "steps": torch.tensor(len(shift_deltas), device=z.device),
    }


def complete_panorama_stub(
    model: Gimbal360CompletionStub,
    perspective: Tensor,
    mask: Tensor,
    *,
    cfg: Gimbal360Config | None = None,
    vae: CircularVAEStub | None = None,
    sampler: SamplerConfig | None = None,
) -> dict[str, Tensor]:
    """Encode perspective ERP crop → shift-equivariant denoise → circular decode."""
    cfg = cfg or model.cfg
    vae = vae or CircularVAEStub(cfg.latent_channels).to(perspective.device)
    z_init = vae.encode(perspective)
    if z_init.shape[-2:] != mask.shape[-2:]:
        z_init = torch.nn.functional.interpolate(z_init, size=mask.shape[-2:], mode="bilinear", align_corners=False)
    sample = shift_equivariant_sample(model, perspective, mask, z_init, sampler=sampler)
    erp = vae.decode(sample["latent"])
    return {**sample, "erp": erp}
