"""Coupled denoising joint-distribution view (Sec. 3.5)."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from ltx_trainer.genpcr.coupled_denoise import couple_latents


def cat_cross_view(x_p: Tensor, x_q: Tensor) -> Tensor:
    """x_PQ = Cat([x_P; x_Q]) along height (latent or image)."""
    return couple_latents(x_p, x_q)


def cat_conditions(d_p: Tensor, d_q: Tensor) -> Tensor:
    """d_PQ = Cat([d_P; d_Q]) for depth or range maps."""
    return couple_latents(d_p, d_q)


def coupled_elbo_summary() -> dict[str, Any]:
    """Textbook anchors for agents (not a full DDPM trainer)."""
    return {
        "objective": "maximize Epdata[pθ(xPQ | dPQ)] via coupled latent diffusion",
        "coupling": "vertical stack enables cross-view self-attention (Eq. 4)",
        "practical_loss": "denoising matching on ε ≈ εθ(xPQ_t, t, dPQ) (Eq. 11)",
        "equivalent_to": ["Eq. 5 DepthMatch finetune", "Eq. 8 LiDARMatch finetune"],
    }
