"""HDR environment map via exposure fusion (Sec. 3.4, Eq. 1–4)."""

from __future__ import annotations

import torch
from torch import Tensor

from ltx_trainer.lightharmony3d.config import FUSION_SAT_THRESHOLD
from ltx_trainer.lightharmony3d.color import normalized_luminance_ldr


def scale_irradiance(l_tilde: Tensor, ev: int) -> Tensor:
    """Eq. (2): E = L̃ · 2^{-EV}."""
    return l_tilde * (2.0 ** (-ev))


def fuse_exposure_bracket(
    images: list[Tensor],
    ev_values: list[int],
    *,
    sat_threshold: float = FUSION_SAT_THRESHOLD,
) -> Tensor:
    """
    Bottom-up luminance merge (Eq. 3) then chromaticity from base EV0 (Eq. 4).

    images[0] is darkest; images[-1] is base EV0 from 3DGS render.
    """
    if len(images) != len(ev_values):
        raise ValueError("images and ev_values length mismatch")
    n = len(images)
    l_tildes = [normalized_luminance_ldr(img) for img in images]
    energies = [scale_irradiance(lt, ev) for lt, ev in zip(l_tildes, ev_values, strict=True)]

    fused = energies[0]
    for i in range(1, n):
        saturated = l_tildes[i] > sat_threshold
        fused = torch.where(saturated, fused, energies[i])

    base = images[-1]
    l_base = l_tildes[-1].clamp(min=1e-6)
    scale = fused / l_base
    hdr = (scale * base).clamp(min=0.0)
    return hdr


def build_hdr_envmap(
    ev0: Tensor,
    underexposed: list[Tensor],
    ev_sequence: tuple[int, ...],
) -> Tensor:
    """Assemble bracket: darkest generated first, EV0 last."""
    if len(underexposed) != len(ev_sequence) - 1:
        raise ValueError("underexposed count must be len(ev_sequence)-1")
    images = list(underexposed) + [ev0]
    return fuse_exposure_bracket(images, list(ev_sequence))
