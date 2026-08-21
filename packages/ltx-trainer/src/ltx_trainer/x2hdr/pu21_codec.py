"""PU21 codec wrappers (X2HDR arXiv:2602.04814)."""

from __future__ import annotations

from torch import Tensor

from ltx_trainer.hdr_ingest import (
    X2HDR_DEFAULT_L_PEAK_CD_M2,
    linear_scene_to_pu21_display,
    pu21_forward_linear_abs,
    pu21_inverse_to_linear_abs,
)


def scene_linear_to_vae_pixels(hdr_linear: Tensor, *, l_peak: float = X2HDR_DEFAULT_L_PEAK_CD_M2) -> Tensor:
    if hdr_linear.dim() == 3:
        return linear_scene_to_pu21_display(hdr_linear.unsqueeze(0), l_peak=l_peak).squeeze(0)
    return linear_scene_to_pu21_display(hdr_linear, l_peak=l_peak)


def vae_pixels_to_scene_linear(pu21_01: Tensor, *, l_peak: float = X2HDR_DEFAULT_L_PEAK_CD_M2) -> Tensor:
    peak = pu21_01.amax().clamp(min=1e-8)
    l_abs = pu21_inverse_to_linear_abs(pu21_01)
    return (l_abs / l_peak * peak).clamp(min=0.0)


def vae_normalize(x: Tensor) -> Tensor:
    return x.clamp(0, 1) * 2.0 - 1.0


def vae_denormalize(x: Tensor) -> Tensor:
    return ((x + 1.0) * 0.5).clamp(0, 1)
