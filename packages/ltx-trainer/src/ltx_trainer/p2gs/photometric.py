"""Per-view exposure and gamma tone mapping (P2GS Sec. 3.3, Eq. 4–6)."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor

EXPOSURE_CLAMP = (1e-6, 10.0)
GAMMA_PRIOR = 2.2


def srgb_to_linear(x: Tensor) -> Tensor:
    """sRGB EOTF inverse (piecewise)."""
    x = x.clamp(0.0, 1.0)
    a = 0.055
    return torch.where(x <= 0.04045, x / 12.92, ((x + a) / (1.0 + a)).pow(2.4))


def linear_to_srgb(x: Tensor) -> Tensor:
    """sRGB OETF."""
    x = x.clamp(min=0.0)
    a = 0.055
    return torch.where(x <= 0.0031308, x * 12.92, (1.0 + a) * x.pow(1.0 / 2.4) - a).clamp(0.0, 1.0)


def apply_exposure(linear_hdr: Tensor, exposure: Tensor | float) -> Tensor:
    """``I_exposed = e · Î_linear`` (Eq. 4), with clamp before tone map."""
    e = exposure if isinstance(exposure, Tensor) else torch.tensor(exposure, device=linear_hdr.device, dtype=linear_hdr.dtype)
    return (e * linear_hdr).clamp(EXPOSURE_CLAMP[0], EXPOSURE_CLAMP[1])


def tone_map_gamma(linear_exposed: Tensor, gamma: Tensor | float) -> Tensor:
    """``T(x) = clamp(x^{1/γ}, 0, 1)`` (Eq. 5)."""
    g = gamma if isinstance(gamma, Tensor) else torch.tensor(gamma, device=linear_exposed.device, dtype=linear_exposed.dtype)
    g = g.clamp(min=1e-3)
    return linear_exposed.clamp(min=0.0).pow(1.0 / g).clamp(0.0, 1.0)


def render_ldr(linear_hdr: Tensor, *, exposure: Tensor | float, gamma: Tensor | float) -> Tensor:
    """Full LDR pipeline: exposure then per-view gamma."""
    return tone_map_gamma(apply_exposure(linear_hdr, exposure), gamma)


class ViewPhotometricParams(nn.Module):
    """Learnable per-view exposure ``e_i`` and tone ``γ_i`` (Sec. 3.3)."""

    def __init__(self, num_views: int, *, gamma_prior: float = GAMMA_PRIOR) -> None:
        super().__init__()
        self.num_views = num_views
        self.gamma_prior = gamma_prior
        # e_i ~ N(1, 0.05^2) via log-space parameterization for positivity
        self._log_exposure = nn.Parameter(torch.zeros(num_views))
        self._log_gamma = nn.Parameter(torch.full((num_views,), math.log(gamma_prior)))

    @property
    def exposure(self) -> Tensor:
        return torch.exp(self._log_exposure).clamp(min=EXPOSURE_CLAMP[0], max=EXPOSURE_CLAMP[1])

    @property
    def gamma(self) -> Tensor:
        return torch.exp(self._log_gamma).clamp(min=1.0, max=4.0)

    def exposure_view(self, view_idx: int) -> Tensor:
        return self.exposure[view_idx]

    def gamma_view(self, view_idx: int) -> Tensor:
        return self.gamma[view_idx]

    def mean_render_params(self) -> tuple[Tensor, Tensor]:
        """Eq. (6): average training views for stable inference."""
        return self.exposure.mean(), self.gamma.mean()

    def render_view(self, linear_hdr: Tensor, view_idx: int) -> Tensor:
        return render_ldr(linear_hdr, exposure=self.exposure_view(view_idx), gamma=self.gamma_view(view_idx))

    def regularization(self) -> Tensor:
        """``L_reg`` terms (Eq. 10) without external weights."""
        e = self.exposure
        g = self.gamma
        l_scale = ((e - 1.0) ** 2).mean()
        l_var = e.var(unbiased=False)
        l_gamma = ((g - self.gamma_prior) ** 2).mean()
        return l_scale, l_var, l_gamma
