"""Camera Conditioning Module — group-disentangled FiLM on spatial latents (Fig. 5)."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from ltx_trainer.deltacam.config import DeltaCamConfig
from ltx_trainer.deltacam.film import DeltaFiLM, cascade_film


class CameraConditioningModule(nn.Module):
    """Modulate VAE-encoded scene proxies with optical / sensory / ISP Δ groups."""

    def __init__(
        self,
        feature_dim: int,
        cfg: DeltaCamConfig | None = None,
        *,
        hidden: int = 128,
    ) -> None:
        super().__init__()
        self.cfg = cfg or DeltaCamConfig()
        self.feature_dim = feature_dim

        def _group_film(keys: tuple[str, ...]) -> DeltaFiLM:
            return DeltaFiLM(len(keys), feature_dim, hidden=hidden)

        self.optical_film = _group_film(self.cfg.optical_params)
        self.sensory_film = _group_film(self.cfg.sensory_params)
        self.isp_film = _group_film(self.cfg.isp_params)

    def _slice_delta(self, delta_full: Tensor, keys: tuple[str, ...]) -> Tensor:
        """Pick columns from full Δ vector ``[B, K]`` or ``[K]``."""
        all_keys = list(self.cfg.ranges.keys())
        idx = [all_keys.index(k) for k in keys]
        if delta_full.dim() == 1:
            return delta_full[idx]
        return delta_full[:, idx]

    def forward(self, h: Tensor, delta_intrinsics: Tensor) -> Tensor:
        """``h``: spatial features; ``delta_intrinsics``: full Δ per frame."""
        d_opt = self._slice_delta(delta_intrinsics, self.cfg.optical_params)
        d_sen = self._slice_delta(delta_intrinsics, self.cfg.sensory_params)
        d_isp = self._slice_delta(delta_intrinsics, self.cfg.isp_params)
        return cascade_film(
            h,
            [d_opt, d_sen, d_isp],
            [self.optical_film, self.sensory_film, self.isp_film],
        )


def apply_ccm(
    spatial_latent: Tensor,
    delta_t: Tensor,
    ccm: CameraConditioningModule,
) -> Tensor:
    """Functional wrapper for one timestep."""
    return ccm(spatial_latent, delta_t)
