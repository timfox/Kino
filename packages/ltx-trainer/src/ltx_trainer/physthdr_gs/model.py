"""PhysHDR-GS dual-branch model (Fig. 2)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

from ltx_trainer.physthdr_gs.cameras import Camera
from ltx_trainer.physthdr_gs.config import PhysHDRConfig
from ltx_trainer.physthdr_gs.gaussians import HDRGaussianField
from ltx_trainer.physthdr_gs.radiance import IlluminationModulator, RadianceComposer
from ltx_trainer.physthdr_gs.rasterize import render_perspective_hdr
from ltx_trainer.physthdr_gs.render import splat_gaussians
from ltx_trainer.physthdr_gs.tonemap import ToneMapper


@dataclass
class PhysHDRForward:
    ihdr: Tensor
    ihdr_scaled: Tensor
    ihdr_relit: Tensor
    ildr: Tensor
    iig: Tensor
    igi: Tensor
    la: Tensor
    la_hat: Tensor
    colors: Tensor
    colors_relit: Tensor
    viewspace: Tensor | None = None


class PhysHDRGS(nn.Module):
    """Physically inspired HDR-NVS with IE and GI branches."""

    def __init__(
        self,
        cfg: PhysHDRConfig | None = None,
        *,
        field: HDRGaussianField | None = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg or PhysHDRConfig()
        h = self.cfg.hidden_dim
        self.field = field or HDRGaussianField(
            self.cfg.num_gaussians,
            init_radius=self.cfg.init_radius,
        )
        self.field._max_points = self.cfg.max_gaussians
        self.composer = RadianceComposer(h)
        self.modulator = IlluminationModulator(h)
        self.tone_mapper = ToneMapper(h)
        self._iteration = 0

    @property
    def fmix_frozen(self) -> bool:
        return self._iteration < self.cfg.freeze_fmix_iters

    def set_iteration(self, it: int) -> None:
        self._iteration = it
        if self.fmix_frozen:
            self.tone_mapper.freeze_fmix()
        else:
            self.tone_mapper.unfreeze_fmix()

    def _rasterize(self, colors: Tensor, camera: Camera | None) -> tuple[Tensor, Tensor | None]:
        if camera is not None and self.cfg.use_perspective:
            xyz = self.field.get_xyz()
            img, viewspace = render_perspective_hdr(
                xyz,
                self.field.get_scales(),
                self.field.get_opacity(),
                colors,
                camera,
                max_gaussians=self.cfg.max_gaussians_render,
                fast=self.cfg.fast_raster,
            )
            return img, viewspace
        img = splat_gaussians(
            self.field.get_xyz(),
            colors,
            self.field.get_opacity(),
            image_size=self.cfg.image_size,
        )
        return img, None

    def forward(
        self,
        exposure: Tensor,
        lighting_level: Tensor | None = None,
        *,
        camera: Camera | None = None,
    ) -> PhysHDRForward:
        la = self.field.get_la()
        hr = self.field.get_hr()
        l = lighting_level if lighting_level is not None else exposure

        colors = self.composer(la, hr)
        ihdr, viewspace = self._rasterize(colors, camera)
        t = exposure if exposure.dim() > 0 else exposure.reshape(1)
        if t.numel() == 1:
            ihdr_scaled = ihdr * t
        else:
            ihdr_scaled = ihdr * t.view(-1, 1, 1, 1)

        if self.cfg.use_gi_branch:
            la_hat = self.modulator(la, l if isinstance(l, Tensor) else torch.tensor(l, device=la.device))
            colors_relit = self.composer(la_hat, hr)
            ihdr_relit, _ = self._rasterize(colors_relit, camera)
        else:
            la_hat = la
            colors_relit = colors
            ihdr_relit = ihdr_scaled

        ildr, iig, igi, _ = self.tone_mapper(
            ihdr_scaled,
            ihdr_relit,
            fmix_enabled=not self.fmix_frozen,
        )
        return PhysHDRForward(
            ihdr=ihdr,
            ihdr_scaled=ihdr_scaled,
            ihdr_relit=ihdr_relit,
            ildr=ildr,
            iig=iig,
            igi=igi,
            la=la,
            la_hat=la_hat,
            colors=colors,
            colors_relit=colors_relit,
            viewspace=viewspace,
        )
