"""LightHarmony3D end-to-end insertion pipeline (Fig. 2)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

from ltx_trainer.lightharmony3d.config import LightHarmony3DConfig
from ltx_trainer.lightharmony3d.gen_env import GenEnvLighting
from ltx_trainer.lightharmony3d.hdr_fusion import build_hdr_envmap
from ltx_trainer.lightharmony3d.panorama import build_ev0_panorama
from ltx_trainer.lightharmony3d.pbr_stub import pbr_render_pair, render_object_layer
from ltx_trainer.lightharmony3d.ray_decoupled import mix_bsdf
from ltx_trainer.lightharmony3d.shadow import composite_with_3dgs, raw_shadow_ratio, shape_shadow_ratio


@dataclass
class LightHarmony3DOutput:
    composite: Tensor
    hdr_env: Tensor
    ev0: Tensor
    shadow_hat: Tensor


class LightHarmony3D(nn.Module):
    """Mesh insertion with GenEnvLighting + PBR shadow compositing over 3DGS."""

    def __init__(self, cfg: LightHarmony3DConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or LightHarmony3DConfig()
        self.gen_env = GenEnvLighting()

    def estimate_hdr(
        self,
        scene_rgb: Tensor,
    ) -> tuple[Tensor, Tensor]:
        ev0 = build_ev0_panorama(scene_rgb, out_h=self.cfg.panorama_h, out_w=self.cfg.panorama_w)
        if not self.cfg.use_gen_env:
            return ev0, ev0

        evs = list(self.cfg.ev_sequence)
        under: list[Tensor] = []
        current = ev0
        for target_ev in evs[:-1]:
            current = self.gen_env(current)
            under.append(current)
        if self.cfg.use_hdr_fusion and len(under) > 0:
            hdr = build_hdr_envmap(ev0, under, tuple(evs))
        else:
            hdr = ev0
        return ev0, hdr

    def forward(
        self,
        background: Tensor,
        object_rgb: Tensor,
        object_mask: Tensor,
        *,
        scene_rgb: Tensor | None = None,
    ) -> LightHarmony3DOutput:
        if background.dim() == 3:
            background = background.unsqueeze(0)
        if object_rgb.dim() == 3:
            object_rgb = object_rgb.unsqueeze(0)
        if object_mask.dim() == 3:
            object_mask = object_mask.unsqueeze(0)

        scene = scene_rgb if scene_rgb is not None else background
        if scene.dim() == 3:
            scene = scene.unsqueeze(0)

        ev0, hdr = self.estimate_hdr(scene)

        if self.cfg.use_ray_decoupled:
            _ = mix_bsdf(1.0, 0.0, ray_type="camera")
            _ = mix_bsdf(1.0, 0.0, ray_type="shadow")

        r0 = pbr_render_pair(background, object_mask, hdr, with_object=False)
        r1 = pbr_render_pair(background, object_mask, hdr, with_object=True)
        obj = render_object_layer(background.shape, object_rgb, object_mask, hdr)

        if self.cfg.use_shadow_ratio:
            s_raw = raw_shadow_ratio(r0, r1)
            s_hat = shape_shadow_ratio(
                s_raw,
                gamma=self.cfg.shadow_gamma,
                smin=self.cfg.shadow_smin,
                lam=self.cfg.shadow_lambda,
            )
        else:
            s_hat = torch.ones_like(background)

        comp = composite_with_3dgs(background, obj, object_mask, s_hat)
        return LightHarmony3DOutput(composite=comp, hdr_env=hdr, ev0=ev0, shadow_hat=s_hat)
