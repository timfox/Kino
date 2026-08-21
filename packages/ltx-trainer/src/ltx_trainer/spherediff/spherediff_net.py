"""SphereDiff MultiDiffusion denoiser stub (Fig. 3, Eq. 4–7)."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ltx_trainer.spherediff.config import SphereDiffConfig
from ltx_trainer.spherediff.fibonacci import spherical_latents_init
from ltx_trainer.spherediff.config import NUM_VIEW_DIRECTIONS
from ltx_trainer.spherediff.multiprompt import prompt_for_direction, view_direction_unit
from ltx_trainer.spherediff.sampling import map_view_latent
from ltx_trainer.spherediff.view_schedule import paper_view_directions
from ltx_trainer.spherediff.weights import fuse_views, perspective_weights


class PerspectiveDenoiserStub(nn.Module):
    """Φ: pretrained perspective diffusion surrogate."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x: Tensor, _prompt: str) -> Tensor:
        return x - 0.1 * self.net(x)


class SphereDiffStub(nn.Module):
    """Ψ_S: spherical MultiDiffuser."""

    def __init__(self, cfg: SphereDiffConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or SphereDiffConfig()
        self.denoiser = PerspectiveDenoiserStub(self.cfg.latent_dim)

    def _view_angles(self) -> list[tuple[float, float]]:
        if self.cfg.use_paper_view_schedule or self.cfg.num_views >= NUM_VIEW_DIRECTIONS:
            return paper_view_directions()
        elevs = [-45.0, 0.0, 45.0]
        azims = [i * (360.0 / max(self.cfg.num_views, 1)) for i in range(self.cfg.num_views)]
        return [(a, e) for e in elevs for a in azims[: max(1, self.cfg.num_views // 3)]]

    def psi_step(
        self,
        feats: Tensor,
        dirs: Tensor,
        prompts: list[str],
        *,
        t: int,
    ) -> Tensor:
        ph, pw = self.cfg.perspective_h, self.cfg.perspective_w
        views: list[Tensor] = []
        weights: list[Tensor] = []
        for az, el in self._view_angles():
            lat = map_view_latent(
                dirs,
                feats,
                azimuth_deg=az,
                elevation_deg=el,
                fov_deg=self.cfg.fov_deg,
                h=ph,
                w=pw,
                dynamic=self.cfg.use_dynamic_sampling,
            )
            view_dir = view_direction_unit(az, el)
            prompt = prompt_for_direction(view_dir, prompts)
            denoised = self.denoiser(lat, prompt)
            w_map = perspective_weights(ph, pw, self.cfg.weight_tau, device=feats.device)
            if not self.cfg.use_weighted_average:
                w_map = torch.ones_like(w_map)
            views.append(denoised)
            weights.append(w_map)
        fused = fuse_views(views, weights)
        # scatter-update spherical feats (simplified: global mean broadcast)
        delta = fused.mean(dim=(2, 3), keepdim=False).mean(0)
        return feats + 0.05 * delta.unsqueeze(0).expand_as(feats)

    def forward(
        self,
        prompts: list[str] | None = None,
    ) -> dict[str, Tensor]:
        prompts = prompts or ["sky", "horizon", "ground", "horizon", "sky"]
        device = next(self.parameters()).device
        dirs, feats = spherical_latents_init(self.cfg, device=device)
        for t in range(self.cfg.denoise_steps, 0, -1):
            feats = self.psi_step(feats, dirs, prompts, t=t)
        rgb = self._latents_to_erp(feats, dirs)
        return {"erp_rgb": rgb, "feats": feats, "dirs": dirs}

    def _latents_to_erp(self, feats: Tensor, dirs: Tensor) -> Tensor:
        """Paint ERP preview from spherical latents."""
        h, w = 64, 128
        erp = torch.zeros(1, 3, h, w, device=feats.device)
        for i in range(feats.shape[0]):
            x, y, z = dirs[i]
            lon = torch.atan2(x, z)
            lat = torch.asin(y.clamp(-1, 1))
            u = int(((lon + math.pi) / (2 * math.pi) * w).item()) % w
            v = int(((lat + math.pi / 2) / math.pi * h).item()) % h
            tint = torch.sigmoid(feats[i].mean()).expand(3)
            erp[0, :, v, u] = tint
        return F.interpolate(erp, size=(64, 128), mode="bilinear", align_corners=False).clamp(0, 1)
