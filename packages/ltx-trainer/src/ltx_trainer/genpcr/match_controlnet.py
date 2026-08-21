"""DepthMatch-ControlNet & LiDARMatch-ControlNet stubs (Sec. 3.3–3.4)."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from ltx_trainer.genpcr.config import GenPcrConfig
from ltx_trainer.genpcr.coupled_denoise import CoupledDenoiseStub, couple_latents, decouple_latents
from ltx_trainer.genpcr.geometry_maps import lidar_to_range_map, points_to_depth_map

COUPLED_PROMPT_DEPTH = (
    "Generate two vertically stacked images captured from different viewpoints in the same scene."
)
COUPLED_PROMPT_LIDAR = (
    "Generate two vertically stacked surround-view panoramas from a self-driving vehicle."
)


class MatchControlNetStub(nn.Module):
    """Synthesize paired RGB images from geometry conditions + optional coupling."""

    def __init__(self, cfg: GenPcrConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or GenPcrConfig()
        c = 3
        self.cond_enc = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 3, padding=1),
        )
        self.rgb_dec = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, c, 3, padding=1),
        )
        self.denoiser = CoupledDenoiseStub(16)

    def geometry_condition(
        self,
        points_p: Tensor,
        points_q: Tensor,
    ) -> tuple[Tensor, Tensor]:
        h, w = self.cfg.image_height, self.cfg.image_width
        if self.cfg.mode == "lidar":
            d_p = lidar_to_range_map(points_p, height=h, width=w)
            d_q = lidar_to_range_map(points_q, height=h, width=w)
        else:
            d_p = points_to_depth_map(points_p, height=h, width=w)
            d_q = points_to_depth_map(points_q, height=h, width=w)
        return d_p, d_q

    def forward(
        self,
        points_p: Tensor,
        points_q: Tensor,
    ) -> dict[str, Tensor]:
        d_p, d_q = self.geometry_condition(points_p, points_q)
        d_p_b = d_p.unsqueeze(0) if d_p.dim() == 3 else d_p
        d_q_b = d_q.unsqueeze(0) if d_q.dim() == 3 else d_q
        c_p = self.cond_enc(d_p_b).squeeze(0)
        c_q = self.cond_enc(d_q_b).squeeze(0)

        if self.cfg.coupled_denoise:
            z_pq = couple_latents(c_p, c_q)
            cond_pq = couple_latents(d_p_b.squeeze(0), d_q_b.squeeze(0))
            z_out = self.denoiser(
                z_pq.unsqueeze(0),
                self.cond_enc(cond_pq.unsqueeze(0)),
            ).squeeze(0)
            f_p, f_q = decouple_latents(z_out)
        else:
            f_p, f_q = c_p, c_q

        ip = self.rgb_dec(f_p.unsqueeze(0)).clamp(0, 1)
        iq = self.rgb_dec(f_q.unsqueeze(0)).clamp(0, 1)
        return {
            "image_p": ip,
            "image_q": iq,
            "depth_p": d_p,
            "depth_q": d_q,
            "prompt": COUPLED_PROMPT_LIDAR if self.cfg.mode == "lidar" else COUPLED_PROMPT_DEPTH,
        }
