"""Explorable Sphere-Aware DiT block (Fig. 2, Sec. 3.3)."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from ltx_trainer.panoworld_x.config import PanoWorldXConfig
from ltx_trainer.panoworld_x.erp_geometry import patch_spherical_coords
from ltx_trainer.panoworld_x.exploration_route import ExplorationRouteEncoder
from ltx_trainer.panoworld_x.spherical_attention import SphereAwareAttention, sphere_attention_mask


class ExplorableSphereAwareDiTBlock(nn.Module):
    """
    Parallel branches: frozen global self-attn stub, Exp-Attn, Sphere-Attn.
    Outputs are summed with zero-init projections on auxiliary branches.
    """

    def __init__(self, cfg: PanoWorldXConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d = cfg.hidden_dim
        self.norm = nn.LayerNorm(d)
        self.global_attn = nn.MultiheadAttention(d, num_heads=4, batch_first=True)
        if cfg.freeze_global_attn:
            for p in self.global_attn.parameters():
                p.requires_grad = False
        self.exp_proj = nn.Linear(d, d)
        nn.init.zeros_(self.exp_proj.weight)
        nn.init.zeros_(self.exp_proj.bias)
        self.exp_attn = nn.MultiheadAttention(d, num_heads=4, batch_first=True)
        self.route_encoder = ExplorationRouteEncoder(out_channels=d)
        self.sphere_attn = SphereAwareAttention(d, cfg.sphere_threshold)
        self.ffn = nn.Sequential(
            nn.Linear(d, d * 4),
            nn.GELU(),
            nn.Linear(d * 4, d),
        )
        self._mask_cache: Tensor | None = None

    def _sphere_mask(self, device: torch.device, dtype: torch.dtype) -> Tensor:
        if self._mask_cache is None:
            theta, phi = patch_spherical_coords(
                self.cfg.height,
                self.cfg.width,
                self.cfg.patch_size,
                device=device,
            )
            self._mask_cache = sphere_attention_mask(theta, phi, self.cfg.sphere_threshold)
        return self._mask_cache.to(device=device, dtype=dtype)

    def forward(self, tokens: Tensor, route: Tensor | None = None) -> Tensor:
        """
        tokens: [B, N, D]; route: optional [B, T, H, W, 6].
        """
        x = self.norm(tokens)
        global_out, _ = self.global_attn(x, x, x, need_weights=False)

        exp_out = torch.zeros_like(x)
        if self.cfg.use_exp_branch and route is not None:
            cond = self.route_encoder(route)
            cond_flat = cond.mean(dim=(2, 3, 4))
            cond_tokens = cond_flat.unsqueeze(1).expand(-1, x.shape[1], -1)
            exp_branch, _ = self.exp_attn(x + cond_tokens, x, x, need_weights=False)
            exp_out = self.exp_proj(exp_branch)

        sphere_out = torch.zeros_like(x)
        if self.cfg.use_sphere_branch:
            mask = self._sphere_mask(x.device, x.dtype)
            sphere_out = self.sphere_attn(x, mask)

        x = tokens + global_out + exp_out + sphere_out
        return x + self.ffn(self.norm(x))
