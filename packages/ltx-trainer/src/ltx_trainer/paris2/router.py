"""DiT-B router for per-step expert selection (Sec. 3.1)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ltx_trainer.paris2.config import Paris2Config


def pool_latent(x: Tensor) -> Tensor:
    """Global mean pool over spatial/temporal dims → (B, C)."""
    dims = tuple(range(2, x.dim()))
    return x.mean(dim=dims)


class Paris2Router(nn.Module):
    """Reads noisy latent, timestep, pooled CLIP (+ optional DINO) → expert weights."""

    def __init__(self, cfg: Paris2Config) -> None:
        super().__init__()
        self.cfg = cfg
        in_dim = cfg.latent_channels + 1 + cfg.clip_dim
        if cfg.dino_dim > 0:
            in_dim += cfg.dino_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, cfg.router_hidden),
            nn.SiLU(),
            nn.Linear(cfg.router_hidden, cfg.num_experts),
        )

    def _router_features(
        self,
        x_t: Tensor,
        t: Tensor,
        clip_pooled: Tensor,
        *,
        dino_first_frame: Tensor | None = None,
    ) -> Tensor:
        b = x_t.shape[0]
        pooled = pool_latent(x_t)
        t_feat = t.view(b, 1).to(pooled.dtype)
        feats: list[Tensor] = [pooled, t_feat, clip_pooled]
        if self.cfg.dino_dim > 0:
            if dino_first_frame is None:
                feats.append(
                    torch.zeros(b, self.cfg.dino_dim, device=pooled.device, dtype=pooled.dtype)
                )
            else:
                feats.append(dino_first_frame)
        return torch.cat(feats, dim=-1)

    def forward(
        self,
        x_t: Tensor,
        t: Tensor,
        clip_pooled: Tensor,
        *,
        dino_first_frame: Tensor | None = None,
    ) -> Tensor:
        """Return routing weights (B, E) summing to 1."""
        logits = self.mlp(self._router_features(x_t, t, clip_pooled, dino_first_frame=dino_first_frame))
        return F.softmax(logits, dim=-1)

    def cluster_logits(
        self,
        x_t: Tensor,
        t: Tensor,
        clip_pooled: Tensor,
    ) -> Tensor:
        """Stage 1 supervised router: cluster classification logits."""
        return self.mlp(self._router_features(x_t, t, clip_pooled))
