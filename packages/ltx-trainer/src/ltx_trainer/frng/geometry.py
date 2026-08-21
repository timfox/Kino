"""Latent-interpolated fine-grained geometry (paper Sec. 4.1, Eq. 5–8).

Trainable tokenizer / de-tokenizer are **MLP stubs** here: real F-RNG feeds patchified IDM G-buffers
and RelitLRM tokens. This module implements bilinear token upsampling on a 2D token grid and a
conditioned fine-geometry MLP output (e.g. extra Gaussian parameter channels).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from ltx_trainer.frng.config import FRNGConfig


def tokens_to_grid(tokens: Tensor, ph: int, pw: int) -> Tensor:
    """(B, P, D) → (B, Ph, Pw, D) with P = Ph*Pw."""
    b, p, d = tokens.shape
    if ph * pw != p:
        raise ValueError(f"token count {p} != grid {ph}x{pw}")
    return tokens.view(b, ph, pw, d)


def grid_to_tokens(grid: Tensor) -> Tensor:
    """(B, Ph, Pw, D) → (B, P, D)."""
    b, ph, pw, d = grid.shape
    return grid.reshape(b, ph * pw, d)


def bilinear_corner_aggregate(grid: Tensor, flat_idx: Tensor) -> Tensor:
    """For each flat patch index, average the four corner tokens (2×2 neighborhood) on the token grid.

    ``grid`` (B, Ph, Pw, D); ``flat_idx`` (B, K) in [0, Ph*Pw).
    """
    b, ph, pw, d = grid.shape
    ih = flat_idx // pw
    iw = flat_idx % pw
    out = []
    for di in (0, 1):
        for dj in (0, 1):
            nh = (ih + di).clamp(max=ph - 1)
            nw = (iw + dj).clamp(max=pw - 1)
            idx = nh * pw + nw
            gathered = torch.gather(grid.reshape(b, ph * pw, d), 1, idx.unsqueeze(-1).expand(-1, -1, d))
            out.append(gathered)
    return torch.stack(out, dim=0).mean(dim=0)


class PriorTokenizer(nn.Module):
    """Eq. (5): single-layer MLP on patchified IDM channels (caller supplies flattened patches)."""

    def __init__(self, in_channels: int, token_dim: int) -> None:
        super().__init__()
        self.net = nn.Linear(in_channels, token_dim)

    def forward(self, patch_features: Tensor) -> Tensor:
        """``patch_features`` (B, P, C_in) → tokens (B, P, D)."""
        return self.net(patch_features)


class FineGeoDetokenizer(nn.Module):
    """Eq. (8): ``G_fine = FineGeoDetokenizer(T_fine_geo | T_fine_prior)`` — single MLP block."""

    def __init__(self, token_dim: int, out_dim: int, *, hidden_mult: int = 4) -> None:
        super().__init__()
        h = max(hidden_mult * token_dim, 32)
        self.net = nn.Sequential(
            nn.Linear(2 * token_dim, h),
            nn.GELU(),
            nn.Linear(h, out_dim),
        )

    def forward(self, t_fine_geo: Tensor, t_fine_prior: Tensor) -> Tensor:
        """Both (B, K, D) → (B, K, out_dim)."""
        return self.net(torch.cat([t_fine_geo, t_fine_prior], dim=-1))


class FineGeometrySynthesis(nn.Module):
    """Top-K salient patches → bilinear token pairs → fine Gaussians (feature vectors)."""

    def __init__(self, cfg: FRNGConfig, *, idm_in_channels: int, fine_out_dim: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.prior_tokenizer = PriorTokenizer(idm_in_channels, cfg.token_dim)
        self.detokenizer = FineGeoDetokenizer(cfg.token_dim, fine_out_dim, hidden_mult=cfg.fine_geo_hidden_mult)

    def forward(
        self,
        t_geo_grid: Tensor,
        idm_patch_features: Tensor,
        salient_flat_idx: Tensor,
        *,
        ph: int,
        pw: int,
    ) -> Tensor:
        """``t_geo_grid`` (B, P, D); ``idm_patch_features`` (B, P, C_in); ``salient_flat_idx`` (B, K)."""
        g = tokens_to_grid(t_geo_grid, ph, pw)
        t_prior = self.prior_tokenizer(idm_patch_features)
        pg = tokens_to_grid(t_prior, ph, pw)
        t_fine_geo = bilinear_corner_aggregate(g, salient_flat_idx)
        t_fine_prior = bilinear_corner_aggregate(pg, salient_flat_idx)
        return self.detokenizer(t_fine_geo, t_fine_prior)
