"""Spherical Spatial Cross-Attention — SSCA (Eq. 7–10)."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from ltx_trainer.panoworld.config import PanoWorldConfig
from ltx_trainer.panoworld.erp_geometry import patch_spherical_directions
from ltx_trainer.panoworld.spherical_encoding import sinusoidal_spherical_encoding


class SphericalSpatialCrossAttention(nn.Module):
    """
    Inject spherical geometry after patch embedding.

    A = MHA(Q=LN(H0), K=LN(S), V=LN(S));  eH0 = H0 + α ⊙ A.
    """

    def __init__(self, cfg: PanoWorldConfig) -> None:
        super().__init__()
        d = cfg.hidden_dim
        enc_dim = 4 * cfg.spherical_freq_bands
        self.sphere_mlp = nn.Sequential(
            nn.Linear(enc_dim, d),
            nn.GELU(),
            nn.Linear(d, d),
        )
        self.attn = nn.MultiheadAttention(d, cfg.num_heads, batch_first=True)
        self.ln_q = nn.LayerNorm(d)
        self.ln_kv = nn.LayerNorm(d)
        self.gate = nn.Parameter(torch.full((d,), cfg.gate_init))
        self._cfg = cfg

    def _build_spherical_tokens(self, b: int, device: torch.device) -> Tensor:
        rays = patch_spherical_directions(
            self._cfg.height,
            self._cfg.width,
            self._cfg.patch_size,
            device=device,
        )
        # recover λ, φ from unit ray
        x, y, z = rays.unbind(-1)
        lam = torch.atan2(x, z)
        phi = torch.asin(y.clamp(-1.0 + 1e-6, 1.0 - 1e-6))
        gamma = sinusoidal_spherical_encoding(lam, phi, self._cfg.spherical_freq_bands)
        s = self.sphere_mlp(gamma)  # [N, d]
        return s.unsqueeze(0).expand(b, -1, -1)

    def forward(self, h0: Tensor) -> Tensor:
        """
        Args:
            h0: patch embeddings [B, N, d]
        Returns:
            geometry-enhanced tokens [B, N, d]
        """
        b = h0.shape[0]
        s = self._build_spherical_tokens(b, h0.device)
        q = self.ln_q(h0)
        kv = self.ln_kv(s)
        attn_out, _ = self.attn(q, kv, kv)
        return h0 + self.gate * attn_out
