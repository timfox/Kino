"""Mesh VQ-VAE pose tokens + trajectory MLP → motion tokens (Eq. 1–4)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ltx_trainer.mesh_token.config import MeshTokenConfig
from ltx_trainer.mesh_token.smpl import pack_trajectory


class PoseTokenRefiner(nn.Module):
    """Self-attention over pose token sequence + linear proj (Eq. 2)."""

    def __init__(self, cfg: MeshTokenConfig) -> None:
        super().__init__()
        d = cfg.pose_token_dim
        hidden = 512
        self.proj_in = nn.Linear(d, hidden)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=16,
            dim_feedforward=hidden * 4,
            batch_first=True,
            activation="gelu",
        )
        self.sa = nn.TransformerEncoder(layer, num_layers=5)
        self.proj_out = nn.Linear(hidden, cfg.dit_hidden)

    def forward(self, zp: Tensor) -> Tensor:
        # zp: B, T, N, L
        b, t, n, l = zp.shape
        x = zp.reshape(b * t, n, l)
        x = self.proj_in(x)
        x = x + self.sa(x)
        return self.proj_out(x).reshape(b, t, n, -1)


class TrajectoryMLP(nn.Module):
    """MLP trajectory embedding Zt (Eq. 3)."""

    def __init__(self, traj_dim: int = 24, hidden: int = 5120) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(traj_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )

    def forward(self, traj_feat: Tensor) -> Tensor:
        return self.net(traj_feat)


class MeshVQEncoderStub(nn.Module):
    """Stub mesh encoder → discrete pose latents before codebook lookup."""

    def __init__(self, cfg: MeshTokenConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder = nn.Linear(cfg.smpl_vertices * 3, cfg.pose_tokens * cfg.pose_token_dim)
        self.codebook = nn.Embedding(cfg.codebook_size, cfg.pose_token_dim)

    def forward(self, canonical_vertices: Tensor) -> Tensor:
        b, t, v, _ = canonical_vertices.shape
        n, l = self.cfg.pose_tokens, self.cfg.pose_token_dim
        flat = canonical_vertices.reshape(b, t, v * 3)
        zp = self.encoder(flat).reshape(b, t, n, l)
        flat_z = zp.reshape(b * t * n, l)
        codes = torch.cdist(flat_z, self.codebook.weight).argmin(dim=-1)
        return self.codebook(codes).reshape(b, t, n, l)


class MotionTokenizer(nn.Module):
    """Full motion tokenization: Zm = Zt + Proj(Zp + SA(Zp)) (Eq. 4)."""

    def __init__(self, cfg: MeshTokenConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or MeshTokenConfig()
        self.mesh_vq = MeshVQEncoderStub(self.cfg)
        self.pose_refiner = PoseTokenRefiner(self.cfg)
        self.traj_mlp = TrajectoryMLP(hidden=self.cfg.dit_hidden)

    def forward(
        self,
        canonical_vertices: Tensor,
        gamma_world: Tensor,
        rot_world: Tensor,
        gamma_cam: Tensor,
        rot_cam: Tensor,
    ) -> Tensor:
        zp = self.mesh_vq(canonical_vertices)
        z_hat_p = self.pose_refiner(zp)
        traj = pack_trajectory(gamma_world, rot_world, gamma_cam, rot_cam)
        zt = self.traj_mlp(traj).unsqueeze(2)
        return zt + z_hat_p
