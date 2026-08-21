"""4D RoPE via low-frequency spatial band repurposing (Sec. 4.2, Eq. 3–7)."""

from __future__ import annotations

import math

import torch
from torch import Tensor

from ltx_trainer.helix4d.config import Helix4DConfig


def rope_frequencies(num_pairs: int, theta: float = 10_000.0) -> Tensor:
    """ω_i = θ^{-2(i-1)/M} for i ∈ {1,…,M}."""
    i = torch.arange(num_pairs, dtype=torch.float32)
    return theta ** (-2.0 * i / max(num_pairs, 1))


def rotation_matrix_1d(pos: Tensor, freqs: Tensor) -> Tensor:
    """Block-diagonal SO(2) rotations; returns (2M, 2M) for M frequency pairs."""
    angles = pos.reshape(()).float() * freqs
    cos = torch.cos(angles)
    sin = torch.sin(angles)
    pairs = []
    for i in range(freqs.numel()):
        pairs.append(torch.tensor([[cos[i], -sin[i]], [sin[i], cos[i]]], dtype=freqs.dtype))
    return torch.block_diag(*pairs)


def spatial_rope_3d(
    coords: Tensor,
    head_dim: int,
    *,
    theta: float = 10_000.0,
) -> Tensor:
    """Eq. (4): R_Ω,p for voxel coords (N, 3). Returns (N, head_dim, head_dim)."""
    if coords.dim() == 1:
        coords = coords.unsqueeze(0)
    m = head_dim // 6
    freqs = rope_frequencies(m, theta)
    mats: list[Tensor] = []
    for i in range(coords.shape[0]):
        axis_blocks = [rotation_matrix_1d(coords[i, axis].float(), freqs) for axis in range(3)]
        mats.append(torch.block_diag(*axis_blocks))
    return torch.stack(mats, dim=0)


def split_spatial_rope(
    rot: Tensor,
    alpha: float,
) -> tuple[Tensor, Tensor]:
    """Eq. (6): split per-axis blocks into high-frequency (kept) and low (identity)."""
    d = rot.shape[-1]
    m_total = d // 2
    m_keep = max(1, int(math.ceil(alpha * (m_total // 3))))
    keep_dim = m_keep * 2 * 3
    high = rot[..., :keep_dim, :keep_dim]
    low_size = d - keep_dim
    ident = torch.eye(low_size, device=rot.device, dtype=rot.dtype)
    ident = ident.expand(rot.shape[:-2] + (low_size, low_size))
    return high, ident


def temporal_rope_1d(
    frame_idx: Tensor,
    num_pairs: int,
    *,
    theta: float = 10_000.0,
    scale: float = 1.0,
) -> Tensor:
    """Temporal block on repurposed bands; frequencies scaled by N/T."""
    freqs = rope_frequencies(num_pairs, theta) * scale
    return rotation_matrix_1d(frame_idx.float(), freqs)


def rope_4d(
    voxel_coords: Tensor,
    frame_idx: Tensor,
    cfg: Helix4DConfig,
) -> Tensor:
    """Eq. (7): R_Ω,(p,t) = R_spatial ⊕ R_temporal with α split."""
    head_dim = cfg.head_dim
    spatial_all = spatial_rope_3d(voxel_coords, head_dim, theta=cfg.rope_theta)
    spatial = spatial_all[0]
    high, _ = split_spatial_rope(spatial, cfg.rope_alpha)
    d = head_dim
    keep_dim = high.shape[-1]
    t_pairs = (d - keep_dim) // 2
    t_scale = cfg.voxel_grid / max(cfg.num_frames, 1)
    temporal = temporal_rope_1d(
        frame_idx,
        max(t_pairs, 1),
        theta=cfg.rope_theta,
        scale=1.0 / t_scale,
    )
    td = temporal.shape[-1]
    full = torch.zeros(*spatial.shape[:-2], d, d, device=spatial.device, dtype=spatial.dtype)
    full[..., :keep_dim, :keep_dim] = high
    full[..., keep_dim : keep_dim + td, keep_dim : keep_dim + td] = temporal
    full[..., keep_dim + td :, keep_dim + td :] = torch.eye(
        d - keep_dim - td, device=spatial.device, dtype=spatial.dtype
    )
    return full


def apply_rope(x: Tensor, rot: Tensor) -> Tensor:
    """Apply rotary embedding: R @ x for feature vectors."""
    return torch.matmul(rot, x.unsqueeze(-1)).squeeze(-1)
