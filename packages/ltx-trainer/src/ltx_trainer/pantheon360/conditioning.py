"""Geometric and semantic conditioning for 360° video diffusion (Sec. 3.3–3.4)."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor


def encode_geometry_scaffold(v_geo: Tensor, *, latent_dim: int = 4) -> Tensor:
    """``v_equi = E(V_geo)`` — stub VAE encode via spatial pool + conv projection."""
    # v_geo: (T, H, W, 3) or (T, 3, H, W)
    if v_geo.shape[-1] == 3:
        x = v_geo.permute(0, 3, 1, 2)
    else:
        x = v_geo
    pooled = F.adaptive_avg_pool2d(x, (32, 64))
    return pooled[:, :latent_dim]


def concat_geometric_latent(
    y_equi_t: Tensor,
    v_equi: Tensor,
) -> Tensor:
    """Concatenate noised latent with geometric scaffold along channel dim."""
    if v_equi.shape[0] != y_equi_t.shape[0]:
        v_equi = v_equi.expand(y_equi_t.shape[0], *v_equi.shape[1:])
    return torch.cat([y_equi_t, v_equi], dim=1)


def clip_perspective_crops(
    frame_erp: Tensor,
    *,
    num_crops: int = 8,
    crop_size: int = 224,
) -> Tensor:
    """Eight perspective crops every 45° yaw for CLIP conditioning (Sec. 3.3)."""
    if frame_erp.dim() != 3:
        raise ValueError("frame_erp must be H×W×3")
    h, w, _ = frame_erp.shape
    crops: list[Tensor] = []
    for i in range(num_crops):
        yaw = 2 * math.pi * i / num_crops
        shift = int((yaw / (2 * math.pi)) * w) % w
        strip = torch.roll(frame_erp, shifts=shift, dims=1)
        cy, cx = h // 2, w // 4
        y0 = max(0, cy - crop_size // 2)
        x0 = max(0, cx - crop_size // 2)
        patch = strip[y0 : y0 + crop_size, x0 : x0 + crop_size]
        if patch.shape[0] < crop_size or patch.shape[1] < crop_size:
            patch = F.interpolate(
                patch.permute(2, 0, 1).unsqueeze(0),
                size=(crop_size, crop_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0).permute(1, 2, 0)
        crops.append(patch)
    return torch.stack(crops, dim=0)


def semantic_condition_vector(crops: Tensor, *, out_dim: int = 32) -> Tensor:
    """Stub CLIP stack: mean-pool crops → linear-ish projection."""
    feat = crops.mean(dim=(0, 1, 2))
    if feat.numel() >= out_dim:
        return feat[:out_dim]
    pad = torch.zeros(out_dim - feat.numel())
    return torch.cat([feat, pad], dim=0)


def diffusion_loss_stub(
    epsilon: Tensor,
    v_pred: Tensor,
    *,
    weight: float = 1.0,
) -> Tensor:
    """``L = ||ε - f_θ(y_equi,t, t, v_equi, c_img)||²`` (Sec. 3.4)."""
    return weight * F.mse_loss(v_pred, epsilon)
