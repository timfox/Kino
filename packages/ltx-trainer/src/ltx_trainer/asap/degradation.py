"""Localized degradation for HAP preference pairs (Sec. 4.1, Fig. 2)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def binary_anatomy_mask(
    height: int,
    width: int,
    *,
    region: str = "hands",
    device: torch.device | None = None,
) -> Tensor:
    """Placeholder mask generator when GroundingSAM2 is external.

    Returns (1, H, W) float mask in {0, 1} for coarse hand/face bands.
    """
    m = torch.zeros(1, height, width, device=device)
    if region in ("hands", "hand"):
        m[:, int(0.55 * height) :, int(0.15 * width) : int(0.45 * width)] = 1.0
        m[:, int(0.55 * height) :, int(0.55 * width) : int(0.85 * width)] = 1.0
    elif region in ("face", "head"):
        m[:, : int(0.35 * height), int(0.3 * width) : int(0.7 * width)] = 1.0
    else:
        m[:, int(0.4 * height) :, :] = 1.0
    return m


def image_space_degraded_negative(
    positive: Tensor,
    corrupted_patch: Tensor,
    mask: Tensor,
) -> Tensor:
    """Blend corrupted anatomy into ``positive`` only where ``mask`` is 1 (image-space degradation)."""
    m = mask.to(dtype=positive.dtype, device=positive.device)
    if positive.dim() == 3:
        # (C, H, W): mask is (1, H, W) or (H, W)
        if m.dim() == 2:
            m = m.unsqueeze(0)
        if m.shape[0] == 1:
            m = m.expand(positive.shape[0], -1, -1)
    elif positive.dim() == 4:
        if m.dim() == 3:
            m = m.unsqueeze(0)
        if m.shape[0] == 1 and positive.shape[0] > 1:
            m = m.expand(positive.shape[0], -1, -1, -1)
        if m.shape[1] == 1 and positive.shape[1] > 1:
            m = m.expand(-1, positive.shape[1], -1, -1)
    else:
        raise ValueError("positive must be (C,H,W) or (B,C,H,W)")
    if m.shape[-2:] != positive.shape[-2:]:
        m = F.interpolate(m.unsqueeze(0), size=positive.shape[-2:], mode="nearest").squeeze(0)
    return positive * (1.0 - m) + corrupted_patch * m


def corrupt_skeleton_connections(
    keypoints: Tensor,
    *,
    finger_extra: int = 1,
) -> Tensor:
    """Skeleton-space degradation: duplicate fingertip keypoints (polydactyly proxy).

    ``keypoints``: (N, K, 2) or (K, 2) normalized xy.
    """
    kp = keypoints.clone()
    if kp.dim() == 2:
        kp = kp.unsqueeze(0)
    # Perturb last keypoints (proxy for hand joints)
    if kp.shape[1] >= 4:
        extra = kp[:, -1:, :].clone()
        offset = torch.tensor([0.02, 0.0], device=kp.device, dtype=kp.dtype)
        extra = extra + offset
        kp = torch.cat([kp, extra.repeat(1, finger_extra, 1)], dim=1)
    return kp.squeeze(0) if keypoints.dim() == 2 else kp


def _align_mask_to_tensor(mask: Tensor, ref: Tensor) -> Tensor:
    """Broadcast ``mask`` (H,W) or (1,H,W) to match ``ref`` (C,H,W) or (B,C,H,W)."""
    m = mask
    if ref.dim() == 3 and m.dim() == 2:
        m = m.unsqueeze(0).expand(ref.shape[0], -1, -1)
    elif ref.dim() == 4 and m.dim() == 2:
        m = m.unsqueeze(0).unsqueeze(0).expand(ref.shape[0], ref.shape[1], -1, -1)
    elif ref.dim() == 4 and m.dim() == 3 and m.shape[0] == 1:
        m = m.expand(ref.shape[0], ref.shape[1], -1, -1)
    return m


def region_mse(
    a: Tensor,
    b: Tensor,
    mask: Tensor,
) -> float:
    """Mean squared error inside ``mask`` (proxy for region LPIPS/SSIM checks)."""
    m = _align_mask_to_tensor(mask, a).bool()
    if m.sum() == 0:
        return 0.0
    diff = (a - b).pow(2)
    return float(diff[m].mean().item())


def background_mse(
    a: Tensor,
    b: Tensor,
    mask: Tensor,
) -> float:
    """MSE outside anatomical mask (should stay low for valid HAP pairs)."""
    m = _align_mask_to_tensor(mask, a).bool()
    inv = ~m
    if inv.sum() == 0:
        return 0.0
    diff = (a - b).pow(2)
    return float(diff[inv].mean().item())
