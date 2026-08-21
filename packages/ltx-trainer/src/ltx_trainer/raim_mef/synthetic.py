"""Synthetic multi-exposure brackets with motion and exposure variation."""

from __future__ import annotations

import torch
from torch import Tensor

# EV stops relative to reference (0 EV)
TRAIN_EV_STOPS = (-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0)
TEST_EV_STOPS = (-2.0, -1.0, 0.0, 1.0, 2.0)


def ev_to_gain(ev: float) -> float:
    return float(2.0**ev)


def apply_exposure(rgb: Tensor, ev: float) -> Tensor:
    """Simulate bracket capture in display/sRGB space."""
    gain = ev_to_gain(ev)
    out = (rgb * gain).clamp(0.0, 1.0)
    return out


def apply_shift(rgb: Tensor, dx: float, dy: float) -> Tensor:
    """Sub-pixel shift via grid_sample (handheld jitter proxy)."""
    import torch.nn.functional as F

    _, h, w = rgb.shape
    theta = torch.tensor(
        [[1.0, 0.0, 2.0 * dx / max(w - 1, 1), 0.0, 1.0, 2.0 * dy / max(h - 1, 1)]],
        dtype=rgb.dtype,
        device=rgb.device,
    ).view(1, 2, 3)
    grid = F.affine_grid(theta, size=(1, 3, h, w), align_corners=True)
    x = rgb.unsqueeze(0)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=True).squeeze(0)


def synthesize_sequence(
    scene: Tensor,
    *,
    ev_stops: tuple[float, ...] | None = None,
    shake_px: float = 2.0,
    moving_blob: bool = True,
) -> tuple[Tensor, Tensor, tuple[float, ...]]:
    """
    Build bracket stack [N,3,H,W] and reference fusion target.

    Returns (stack, gt_fusion, ev_stops).
    """
    evs = ev_stops or TEST_EV_STOPS
    frames: list[Tensor] = []
    ref = scene.clamp(0, 1)
    h, w = ref.shape[-2:]
    for i, ev in enumerate(evs):
        dx = shake_px * (0.3 if i % 2 == 0 else -0.2) * (i + 1) / len(evs)
        dy = shake_px * (0.2 if i % 3 == 0 else -0.3) * (i + 1) / len(evs)
        frame = apply_shift(ref, dx, dy)
        if moving_blob and i > 0:
            # Local motion ghosting proxy
            blob = torch.zeros_like(frame)
            y0, x0 = h // 3 + i * 2, w // 4 + i * 3
            y1, x1 = min(h, y0 + 12), min(w, x0 + 12)
            blob[:, y0:y1, x0:x1] = 0.9
            frame = (frame * (1 - blob) + blob * (0.7 + 0.1 * i)).clamp(0, 1)
        frames.append(apply_exposure(frame, ev))
    stack = torch.stack(frames, dim=0)
    # GT: exposure fusion of aligned static scene (no local motion in gt)
    weights = torch.tensor([ev_to_gain(ev) for ev in evs], device=scene.device)
    weights = weights / weights.sum()
    gt = sum(w * apply_exposure(ref, ev) for w, ev in zip(weights, evs, strict=True))
    gt = gt.clamp(0, 1)
    return stack, gt, evs
