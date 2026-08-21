"""Synthetic SI-HDR bracket bursts for training smoke."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from ltx_trainer.lucky_hdr.capture_ae import bracket_evs
from ltx_trainer.lucky_hdr.tonemap import tone_map_mu


def _apply_shake(img: Tensor, *, px: float, seed: int) -> Tensor:
    if px <= 0:
        return img
    g = torch.Generator(device=img.device)
    g.manual_seed(seed)
    dx = (torch.rand((), generator=g, device=img.device) * 2 - 1) * px
    dy = (torch.rand((), generator=g, device=img.device) * 2 - 1) * px
    h, w = img.shape[-2:]
    yy, xx = torch.meshgrid(
        torch.arange(h, device=img.device, dtype=img.dtype),
        torch.arange(w, device=img.device, dtype=img.dtype),
        indexing="ij",
    )
    grid_x = 2.0 * (xx + dx) / max(w - 1, 1) - 1.0
    grid_y = 2.0 * (yy + dy) / max(h - 1, 1) - 1.0
    grid = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0)
    return F.grid_sample(img.unsqueeze(0), grid, mode="bilinear", padding_mode="border", align_corners=True).squeeze(0)


def synthesize_bracket_burst(
    hdr: Tensor,
    *,
    num_frames: int = 3,
    shake_px: float = 1.5,
    seed: int = 0,
) -> tuple[Tensor, Tensor, list[float]]:
    """Build linear bracket stack, tone-mapped GT, and EV list."""
    evs = bracket_evs(n=num_frames, span=2.0)
    ref = evs[num_frames // 2]
    stack = torch.stack([(hdr * (2.0 ** (ev - ref))).clamp(0.0, 1.0) for ev in evs], dim=0)
    shaken = torch.stack([_apply_shake(stack[i], px=shake_px, seed=seed + i) for i in range(num_frames)], dim=0)
    gt = tone_map_mu(hdr.clamp(0.0, 1.0))
    return shaken, gt, evs
