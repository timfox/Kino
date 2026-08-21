"""Synthetic HDR/LDR pairs for Fusion UNet training."""

from __future__ import annotations

import torch
from torch import Tensor

from ltx_trainer.vdp_hdr.bracket import BracketConfig, hdr_to_ldr_bracket


def synthesize_hdr_pair(
    height: int,
    width: int,
    *,
    cfg: BracketConfig | None = None,
    device: torch.device | str = "cpu",
) -> tuple[Tensor, Tensor, Tensor]:
    """Random smooth HDR, GT bracket, and a random conditioning LDR frame.

    Returns:
        ``hdr [C,H,W]``, ``ldr_cond [C,H,W]`` (one frame from bracket), ``bracket [N,C,H,W]``.
    """
    cfg = cfg or BracketConfig()
    g = torch.Generator(device=device if isinstance(device, str) else None)
    g.manual_seed(int(torch.randint(0, 2**31, (1,)).item()))
    # Smooth random radiance + hotspots
    base = torch.rand(1, 1, height, width, generator=g) * 0.4 + 0.05
    base = torch.nn.functional.interpolate(
        base,
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )
    hdr = base.repeat(1, 3, 1, 1).squeeze(0)
    spot = torch.zeros_like(hdr)
    cy, cx = height // 3, width // 3
    spot[:, cy : cy + height // 6, cx : cx + width // 6] = 2.5
    hdr = (hdr + spot).clamp(min=0.0)
    bracket, _t = hdr_to_ldr_bracket(hdr.unsqueeze(0), cfg)
    bracket = bracket.squeeze(0)
    idx = torch.randint(0, cfg.num_frames, (1,)).item()
    ldr_cond = bracket[idx]
    return hdr, ldr_cond, bracket
