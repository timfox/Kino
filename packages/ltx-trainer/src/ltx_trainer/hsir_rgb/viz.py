"""Fig. 1 visualization helpers — RGB band slices from HS cubes."""

from __future__ import annotations

import torch
from torch import Tensor

from ltx_trainer.hsir_rgb.config import HSIRRgbConfig
from ltx_trainer.hsir_rgb.metrics import hs_psnr


def bands_one_based_to_indices(bands_1based: tuple[int, ...], *, num_bands: int) -> tuple[int, ...]:
    """Convert paper 1-based band ids to 0-based channel indices."""
    out: list[int] = []
    for b in bands_1based:
        idx = int(b) - 1
        if idx < 0 or idx >= num_bands:
            raise ValueError(f"Band {b} out of range for C={num_bands}")
        out.append(idx)
    return tuple(out)


def extract_rgb_viz_cube(
    cube: Tensor,
    bands_1based: tuple[int, ...] | None = None,
    *,
    cfg: HSIRRgbConfig | None = None,
) -> Tensor:
    """Select three HS bands as an RGB tensor ``[B, 3, H, W]`` (Fig. 1)."""
    cfg = cfg or HSIRRgbConfig()
    bands = bands_1based or cfg.harvard_rgb_viz_bands
    idx = bands_one_based_to_indices(bands, num_bands=cube.shape[1])
    if cube.dim() != 4:
        raise ValueError("cube must be [B,C,H,W]")
    picked = torch.stack([cube[:, i] for i in idx], dim=1)
    if picked.shape[1] != 3:
        raise ValueError("viz expects exactly three bands")
    return picked


def psnr_on_viz_bands(
    x_hat: Tensor,
    x: Tensor,
    bands_1based: tuple[int, ...] | None = None,
    *,
    cfg: HSIRRgbConfig | None = None,
) -> float:
    """PSNR on Fig. 1 band subset only (paper reports PSNR on {24, 14, 4})."""
    cfg = cfg or HSIRRgbConfig()
    xh = extract_rgb_viz_cube(x_hat, bands_1based, cfg=cfg)
    xr = extract_rgb_viz_cube(x, bands_1based, cfg=cfg)
    return hs_psnr(xh, xr)
