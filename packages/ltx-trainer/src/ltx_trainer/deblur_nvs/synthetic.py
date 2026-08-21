"""Synthetic multi-view batches for DeblurNVS smoke tests."""

from __future__ import annotations

import torch
from torch import Tensor

from ltx_trainer.deblur_nvs.blur_synthesis import synthesize_blur_pair
from ltx_trainer.deblur_nvs.config import DeblurNVSConfig


def _checkerboard(h: int, w: int, squares: int = 8) -> Tensor:
    y = torch.arange(h).float().view(h, 1)
    x = torch.arange(w).float().view(1, w)
    board = ((y // (h // squares) + x // (w // squares)) % 2).float()
    rgb = torch.stack([board, 1 - board, board * 0.5 + 0.25], dim=0)
    return rgb


def synthetic_views(
    cfg: DeblurNVSConfig,
    *,
    batch_size: int = 1,
    device: torch.device | str = "cpu",
    seed: int = 0,
) -> dict[str, Tensor]:
    """Return sharp/blur context views + target camera + sharp target."""
    dev = torch.device(device)
    h, w = cfg.height, cfg.width
    sharp_ctx: list[Tensor] = []
    blur_ctx: list[Tensor] = []
    for k in range(cfg.context_views):
        base = _checkerboard(h, w, squares=6 + k)
        sharp, blur, _ = synthesize_blur_pair(base, seed=seed + k)
        sharp_ctx.append(sharp)
        blur_ctx.append(blur)
    sharp_stack = torch.stack(sharp_ctx, dim=0).unsqueeze(0).repeat(batch_size, 1, 1, 1, 1).to(dev)
    blur_stack = torch.stack(blur_ctx, dim=0).unsqueeze(0).repeat(batch_size, 1, 1, 1, 1).to(dev)
    target_sharp = _checkerboard(h, w, squares=10).unsqueeze(0).repeat(batch_size, 1, 1, 1).to(dev)
    # 16-D camera token stub (native DA3 coords)
    camera = torch.zeros(batch_size, 16, device=dev)
    camera[:, 0] = 1.0
    camera[:, 7] = 0.35
    return {
        "sharp_context": sharp_stack,
        "blur_context": blur_stack,
        "target_sharp": target_sharp,
        "camera": camera,
    }
