"""Synthetic compositing with distractor foregrounds (Sec. 5.1)."""

from __future__ import annotations

import random

import torch
import torch.nn.functional as F
from torch import Tensor


def _random_blob(h: int, w: int, device: torch.device) -> Tensor:
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, h, device=device),
        torch.linspace(-1, 1, w, device=device),
        indexing="ij",
    )
    cx, cy = random.uniform(-0.4, 0.4), random.uniform(-0.4, 0.4)
    sx, sy = random.uniform(0.15, 0.45), random.uniform(0.15, 0.45)
    alpha = torch.exp(-(((xx - cx) ** 2) / (2 * sx**2) + ((yy - cy) ** 2) / (2 * sy**2)))
    return alpha.clamp(0, 1)


def augment_matting_pair(
    image: Tensor,
    background: Tensor,
    alpha: Tensor,
    *,
    hflip_prob: float = 0.5,
    color_jitter: bool = True,
    affine: bool = True,
) -> tuple[Tensor, Tensor, Tensor]:
    """Paper-style augmentations on foreground + background (Sec. 5.1)."""
    if random.random() < hflip_prob:
        image = torch.flip(image, [-1])
        background = torch.flip(background, [-1])
        alpha = torch.flip(alpha, [-1])

    if color_jitter:
        for t in (image, background):
            # saturation / brightness on RGB
            gray = t.mean(dim=0, keepdim=True)
            sat = random.uniform(0.85, 1.15)
            t.mul_(sat).add_(gray * (1 - sat))
            t.add_(random.uniform(-0.08, 0.08))
            t.clamp_(0, 1)

    if affine:
        angle = random.uniform(-12, 12)
        scale = random.uniform(0.92, 1.08)
        shear = random.uniform(-0.05, 0.05)
        c, h, w = image.shape
        theta = torch.tensor(
            [
                [scale, shear, random.uniform(-0.05, 0.05)],
                [shear, scale, random.uniform(-0.05, 0.05)],
            ],
            dtype=image.dtype,
            device=image.device,
        ).unsqueeze(0)
        grid = F.affine_grid(theta, [1, c, h, w], align_corners=False)
        image = F.grid_sample(image.unsqueeze(0), grid, align_corners=False).squeeze(0)
        background = F.grid_sample(background.unsqueeze(0), grid, align_corners=False).squeeze(0)
        alpha = F.grid_sample(alpha.unsqueeze(0), grid, align_corners=False).squeeze(0)
        _ = angle  # reserved for future rotate-only on fg

    return image, background, alpha


def composite_training_sample(
    h: int = 256,
    w: int = 256,
    *,
    num_subjects: int | None = None,
    num_distractors: int | None = None,
    augment: bool = False,
    device: torch.device | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return ``(image, background_plate, alpha)`` tensors ``[3,H,W]`` / ``[1,H,W]`` in ``[0,1]``."""
    device = device or torch.device("cpu")
    bg = torch.rand(3, h, w, device=device)
    plate = bg.clone()
    num_subjects = num_subjects if num_subjects is not None else random.randint(1, 3)
    num_distractors = num_distractors if num_distractors is not None else random.randint(0, 3)

    alpha = torch.zeros(h, w, device=device)
    image = bg.clone()

    for _ in range(num_distractors):
        a = _random_blob(h, w, device)
        fg_rgb = torch.rand(3, 1, 1, device=device)
        plate = plate * (1 - a) + fg_rgb * a

    for _ in range(num_subjects):
        a = _random_blob(h, w, device)
        fg_rgb = torch.rand(3, 1, 1, device=device)
        alpha = torch.maximum(alpha, a)
        image = image * (1 - a) + fg_rgb * a

    alpha = alpha.unsqueeze(0)
    if augment:
        image, plate, alpha = augment_matting_pair(image, plate, alpha)
    return image, plate, alpha
