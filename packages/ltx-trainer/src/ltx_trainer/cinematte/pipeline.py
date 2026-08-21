"""Inference, compositing, and background-shift augmentation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor

from ltx_trainer.cinematte.model import CineMatte, CineMatteConfig


@dataclass
class MatteResult:
    alpha: Tensor  # [1,H,W] or [B,1,H,W]
    foreground: Tensor | None = None
    composite: Tensor | None = None


def _load_rgb(path: Path, device: torch.device, size: tuple[int, int] | None = None) -> Tensor:
    from PIL import Image
    from torchvision.transforms.functional import to_tensor

    with Image.open(path) as im:
        im = im.convert("RGB")
        if size is not None:
            im = im.resize(size, Image.Resampling.BILINEAR)
        t = to_tensor(im).to(device)
    return t


def matte_image(
    model: CineMatte,
    image: Tensor,
    background: Tensor,
    *,
    return_composite: bool = False,
    new_background: Tensor | None = None,
) -> MatteResult:
    """Run CineMatte on batched or single ``[3,H,W]`` tensors in ``[0,1]``."""
    if image.dim() == 3:
        image = image.unsqueeze(0)
    if background.dim() == 3:
        background = background.unsqueeze(0)
    model.eval()
    with torch.no_grad():
        alpha = model(image, background)
    fg = None
    comp = None
    if return_composite:
        bg_out = new_background if new_background is not None else background
        if bg_out.dim() == 3:
            bg_out = bg_out.unsqueeze(0)
        comp = alpha * image + (1.0 - alpha) * bg_out
    return MatteResult(alpha=alpha.squeeze(0), composite=comp.squeeze(0) if comp is not None else None)


def matte_video_frame(
    model: CineMatte,
    frame: Tensor,
    background_plate: Tensor,
    *,
    new_background: Tensor | None = None,
) -> MatteResult:
    """Single VP frame: inner-frustum background plate + optional relight background."""
    return matte_image(
        model,
        frame,
        background_plate,
        return_composite=new_background is not None,
        new_background=new_background,
    )


def apply_background_shift(bg: Tensor, *, angle_deg: float = 0.0, scale: float = 1.0, shear: float = 0.0) -> Tensor:
    """Stress-test augmentation from Table 3 (affine background misalignment)."""
    b = bg.unsqueeze(0) if bg.dim() == 3 else bg
    _, _, h, w = b.shape
    rad = angle_deg * math.pi / 180.0
    cos_a, sin_a = math.cos(rad), math.sin(rad)
    a = scale * cos_a
    c = scale * sin_a
    # 2x3 affine: [[a, shear, tx], [c, b, ty]]
    theta = torch.tensor([[a, shear, 0.0], [c, a, 0.0]], dtype=b.dtype, device=b.device).unsqueeze(0)
    theta = theta.expand(b.shape[0], -1, -1)
    grid = F.affine_grid(theta, b.size(), align_corners=False)
    return F.grid_sample(b, grid, mode="bilinear", padding_mode="border", align_corners=False).squeeze(0)


def load_cinematte_checkpoint(path: Path | str, *, device: str = "cpu") -> CineMatte:
    ckpt = torch.load(Path(path).expanduser(), map_location=device, weights_only=False)
    cfg_raw = ckpt.get("config", {})
    cfg = CineMatteConfig(**cfg_raw) if cfg_raw else CineMatteConfig()
    model = CineMatte(cfg)
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval()
