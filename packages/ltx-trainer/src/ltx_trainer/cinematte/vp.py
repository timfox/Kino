"""LED virtual production helpers (inner-frustum background plates)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

from ltx_trainer.cinematte.model import CineMatte
from ltx_trainer.cinematte.pipeline import load_cinematte_checkpoint, matte_image


@dataclass
class VPCompositeResult:
    alpha: Tensor
    foreground_rgba: Tensor
    composite: Tensor | None


def resolve_vp_checkpoint() -> Path | None:
    """``GOPEX_CINEMATTE_CHECKPOINT`` or ``GOPEX_CINEMATTE_CKPT``."""
    for key in ("GOPEX_CINEMATTE_CHECKPOINT", "GOPEX_CINEMATTE_CKPT"):
        val = os.environ.get(key, "").strip()
        if val:
            p = Path(val).expanduser()
            if p.is_file():
                return p
    return None


def matte_vp_frame(
    frame: Tensor,
    inner_frustum_plate: Tensor,
    *,
    model: CineMatte | None = None,
    checkpoint: str | Path | None = None,
    device: str = "cuda",
    new_background: Tensor | None = None,
) -> VPCompositeResult:
    """
    Matte one VP camera frame against the captured LED inner-frustum plate.

    Args:
        frame: ``[3,H,W]`` actor + props on stage.
        inner_frustum_plate: ``[3,H,W]`` background as seen by camera (may be misaligned).
    """
    dev = device if torch.cuda.is_available() or device == "cpu" else "cpu"
    if model is None:
        ckpt = Path(checkpoint) if checkpoint else resolve_vp_checkpoint()
        if ckpt is not None:
            model = load_cinematte_checkpoint(ckpt, device=dev)
        else:
            from ltx_trainer.cinematte.model import CineMatteConfig

            backbone = os.environ.get("GOPEX_CINEMATTE_BACKBONE", "stub")
            model = CineMatte(CineMatteConfig(backbone=backbone)).to(dev)

    result = matte_image(
        model,
        frame,
        inner_frustum_plate,
        return_composite=new_background is not None,
        new_background=new_background,
    )
    alpha = result.alpha
    fg = frame * alpha + (1 - alpha) * inner_frustum_plate
    rgba = torch.cat([fg, alpha], dim=0)
    return VPCompositeResult(alpha=alpha, foreground_rgba=rgba, composite=result.composite)
