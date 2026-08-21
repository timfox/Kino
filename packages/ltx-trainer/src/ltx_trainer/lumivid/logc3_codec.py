"""LogC3 compress / decompress for LumiVid VAE alignment (Sec. 3.1)."""

from __future__ import annotations

import torch
from ltx_core.hdr import LogC3
from torch import Tensor

from ltx_trainer.hdr_ingest import linear_scene_to_logc3_display, prepare_scene_linear_for_vae

_LOGC3 = LogC3()


def scene_linear_to_vae_pixels(scene_linear: Tensor) -> Tensor:
    """Scene-linear RGB → LogC3 display codes in [0, 1] (frozen VAE input before [-1,1] norm)."""
    return linear_scene_to_logc3_display(scene_linear)


def vae_pixels_to_scene_linear(logc3_display: Tensor) -> Tensor:
    """LogC3 [0, 1] codes → scene-linear radiance (inverse LogC3)."""
    return _LOGC3.decompress(logc3_display.clamp(0.0, 1.0))


def vae_normalize(pixels_01: Tensor) -> Tensor:
    """Map [0, 1] display to [-1, 1] VAE domain."""
    return pixels_01.clamp(0.0, 1.0) * 2.0 - 1.0


def vae_denormalize(vae_rgb: Tensor) -> Tensor:
    return ((vae_rgb.clamp(-1.0, 1.0) + 1.0) * 0.5).clamp(0.0, 1.0)
