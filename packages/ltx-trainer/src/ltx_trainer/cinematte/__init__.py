"""CineMatte: background matting for virtual production (He et al. arXiv:2605.18328)."""

from ltx_trainer.cinematte.losses import CineMatteLoss, CineMatteLossConfig
from ltx_trainer.cinematte.metrics import (
    connectivity_error,
    dtssd,
    gradient_error,
    matting_mad,
    matting_mse,
)
from ltx_trainer.cinematte.model import CineMatte, CineMatteConfig
from ltx_trainer.cinematte.pipeline import matte_image, matte_video_frame

__all__ = [
    "CineMatte",
    "CineMatteConfig",
    "CineMatteLoss",
    "CineMatteLossConfig",
    "connectivity_error",
    "dtssd",
    "gradient_error",
    "matte_image",
    "matte_video_frame",
    "matting_mad",
    "matting_mse",
]
