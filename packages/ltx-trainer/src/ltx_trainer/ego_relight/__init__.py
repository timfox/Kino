"""EgoRelight: egocentric relightable full-body avatars (arXiv:2605.28401)."""

from ltx_trainer.ego_relight.config import EgoRelightConfig
from ltx_trainer.ego_relight.hdr import finlayson_ldr_to_hdr, optimize_hdr_color_correction
from ltx_trainer.ego_relight.lighting import (
    diffuse_maps,
    ray_encoding,
    sample_specular_rays,
)
from ltx_trainer.ego_relight.pipeline import (
    evaluation_demo,
    framework_card,
    relight_frame,
    run_perception_smoke,
)

__all__ = [
    "EgoRelightConfig",
    "diffuse_maps",
    "evaluation_demo",
    "finlayson_ldr_to_hdr",
    "framework_card",
    "optimize_hdr_color_correction",
    "ray_encoding",
    "relight_frame",
    "run_perception_smoke",
    "sample_specular_rays",
]
