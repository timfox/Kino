"""H-OmniStereo: zero-shot omnidirectional stereo with heading-aligned normals (arXiv:2605.14963)."""

from ltx_trainer.h_omnistereo.benchmarks import PAPER_ARXIV, PAPER_TITLE, benchmarks_bundle
from ltx_trainer.h_omnistereo.config import HOmniStereoConfig
from ltx_trainer.h_omnistereo.heading_normal import camera_to_heading_aligned, longitude_grid
from ltx_trainer.h_omnistereo.paper import evaluation_demo, framework_card
from ltx_trainer.h_omnistereo.pipeline import evaluation_demo_run, predict_disparity
from ltx_trainer.h_omnistereo.spherical_geo import spherical_disparity

__all__ = [
    "HOmniStereoConfig",
    "PAPER_ARXIV",
    "PAPER_TITLE",
    "benchmarks_bundle",
    "camera_to_heading_aligned",
    "evaluation_demo",
    "evaluation_demo_run",
    "framework_card",
    "longitude_grid",
    "predict_disparity",
    "spherical_disparity",
]
