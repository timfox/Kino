"""X2HDR — SDR-to-HDR via PU21-aligned latent diffusion (arXiv:2602.04814)."""

from ltx_trainer.x2hdr.benchmarks import benchmarks_bundle
from ltx_trainer.x2hdr.fold import annotate_video_latent_data
from ltx_trainer.x2hdr.mock import evaluation_smoke
from ltx_trainer.x2hdr.model import X2Hdr, X2HdrConfig
from ltx_trainer.x2hdr.paper import framework_card
from ltx_trainer.x2hdr.pipeline import evaluation_demo, pipeline_demo, train_step

__all__ = [
    "X2Hdr",
    "X2HdrConfig",
    "annotate_video_latent_data",
    "benchmarks_bundle",
    "evaluation_demo",
    "evaluation_smoke",
    "framework_card",
    "pipeline_demo",
    "train_step",
]
