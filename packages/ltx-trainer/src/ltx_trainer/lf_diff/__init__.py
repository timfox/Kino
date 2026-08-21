"""LF-Diff — linearized diffusion HDR from exposure brackets (arXiv:2503.07351)."""

from ltx_trainer.lf_diff.benchmarks import benchmarks_bundle
from ltx_trainer.lf_diff.fold import annotate_video_latent_data
from ltx_trainer.lf_diff.mock import evaluation_smoke
from ltx_trainer.lf_diff.model import LfDiff, LfDiffConfig
from ltx_trainer.lf_diff.paper import framework_card
from ltx_trainer.lf_diff.pipeline import evaluation_demo, pipeline_demo, train_step

__all__ = [
    "LfDiff",
    "LfDiffConfig",
    "annotate_video_latent_data",
    "benchmarks_bundle",
    "evaluation_demo",
    "evaluation_smoke",
    "framework_card",
    "pipeline_demo",
    "train_step",
]
