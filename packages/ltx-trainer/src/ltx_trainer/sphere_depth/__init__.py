"""Sphere-Depth: pose-aware 360° monocular depth benchmark (arXiv:2604.23432)."""

from ltx_trainer.sphere_depth.benchmarks import PAPER_ARXIV, PAPER_TITLE, benchmarks_bundle
from ltx_trainer.sphere_depth.calibration import learn_scaling_lambda, landmark_mse
from ltx_trainer.sphere_depth.config import BENCHMARK_MODELS, CODE_URL, SphereDepthConfig
from ltx_trainer.sphere_depth.paper import evaluation_demo, framework_card
from ltx_trainer.sphere_depth.pipeline import evaluation_demo_run
from ltx_trainer.sphere_depth.pose import apply_pose_perturbation

__all__ = [
    "BENCHMARK_MODELS",
    "CODE_URL",
    "PAPER_ARXIV",
    "PAPER_TITLE",
    "SphereDepthConfig",
    "apply_pose_perturbation",
    "benchmarks_bundle",
    "evaluation_demo",
    "evaluation_demo_run",
    "framework_card",
    "landmark_mse",
    "learn_scaling_lambda",
]
