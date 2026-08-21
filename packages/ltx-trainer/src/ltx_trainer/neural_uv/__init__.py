"""Neural UV Atlas — continuous reparameterization for fixed-chart UV repair."""

from ltx_trainer.neural_uv.benchmarks import benchmarks_bundle
from ltx_trainer.neural_uv.config import NeuralUVConfig
from ltx_trainer.neural_uv.pipeline import evaluation_demo, evaluation_smoke, framework_card

__all__ = [
    "NeuralUVConfig",
    "benchmarks_bundle",
    "evaluation_demo",
    "evaluation_smoke",
    "framework_card",
]
