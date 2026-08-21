"""Natural experiments: DCDI + Markov-blanket feature selection (arXiv:2606.03251)."""

from ltx_trainer.natural_experiments.config import NaturalExperimentsConfig
from ltx_trainer.natural_experiments.mock import evaluation_smoke
from ltx_trainer.natural_experiments.pipeline import evaluation_demo, framework_card

__all__ = [
    "NaturalExperimentsConfig",
    "evaluation_demo",
    "evaluation_smoke",
    "framework_card",
]
