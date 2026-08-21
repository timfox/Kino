"""SpaceDG: spatial intelligence under visual degradation (Zhou et al. arXiv:2605.22536)."""

from ltx_trainer.spacedg.degradations import (
    DEGRADATION_PARAM_RANGES,
    apply_degradation,
)
from ltx_trainer.spacedg.schema import DegradationType
from ltx_trainer.spacedg.eval import evaluate_predictions
from ltx_trainer.spacedg.metrics import mean_relative_accuracy, point_biserial_abs
from ltx_trainer.spacedg.pipeline import SpaceDGEngine, synthesize_benchmark_item
from ltx_trainer.spacedg.schema import AnswerFormat, QAPair, QuestionType, SpatialTaskGroup

__all__ = [
    "AnswerFormat",
    "DEGRADATION_PARAM_RANGES",
    "DegradationType",
    "QAPair",
    "QuestionType",
    "SpaceDGEngine",
    "SpatialTaskGroup",
    "apply_degradation",
    "evaluate_predictions",
    "mean_relative_accuracy",
    "point_biserial_abs",
    "synthesize_benchmark_item",
]
