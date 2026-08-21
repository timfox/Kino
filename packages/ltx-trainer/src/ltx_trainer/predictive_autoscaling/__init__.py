"""Predictive autoscaling survey package (Kumar et al., arXiv:2606.07046)."""

from ltx_trainer.predictive_autoscaling.config import PredictiveAutoscalingConfig
from ltx_trainer.predictive_autoscaling.mock import evaluation_smoke
from ltx_trainer.predictive_autoscaling.paper import knowledge_bundle, paper_card
from ltx_trainer.predictive_autoscaling.pipeline import run_demo
from ltx_trainer.predictive_autoscaling.taxonomy import taxonomy_card

__all__ = [
    "PredictiveAutoscalingConfig",
    "evaluation_smoke",
    "knowledge_bundle",
    "paper_card",
    "run_demo",
    "taxonomy_card",
]
