"""Clairvoyant — predictive SJF sidecar for serial LLM backends (arXiv:2606.07248)."""

from ltx_trainer.clairvoyant.config import (
    ClairvoyantConfig,
    MEDIUM_MAX_TOKENS,
    PAPER_ARXIV,
    PAPER_TITLE,
    PAPER_URL,
    SHORT_MAX_TOKENS,
    UPSTREAM_REPO,
)
from ltx_trainer.clairvoyant.features import extract_features, feature_names
from ltx_trainer.clairvoyant.mock import evaluation_smoke
from ltx_trainer.clairvoyant.paper import evaluation_demo, framework_card, knowledge_blob
from ltx_trainer.clairvoyant.predictor import predict_class, predict_plong, predict_proba, predict_record
from ltx_trainer.clairvoyant.scheduler import SJFScheduler, validate_short_before_long
from ltx_trainer.clairvoyant.ranking import pairwise_ranking_accuracy

__all__ = [
    "ClairvoyantConfig",
    "MEDIUM_MAX_TOKENS",
    "PAPER_ARXIV",
    "PAPER_TITLE",
    "PAPER_URL",
    "SHORT_MAX_TOKENS",
    "UPSTREAM_REPO",
    "SJFScheduler",
    "evaluation_demo",
    "evaluation_smoke",
    "extract_features",
    "feature_names",
    "framework_card",
    "knowledge_blob",
    "pairwise_ranking_accuracy",
    "predict_class",
    "predict_plong",
    "predict_proba",
    "predict_record",
    "validate_short_before_long",
]
