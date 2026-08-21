"""CPU smoke for Clairvoyant predictive SJF sidecar."""

from __future__ import annotations

from typing import Any

from ltx_trainer.clairvoyant.benchmarks import benchmarks_bundle
from ltx_trainer.clairvoyant.config import PAPER_ARXIV
from ltx_trainer.clairvoyant.features import feature_names
from ltx_trainer.clairvoyant.paper import evaluation_demo, framework_card, knowledge_blob
from ltx_trainer.clairvoyant.pipeline import evaluation_demo_run
from ltx_trainer.clairvoyant.predictor import predict_plong
from ltx_trainer.clairvoyant.ranking import pairwise_ranking_accuracy


def evaluation_smoke() -> dict[str, Any]:
    demo = evaluation_demo_run()
    short_pl = [predict_plong(p) for p in (
        "What is 2+2?",
        "Define HTTP.",
        "Who is Marie Curie?",
    )]
    long_pl = [predict_plong(p) for p in (
        "Write a full Python module implementing a REST API with authentication.",
        "Generate a comprehensive essay on climate policy.",
    )]
    rank_acc = pairwise_ranking_accuracy(short_pl, long_pl)
    return {
        "package": "clairvoyant",
        "status": "ok",
        "arxiv": PAPER_ARXIV,
        "n_features": len(feature_names()),
        "dispatch_sjf_valid": demo["dispatch_validation"]["sjf_order_valid"],
        "pairwise_ranking_accuracy_stub": round(rank_acc, 4),
        "predictor_p_long_short_max": round(max(short_pl), 4),
        "predictor_p_long_long_min": round(min(long_pl), 4),
        "demo": demo,
        "framework": framework_card(),
        "benchmarks": benchmarks_bundle(),
        "knowledge": knowledge_blob(),
        "evaluation_demo": evaluation_demo(),
    }
