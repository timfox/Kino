"""Paper cards and knowledge exports."""

from __future__ import annotations

from typing import Any

from ltx_trainer.predictive_autoscaling.benchmarks import (
    TABLE_I_SURVEY_COMPARISON,
    literature_funnel,
    table_iii_predictive_models,
    table_v_drift_mechanisms,
)
from ltx_trainer.predictive_autoscaling.config import PAPER_ARXIV, PAPER_TITLE, PAPER_URL
from ltx_trainer.predictive_autoscaling.constants import AUTHORS, RESEARCH_QUESTIONS
from ltx_trainer.predictive_autoscaling.federated import fl_autoscaling_card, kubeflower_overview
from ltx_trainer.predictive_autoscaling.future import future_research_card
from ltx_trainer.predictive_autoscaling.kubernetes_native import (
    k8s_autoscaler_catalog,
    operator_reconciliation_steps,
    predictive_autoscaler_crd_example,
)
from ltx_trainer.predictive_autoscaling.mape import mape_overview
from ltx_trainer.predictive_autoscaling.models import prediction_model_catalog
from ltx_trainer.predictive_autoscaling.taxonomy import taxonomy_card


def paper_card() -> dict[str, Any]:
    return {
        "arxiv": PAPER_ARXIV,
        "title": PAPER_TITLE,
        "url": PAPER_URL,
        "authors": list(AUTHORS),
        "contributions": [
            "unified cloud-native + MAPE foundation",
            "four-dimensional autoscaling taxonomy",
            "predictive models + CRD/operator integration",
            "FL-specific autoscaling and KubeFlower",
            "ADI drift-aware and FRSC stability control",
            "open challenges and future directions",
        ],
        "research_questions": list(RESEARCH_QUESTIONS),
    }


def knowledge_bundle() -> dict[str, Any]:
    return {
        "paper": paper_card(),
        "literature_funnel": literature_funnel(),
        "taxonomy": taxonomy_card(),
        "mape": mape_overview(),
        "kubernetes": k8s_autoscaler_catalog(),
        "crd_example": predictive_autoscaler_crd_example(),
        "operator_steps": operator_reconciliation_steps(),
        "prediction_models": prediction_model_catalog(),
        "federated": fl_autoscaling_card(),
        "kubeflower": kubeflower_overview(),
        "table_i": TABLE_I_SURVEY_COMPARISON,
        "table_iii": table_iii_predictive_models(),
        "table_v": table_v_drift_mechanisms(),
        "future": future_research_card(),
    }
