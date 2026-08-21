"""Predictive autoscaling survey constants (Kumar et al., arXiv:2606.07046)."""

from __future__ import annotations

PAPER_ARXIV = "2606.07046"
PAPER_TITLE = (
    "Predictive Autoscaling in Cloud-Native and Federated Cloud-Edge Computing "
    "Environments: A Taxonomy and Future Directions"
)
PAPER_URL = f"https://arxiv.org/abs/{PAPER_ARXIV}"
AUTHORS = ("Bablu Kumar", "Anshul Verma", "Rajkumar Buyya")
AFFILIATION = "qCLOUDS, University of Melbourne / BHU Varanasi"

# Systematic review funnel (Fig. 1)
LITERATURE_IDENTIFIED = 450
LITERATURE_AFTER_DEDUP = 390
LITERATURE_SCREENED = 230
LITERATURE_INCLUDED = 160
LITERATURE_EXCLUDED = 30
LITERATURE_FINAL = 130

# Four-dimensional taxonomy (Sec. IV, Fig. 7)
TAXONOMY_SCALING_TRIGGERS = (
    "threshold",
    "rule_policy",
    "event_driven",
    "predictive",
    "hybrid",
)
TAXONOMY_SCALING_TARGETS = (
    "hpa",
    "vpa",
    "cluster_autoscaler",
    "keda",
    "other",
)
TAXONOMY_PREDICTION_MODELS = (
    "statistical",
    "machine_learning",
    "deep_learning",
    "transformer",
    "mv_transformer",
)
TAXONOMY_EVALUATION = (
    "latency",
    "elasticity",
    "cost_efficiency",
    "prediction_accuracy",
    "stability_robustness",
    "quality_of_experience",
)

MAPE_PHASES = ("monitor", "analyze", "plan", "execute")

RESEARCH_QUESTIONS = (
    "RQ1: predictive models for proactive autoscaling accuracy",
    "RQ2: Kubernetes CRDs, operators, reconciliation for autonomous pipelines",
    "RQ3: FL workload autoscaling under DP and heterogeneity",
    "RQ4: autoscaling drift, ADI, uncertainty-aware correction and FRSC",
)
