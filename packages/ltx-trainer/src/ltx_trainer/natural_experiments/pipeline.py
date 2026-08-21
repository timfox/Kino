"""Natural experiments paper tables and evaluation demo."""

from __future__ import annotations

from typing import Any

from ltx_trainer.natural_experiments.baselines import baselines_smoke
from ltx_trainer.natural_experiments.config import NaturalExperimentsConfig
from ltx_trainer.natural_experiments.dcdi import dcdi_pipeline_smoke
from ltx_trainer.natural_experiments.downstream import downstream_smoke
from ltx_trainer.natural_experiments.partition import partition_summary
import numpy as np


def framework_card(cfg: NaturalExperimentsConfig | None = None) -> dict[str, Any]:
    cfg = cfg or NaturalExperimentsConfig()
    return {
        "paper": cfg.paper_arxiv,
        "title": cfg.title,
        "method": "DCDI causal discovery → Markov blanket → MLP p(y|x_MB)",
        "dcdi_modes": list(cfg.dcdi_modes),
        "intervention_rule": "class-0 observational; other classes soft-known (ISK)",
        "natural_experiment_rule": "causal-sk F1 > causal-obs F1 on held-out test",
        "dcdi_repo": cfg.dcdi_repo,
        "datasets": cfg.data_tabddpm,
        "sachs_target": cfg.sachs_target,
        "sachs_ground_truth_mb": list(cfg.sachs_ground_truth_mb),
        "real_world_count": cfg.num_real_world_datasets,
        "natural_experiment_count": cfg.num_natural_experiment_datasets,
        "natural_experiment_datasets": list(cfg.natural_experiment_dataset_names),
    }


def table1_sachs_synthetic() -> list[dict[str, Any]]:
    """Anchors from paper Table 1 (selected rows)."""
    return [
        {
            "setting": "obs+interv all variables",
            "method": "Ground truth",
            "f1": 0.79,
            "mb": [1, 6, 7, 9],
            "shd_g": 0,
            "ed_mb": 0,
        },
        {
            "setting": "obs+interv all variables",
            "method": "Causal ISK",
            "f1": 0.79,
            "mb": [1, 6, 7, 9],
            "shd_g": 5,
            "ed_mb": 0,
        },
        {
            "setting": "observational",
            "method": "Causal O",
            "f1": 0.79,
            "mb": [1, 6, 7, 9],
            "shd_g": 8,
            "ed_mb": 0,
        },
        {
            "setting": "obs with hidden confounders",
            "method": "Causal ISK",
            "f1": 0.77,
            "mb": [0, 1, 3, 7, 9],
            "shd_g": 20,
            "ed_mb": 0,
        },
    ]


def table2_real_world_stats(cfg: NaturalExperimentsConfig | None = None) -> list[dict[str, Any]]:
    cfg = cfg or NaturalExperimentsConfig()
    return [dict(row) for row in cfg.real_world_datasets]


def table2_sachs_test_f1() -> dict[str, dict[str, float]]:
    """Paper Table 2 right — Sachs test-set F1 by method and test composition."""
    return {
        "Causal O": {"O": 0.795, "I": 0.844, "O+I": 0.820},
        "Causal ISK": {"O": 0.793, "I": 0.848, "O+I": 0.822},
        "natural_exp_on_I": True,
        "natural_exp_on_O+I": True,
    }


def table3_real_world_f1() -> list[dict[str, Any]]:
    """Paper Table 3 — best-sparse F1 (mean %); subset + natural-experiment rows."""
    rows = [
        {"dataset": "diabetes", "all_feat": 79.06, "causal_obs": 79.06, "causal_sk": 81.00, "natural_exp": True},
        {"dataset": "higgs-small", "all_feat": 61.76, "causal_obs": 60.60, "causal_sk": 62.13, "natural_exp": True},
        {"dataset": "credit-card-fraud", "all_feat": 99.90, "causal_obs": 99.88, "causal_sk": 99.91, "natural_exp": True},
        {"dataset": "miniboone", "all_feat": 70.05, "causal_obs": 79.98, "causal_sk": 80.58, "natural_exp": False},
        {"dataset": "wilt", "all_feat": 92.02, "causal_obs": 92.02, "causal_sk": 92.02, "natural_exp": False},
        {"dataset": "dermatology", "all_feat": 97.27, "causal_obs": 97.27, "causal_sk": 97.59, "natural_exp": False},
    ]
    return rows


def table4_bruteforce_ranks() -> list[dict[str, Any]]:
    """Paper Table 4 — causal-sk MB rank vs exhaustive search."""
    return [
        {"dataset": "wilt", "rank": "5/31", "ed_mb": 2},
        {"dataset": "diabetes", "rank": "1/255", "ed_mb": 0},
        {"dataset": "buddy", "rank": "188/511", "ed_mb": 2},
        {"dataset": "churn-modelling", "rank": "55/2047", "ed_mb": 2},
        {"dataset": "cardio", "rank": "1/2047", "ed_mb": 0},
    ]


def table8_avg_shd_ed() -> list[dict[str, Any]]:
    """Appendix Table 8 — average SHD / ED_MB across Sachs settings."""
    return [
        {"method": "Causal O", "avg_shd_g": 14.75, "avg_ed_mb": 0.50},
        {"method": "Causal ISK", "avg_shd_g": 12.75, "avg_ed_mb": 0.25},
        {"method": "Causal IHK", "avg_shd_g": 13.25, "avg_ed_mb": 1.00},
        {"method": "Causal IHU", "avg_shd_g": 12.75, "avg_ed_mb": 1.00},
    ]


def evaluation_demo(cfg: NaturalExperimentsConfig | None = None) -> dict[str, Any]:
    cfg = cfg or NaturalExperimentsConfig()
    labels = np.array([0] * 60 + [1] * 30 + [2] * 10)
    return {
        "framework": framework_card(cfg),
        "dcdi_smoke": dcdi_pipeline_smoke(cfg),
        "baselines_smoke": baselines_smoke(),
        "downstream_smoke": downstream_smoke(cfg),
        "partition": partition_summary(labels),
        "table1_sachs": table1_sachs_synthetic(),
        "table2_datasets": table2_real_world_stats(cfg),
        "table2_sachs_test": table2_sachs_test_f1(),
        "table3_f1": table3_real_world_f1(),
        "table4_bruteforce": table4_bruteforce_ranks(),
        "table8_shd": table8_avg_shd_ed(),
        "diabetes_insulin_excluded": True,
        "conclusion": (
            "3/11 real-world datasets show natural experiments when "
            "soft-known (ISK) causal feature selection beats observational (O)."
        ),
    }
