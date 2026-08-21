"""Framework card and benchmark bundles for CHF LUS readmission pilot."""

from __future__ import annotations

from typing import Any

from ltx_trainer.chf_lus.config import ChfLusConfig
from ltx_trainer.chf_lus.layout import LIMITATIONS, PIPELINE_STAGES, VIEW_ANATOMY
from ltx_trainer.chf_lus.mock import evaluation_smoke
from ltx_trainer.chf_lus.tables import (
    biomarker_selection_frequency,
    ehr_biomarker_comparison,
    fusion_heatmap_mlp,
    scan_counts_by_day,
    table1_all_views_by_classifier,
    table1_view_classifier_mlp,
    table2_day_pair_mlp,
    table3_biomarker_svm,
)


def framework_card(cfg: ChfLusConfig | None = None) -> dict[str, Any]:
    cfg = cfg or ChfLusConfig()
    return {
        "name": cfg.paper_title,
        "arxiv": f"arXiv:{cfg.paper_arxiv}",
        "authors": "Armouti et al. (CMU, LSUHSC, collaborators)",
        "task": "Predict 30-day CHF readmission from point-of-care lung ultrasound during hospitalization.",
        "cohorts": {
            "pretrain_lus": f"{cfg.pretrain_patients} patients, {cfg.pretrain_clips} clips (Gare et al. S/F pretraining)",
            "chf_pilot": (
                f"{cfg.chf_patients} patients ({cfg.chf_readmitted} readmitted, "
                f"{cfg.chf_not_readmitted} not)"
            ),
        },
        "imaging": {
            "views": list(cfg.views[:-1]),
            "view_anatomy": VIEW_ANATOMY,
            "scan_counts_by_day": scan_counts_by_day(),
        },
        "encoder": {
            "backbone": cfg.backbone,
            "embedding_dim": cfg.embedding_dim,
            "pretrain": cfg.pretrain_target,
            "temporal_primary": cfg.best_temporal,
            "fusion_primary": cfg.best_fusion,
        },
        "evaluation": {
            "protocol": f"{cfg.outer_folds}×{cfg.inner_folds} nested CV, patient-level splits",
            "metric": cfg.primary_metric,
            "bootstrap_ci": cfg.bootstrap_iters,
        },
        "headline": {
            "best_f1": cfg.best_f1,
            "ci_95": list(cfg.best_f1_ci),
            "classifier": cfg.best_classifier,
            "configuration": f"{cfg.best_view}, {cfg.best_temporal}, {cfg.best_fusion}",
        },
        "findings": [
            "Dependent lower-lung views (L3, R3) carry strongest prognostic signal.",
            "Temporal difference (Day2−Day1) beats single-timepoint or concatenation.",
            "Multi-view embedding concatenation achieves best F1 (0.80).",
            "Pleural-line breaks/indents rival canonical A-line/B-line biomarkers.",
        ],
        "pipeline_stages": list(PIPELINE_STAGES),
        "irb": cfg.irb_protocol,
        "limitations": list(LIMITATIONS),
    }


def benchmarks_bundle() -> dict[str, Any]:
    return {
        "table1_mlp_per_view": table1_view_classifier_mlp(),
        "table1_all_views_classifiers": table1_all_views_by_classifier(),
        "fig3_fusion_heatmap_mlp": fusion_heatmap_mlp(),
        "table2_day_pairs": table2_day_pair_mlp(),
        "table3_biomarker_vs_embedding": table3_biomarker_svm(),
        "fig2_ehr_biomarker": ehr_biomarker_comparison(),
        "fig2_biomarker_radar": biomarker_selection_frequency(),
        "scan_counts_by_day": scan_counts_by_day(),
    }


def evaluation_demo() -> dict[str, Any]:
    return {"smoke": evaluation_smoke(), "headline_f1": ChfLusConfig().best_f1}


def pipeline_demo() -> dict[str, Any]:
    return {"framework": framework_card(), "benchmarks_keys": list(benchmarks_bundle())}


def headline_results() -> dict[str, Any]:
    cfg = ChfLusConfig()
    return {
        "best_f1": cfg.best_f1,
        "ci_95": cfg.best_f1_ci,
        "classifier": cfg.best_classifier,
        "best_views": ["Left-3", "Right-3", "All Views"],
    }
