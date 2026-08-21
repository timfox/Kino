"""Configuration for CHF LUS 30-day readmission pilot (arXiv:2605.18878)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ChfLusConfig:
    paper_arxiv: str = "2605.18878"
    paper_title: str = (
        "Prognostic Value of Lung Ultrasound Biomarkers for Readmission Risk in Congestive Heart Failure"
    )
    irb_protocol: str = "LSU IRB #1509 — Artificial Intelligence Interpretation of Lung Ultrasound Images"

    # Cohorts
    pretrain_patients: int = 221
    pretrain_clips: int = 1001
    chf_patients: int = 30
    chf_readmitted: int = 9
    chf_not_readmitted: int = 21

    # Encoder
    backbone: str = "TSM-ResNet-18"
    embedding_dim: int = 512
    biomarker_dim: int = 38
    pretrain_target: str = "S/F ratio regression"

    # Evaluation
    outer_folds: int = 5
    inner_folds: int = 5
    bootstrap_iters: int = 2000
    primary_metric: str = "weighted_f1"
    best_f1: float = 0.80
    best_f1_ci: tuple[float, float] = (0.62, 0.96)
    best_classifier: str = "MLP"
    best_view: str = "All Views"
    best_temporal: str = "temporal_difference"
    best_fusion: str = "feature_concatenate"

    views: tuple[str, ...] = (
        "Left-1",
        "Left-2",
        "Left-3",
        "Right-1",
        "Right-2",
        "Right-3",
        "All Views",
    )
    classifiers: tuple[str, ...] = (
        "Decision-Tree",
        "Random-Forest",
        "SVM",
        "MLP",
        "MLP-Large",
        "TabPFN",
    )

    biomarker_groups: tuple[str, ...] = field(
        default_factory=lambda: (
            "PL Location",
            "B-Line",
            "A-Line",
            "B-Line Origin",
            "PL Thickness",
            "PL Breaks",
            "Consolidation",
            "Effusion",
            "PL Indents",
        )
    )
