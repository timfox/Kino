"""LTX / GOPEX integration plan for natural-experiment tabular pipelines."""

from __future__ import annotations

from typing import Any

from ltx_trainer.natural_experiments.config import NaturalExperimentsConfig


def ltx_plan(cfg: NaturalExperimentsConfig | None = None) -> dict[str, Any]:
    cfg = cfg or NaturalExperimentsConfig()
    return {
        "paper": cfg.paper_arxiv,
        "use_cases": [
            "Tabular clip metadata: flag fraud/disease cohorts as soft interventions",
            "Caption conditioning: prefer MB features when training classifiers on sidecars",
            "Dataset audit: ISK vs O F1 gap before merging observational + interventional rows",
        ],
        "hooks": {
            "preprocess": "partition_by_class → DCDI ISK graph → MB feature mask",
            "train": "MLP on MB only; oversample minority classes (paper §5.2)",
            "eval": "compare causal-sk vs causal-obs F1; brute-force on n_features ≤ 11",
        },
        "data_sources": [cfg.data_tabddpm, cfg.dcdi_repo],
        "natural_experiment_datasets": list(cfg.natural_experiment_dataset_names),
    }
