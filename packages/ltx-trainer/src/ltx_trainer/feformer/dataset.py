"""Dataset summaries (Sec. 4.1)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.feformer.config import DATASETS, PATCH_SIZE
from ltx_trainer.feformer.metrics import table1_datasets


def dataset_summary() -> dict[str, Any]:
    return {
        "datasets": table1_datasets(),
        "patch_size": PATCH_SIZE,
        "training_epochs": 1000,
        "batch_size": 2,
        "loss": "cross_entropy + dice",
        "cross_validation": "5-fold",
        "names": list(DATASETS),
    }
