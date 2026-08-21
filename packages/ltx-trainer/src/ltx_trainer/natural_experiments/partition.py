"""Observational vs soft-known interventional split by class label (§4.2)."""

from __future__ import annotations

import numpy as np

from ltx_trainer.natural_experiments.config import NaturalExperimentsConfig


def partition_by_class(
    labels: np.ndarray,
    cfg: NaturalExperimentsConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Class-0 → observational indices; all other classes → interventional.

    Matches the paper's multi-class treatment for DCDI ISK.
    """
    cfg = cfg or NaturalExperimentsConfig()
    obs = np.where(labels == cfg.observational_class_index)[0]
    interv = np.where(labels != cfg.observational_class_index)[0]
    return obs, interv


def partition_summary(labels: np.ndarray) -> dict[str, int]:
    obs, interv = partition_by_class(labels)
    return {
        "observational": int(len(obs)),
        "interventional": int(len(interv)),
        "total": int(len(labels)),
    }
