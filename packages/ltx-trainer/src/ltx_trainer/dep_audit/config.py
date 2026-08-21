"""Multi-probe depression benchmark audit (Zenodo:10.5281/zenodo.19813141)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DepAuditConfig:
    paper_zenodo: str = "10.5281/zenodo.19813141"
    paper_code_zenodo: str = "10.5281/zenodo.19813142"
    official_split_train: int = 163
    official_split_dev: int = 56
    official_split_test: int = 56
    loso_tl_macro_f1: float = 0.723
    config_sweep_count: int = 96
    bootstrap_replicates: int = 4000
    topic_seeds: tuple[int, ...] = (13, 23, 37, 42, 79)
