"""TADA: Target Alignment through Data Adaptation for JPEG steganalysis (arXiv:2605.21523)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TADAConfig:
    paper_arxiv: str = "arXiv:2605.21523"
    paper_doi: str = "10.1145/3785353.3815080"
    github: str = "https://github.com/RonyAbecidan/TADA"
    demosaick: str = "amaze (fixed; paper Sec. 3.1)"
    emulator_kernel_size: int = 3  # toy; operational uses 5×5
    loss_lambda_cov: float = 1.0
    loss_mu_dist: float = 1.0
    loss_gamma_realism: float = 1.0
    patch_shape: tuple[int, int] = (8, 16)
    training_seed: int = 2026
    sgd_lr: float = 1e-3
    max_epochs: int = 3000
    early_stop_patience: int = 200
    embedding_payload: str = "UERD bpnzac=1"
    alaska_raw_count: int = 2000
    operational_unlabeled: int = 500
