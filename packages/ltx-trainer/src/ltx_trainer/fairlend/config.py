"""FairLend / HMDA fair-binning configuration (Rathod et al., arXiv:2606.12435)."""

from __future__ import annotations

from dataclasses import dataclass

PAPER_ARXIV = "2606.12435"
PAPER_TITLE = (
    "Auditing Discriminatory Patterns in Mortgage Lending Through "
    "Association Rules and Fair Binning"
)
PAPER_AUTHORS = "Archit Rathod, Dhwani Chande, Het Nagda (UIC)"
PAPER_URL = f"https://arxiv.org/abs/{PAPER_ARXIV}"
PAPER_REPO = "https://github.com/Archit1706/FairLend-Miners"
BENCHMARK = "HMDA 2023 Snapshot — Chicago MSA (derived_msa_md 16984)"


@dataclass
class FairLendConfig:
    msa_code: str = "16984"
    n_applications: int = 103_481
    n_clustering: int = 99_524
    overall_denial_rate: float = 0.239
    income_clip_low_k: float = 16.0
    income_clip_high_k: float = 1025.0
    n_bins: int = 5
    n_race_groups: int = 7
    fp_min_support: float = 0.10
    fp_min_confidence: float = 0.50
    fp_min_lift: float = 1.0
    dir_threshold: float = 0.80
    dir_min_group_size: int = 30
    dir_audited_pairs: int = 45
    kmeans_k: int = 5
    kmeans_k_min: int = 4
