"""Configuration for R-FUML (arXiv:2605.24475)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RFUMLConfig:
    paper_arxiv: str = "arXiv:2605.24475"
    email_corresponding: str = "penghu.ml@gmail.com"

    # Datasets (Sec. IV-A).
    datasets: tuple[str, ...] = (
        "HW",
        "MSRC",
        "NUSOBJ",
        "Fashion",
        "Scene",
        "LandUse",
        "Leaves",
        "PIE",
    )
    train_test_ratio: tuple[int, int] = (8, 2)
    vc_rates: tuple[int, ...] = (0, 20, 40, 60)

    # Membership mapping (Eq. 8).
    lp_norm: int = 2

    # RLVC (Sec. III-D).
    gmm_threshold_beta: float = 0.5
    warmup_epochs_default: int = 10
    cyclical_epochs_per_cycle: int = 10
    recommended_cycle_count: int = 10  # Fig. 8 — 100 epochs in Stage 2

    # Loss (Eq. 11).
    gamma_warmup_epochs: int = 10

    # Baseline count (Sec. IV).
    n_baselines: int = 15

    # Table I excerpt datasets for smoke tables.
    table_i_datasets: tuple[str, ...] = ("HW", "Fashion", "Scene", "LandUse")

    trusted_baselines: tuple[str, ...] = (
        "DUA-Nets",
        "ETMC",
        "UIMC",
        "ECML",
        "TMNR",
        "CCML",
        "TUNED",
        "SAEML",
        "FUML",
        "R-FUML",
    )

    components: tuple[str, ...] = field(
        default_factory=lambda: (
            "Fuzzy memberships → category credibility → entropy uncertainty",
            "RMF (Robust Multi-view Fusion)",
            "RLVC (Robust Learning Against VC)",
            "Lccl category credibility learning loss",
        )
    )
