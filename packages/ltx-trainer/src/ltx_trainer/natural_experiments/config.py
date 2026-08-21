"""Natural experiments via DCDI causal feature selection (Gare et al., arXiv:2606.03251)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class NaturalExperimentsConfig:
    paper_arxiv: str = "arXiv:2606.03251"
    title: str = (
        "Do Real-World Datasets Contain Natural Experiments? "
        "An Empirical Study Using Causal Feature Selection"
    )
    dcdi_repo: str = "https://github.com/slachapelle/dcdi"
    data_tabddpm: str = "https://www.dropbox.com/s/rpckvcs3vx7j605/data.tar?dl=0"

    # DCDI intervention modes (§4.1)
    dcdi_modes: tuple[str, ...] = ("O", "ISK", "IHK", "IHU")

    # Sachs synthetic (BnLearn) — Mek target, 3-way class
    sachs_nodes: int = 11
    sachs_target: str = "Mek"
    sachs_ground_truth_mb: tuple[int, ...] = (1, 6, 7, 9)  # node indices per paper Table 1

    # Real-world split (64:16:20) and natural-experiment rule
    train_frac: float = 0.64
    val_frac: float = 0.16
    observational_class_index: int = 0  # class-0 = observational, rest = soft-known intervention

    # Count of datasets flagged with natural experiments (§5.2.1)
    num_real_world_datasets: int = 11
    num_natural_experiment_datasets: int = 3
    natural_experiment_dataset_names: tuple[str, ...] = (
        "diabetes",
        "higgs-small",
        "credit-card-fraud",
    )

    real_world_datasets: tuple[dict[str, int | str], ...] = field(
        default_factory=lambda: (
            {"name": "adult", "samples": 48842, "features": 14, "classes": 2},
            {"name": "buddy", "samples": 18834, "features": 9, "classes": 3},
            {"name": "cardio", "samples": 70000, "features": 11, "classes": 2},
            {"name": "churn-modelling", "samples": 10000, "features": 11, "classes": 2},
            {"name": "credit-card-fraud", "samples": 284807, "features": 30, "classes": 2},
            {"name": "dermatology", "samples": 358, "features": 34, "classes": 6},
            {"name": "diabetes", "samples": 768, "features": 8, "classes": 2},
            {"name": "gesture-phase", "samples": 9873, "features": 32, "classes": 5},
            {"name": "higgs-small", "samples": 98049, "features": 28, "classes": 2},
            {"name": "miniboone", "samples": 130064, "features": 50, "classes": 2},
            {"name": "wilt", "samples": 4839, "features": 5, "classes": 2},
        )
    )
