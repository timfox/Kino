"""Hyperparameters for physics steering (Alam, arXiv:2605.24322)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PhysicsSteeringConfig:
    """VideoMAE-base defaults where noted; small dims for unit tests."""

    paper_arxiv: str = "arXiv:2605.24322"
    model_id: str = "MCG-NJU/videomae-base"
    hidden_dim: int = 768
    num_layers: int = 12
    num_frames: int = 16
    spatial_size: int = 224

    pez_epsilon: float = 0.05
    """Eq. (3): layers within ε of peak probe accuracy."""

    primary_pez_layer: int = 5
    top_pez_layers: tuple[int, ...] = (5, 0, 1)

    pca_components: int = 64
    """PCA before logistic fit when N ≪ D (Sec. 3.3)."""

    logistic_c: float = 1.0
    logistic_max_iter: int = 1000

    steering_saturation_alpha: float = 5.0
    """|α|≈5 saturates P(impossible) per Tab. 2."""

    intphys_blocks: tuple[str, ...] = ("O1", "O2", "O3")
    train_size: int = 216
    val_size: int = 72
    test_size: int = 72

    probe_train_steps: int = 200
    """Torch LBFGS-style steps for smoke probes (full paper uses sklearn)."""

    orthogonal_probe_max_iters: int = 5

    ltx_bridge_dim: int = 64
    """Hidden dim for LTX validation QA bridge smoke."""
