"""Paper anchors for latent spin-glass phase diagnosis (arXiv:2606.02600)."""

from __future__ import annotations

from typing import Any


def knowledge_card() -> dict[str, Any]:
    return {
        "title": "High-Dimensional Latents Should Be Diagnosed Through Phase Structure",
        "authors": "Ascárate, Lebrat, Santa Cruz, Fookes, Salvado (QUT)",
        "arxiv": "2606.02600",
        "mapping": "VAE latent ⇔ spherical spin glass; prior ⇔ external field",
        "diagnostics": [
            "overlap P(R)",
            "χovl(α) susceptibility",
            "block-spin coarse-graining to R^3",
            "k-NOT nested angular order",
        ],
        "downstream": ["generation self-FID+MSE", "k-NN AD / OOD FPR95"],
        "latent_dim": 128,
        "training_epochs": 300,
    }


def fig2_replica_angle_modes() -> list[dict[str, str | float]]:
    return [
        {"mode": "zero", "label": "equator / no compression", "target_angle_rad": 1.57},
        {"mode": "half", "label": "field (1,...,1)", "target_angle_rad": 1.05},
        {"mode": "full", "label": "north pole", "target_angle_rad": 0.35},
    ]


def fig4_generation_cifar10() -> dict[str, float]:
    return {"delta_self_fid": 10.0, "delta_mse": 1.0}


def fig5_ad_mars_rover() -> dict[str, float]:
    return {"vae_auroc": 0.66, "compressed_auroc": 0.76}


def fig5_ad_galaxy_zoo() -> dict[str, float]:
    return {"vae_auroc": 0.74, "compressed_auroc": 0.79}


def learning_curve_exponents_stub() -> dict[int, float]:
    """Not in this paper — placeholder for cross-paper spin-glass tooling."""
    return {}
