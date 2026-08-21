"""Framework card, demos, smoke for latent spin-glass phase diagnosis (arXiv:2606.02600)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.latent_phase.anomaly import ad_improvement, unsupervised_ad_results
from ltx_trainer.latent_phase.config import LatentPhaseConfig
from ltx_trainer.latent_phase.diagnostics import susceptibility_sweep
from ltx_trainer.latent_phase.generation import generation_improvement, generation_results
from ltx_trainer.latent_phase.paper_tables import (
    fig4_generation_cifar10,
    fig5_ad_galaxy_zoo,
    fig5_ad_mars_rover,
    knowledge_card,
)
from ltx_trainer.latent_phase.simulation import phase_diagnosis_demo


def framework_card(cfg: LatentPhaseConfig | None = None) -> dict[str, Any]:
    cfg = cfg or LatentPhaseConfig()
    p = cfg.prior
    return {
        "name": "Latent-Phase",
        "paper": f"arXiv:{cfg.paper_arxiv}",
        "title": "Diagnose AE/VAE latents via spin-glass phase structure",
        "task": "Overlap + susceptibility + block-spin → generation & AD gains",
        "pipeline": [
            "Latent energy Hx(μ) = reconstruction + hyperspherical prior (Eq. 4–5)",
            "Overlap P(R) and χovl(α) order parameters (Eq. 7, 10)",
            "Block-spin coarse-graining to R^3 (Eq. 8)",
            "k-NOT nested angular order (Eq. 9)",
            "Edge-of-stability training via β annealing (Appendix D)",
            "Downstream: self-FID/MSE generation + k-NN AD/OOD",
        ],
        "latent": {
            "dim": p.latent_dim,
            "compression_mode": p.mode,
            "block_spin_dim": p.block_spin_dim,
            "k_nn": p.k_nn,
        },
        "schedule": {
            "epochs": cfg.schedule.total_epochs,
            "beta_warmup_epochs": cfg.schedule.beta_warmup_epochs,
        },
        "edge_alpha_star": cfg.edge_alpha_star,
    }


def paper_limitations() -> list[str]:
    return [
        "Numpy stub — no trained VAE, hyperspherical-coordinate loss, or Freud-style BOO.",
        "Overlap and susceptibility are synthetic samplers, not batch statistics from checkpoints.",
        "Generation self-FID and AD AUROC/FPR95 are table anchors from companion papers.",
        "Block-spin stability is a spread proxy, not MNIST k-NN accuracy under RG transforms.",
        "Mixed p-spin Taylor decay (P0) uses geometric synthetic coefficients, not autograd on Hx.",
    ]


def evaluation_demo(cfg: LatentPhaseConfig | None = None) -> dict[str, Any]:
    cfg = cfg or LatentPhaseConfig()
    demo = phase_diagnosis_demo(cfg=cfg)
    return {
        "framework": framework_card(cfg),
        "limitations": paper_limitations(),
        "knowledge": knowledge_card(),
        "fig4_generation": fig4_generation_cifar10(),
        "fig5_mars": fig5_ad_mars_rover(),
        "fig5_gz": fig5_ad_galaxy_zoo(),
        "generation_table": generation_results(),
        "ad_table": unsupervised_ad_results(),
        "chi_sweep": susceptibility_sweep(alpha_star=cfg.edge_alpha_star),
        "phase_demo": demo,
    }


def evaluation_smoke(cfg: LatentPhaseConfig | None = None) -> dict[str, Any]:
    demo = evaluation_demo(cfg)
    cfg = cfg or LatentPhaseConfig()
    phase = demo["phase_demo"]

    assert phase["overlap_ordered"]["mean"] > phase["overlap_disordered"]["mean"]
    assert phase["block_spin_ordered_stability"] > phase["block_spin_disordered_stability"]
    assert phase["chi_peak"] > 0.2
    assert abs(phase["alpha_star"] - cfg.edge_alpha_star) < 0.25
    assert phase["taylor_decay"]["passes_decay_test"]

    gen = phase["generation_cifar10"]
    assert gen["delta_self_fid"] >= 8.0
    assert gen["delta_mse"] >= 0.5

    ad = phase["ad_galaxy_zoo"]
    assert ad["delta_auroc"] >= 0.04

    mars = demo["fig5_mars"]
    assert mars["compressed_auroc"] - mars["vae_auroc"] >= 0.08

    return {
        "status": "ok",
        "paper": f"arXiv:{cfg.paper_arxiv}",
        "chi_peak": phase["chi_peak"],
        "alpha_star": phase["alpha_star"],
        "delta_self_fid_cifar10": gen["delta_self_fid"],
        "delta_auroc_gz": ad["delta_auroc"],
        "demo_keys": list(demo.keys()),
    }
