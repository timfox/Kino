"""Cert-LAS framework card, knowledge, and evaluation demos."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.cert_las.auditing import audit_backdoor_method
from ltx_trainer.cert_las.benchmarks import benchmarks_bundle
from ltx_trainer.cert_las.config import CertLASConfig
from ltx_trainer.cert_las.lfs import (
    allocate_layer_sigmas,
    budget_equivalent_uniform_sigma,
    layer_dims,
    layer_fine_tuning_sensitivity,
    synthetic_unet_layer_deltas,
)
from ltx_trainer.cert_las.smoothing import certified_radius_grid, mahalanobis_norm, sample_layer_noise
from ltx_trainer.cert_las.verify import ownership_threshold, simulate_wr_rp, verify_ownership
from ltx_trainer.cert_las.watermark import exponential_growth_schedule, kl_watermark_loss


def framework_card(cfg: CertLASConfig | None = None) -> dict[str, Any]:
    cfg = cfg or CertLASConfig()
    return {
        "name": "Cert-LAS",
        "paper": cfg.paper_arxiv,
        "title": "Certified Model Ownership Verification for T2I Diffusion via Layer-Adaptive Smoothing",
        "stages": {
            "embed": "LFS-guided layer-adaptive noise + diffusion-classifier KL watermark",
            "verify": "WR vs RP paired t-test + certified Mahalanobis radius",
        },
        "backbone": cfg.backbone,
        "class_prompts": list(cfg.class_prompts),
        "trigger_free": True,
        "key_hyperparams": {
            "sigma_u": cfg.sigma_uniform,
            "target_lambda": cfg.target_lambda,
            "verify_MN": [cfg.verify_M, cfg.verify_N],
        },
    }


def knowledge_card(cfg: CertLASConfig | None = None) -> dict[str, Any]:
    cfg = cfg or CertLASConfig()
    bench = benchmarks_bundle()
    return {
        "framework": framework_card(cfg),
        "key_results": {
            "t_at_1e6": bench["table2_watermark"]["Cert-LAS (w/o smooth)"]["t_at_1e6"],
            "vsr_sigma1": bench["table2_watermark"]["Cert-LAS (w/ smooth)"]["vsr"],
            "R_bar_sigma1": bench["certified_radius_sigma1"],
            "stealth_sin_p": bench["table2_watermark"]["Cert-LAS (w/o smooth)"]["sin_p"],
            "l2_budget_08": bench["table7_l2_08"]["Cert-LAS (w/ smooth)"],
        },
        "integration": {
            "env": "GOPEX_CERT_LAS=1",
            "fold_hook": "GOPEX_AV_FOLD_HOOKS=...,cert_las",
            "cli": "./scripts/gopex-cert-las.sh",
        },
    }


def evaluation_demo(cfg: CertLASConfig | None = None) -> dict[str, Any]:
    cfg = cfg or CertLASConfig()
    deltas = synthetic_unet_layer_deltas()
    lfs = layer_fine_tuning_sensitivity(deltas)
    dims = layer_dims(deltas)
    sigmas = allocate_layer_sigmas(lfs, dims, sigma_u=cfg.sigma_uniform)
    sigma_u_check = budget_equivalent_uniform_sigma(sigmas, dims)

    noise = sample_layer_noise(sigmas, dims, rng=np.random.default_rng(3), scale_k=cfg.sigma_scale_k)
    peturbed = {k: deltas[k] + noise[k] for k in deltas}
    r_norm = mahalanobis_norm(
        {k: peturbed[k] - deltas[k] for k in deltas},
        {k: sigmas[k] * cfg.sigma_scale_k for k in sigmas},
    )

    wr, rp = simulate_wr_rp(
        p_suspect=0.92,
        p_ref=0.10,
        M=cfg.verify_M,
        N=cfg.verify_N,
        seed=7,
    )
    verdict = verify_ownership(wr, rp, M=cfg.verify_M, N=cfg.verify_N, zeta=cfg.rp_zeta_upper)
    tau = ownership_threshold(M=cfg.verify_M, N=cfg.verify_N, zeta=cfg.rp_zeta_upper)

    ps = [0.85, 0.80, 0.75, 0.70, 0.65]
    r_star = certified_radius_grid(ps, tau=tau, k=cfg.sigma_scale_k)

    rng = np.random.default_rng(5)
    probs = np.array([0.45, 0.55])
    kl = kl_watermark_loss(probs, cfg=cfg)
    m_t, omega = exponential_growth_schedule(100, Tg=cfg.exponential_Tg, m_max=cfg.m_noise_max)

    audit = audit_backdoor_method(
        method="cert_las",
        prompt="a photo of a cat on grass",
        m_plus=0.42,
        m_minus=0.41,
    )

    return {
        "lfs_top_layer": max(lfs, key=lfs.get),
        "sigma_u_matched": round(sigma_u_check, 6),
        "mahalanobis_noise_norm": round(r_norm, 4),
        "wr": round(wr, 4),
        "rp": round(rp, 4),
        "ownership": verdict,
        "threshold_tau": round(tau, 4),
        "certified_R_bar": round(r_star, 4),
        "kl_loss_stub": round(kl, 4),
        "exp_schedule_m_t": m_t,
        "exp_schedule_omega_scale": omega,
        "audit_scores": audit,
        "paper_tables": benchmarks_bundle(),
    }


def evaluation_smoke(cfg: CertLASConfig | None = None) -> dict[str, Any]:
    cfg = cfg or CertLASConfig()
    ev = evaluation_demo(cfg)
    bench = ev["paper_tables"]
    return {
        "package": "ltx_trainer.cert_las",
        "status": "smoke_ok",
        "paper": cfg.paper_arxiv,
        "verified": ev["ownership"]["verified"],
        "cert_las_stealth": bench["cert_las_beats_backdoor_stealth"],
        "cert_las_l2_robust": bench["cert_las_robust_l2"],
    }
