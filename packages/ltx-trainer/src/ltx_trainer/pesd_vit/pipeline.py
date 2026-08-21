"""PESD-ViT framework card, knowledge, and evaluation demos."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.pesd_vit.adapter import TaskAdapter, adapter_param_count
from ltx_trainer.pesd_vit.activations import task_activation_demo
from ltx_trainer.pesd_vit.benchmarks import benchmarks_bundle
from ltx_trainer.pesd_vit.config import PesdVitConfig
from ltx_trainer.pesd_vit.correlation import NAS_PEARSON_OURS, negative_transfer_risk
from ltx_trainer.pesd_vit.mtl_loss import total_training_loss
from ltx_trainer.pesd_vit.ortho import orthogonal_decoupling_loss, subspace_overlap_matrix


def framework_card(cfg: PesdVitConfig | None = None) -> dict[str, Any]:
    cfg = cfg or PesdVitConfig()
    params = adapter_param_count(cfg.feature_dim, cfg.adapter_rank, cfg.num_tasks)
    return {
        "name": "PESD-ViT",
        "paper": cfg.paper_arxiv,
        "title": "Parameter-Efficient Subspace Decoupling ViT for NAFLD Histological Scoring",
        "backbone": cfg.backbone,
        "tasks": list(cfg.task_names),
        "adapter_rank": cfg.adapter_rank,
        "ortho_lambda": cfg.ortho_lambda,
        "trainable_params": params,
        "dataset_patches": cfg.dataset_patches,
    }


def knowledge_card(cfg: PesdVitConfig | None = None) -> dict[str, Any]:
    cfg = cfg or PesdVitConfig()
    bench = benchmarks_bundle()
    return {
        "framework": framework_card(cfg),
        "nas_correlation": NAS_PEARSON_OURS.tolist(),
        "negative_transfer_risk": negative_transfer_risk(NAS_PEARSON_OURS),
        "key_results": {
            "ours_steatosis_acc": bench["table3_vs_inception"]["Ours"]["steatosis"],
            "ours_inflammation_acc": bench["table3_vs_inception"]["Ours"]["inflammation"],
            "beats_inception": bench["beats_inception_all_tasks"],
            "best_ortho_lambda": cfg.ortho_lambda,
        },
        "integration": {
            "env": "GOPEX_PESD_VIT=1",
            "fold_hook": "GOPEX_AV_FOLD_HOOKS=...,pesd_vit",
            "cli": "./scripts/gopex-pesd-vit.sh",
        },
    }


def evaluation_demo(cfg: PesdVitConfig | None = None) -> dict[str, Any]:
    cfg = cfg or PesdVitConfig()
    rng = np.random.default_rng(0)
    d, r = cfg.feature_dim, cfg.adapter_rank

    adapters = {
        t: TaskAdapter.init(t, d, r, rng=rng)
        for t in cfg.task_names
    }
    downs = {t: adapters[t].down for t in adapters}
    l_ortho = orthogonal_decoupling_loss(downs)
    overlap = subspace_overlap_matrix(downs)

    x = rng.standard_normal(d)
    adapted = {t: adapters[t].forward(x) for t in adapters}

    task_losses = [0.42, 0.18, 0.55]
    log_sigma2 = np.array([0.0, -0.5, -0.2])
    loss_pack = total_training_loss(
        task_losses,
        log_sigma2.tolist(),
        l_ortho,
        ortho_lambda=cfg.ortho_lambda,
        manual_weights=cfg.task_manual_weights,
    )

    latent = rng.standard_normal((49, 32))
    activations = {t: task_activation_demo(latent, t) for t in cfg.task_names}

    return {
        "adapter_forward_norms": {t: float(np.linalg.norm(adapted[t])) for t in adapted},
        "l_ortho": round(l_ortho, 6),
        "subspace_overlap": overlap,
        "loss": loss_pack,
        "activations_shapes": {t: activations[t]["norm"].shape for t in activations},
        "paper_tables": benchmarks_bundle(),
        "pearson_risk": negative_transfer_risk(NAS_PEARSON_OURS),
    }


def evaluation_smoke(cfg: PesdVitConfig | None = None) -> dict[str, Any]:
    cfg = cfg or PesdVitConfig()
    ev = evaluation_demo(cfg)
    bench = ev["paper_tables"]
    return {
        "package": "ltx_trainer.pesd_vit",
        "status": "smoke_ok",
        "paper": cfg.paper_arxiv,
        "beats_inception": bench["beats_inception_all_tasks"],
        "ortho_helps_inflammation": bench["ortho_lambda_01_best"],
        "l_ortho_finite": ev["l_ortho"] >= 0.0,
    }
