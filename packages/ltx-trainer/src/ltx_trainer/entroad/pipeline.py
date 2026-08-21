"""EntroAD framework card, knowledge, and evaluation demos."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.entroad.benchmarks import benchmarks_bundle
from ltx_trainer.entroad.config import EntroADConfig
from ltx_trainer.entroad.dual_branch import (
    branch_prompt_bias,
    cosine_branch_maps,
    fuse_branch_maps,
    image_score_from_map,
)
from ltx_trainer.entroad.entropy import structural_entropy_map
from ltx_trainer.entroad.losses import stage2_branch_loss
from ltx_trainer.entroad.routing import routed_token_pair


def framework_card(cfg: EntroADConfig | None = None) -> dict[str, Any]:
    cfg = cfg or EntroADConfig()
    return {
        "name": "EntroAD",
        "paper": cfg.paper_arxiv,
        "title": "Structural Entropy-Guided Prompt Adaptation for Zero-Shot Anomaly Detection",
        "task": "cross_dataset_ZSAD",
        "stages": {
            "stage1": "memory-guided retrieval (image + patch)",
            "stage2": "structural entropy routing + dual-branch CLIP prompts",
        },
        "backbone": cfg.backbone,
        "train_on": cfg.train_source,
        "eval_datasets": list(cfg.test_datasets),
        "key_hyperparams": {
            "router_T": cfg.router_temperature,
            "lambda_A": cfg.lambda_a,
            "fusion_alpha": cfg.fusion_alpha,
        },
    }


def knowledge_card(cfg: EntroADConfig | None = None) -> dict[str, Any]:
    cfg = cfg or EntroADConfig()
    bench = benchmarks_bundle()
    return {
        "framework": framework_card(cfg),
        "key_results": {
            "image_avg_auroc_ap": bench["table1_image_average"]["EntroAD"],
            "pixel_avg_auroc_aupro": bench["table2_pixel_average"]["EntroAD"],
            "mvtec_image": bench["table1_entroad"]["MVTec-AD"],
            "gating_ablation_drop_aupro": bench["gating_critical_mvtec_pix"],
        },
        "integration": {
            "env": "GOPEX_ENTROAD=1",
            "fold_hook": "GOPEX_AV_FOLD_HOOKS=...,entroad",
            "cli": "./scripts/gopex-entroad.sh",
        },
    }


def evaluation_demo(cfg: EntroADConfig | None = None) -> dict[str, Any]:
    cfg = cfg or EntroADConfig()
    rng = np.random.default_rng(11)
    n = 16
    d = 16
    z = rng.standard_normal((n, d))
    r = z + rng.standard_normal((n, d)) * 0.05
    attn = np.clip(z @ z.T, 0, None)
    e = structural_entropy_map(attn[None, ...], drop_cls=False)
    mem_k = rng.standard_normal((8, d))
    mem_v = np.array([[1.0, 0.0]] * 8)

    tokens = routed_token_pair(z, r, e, mem_k, mem_v, cfg=cfg)
    b_a = branch_prompt_bias(tokens["t_n"], tokens["t_a"], seed=1)
    b_b = branch_prompt_bias(tokens["t_n"], tokens["t_a"], seed=2)
    un = rng.standard_normal(d)
    ua = rng.standard_normal(d)
    m_n_a, m_a_a = cosine_branch_maps(z, un + b_a[:d], ua + b_a[:d])
    m_n_b, m_a_b = cosine_branch_maps(z, un + b_b[:d], ua + b_b[:d])
    fused = fuse_branch_maps(m_a_a, m_a_b, cfg=cfg)
    score = image_score_from_map(fused, retrieval_score=0.6, cfg=cfg)

    gt = np.zeros(n)
    gt[:4] = 1.0
    loss_a = stage2_branch_loss(m_a_a, m_n_a, gt)

    return {
        "n_patches": n,
        "entropy_mean": float(e.mean()),
        "gate": tokens["gate"],
        "fused_map_mean": float(fused.mean()),
        "image_anomaly_score": score,
        "stage2_branch_loss_stub": loss_a,
        "paper_tables": benchmarks_bundle(),
    }


def evaluation_smoke(cfg: EntroADConfig | None = None) -> dict[str, Any]:
    cfg = cfg or EntroADConfig()
    ev = evaluation_demo(cfg)
    bench = ev["paper_tables"]
    return {
        "package": "ltx_trainer.entroad",
        "status": "smoke_ok",
        "paper": cfg.paper_arxiv,
        "gate_finite": bool(np.isfinite(ev["gate"])),
        "paper_sota_image": bench["entroad_beats_mrad_image_auroc"],
        "paper_sota_pixel": bench["entroad_beats_mrad_pixel_aupro"],
    }
