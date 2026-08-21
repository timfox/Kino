"""CAMERA framework card, tables, and smoke demos (arXiv:2605.20032)."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.camera.config import CAMERAConfig
from ltx_trainer.camera.detector import fraud_scores
from ltx_trainer.camera.layout import LIMITATIONS
from ltx_trainer.camera.losses import expert_loss, gating_entropy_loss, oc_bce_loss, total_loss
from ltx_trainer.camera.mock import init_expert_weights, toy_tag_graph
from ltx_trainer.camera.moe import ego_decoupled_moe_layer


def framework_card(cfg: CAMERAConfig | None = None) -> dict[str, Any]:
    cfg = cfg or CAMERAConfig()
    return {
        "name": "CAMERA",
        "paper": cfg.paper_arxiv,
        "github": cfg.github,
        "task": "Unsupervised Text-Attributed Graph Fraud Detection (TAGFD) under semantic camouflage",
        "idea": (
            "Ego-decoupled Mixture-of-Experts with graph/semantic/global experts, "
            "context-informed gating over ego+neighborhood, and rarity-driven one-class training "
            "without fraud labels."
        ),
        "experts": ["graph (structural residual)", "semantic (autoencoder residual)", "global (prototype residual)"],
        "loss": "L = L_expert + α·L_gating + β·L_OC (Eq. 12)",
        "detector": "s_i = σ(||h_i^{final}||_2) parameter-free OC scorer",
        "datasets": ["Reddit", "Instagram", "AmazonVideo", "YelpChi"],
        "defaults": cfg.__dict__,
    }


def dataset_statistics() -> list[dict[str, int | str]]:
    """Table 4 — dataset statistics (Appendix E)."""
    return [
        {"dataset": "Reddit", "nodes": 18_574, "edges": 64_469, "avg_text_len": 142},
        {"dataset": "Instagram", "nodes": 8_026, "edges": 85_520, "avg_text_len": 15},
        {"dataset": "AmazonVideo", "nodes": 37_126, "edges": 3_658_396, "avg_text_len": 98},
        {"dataset": "YelpChi", "nodes": 67_395, "edges": 16_553_904, "avg_text_len": 139},
    ]


def table_i_main_comparison() -> list[dict[str, Any]]:
    """Table 1 — AUROC / AUPRC (CAMERA row + selected baselines)."""
    return [
        {
            "method": "HUGE",
            "auroc_reddit": 60.64,
            "auroc_instagram": 44.07,
            "auroc_amazon": 51.77,
            "auroc_yelpchi": None,
            "auprc_reddit": 12.80,
        },
        {
            "method": "CoLL",
            "auroc_reddit": 59.26,
            "auroc_instagram": 44.02,
            "auroc_amazon": 50.27,
            "auroc_yelpchi": 47.25,
        },
        {
            "method": "CAMERA",
            "auroc_reddit": 65.09,
            "auroc_instagram": 58.21,
            "auroc_amazon": 63.05,
            "auroc_yelpchi": 61.74,
            "auprc_reddit": 18.99,
            "auprc_instagram": 14.23,
            "auprc_amazon": 17.37,
            "auprc_yelpchi": 18.47,
        },
    ]


def table_ii_text_encoder_ablation() -> list[dict[str, float | str]]:
    """Table 2 — text encoder ablation (AUROC %)."""
    return [
        {"encoder": "OpenAI (Ours)", "reddit": 65.09, "instagram": 58.21, "amazon": 63.05, "yelpchi": 61.74},
        {"encoder": "SentenceBERT", "reddit": 61.45, "instagram": 53.23, "amazon": 53.95, "yelpchi": 53.79},
        {"encoder": "BoW", "reddit": 48.88, "instagram": 51.52, "amazon": 51.58, "yelpchi": 39.93},
    ]


def table_iii_expert_ablation() -> list[dict[str, Any]]:
    """Table 3 — expert participation ablation (AUROC %)."""
    return [
        {
            "graph": True,
            "global": True,
            "semantic": True,
            "reddit": 65.09,
            "instagram": 58.21,
            "amazon": 63.05,
            "yelpchi": 61.74,
        },
        {
            "graph": False,
            "global": True,
            "semantic": True,
            "reddit": 55.74,
            "instagram": 56.55,
            "amazon": 58.97,
            "yelpchi": 60.32,
        },
        {
            "graph": True,
            "global": False,
            "semantic": True,
            "reddit": 59.96,
            "instagram": 56.52,
            "amazon": 57.69,
            "yelpchi": 60.24,
        },
    ]


def figure5_expert_weight_excerpt() -> dict[str, Any]:
    """Fig. 5 — dataset-level expert weight patterns (paper excerpt)."""
    return {
        "reddit_layer1": {"graph": 0.99, "semantic": 0.01, "global": 0.01},
        "reddit_layer2": {"graph": 0.01, "semantic": 0.01, "global": 0.99},
        "instagram": {"graph": 0.33, "semantic": 0.33, "global": 0.34},
        "yelpchi_layer1": {"graph": 0.71, "semantic": 0.27, "global": 0.02},
    }


def hyperparameter_settings() -> list[dict[str, float | int | str]]:
    """Table 5 — per-dataset hyperparameters."""
    return [
        {"dataset": "Reddit", "epochs": 1200, "alpha": 5.0, "beta": 0.1, "lr": 1e-3},
        {"dataset": "Instagram", "epochs": 15, "alpha": 10.0, "beta": 10.0, "lr": 5e-5},
        {"dataset": "AmazonVideo", "epochs": 15, "alpha": 0.1, "beta": 1.0, "lr": 5e-5},
        {"dataset": "YelpChi", "epochs": 450, "alpha": 0.1, "beta": 10.0, "lr": 1e-3},
    ]


def pipeline_demo(cfg: CAMERAConfig | None = None) -> dict[str, Any]:
    cfg = cfg or CAMERAConfig()
    d = cfg.hidden_dim
    h0, adj = toy_tag_graph(n=24, d=d, seed=1)
    w = init_expert_weights(d, seed=2)
    residuals_layers: list[dict[str, np.ndarray]] = []
    h = h0
    gates_all: list[np.ndarray] = []
    for _ in range(cfg.num_moe_layers):
        h, gates, res = ego_decoupled_moe_layer(
            h,
            adj,
            sem_enc_w=w["sem_enc"],
            sem_dec_w=w["sem_dec"],
            global_mlp_w=w["global_mlp"],
            gate_w=w["gate_w"],
            all_h=h,
        )
        residuals_layers.append(res)
        gates_all.append(gates)
    scores = fraud_scores(h)
    lex = expert_loss(residuals_layers)
    lg = gating_entropy_loss(np.vstack(gates_all))
    loc = oc_bce_loss(scores)
    return {
        "num_nodes": h.shape[0],
        "mean_fraud_score": float(scores.mean()),
        "losses": {
            "expert": lex,
            "gating": lg,
            "oc": loc,
            "total": total_loss(expert=lex, gating=lg, oc=loc, alpha=cfg.loss_alpha_gating, beta=cfg.loss_beta_oc),
        },
        "gate_mean_layer_last": gates_all[-1].mean(axis=0).tolist(),
    }


def evaluation_demo(cfg: CAMERAConfig | None = None) -> dict[str, Any]:
    demo = pipeline_demo(cfg)
    demo["framework"] = framework_card(cfg)
    demo["limitations"] = LIMITATIONS
    demo["camera_instagram_auroc"] = 58.21
    return demo


def benchmarks_bundle() -> dict[str, Any]:
    return {
        "dataset_statistics": dataset_statistics(),
        "table_i_main": table_i_main_comparison(),
        "table_ii_encoder": table_ii_text_encoder_ablation(),
        "table_iii_experts": table_iii_expert_ablation(),
        "figure5_weights": figure5_expert_weight_excerpt(),
        "hyperparameters": hyperparameter_settings(),
    }
