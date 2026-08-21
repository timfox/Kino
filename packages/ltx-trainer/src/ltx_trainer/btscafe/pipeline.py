"""BTS-CAFE framework card and paper tables (arXiv:2605.29862)."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.btscafe.config import BTSCafeConfig
from ltx_trainer.btscafe.devices import device_registry, lodo_splits
from ltx_trainer.btscafe.federated import gradient_alignment_smoke, local_loss
from ltx_trainer.btscafe.lodo_eval import lodo_eval_smoke
from ltx_trainer.btscafe.gin import gin_augment
from ltx_trainer.btscafe.interventions import embedding_intervention_analysis
from ltx_trainer.btscafe.layout import LIMITATIONS
from ltx_trainer.btscafe.text_aug import counterfactual_text_augment


def framework_card(cfg: BTSCafeConfig | None = None) -> dict[str, Any]:
    cfg = cfg or BTSCafeConfig()
    return {
        "name": "BTS-CAFE",
        "paper": cfg.paper_arxiv,
        "idea": (
            "FedDG for respiratory sound classification under stethoscope-induced "
            "device shifts. Combines causality-inspired GIN style intervention, "
            "counterfactual text augmentation, and single-sample gradient alignment "
            "on the BTS multimodal CLAP backbone."
        ),
        "scm_paths": {
            "causal": "A → S; C,S → X; C → Y",
            "shortcut": "S ↛ Y (empirical); A,D → T → Ŷ (metadata)",
            "interventions": ["GIN do(S) proxy", "counterfactual T", "gradient alignment"],
        },
        "datasets": ["ICBHI", "SPRSound"],
        "fed_settings": ["LODO AKGC417L/Meditron/Yunting", "Littmann-family leave-out"],
        "hyperparams": {
            "fed_rounds": cfg.fed_rounds,
            "learning_rate": cfg.learning_rate,
            "t_aug": cfg.t_aug,
            "lambda_align": cfg.lambda_align,
            "gain_range": [cfg.gain_min, cfg.gain_max],
            "alpha_min": cfg.alpha_min,
            "p_text_neutralize": cfg.p_text_neutralize,
        },
        "limitations": list(LIMITATIONS),
    }


def figure_ii_embedding_analysis() -> dict[str, Any]:
    """Fig. 2 — CLAP embedding device vs disease k-NN trade-off."""
    cfg = BTSCafeConfig()
    return {
        "raw": {"device_acc": cfg.raw_device_acc, "disease_acc": cfg.raw_disease_acc},
        "mean_subtract": {"device_acc": cfg.mean_device_acc, "disease_acc": cfg.mean_disease_acc},
        "whiten": {"device_acc": cfg.whiten_device_acc, "disease_acc": cfg.whiten_disease_acc},
        "note": "Deterministic style removal degrades pathology signal; GIN diversifies S instead.",
    }


def table_ii_main_results() -> list[dict[str, Any]]:
    """Table 2 — FedDG comparison (ICBHI Score = (Sp+Se)/2, excerpt OOD columns)."""
    return [
        {"method": "FedAvg", "OOD_AKGC417L": 45.71, "OOD_Meditron": 52.33, "OOD_Yunting": 60.58, "LittC2SE": 38.11, "Litt3200": 61.19},
        {"method": "FedSR", "OOD_AKGC417L": 57.27, "OOD_Meditron": 52.85, "OOD_Yunting": 61.86, "LittC2SE": 39.73, "Litt3200": 63.95},
        {"method": "FedCAug", "OOD_AKGC417L": 45.06, "OOD_Meditron": 50.32, "OOD_Yunting": 58.81, "LittC2SE": 41.28, "Litt3200": 65.71},
        {"method": "SpecAugment", "OOD_AKGC417L": 58.40, "OOD_Meditron": 53.47, "OOD_Yunting": 60.09, "LittC2SE": 41.28, "Litt3200": 64.42},
        {"method": "BTS", "OOD_AKGC417L": 60.93, "OOD_Meditron": 52.31, "OOD_Yunting": 62.20, "LittC2SE": 38.11, "Litt3200": 61.19},
        {"method": "BTS-CAFE", "OOD_AKGC417L": 52.82, "OOD_Meditron": 54.60, "OOD_Yunting": 65.69, "LittC2SE": 43.15, "Litt3200": 66.24},
    ]


def table_iii_ablation() -> list[dict[str, Any]]:
    """Table 3 — component ablation (ICBHI Score OOD)."""
    return [
        {"variant": "Full Model", "AKGC417L": 52.82, "Meditron": 54.60, "Yunting": 65.69, "LittC2SE": 43.15, "Litt3200": 66.24},
        {"variant": "w/o GIN", "AKGC417L": 48.15, "Meditron": 50.03, "Yunting": 60.42, "LittC2SE": 39.84, "Litt3200": 62.71},
        {"variant": "w/o Text Augmentation", "AKGC417L": 50.64, "Meditron": 52.18, "Yunting": 63.51, "LittC2SE": 41.36, "Litt3200": 64.11},
        {"variant": "w/o Gradient Alignment", "AKGC417L": 50.91, "Meditron": 52.87, "Yunting": 63.14, "LittC2SE": 41.72, "Litt3200": 64.42},
        {"variant": "Classifier-only alignment", "AKGC417L": 48.72, "Meditron": 49.58, "Yunting": 59.76, "LittC2SE": 38.92, "Litt3200": 61.83},
        {"variant": "Full mini-batch alignment", "AKGC417L": 47.95, "Meditron": 48.86, "Yunting": 58.91, "LittC2SE": 38.27, "Litt3200": 60.94},
    ]


def table_iv_backbone_comparison() -> list[dict[str, Any]]:
    """Table 4 — RSC methods under FedDG (OOD ICBHI Score)."""
    return [
        {"backbone": "AST-CE", "AKGC417L": 48.38, "Meditron": 61.88, "Yunting": 60.00, "LittC2SE": 34.17, "Litt3200": 42.58},
        {"backbone": "SG-SCL", "AKGC417L": 58.64, "Meditron": 57.38, "Yunting": 62.25, "LittC2SE": 38.35, "Litt3200": 52.40},
        {"backbone": "BTS", "AKGC417L": 60.93, "Meditron": 52.31, "Yunting": 62.20, "LittC2SE": 38.11, "Litt3200": 61.19},
        {"backbone": "BTS-CARD", "AKGC417L": 60.72, "Meditron": 51.47, "Yunting": 60.38, "LittC2SE": 40.92, "Litt3200": 61.74},
        {"backbone": "BTS-CAFE", "AKGC417L": 52.82, "Meditron": 54.60, "Yunting": 65.69, "LittC2SE": 43.15, "Litt3200": 66.24},
    ]


def headline_results() -> dict[str, Any]:
    cfg = BTSCafeConfig()
    return {
        "best_ood_setting1": {
            "AKGC417L": cfg.ood_score_akgc417l,
            "Meditron": cfg.ood_score_meditron,
            "Yunting": cfg.ood_score_yunting,
        },
        "best_ood_setting2": {"LittC2SE": cfg.ood_score_littc2se, "Litt3200": cfg.ood_score_litt3200},
        "vs_baselines": "Best OOD ICBHI Score on all three Setting #1 held-out devices",
        "primary_driver": "GIN style diversification (largest ablation drop w/o GIN)",
        "backbone": "BTS multimodal CLAP + metadata text branch",
    }


def pipeline_demo(cfg: BTSCafeConfig | None = None, *, seed: int = 42) -> dict[str, Any]:
    cfg = cfg or BTSCafeConfig()
    rng = np.random.default_rng(seed)
    spec = rng.standard_normal((64, 128)) * 0.1

    gin_out = gin_augment(spec, cfg=cfg, seed=seed)
    text = (
        "This sound was recorded from the left anterior chest of an adult male patient, "
        "using a Litt3200 stethoscope."
    )
    text_out = counterfactual_text_augment(text, cfg=cfg, seed=seed)
    align = gradient_alignment_smoke(num_clients=cfg.num_clients, seed=seed)
    loss = local_loss(0.42, 0.38, align["mean_align_penalty"], round_idx=10, cfg=cfg)

    n, d = 40, 32
    emb = rng.standard_normal((n, d))
    devices = rng.integers(0, 3, size=n)
    diseases = rng.integers(0, 4, size=n)
    emb_analysis = embedding_intervention_analysis(emb, devices, diseases, k=5, cfg=cfg)

    lodo = lodo_eval_smoke(cfg)

    return {
        "num_devices": len(device_registry()),
        "gin_alpha_mean": gin_out["alpha_mean"],
        "text_device_neutralized": text_out["device_neutralized"],
        "gradient_align_penalty": align["mean_align_penalty"],
        "local_loss": loss["loss"],
        "gin_active_round10": loss["gin_active"],
        "embedding_raw_device_acc": emb_analysis["raw"]["device_acc"],
        "lodo_mean_score": lodo["mean_score"],
        "lodo_n_folds": lodo["n_folds"],
    }


def evaluation_demo(*, seed: int = 42) -> dict[str, Any]:
    return {
        "headline": headline_results(),
        "demo": pipeline_demo(seed=seed),
        "framework": framework_card(),
    }


def benchmarks_bundle() -> dict[str, Any]:
    return {
        "devices": device_registry(),
        "lodo_splits": lodo_splits(),
        "figure_ii_embedding_analysis": figure_ii_embedding_analysis(),
        "table_ii_main_results": table_ii_main_results(),
        "table_iii_ablation": table_iii_ablation(),
        "table_iv_backbone_comparison": table_iv_backbone_comparison(),
        "headline": headline_results(),
        "limitations": list(LIMITATIONS),
    }
