"""CAFNet framework card and paper tables (arXiv:2605.29531)."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.cafnet.config import CafNetConfig, DeepfakeClass
from ltx_trainer.cafnet.features import extract_feature_triplet, feature_shapes
from ltx_trainer.cafnet.layout import LIMITATIONS
from ltx_trainer.cafnet.losses import cafnet_loss, class_index
from ltx_trainer.cafnet.boundary_eval import boundary_eval_smoke
from ltx_trainer.cafnet.model import cafnet_forward, mfaan_forward


def framework_card(cfg: CafNetConfig | None = None) -> dict[str, Any]:
    cfg = cfg or CafNetConfig()
    return {
        "name": "CAFNet",
        "paper": cfg.paper_arxiv,
        "github": cfg.github,
        "idea": (
            "Cross-attentive fusion of MFCC, LFCC, and Chroma-STFT for joint ternary "
            "classification (real / fully-fake / half-truth) and splice-boundary "
            "localisation on MLADDC T3 in a single 576k-parameter forward pass."
        ),
        "dataset": {
            "name": "MLADDC",
            "tracks": {"T2": "binary 14 languages", "T3": "half-truth 20 languages"},
            "clip_seconds": cfg.clip_seconds,
            "sample_rate_hz": cfg.sample_rate_hz,
        },
        "features": feature_shapes(cfg),
        "architecture": {
            "encoder": "EnhancedPath (DSConv1d + self-attention)",
            "fusion": f"CrossAttentionFusion ({cfg.attn_heads}-head)",
            "classification": "3-class main + auxiliary heads",
            "temporal": f"BiLSTM ({cfg.bilstm_units}×2) boundary regression",
            "params": cfg.cafnet_params,
        },
        "loss": "L = Lcls + 0.4 Laux + 0.3 Ltemp",
        "limitations": list(LIMITATIONS),
    }


def table_iii_binary_t2() -> list[dict[str, Any]]:
    """Table 3 — MLADDC T2 binary test."""
    return [
        {"model": "MLADDC baseline", "acc": 68.44, "eer": 40.90, "auc": None, "params": None},
        {"model": "XLS-R 300M", "acc": 78.31, "eer": 4.73, "auc": 0.9901, "params": "300M"},
        {"model": "AST 87M", "acc": 93.03, "eer": 7.13, "auc": 0.9810, "params": "87M"},
        {"model": "MFAAN", "acc": 96.37, "eer": 2.21, "auc": None, "params": "323K"},
        {"model": "CAFNet", "acc": 96.76, "eer": 3.20, "auc": 0.9956, "params": "576K"},
    ]


def table_iv_unified() -> dict[str, Any]:
    """Table 4 — CAFNet T2+T3 unified performance."""
    return {
        "accuracy_pct": 92.71,
        "macro_auc_ovr": 0.9910,
        "eer_real_vs_nonreal_pct": 6.07,
        "temporal_mae_s": 0.075,
        "temporal_mae_start_s": 0.083,
        "temporal_mae_end_s": 0.068,
    }


def table_v_per_class() -> list[dict[str, Any]]:
    """Table 5 — per-class metrics."""
    return [
        {"class": "Real", "precision": 0.7651, "recall": 0.9352, "f1": 0.8416, "support": 5600},
        {"class": "Fake", "precision": 0.9691, "recall": 0.9733, "f1": 0.9712, "support": 11200},
        {"class": "Half-truth", "precision": 0.9704, "recall": 0.8919, "f1": 0.9295, "support": 16000},
    ]


def table_vi_localisation() -> dict[str, Any]:
    """Table 6 — boundary localisation on T3."""
    return {
        "start_mae_s": 0.083,
        "start_median_s": 0.060,
        "start_p90_s": 0.153,
        "end_mae_s": 0.068,
        "end_median_s": 0.040,
        "end_p90_s": 0.135,
        "overall_mae_s": 0.075,
        "overall_median_s": 0.052,
        "overall_p90_s": 0.131,
        "within_0_25s_pct": 96.6,
    }


def table_viii_ablation() -> list[dict[str, Any]]:
    """Table 8 — feature ablation on T2."""
    return [
        {"features": "MLADDC baseline LFCC", "val_acc": 68.44, "eer": 40.90},
        {"features": "LFCC only", "val_acc": 96.74, "eer": 2.34},
        {"features": "MFCC + LFCC", "val_acc": 97.96, "eer": 2.27},
        {"features": "MFCC + LFCC + Chroma (MFAAN)", "val_acc": 96.37, "eer": 2.21},
    ]


def table_x_cross_dataset() -> list[dict[str, Any]]:
    """Table 10 — zero-shot cross-dataset."""
    return [
        {"dataset": "MLADDC T2 (train)", "acc": 96.76, "auc": 0.9956, "eer": 3.20},
        {"dataset": "FoR", "acc": 54.25, "auc": 0.9289, "eer": 10.34},
        {"dataset": "WaveFake", "acc": 17.30, "auc": 0.4948, "eer": 50.38},
        {"dataset": "ASVspoof 2019", "acc": 84.68, "auc": 0.5042, "eer": 48.51},
        {"dataset": "In-the-Wild", "acc": 53.62, "auc": 0.5622, "eer": 45.90},
    ]


def table_xi_finetune_collapse() -> list[dict[str, Any]]:
    """Table 11 — cross-dataset AUC before/after MLADDC fine-tuning."""
    return [
        {"dataset": "FoR", "pre_auc": 0.9908, "post_auc": 0.0503},
        {"dataset": "WaveFake", "pre_auc": 0.4948, "post_auc": 0.5291},
        {"dataset": "ASVspoof", "pre_auc": 0.9289, "post_auc": 0.3136},
        {"dataset": "In-the-Wild", "pre_auc": None, "post_auc": 0.4429},
    ]


def headline_results() -> dict[str, Any]:
    return {
        "unified_acc": "92.71% on MLADDC T2+T3",
        "localisation": "MAE 0.075 s, median 0.052 s (first T3 baseline)",
        "binary_t2": "96.76% acc vs XLS-R 78.31% at 500× fewer params",
        "generalisation": "Fine-tuning collapses cross-domain AUC (FoR 0.99→0.05)",
    }


def pipeline_demo(cfg: CafNetConfig | None = None, *, seed: int = 42) -> dict[str, Any]:
    cfg = cfg or CafNetConfig()
    rng = np.random.default_rng(seed)
    n = int(cfg.sample_rate_hz * cfg.clip_seconds)
    wave = rng.standard_normal(n) * 0.05
    feats = extract_feature_triplet(wave, cfg)
    caf = cafnet_forward(feats, cfg=cfg, seed=seed)
    mfa = mfaan_forward(feats, seed=seed)
    # half-truth example with synthetic boundaries
    true_bounds = (0.35, 0.60)
    pred_bounds = (caf["boundary_norm"]["start"], caf["boundary_norm"]["end"])
    loss = cafnet_loss(
        np.array(caf["logits_main"]),
        np.array(caf["logits_aux"]),
        class_index(DeepfakeClass.HALF_TRUTH),
        boundary_pred=pred_bounds,
        boundary_true=true_bounds,
        cfg=cfg,
    )
    boundary_eval = boundary_eval_smoke(cfg, seed=seed)
    return {
        "feature_shapes": {k: list(v.shape) for k, v in feats.items()},
        "cafnet_pred_class": caf["pred_class"],
        "cafnet_boundary_s": caf["boundary_seconds"],
        "cafnet_params": caf["params"],
        "mfaan_pred_fake": mfa["pred_fake"],
        "mfaan_params": mfa["params"],
        "loss_half_truth": loss,
        "boundary_eval": boundary_eval,
    }


def evaluation_demo(*, seed: int = 42) -> dict[str, Any]:
    return {
        "headline": headline_results(),
        "demo": pipeline_demo(seed=seed),
        "framework": framework_card(),
    }


def benchmarks_bundle() -> dict[str, Any]:
    return {
        "table_iii_binary_t2": table_iii_binary_t2(),
        "table_iv_unified": table_iv_unified(),
        "table_v_per_class": table_v_per_class(),
        "table_vi_localisation": table_vi_localisation(),
        "table_viii_ablation": table_viii_ablation(),
        "table_x_cross_dataset": table_x_cross_dataset(),
        "table_xi_finetune_collapse": table_xi_finetune_collapse(),
        "headline": headline_results(),
        "limitations": list(LIMITATIONS),
    }
