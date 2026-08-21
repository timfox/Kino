"""Framework card and benchmark tables."""

from __future__ import annotations

from typing import Any

from ltx_trainer.seam.augmentation import augmentation_demo
from ltx_trainer.seam.config import SeamConfig
from ltx_trainer.seam.model import model_demo
from ltx_trainer.seam.preprocessing import preprocessing_demo
from ltx_trainer.seam.sampling import sampling_demo


def framework_card(cfg: SeamConfig | None = None) -> dict[str, Any]:
    cfg = cfg or SeamConfig()
    return {
        "paper": cfg.paper_arxiv,
        "title": cfg.title,
        "framework": cfg.framework,
        "task": "scripted_vs_spontaneous",
        "deployment": "real_time_interview_guardrails",
        "backbone": cfg.backbone,
        "components": [
            "uniform_waveform_preprocessing",
            "seam_aware_provenance_sampling",
            "non_speech_noise_bank_augmentation",
            "distilhubert_top1_unfreeze",
            "transfer_oriented_evaluation",
        ],
        "corpora": {
            "spontaneous": list(cfg.spontaneous_corpora),
            "scripted": list(cfg.scripted_corpora),
        },
        "window_s": cfg.window_s,
        "headline": {
            "ext_auc": cfg.ext_auc,
            "ext_auc_std": cfg.ext_auc_std,
            "int4_vram_mb": cfg.int4_vram_mb,
        },
    }


def table1_full_training() -> list[dict[str, Any]]:
    """Table 1 — full training regime (mean ± std across 3 seeds)."""
    return [
        {"seed": 1337, "eval_acc": 0.9690, "eval_auc": 0.9820, "test_acc": 0.9639, "test_auc": 0.9781, "ext_acc": 0.9540, "ext_auc": 0.9725},
        {"seed": 1338, "eval_acc": 0.9602, "eval_auc": 0.9762, "test_acc": 0.9548, "test_auc": 0.9715, "ext_acc": 0.9431, "ext_auc": 0.9669},
        {"seed": 1339, "eval_acc": 0.9728, "eval_auc": 0.9835, "test_acc": 0.9681, "test_auc": 0.9802, "ext_acc": 0.9580, "ext_auc": 0.9745},
        {"seed": "mean", "eval_acc": 0.9673, "eval_auc": 0.9806, "test_acc": 0.9623, "test_auc": 0.9766, "ext_acc": 0.9517, "ext_auc": 0.9713},
        {"seed": "std", "eval_acc": 0.0065, "eval_auc": 0.0039, "test_acc": 0.0068, "test_auc": 0.0045, "ext_acc": 0.0077, "ext_auc": 0.0039},
    ]


def table2_shortcut_ablation() -> list[dict[str, Any]]:
    """Table 2 — shortcut-prevention ablation (fixed-budget)."""
    return [
        {"setting": "baseline on", "eval_auc": 0.9550, "test_auc": 0.9287, "ext_acc": 0.8527, "ext_auc": 0.8991},
        {"setting": "seam off", "eval_auc": 0.9638, "test_auc": 0.9328, "ext_acc": 0.8179, "ext_auc": 0.8674},
        {"setting": "noise off", "eval_auc": 0.9792, "test_auc": 0.9491, "ext_acc": 0.7089, "ext_auc": 0.7518},
        {"setting": "noise off + seam off", "eval_auc": 0.9848, "test_auc": 0.9557, "ext_acc": 0.6882, "ext_auc": 0.7324},
    ]


def table3_window_length() -> list[dict[str, Any]]:
    """Table 3 — analysis window length (fixed-budget)."""
    return [
        {"window_s": 2, "test_acc": 0.7229, "test_auc": 0.7739, "ext_acc": 0.6572, "ext_auc": 0.7485},
        {"window_s": 4, "test_acc": 0.8629, "test_auc": 0.8889, "ext_acc": 0.8067, "ext_auc": 0.8646},
        {"window_s": 8, "test_acc": 0.8949, "test_auc": 0.9287, "ext_acc": 0.8527, "ext_auc": 0.8991},
        {"window_s": 12, "test_acc": 0.8786, "test_auc": 0.9067, "ext_acc": 0.8349, "ext_auc": 0.8813},
    ]


def table4_adaptation_depth() -> list[dict[str, Any]]:
    """Table 4 — partial fine-tuning depth (fixed-budget)."""
    return [
        {"setting": "head", "eval_auc": 0.7799, "test_acc": 0.7078, "test_auc": 0.7625},
        {"setting": "tr1", "eval_auc": 0.9550, "test_acc": 0.8949, "test_auc": 0.9287},
        {"setting": "tr2", "eval_auc": 0.9296, "test_acc": 0.8502, "test_auc": 0.9031},
        {"setting": "cnn2", "eval_auc": 0.7631, "test_acc": 0.6887, "test_auc": 0.7354},
    ]


def table5_quantization() -> list[dict[str, Any]]:
    """Table 5 — post-training quantization on external set."""
    return [
        {"precision": "AMP (base)", "vram_mb": 90.37, "ext_acc": 0.9517, "ext_auc": 0.9713, "latency_ms_per_win": 6.99},
        {"precision": "INT8", "vram_mb": 48.74, "ext_acc": 0.9530, "ext_auc": 0.9700, "latency_ms_per_win": 7.86},
        {"precision": "INT4", "vram_mb": 41.80, "ext_acc": 0.9535, "ext_auc": 0.9743, "latency_ms_per_win": 7.25},
    ]


def table6_backbone_screening() -> list[dict[str, Any]]:
    """Table 6 — frozen SSL backbone screening."""
    return [
        {"backbone": "DistilHuBERT", "test_acc": 0.754, "test_auc": 0.848, "params_m": 23.49, "size_mb": 93.97, "rtf": 5.60e-4},
        {"backbone": "Distil-wav2vec2", "test_acc": 0.736, "test_auc": 0.822, "params_m": 51.84, "size_mb": 207.52, "rtf": 6.76e-4},
        {"backbone": "HuBERT Base", "test_acc": 0.779, "test_auc": 0.868, "params_m": 94.37, "size_mb": 377.57, "rtf": 8.37e-4},
        {"backbone": "wav2vec2 Base", "test_acc": 0.778, "test_auc": 0.868, "params_m": 94.37, "size_mb": 377.61, "rtf": 8.33e-4},
        {"backbone": "WavLM Base+", "test_acc": 0.803, "test_auc": 0.895, "params_m": 94.38, "size_mb": 377.62, "rtf": 1.53e-3},
    ]


def benchmarks_bundle() -> dict[str, Any]:
    return {
        "table1_full_training": table1_full_training(),
        "table2_shortcut_ablation": table2_shortcut_ablation(),
        "table3_window_length": table3_window_length(),
        "table4_adaptation_depth": table4_adaptation_depth(),
        "table5_quantization": table5_quantization(),
        "table6_backbone_screening": table6_backbone_screening(),
    }


def headline_results(cfg: SeamConfig | None = None) -> dict[str, Any]:
    cfg = cfg or SeamConfig()
    return {
        "ext_auc": cfg.ext_auc,
        "ext_auc_std": cfg.ext_auc_std,
        "test_auc": cfg.test_auc,
        "int4_vram_mb": cfg.int4_vram_mb,
        "int4_ext_auc": cfg.int4_ext_auc,
    }


def evaluation_demo(*, seed: int = 0, cfg: SeamConfig | None = None) -> dict[str, Any]:
    cfg = cfg or SeamConfig()
    return {
        "framework": framework_card(cfg),
        "preprocessing": preprocessing_demo(seed=seed, cfg=cfg),
        "sampling": sampling_demo(seed=seed, cfg=cfg),
        "augmentation": augmentation_demo(seed=seed, cfg=cfg),
        "model": model_demo(seed=seed, cfg=cfg),
        "headline": headline_results(cfg),
    }
