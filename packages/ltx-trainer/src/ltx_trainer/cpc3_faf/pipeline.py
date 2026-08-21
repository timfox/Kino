"""CPC3 FAF framework card and paper tables (arXiv:2605.23619)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.cpc3_faf.config import Cpc3FafConfig
from ltx_trainer.cpc3_faf.layout import LIMITATIONS
from ltx_trainer.cpc3_faf.mock import evaluation_smoke


def framework_card(cfg: Cpc3FafConfig | None = None) -> dict[str, Any]:
    cfg = cfg or Cpc3FafConfig()
    return {
        "name": "Frame-Aligned Fusion (Canary + WavLM)",
        "paper": cfg.paper_arxiv,
        "challenge": cfg.challenge,
        "author": "Kazushi Nakazawa (Advanced Media, Inc.)",
        "task": "Non-intrusive intelligibility prediction for hearing-aid-processed binaural speech",
        "encoders": {
            "canary": {"model": cfg.canary_model, "layers": list(cfg.canary_layers), "hz": cfg.canary_hz},
            "wavlm": {"model": cfg.wavlm_model, "layers": list(cfg.wavlm_layers_main), "hz": cfg.wavlm_hz},
        },
        "fusion_family": [
            "Canary-only / WavLM-only baselines",
            "Uniform score averaging",
            "Pool-late fusion",
            "Frame-aligned fusion (avg or conv preparation)",
            "Cross-attention / reverse alignment",
        ],
        "best_system": {
            "variant": "Frame-aligned fusion, conv preparation",
            "eval_rmse": cfg.best_eval_rmse,
            "eval_corr": cfg.best_eval_corr,
            "trainable_params_m": 1.30,
        },
        "training": {
            "optimizer": "AdamW",
            "lr": cfg.learning_rate,
            "batch_size": cfg.batch_size,
            "loss": "MSE on normalized targets",
            "folds": "5-fold CV grouped by scene, 5 seeds",
        },
        "limitations": LIMITATIONS,
    }


def table_i_main_comparison() -> list[dict[str, Any]]:
    """Table I — main comparison (evaluation set excerpts)."""
    return [
        {
            "system": "Canary-only baseline",
            "params_m": 1.60,
            "eval_rmse": 25.64,
            "eval_corr": 0.784,
        },
        {
            "system": "WavLM-only baseline",
            "params_m": 1.60,
            "eval_rmse": 26.62,
            "eval_corr": 0.766,
        },
        {
            "system": "Uniform score avg.",
            "params_m": 3.20,
            "eval_rmse": 25.53,
            "eval_corr": 0.784,
        },
        {
            "system": "Pool-late fusion",
            "params_m": 1.69,
            "eval_rmse": 25.57,
            "eval_corr": 0.786,
        },
        {
            "system": "Frame-aligned fusion (Avg)",
            "params_m": 1.15,
            "eval_rmse": 25.03,
            "eval_corr": 0.794,
        },
        {
            "system": "Frame-aligned fusion (Conv)",
            "params_m": 1.30,
            "eval_rmse": 24.96,
            "eval_corr": 0.796,
        },
        {
            "system": "Cross-attention fusion",
            "params_m": 1.52,
            "eval_rmse": 25.62,
            "eval_corr": 0.785,
        },
    ]


def table_ii_diagnostics() -> dict[str, Any]:
    """Table II — reverse alignment, WavLM layers, temporal shift."""
    return {
        "reverse_alignment": [
            {"method": "Canary-up, linear", "eval_rmse": 25.26, "corr": 0.791},
            {"method": "Canary-up, transp. conv.", "eval_rmse": 25.46, "corr": 0.788},
        ],
        "wavlm_only_layers": [
            {"layers": "5–12", "eval_rmse": 28.12, "corr": 0.743},
            {"layers": "17–24", "eval_rmse": 26.62, "corr": 0.766},
        ],
        "temporal_shift_ms": [
            {"shift": "-320 ms", "eval_rmse": 25.12},
            {"shift": "0 ms", "eval_rmse": 24.96},
            {"shift": "+160 ms", "eval_rmse": 24.93},
        ],
    }


def table_iii_robustness() -> dict[str, Any]:
    """Table III — severity and enhancement-system macro summary."""
    return {
        "severity_frame_aligned_conv": [
            {"severity": "Mild", "rmse": 24.20, "corr": 0.783},
            {"severity": "Moderate", "rmse": 25.26, "corr": 0.793},
            {"severity": "Mod.-severe", "rmse": 25.48, "corr": 0.806},
        ],
        "enhancement_macro": [
            {"system": "Canary-only", "rmse": 24.57, "corr": 0.641, "mae": 17.99},
            {"system": "Uniform score avg.", "rmse": 24.43, "corr": 0.647, "mae": 18.85},
            {"system": "Frame-align. Conv", "rmse": 23.95, "corr": 0.661, "mae": 17.80},
        ],
        "win_rates_vs_canary": {"frame_align_rmse": "9/9", "frame_align_corr": "9/9", "frame_align_mae": "6/9"},
    }


def headline_results() -> dict[str, Any]:
    cfg = Cpc3FafConfig()
    return {
        "best_eval_rmse": cfg.best_eval_rmse,
        "best_eval_corr": cfg.best_eval_corr,
        "improvement_over_canary_rmse": round(cfg.canary_only_eval_rmse - cfg.best_eval_rmse, 2),
        "improvement_over_score_avg_rmse": round(cfg.score_avg_eval_rmse - cfg.best_eval_rmse, 2),
    }


def evaluation_demo(cfg: Cpc3FafConfig | None = None) -> dict[str, Any]:
    cfg = cfg or Cpc3FafConfig()
    return {"config": cfg.__dict__, "smoke": evaluation_smoke()}


def benchmarks_bundle() -> dict[str, Any]:
    return {
        "table_i_main_comparison": table_i_main_comparison(),
        "table_ii_diagnostics": table_ii_diagnostics(),
        "table_iii_robustness": table_iii_robustness(),
        "headlines": headline_results(),
    }
