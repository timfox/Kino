"""CPC3 WLAM framework card and paper tables (arXiv:2605.23604)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.cpc3_wlam.config import Cpc3WlamConfig
from ltx_trainer.cpc3_wlam.layout import LIMITATIONS
from ltx_trainer.cpc3_wlam.mock import evaluation_smoke


def framework_card(cfg: Cpc3WlamConfig | None = None) -> dict[str, Any]:
    cfg = cfg or Cpc3WlamConfig()
    return {
        "name": "Reference-conditioned Word-Level Modeling with Alignment-Aware Acoustic Fusion",
        "paper": cfg.paper_arxiv,
        "challenge": cfg.challenge,
        "author": "Kazushi Nakazawa (Advanced Media, Inc.)",
        "task": "Text-assisted intelligibility prediction for hearing-impaired listeners",
        "backbone": {
            "small": cfg.backbone_small,
            "medium": cfg.backbone_medium,
            "frozen": True,
        },
        "branches": [
            "Teacher-forced decoder word states",
            "Character-level dynamic top-10 local alignment branch",
            "Utterance-level global acoustic calibration branch",
            "Severity embedding branch",
        ],
        "best_system": {
            "variant": "Joint local/global acoustic fusion",
            "eval_f1": cfg.best_eval_f1,
            "eval_mcc": cfg.best_eval_mcc,
            "eval_corr": cfg.best_eval_corr,
            "eval_rmse": cfg.best_eval_rmse,
        },
        "limitations": LIMITATIONS,
    }


def table_i_main_comparison() -> list[dict[str, Any]]:
    return [
        {"system": "Text-conditioned decoder baseline", "f1": 0.760, "mcc": 0.601, "corr": 0.795, "rmse": 24.92},
        {"system": "+ Word-aligned local acoustic fusion", "f1": 0.776, "mcc": 0.623, "corr": 0.803, "rmse": 24.55},
        {"system": "+ Utterance-level global acoustic fusion", "f1": 0.767, "mcc": 0.609, "corr": 0.802, "rmse": 24.55},
        {"system": "+ Joint local/global acoustic fusion", "f1": 0.778, "mcc": 0.626, "corr": 0.806, "rmse": 24.39},
    ]


def table_ii_severity() -> list[dict[str, Any]]:
    return [
        {"severity": "Mild", "baseline_rmse": 23.63, "joint_rmse": 22.98, "baseline_corr": 0.793, "joint_corr": 0.807},
        {"severity": "Moderate", "baseline_rmse": 25.12, "joint_rmse": 24.67, "baseline_corr": 0.794, "joint_corr": 0.804},
        {"severity": "Moderately severe", "baseline_rmse": 29.15, "joint_rmse": 28.31, "baseline_corr": 0.735, "joint_corr": 0.759},
    ]


def table_iii_diagnostics() -> dict[str, Any]:
    return {
        "alignment_quality": [
            {"system": "Subword-BPE all-head local alignment", "f1": 0.772, "mcc": 0.616, "corr": 0.800, "rmse": 24.66},
            {"system": "Character dynamic top-10 alignment", "f1": 0.776, "mcc": 0.623, "corr": 0.803, "rmse": 24.55},
            {"system": "Oracle-clean alignment", "f1": 0.777, "mcc": 0.624, "corr": 0.804, "rmse": 24.49},
        ],
        "reference_conditioning": [
            {"system": "Hypothesis-derived baseline", "f1": 0.723, "mcc": 0.553, "corr": 0.707, "rmse": 31.32},
            {"system": "Teacher-forced baseline", "f1": 0.760, "mcc": 0.601, "corr": 0.795, "rmse": 24.92},
        ],
        "backbone_scaling": [
            {"system": "Whisper-small joint fusion", "f1": 0.778, "mcc": 0.626, "corr": 0.806, "rmse": 24.39},
            {"system": "Whisper-medium joint fusion", "f1": 0.781, "mcc": 0.628, "corr": 0.807, "rmse": 24.41},
        ],
    }


def headline_results() -> dict[str, Any]:
    cfg = Cpc3WlamConfig()
    return {
        "joint_eval_rmse": cfg.best_eval_rmse,
        "joint_eval_corr": cfg.best_eval_corr,
        "joint_eval_f1": cfg.best_eval_f1,
        "joint_eval_mcc": cfg.best_eval_mcc,
        "rmse_gain_vs_baseline": round(cfg.baseline_rmse - cfg.best_eval_rmse, 2),
        "corr_gain_vs_baseline": round(cfg.best_eval_corr - cfg.baseline_corr, 3),
    }


def evaluation_demo(cfg: Cpc3WlamConfig | None = None) -> dict[str, Any]:
    cfg = cfg or Cpc3WlamConfig()
    return {"config": cfg.__dict__, "smoke": evaluation_smoke()}


def benchmarks_bundle() -> dict[str, Any]:
    return {
        "table_i_main_comparison": table_i_main_comparison(),
        "table_ii_severity": table_ii_severity(),
        "table_iii_diagnostics": table_iii_diagnostics(),
        "headlines": headline_results(),
    }
