"""KSAA-2026 Task 2 framework card and paper tables (arXiv:2605.25928)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.ksaa_diac.architecture import architecture_summary
from ltx_trainer.ksaa_diac.config import KsaaDiacConfig
from ltx_trainer.ksaa_diac.layout import LIMITATIONS
from ltx_trainer.ksaa_diac.mock import evaluation_smoke


def framework_card(cfg: KsaaDiacConfig | None = None) -> dict[str, Any]:
    cfg = cfg or KsaaDiacConfig()
    return {
        "name": "Thaka KSAA-2026 Task 2",
        "paper": cfg.paper_arxiv,
        "team": cfg.team,
        "task": cfg.task,
        "base_model": "CATT-Whisper (Ghannam et al., 2025)",
        "rank": "1st on test WER (primary metric)",
        "training_regularization": [
            "R-Drop (α=2.08)",
            "Focal Loss (γ=0.34) + label smoothing",
            "Optuna hyperparameters + high weight decay",
            "SpecAugment + Gaussian noise",
        ],
        "inference": (
            f"{cfg.num_checkpoints} checkpoints × {cfg.mc_passes_per_checkpoint} "
            f"MC Dropout passes = {cfg.num_checkpoints * cfg.mc_passes_per_checkpoint} softmax averages"
        ),
        "data": {
            "train_filtered": cfg.train_samples_filtered,
            "dev": cfg.dev_samples,
            "test": cfg.test_samples,
            "filter": f"diacritic ratio ≥ {cfg.min_diacritic_ratio}",
        },
        "architecture": architecture_summary(cfg),
        "hyperparameters": table_i_hyperparameters(),
    }


def table_i_hyperparameters() -> dict[str, Any]:
    """Table 1 — Optuna-selected hyperparameters."""
    cfg = KsaaDiacConfig()
    return {
        "learning_rate": cfg.learning_rate,
        "rdrop_alpha": cfg.rdrop_alpha,
        "focal_gamma": cfg.focal_gamma,
        "label_smoothing": cfg.label_smoothing,
        "weight_decay": cfg.weight_decay,
        "speech_emb_dropout": cfg.speech_emb_dropout,
        "batch_size": cfg.batch_size,
        "epochs": cfg.epochs,
        "warmup_epochs": 3,
        "min_lr_factor": 0.002,
        "specaugment_freq": cfg.specaug_freq,
        "specaugment_time": cfg.specaug_time,
        "noise_snr_db": "10–30",
        "whisper_unfrozen_blocks": 0,
        "optuna_trials": cfg.optuna_trials,
    }


def table_ii_leaderboard() -> list[dict[str, Any]]:
    """Table 2 — KSAA-2026 Task 2 test set (WER primary)."""
    return [
        {"system": "meshal (Ours)", "der": 6.87, "wer": 23.26, "ser": 66.16, "rank": 1},
        {"system": "nadaadelmousa", "der": 7.04, "wer": 24.39, "ser": 71.65, "rank": 2},
        {"system": "naif_alharthi", "der": 7.51, "wer": 25.34, "ser": 73.48, "rank": 3},
        {"system": "nahian_abu", "der": 8.23, "wer": 30.37, "ser": 80.79, "rank": 4},
        {"system": "Hassan", "der": 10.56, "wer": 34.47, "ser": 79.88, "rank": 5},
        {"system": "omarnj10", "der": 27.94, "wer": 44.05, "ser": 98.78, "rank": 6},
        {"system": "astral_fate", "der": 31.67, "wer": 84.50, "ser": 99.70, "rank": 7},
        {"system": "Baseline (FT text+ASR)", "der": 9.91, "wer": 31.84, "ser": 82.93, "rank": None},
        {"system": "Baseline (text+ASR)", "der": 13.50, "wer": 40.24, "ser": 82.32, "rank": None},
        {"system": "Baseline (text-only)", "der": 17.66, "wer": 49.85, "ser": 91.77, "rank": None},
    ]


def table_iii_ablation() -> list[dict[str, Any]]:
    """Table 3 — cumulative dev ablation."""
    return [
        {"configuration": "CATT-Whisper (pretrained)", "der": 17.76, "wer": 54.06},
        {"configuration": "CATT-Whisper (fine-tuned)", "der": 8.59, "wer": 30.43},
        {"configuration": "+ Regularized recipe", "der": 7.57, "wer": 27.18},
        {"configuration": "+ 4-model MC Dropout ensemble", "der": 7.17, "wer": 26.02},
    ]


def table_iv_example_snippet() -> dict[str, str]:
    """Table 4 — dev example (abbreviated labels)."""
    return {
        "input_undiacritized": "الظاهر انه لا خلاف في الحقيقة للتفاق على امتناع ادراك حقيقة الذات",
        "ours_diacritized": "(fully diacritized — see paper Table 4)",
        "gold": "(reference — see paper Table 4)",
        "note": "Complex case endings and shadda correctly restored",
    }


def headline_results() -> dict[str, Any]:
    abl = table_iii_ablation()
    return {
        "finding": (
            "Regularized fine-tuning (R-Drop + Focal + Optuna) yields 3.25 pp WER gain on dev; "
            "MC Dropout ensemble adds 1.16 pp. Test WER 23.26% (1st place)."
        ),
        "test_wer": 23.26,
        "test_der": 6.87,
        "dev_wer_pretrained": abl[0]["wer"],
        "dev_wer_final": abl[-1]["wer"],
        "wer_gain_regularization_pp": round(abl[1]["wer"] - abl[2]["wer"], 2),
        "wer_gain_mc_pp": round(abl[2]["wer"] - abl[3]["wer"], 2),
    }


def evaluation_demo(cfg: KsaaDiacConfig | None = None) -> dict[str, Any]:
    cfg = cfg or KsaaDiacConfig()
    return {"paper": cfg.paper_arxiv, "smoke": evaluation_smoke()}


def benchmarks_bundle() -> dict[str, Any]:
    return {
        "limitations": LIMITATIONS,
        "framework": framework_card(),
        "table_i_hyperparameters": table_i_hyperparameters(),
        "table_ii_leaderboard": table_ii_leaderboard(),
        "table_iii_ablation": table_iii_ablation(),
        "table_iv_example": table_iv_example_snippet(),
        "headlines": headline_results(),
    }


def pipeline_demo() -> dict[str, Any]:
    return {
        "framework": framework_card(),
        "evaluation": evaluation_demo(),
        "headlines": headline_results(),
    }
