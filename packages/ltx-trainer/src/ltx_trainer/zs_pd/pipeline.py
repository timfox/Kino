"""Zero-shot PD framework card and Table 3 (arXiv:2605.24806)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.zs_pd.config import ZsPdConfig
from ltx_trainer.zs_pd.datasets import table_i_datasets
from ltx_trainer.zs_pd.layout import LIMITATIONS
from ltx_trainer.zs_pd.mock import evaluation_smoke


def framework_card(cfg: ZsPdConfig | None = None) -> dict[str, Any]:
    cfg = cfg or ZsPdConfig()
    return {
        "name": "Zero-Shot Parkinson's Disease Detection from Speech",
        "paper": cfg.paper_arxiv,
        "authors": "Muhammad Ashad Kabir, Sirajam Munira",
        "modalities": [
            "Handcrafted 71-D acoustic features → LLaMA 3 8B (text)",
            "Raw waveform → Qwen2-Audio, Pengi, Audio-Reasoner (LALM/LARM)",
        ],
        "preprocessing": f"{cfg.sample_rate_hz} Hz, 10 s non-overlapping segments",
        "inference": "Zero-shot, temperature=0, segment → subject majority vote",
        "metrics": [
            "Balanced accuracy",
            "AUROC",
            "Sensitivity",
            "Specificity",
            "Brier score",
        ],
        "bootstrap_ci": f"{cfg.bootstrap_replicates} stratified BCa replicates",
        "datasets": [d["name"] for d in table_i_datasets()],
        "limitations": LIMITATIONS,
    }


def table_iii_results() -> list[dict[str, Any]]:
    """Table 3 — subject-level zero-shot performance (point estimates)."""
    return [
        {
            "dataset": "BenSParX",
            "model_type": "LLM",
            "model": "LLaMA 3",
            "balanced_accuracy": 83.33,
            "auroc": 0.901,
            "sensitivity": 86.67,
            "specificity": 80.00,
            "brier": 0.228,
        },
        {
            "dataset": "BenSParX",
            "model_type": "LALM",
            "model": "Qwen2-Audio",
            "balanced_accuracy": 50.00,
            "auroc": 0.536,
            "sensitivity": 16.67,
            "specificity": 83.33,
            "brier": 0.258,
        },
        {
            "dataset": "BenSParX",
            "model_type": "LARM",
            "model": "Audio-Reasoner",
            "balanced_accuracy": 50.83,
            "auroc": 0.534,
            "sensitivity": 91.67,
            "specificity": 10.00,
            "brier": 0.396,
        },
        {
            "dataset": "MDVR-KCL",
            "model_type": "LLM",
            "model": "LLaMA 3",
            "balanced_accuracy": 50.74,
            "auroc": 0.702,
            "sensitivity": 6.25,
            "specificity": 95.24,
            "brier": 0.242,
        },
        {
            "dataset": "MDVR-KCL",
            "model_type": "LARM",
            "model": "Audio-Reasoner",
            "balanced_accuracy": 69.49,
            "auroc": 0.609,
            "sensitivity": 43.75,
            "specificity": 95.24,
            "brier": 0.252,
        },
        {
            "dataset": "IPVS",
            "model_type": "LLM",
            "model": "LLaMA 3",
            "balanced_accuracy": 51.79,
            "auroc": 0.805,
            "sensitivity": 3.57,
            "specificity": 100.0,
            "brier": 0.249,
        },
        {
            "dataset": "IPVS",
            "model_type": "LALM",
            "model": "Qwen2-Audio",
            "balanced_accuracy": 54.87,
            "auroc": 0.472,
            "sensitivity": 64.29,
            "specificity": 45.45,
            "brier": 0.254,
        },
        {
            "dataset": "NeuroVoz",
            "model_type": "LLM",
            "model": "LLaMA 3",
            "balanced_accuracy": 52.58,
            "auroc": 0.486,
            "sensitivity": 39.13,
            "specificity": 66.04,
            "brier": 0.247,
        },
        {
            "dataset": "NeuroVoz",
            "model_type": "LALM",
            "model": "Qwen2-Audio",
            "balanced_accuracy": 63.04,
            "auroc": 0.519,
            "sensitivity": 26.09,
            "specificity": 100.0,
            "brier": 0.219,
        },
        {
            "dataset": "NeuroVoz",
            "model_type": "LARM",
            "model": "Audio-Reasoner",
            "balanced_accuracy": 63.90,
            "auroc": 0.676,
            "sensitivity": 39.13,
            "specificity": 88.68,
            "brier": 0.207,
        },
    ]


def best_per_dataset() -> list[dict[str, Any]]:
    """Highest balanced accuracy per corpus from Table 3."""
    rows = table_iii_results()
    by_ds: dict[str, dict[str, Any]] = {}
    for r in rows:
        ds = r["dataset"]
        if ds not in by_ds or r["balanced_accuracy"] > by_ds[ds]["balanced_accuracy"]:
            by_ds[ds] = r
    return list(by_ds.values())


def headline_results() -> dict[str, Any]:
    best = {r["dataset"]: r for r in best_per_dataset()}
    return {
        "finding": (
            "Input modality affects zero-shot PD detection: handcrafted features via "
            "LLaMA 3 are strongest on Bengali (BenSParX); raw audio helps on NeuroVoz "
            "and MDVR-KCL but with variable calibration (Brier)."
        ),
        "bensparx_best": best["BenSParX"],
        "neurovoz_best": best["NeuroVoz"],
        "feature_stable_low_resource": True,
        "audio_dataset_dependent": True,
    }


def evaluation_demo(cfg: ZsPdConfig | None = None) -> dict[str, Any]:
    cfg = cfg or ZsPdConfig()
    return {"paper": cfg.paper_arxiv, "smoke": evaluation_smoke()}


def benchmarks_bundle() -> dict[str, Any]:
    return {
        "limitations": LIMITATIONS,
        "framework": framework_card(),
        "table_i_datasets": table_i_datasets(),
        "table_iii_results": table_iii_results(),
        "best_per_dataset": best_per_dataset(),
        "headlines": headline_results(),
    }


def pipeline_demo() -> dict[str, Any]:
    return {
        "framework": framework_card(),
        "evaluation": evaluation_demo(),
        "headlines": headline_results(),
    }
