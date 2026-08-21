"""Framework card + paper excerpts — multi-band EDC ConvNet (arXiv:2605.20968)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.edc_predict.config import EdcPredictConfig
from ltx_trainer.edc_predict.layout import LIMITATIONS
from ltx_trainer.edc_predict.mock import evaluation_smoke


def table1_room_feature_ranges() -> list[dict[str, Any]]:
    """Table 1 — simulated shoebox feature ranges."""
    return [
        {"parameter": "Dimensions (L × W × H)", "range": "3–6 m × 3–6 m × 2.5–4 m"},
        {"parameter": "Source–Receiver distances", "range": "1–4 m"},
        {"parameter": "Wall absorptions (1/3-oct)", "range": "0.14–0.65 (100 Hz–20 kHz)"},
        {"parameter": "Total room configurations", "range": "6000"},
    ]


def table2_convnet_metrics() -> list[dict[str, Any]]:
    """Table 2 — predictive performance on test set."""
    return [
        {"parameter": "EDT (s)", "rmse": 0.10, "mae": 0.07, "r2": 0.79},
        {"parameter": "T20 (s)", "rmse": 0.06, "mae": 0.04, "r2": 0.93},
        {"parameter": "T30 (s)", "rmse": 0.07, "mae": 0.05, "r2": 0.90},
        {"parameter": "C50 (dB)", "rmse": 0.47, "mae": 0.35, "r2": 0.67},
    ]


def framework_card(cfg: EdcPredictConfig | None = None) -> dict[str, Any]:
    cfg = cfg or EdcPredictConfig()
    return {
        "name": cfg.title,
        "paper": cfg.paper_arxiv,
        "authors": "Imran Muhammad, Gerald Schuller (TU Ilmenau, Applied Media Systems)",
        "repo": cfg.repo,
        "problem": "Predict frequency-dependent EDCs from compact room descriptors; avoid fragile full-RIR phase synthesis.",
        "method": {
            "dataset": f"{cfg.n_rooms} Pyroomacoustics shoebox rooms; 1/3-oct absorption per surface ({cfg.n_third_octave_bands} bands).",
            "inputs": f"{cfg.n_input_features}-D feature vector (geometry, source/receiver positions, averaged wall absorption).",
            "encoder": "MLP maps features to latent.",
            "decoder": "1D-CNN with linear interpolation upsampling to sequence length; sigmoid output in (0,1).",
            "loss": f"log-domain MSE + α·MSE(Δ) with stride k={cfg.slope_stride_k}, α={cfg.loss_alpha} (Eq. 1–2).",
            "rir_reconstruction": f"magnitude from EDC derivative + RSS polarity (p={cfg.rss_stickiness_p}).",
        },
        "efficiency": {
            "lstm_params_m": cfg.lstm_params_m,
            "convnet_params_m": cfg.convnet_params_m,
            "reduction_note": "~90% parameter reduction vs prior LSTM work",
        },
        "perception": f"T30 errors largely within {int(cfg.t30_jnd_fraction * 100)}% JND vs ground-truth sims.",
        "headlines": headline_results(cfg),
        "limitations": LIMITATIONS,
    }


def headline_results(cfg: EdcPredictConfig | None = None) -> dict[str, Any]:
    cfg = cfg or EdcPredictConfig()
    return {
        "third_octave_bands": cfg.n_third_octave_bands,
        "rooms_simulated": cfg.n_rooms,
        "convnet_params_millions": cfg.convnet_params_m,
        "t30_r2": 0.90,
        "t30_rmse_s": 0.07,
    }


def evaluation_demo(cfg: EdcPredictConfig | None = None) -> dict[str, Any]:
    cfg = cfg or EdcPredictConfig()
    return {"paper": cfg.paper_arxiv, "smoke": evaluation_smoke(cfg)}


def benchmarks_bundle(cfg: EdcPredictConfig | None = None) -> dict[str, Any]:
    cfg = cfg or EdcPredictConfig()
    return {
        "framework": framework_card(cfg),
        "table1_room_features": table1_room_feature_ranges(),
        "table2_metrics": table2_convnet_metrics(),
        "headlines": headline_results(cfg),
        "limitations": LIMITATIONS,
    }
