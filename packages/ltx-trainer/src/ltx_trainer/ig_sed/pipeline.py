"""Framework card and paper benchmark excerpts (arXiv:2605.23293)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.ig_sed.config import IgSedConfig
from ltx_trainer.ig_sed.layout import LIMITATIONS
from ltx_trainer.ig_sed.mock import evaluation_smoke


def framework_card(cfg: IgSedConfig | None = None) -> dict[str, Any]:
    cfg = cfg or IgSedConfig()
    return {
        "name": "IG temporal detection from clip-level sound classifiers",
        "paper": cfg.paper_arxiv,
        "authors": "Martynas Dumpis (Vilnius Tech), Tuomas Virtanen (Tampere)",
        "setting": (
            "Post-hoc Integrated Gradients on CNN14 multi-label classifier trained with "
            "clip-level weak labels only; evaluated on synthetic DESED/Scaper polyphonic soundscapes."
        ),
        "pipeline": {
            "input": f"{cfg.clip_duration_s}s @ {cfg.sample_rate_hz} Hz → log-mel → frozen CNN14",
            "clip_output": f"{cfg.num_classes} sigmoid class scores (threshold {cfg.clip_threshold})",
            "attribution": f"Captum IG, n={cfg.ig_steps} steps, zero-silence baseline",
            "detection": f"binarize |attr| at {cfg.frame_resolution_ms} ms frames via validation percentile",
        },
        "baselines": ["FW-WS (clip-level MIL)", "FW-SS (frame labels)", "Random", "Energy"],
        "dataset": {
            "source": "DESED + Scaper synthetic domestic soundscapes",
            "train": cfg.train_clips,
            "val": cfg.val_clips,
            "test": cfg.test_clips,
            "classes": list(cfg.classes),
        },
        "headlines": {
            "ig_mean_iou": cfg.mean_iou_ig,
            "ig_mean_f1": cfg.mean_f1_ig,
            "ig_pointing_game": cfg.pointing_game_ig,
            "fw_ws_iou": cfg.mean_iou_fw_ws,
            "fw_ss_iou": cfg.mean_iou_fw_ss,
            "ig_optimal_percentile_iou": cfg.ig_peak_iou_percentile,
        },
        "limitations": LIMITATIONS,
    }


def table_1_classification() -> list[dict[str, Any]]:
    """Table I: multi-label classification F1 on test set."""
    return [
        {"class": "Alarm bell ringing", "precision": 1.00, "recall": 0.75, "f1": 0.86},
        {"class": "Blender", "precision": 1.00, "recall": 0.15, "f1": 0.27},
        {"class": "Cat", "precision": 1.00, "recall": 0.81, "f1": 0.90},
        {"class": "Dishes", "precision": 1.00, "recall": 0.57, "f1": 0.72},
        {"class": "Dog", "precision": 1.00, "recall": 0.56, "f1": 0.71},
        {"class": "Electric shaver", "precision": 0.91, "recall": 0.77, "f1": 0.83},
        {"class": "Frying", "precision": 1.00, "recall": 0.17, "f1": 0.29},
        {"class": "Running water", "precision": 1.00, "recall": 0.46, "f1": 0.62},
        {"class": "Speech", "precision": 0.92, "recall": 0.98, "f1": 0.95},
        {"class": "Vacuum cleaner", "precision": 0.86, "recall": 0.46, "f1": 0.60},
    ]


def table_2_temporal_detection() -> list[dict[str, Any]]:
    """Table II: temporal detection comparison (macro-averaged)."""
    return [
        {"method": "IG", "mean_iou": 0.39, "f1": 0.52, "std_iou": 0.23, "pointing_game": 0.826},
        {"method": "FW-WS", "mean_iou": 0.42, "f1": 0.55, "std_iou": 0.24, "pointing_game": 0.973},
        {"method": "FW-SS", "mean_iou": 0.45, "f1": 0.58, "std_iou": 0.24, "pointing_game": 0.979},
        {"method": "Random baseline", "mean_iou": 0.19, "f1": 0.30, "std_iou": 0.11, "pointing_game": 0.283},
        {"method": "Energy baseline", "mean_iou": 0.16, "f1": 0.24, "std_iou": 0.16, "pointing_game": 0.159},
    ]


def table_3_per_class() -> list[dict[str, Any]]:
    """Table III excerpt: per-class IoU and F1 for IG and framewise baselines."""
    return [
        {"class": "Alarm bell ringing", "ig_iou": 0.44, "ig_f1": 0.57, "fw_ws_iou": 0.45, "fw_ws_f1": 0.58, "fw_ss_iou": 0.46, "fw_ss_f1": 0.59},
        {"class": "Blender", "ig_iou": 0.63, "ig_f1": 0.75, "fw_ws_iou": 0.67, "fw_ws_f1": 0.78, "fw_ss_iou": 0.69, "fw_ss_f1": 0.82},
        {"class": "Cat", "ig_iou": 0.47, "ig_f1": 0.61, "fw_ws_iou": 0.54, "fw_ws_f1": 0.67, "fw_ss_iou": 0.55, "fw_ss_f1": 0.68},
        {"class": "Dishes", "ig_iou": 0.20, "ig_f1": 0.31, "fw_ws_iou": 0.20, "fw_ws_f1": 0.31, "fw_ss_iou": 0.24, "fw_ss_f1": 0.36},
        {"class": "Dog", "ig_iou": 0.45, "ig_f1": 0.57, "fw_ws_iou": 0.32, "fw_ws_f1": 0.45, "fw_ss_iou": 0.34, "fw_ss_f1": 0.46},
        {"class": "Electric shaver", "ig_iou": 0.67, "ig_f1": 0.79, "fw_ws_iou": 0.66, "fw_ws_f1": 0.77, "fw_ss_iou": 0.66, "fw_ss_f1": 0.77},
        {"class": "Frying", "ig_iou": 0.40, "ig_f1": 0.57, "fw_ws_iou": 0.52, "fw_ws_f1": 0.68, "fw_ss_iou": 0.52, "fw_ss_f1": 0.68},
        {"class": "Running water", "ig_iou": 0.49, "ig_f1": 0.62, "fw_ws_iou": 0.45, "fw_ws_f1": 0.58, "fw_ss_iou": 0.49, "fw_ss_f1": 0.62},
        {"class": "Speech", "ig_iou": 0.32, "ig_f1": 0.46, "fw_ws_iou": 0.41, "fw_ws_f1": 0.55, "fw_ss_iou": 0.43, "fw_ss_f1": 0.57},
        {"class": "Vacuum cleaner", "ig_iou": 0.51, "ig_f1": 0.65, "fw_ws_iou": 0.45, "fw_ws_f1": 0.60, "fw_ss_iou": 0.52, "fw_ss_f1": 0.65},
    ]


def figure_3_threshold_sensitivity() -> dict[str, Any]:
    """Figure 3 headlines: optimal percentile thresholds on validation."""
    return {
        "ig_peak_iou_percentile": 56,
        "ig_peak_iou": 0.39,
        "ig_peak_f1_percentile": 57,
        "ig_peak_f1": 0.52,
        "ig_iou_at_80th_percentile": 0.34,
        "fw_ws_peak_iou_percentile": 43,
        "fw_ws_peak_iou": 0.54,
        "fw_ws_peak_f1": 0.67,
        "fw_ss_peak_iou_percentile": 28,
        "fw_ss_peak_iou": 0.65,
        "fw_ss_peak_f1_percentile": 29,
        "fw_ss_peak_f1": 0.77,
        "note": "Fixed image-XAI thresholds (e.g. 20% of max) are suboptimal for audio boundaries.",
    }


def headline_results() -> dict[str, Any]:
    cfg = IgSedConfig()
    return {
        "ig_mean_iou": cfg.mean_iou_ig,
        "ig_mean_f1": cfg.mean_f1_ig,
        "ig_pointing_game": cfg.pointing_game_ig,
        "fw_ws_mean_iou": cfg.mean_iou_fw_ws,
        "fw_ss_mean_iou": cfg.mean_iou_fw_ss,
        "speech_clip_f1": 0.95,
        "speech_ig_iou": 0.32,
    }


def evaluation_demo(cfg: IgSedConfig | None = None) -> dict[str, Any]:
    cfg = cfg or IgSedConfig()
    return {"config": cfg.__dict__, "smoke": evaluation_smoke(cfg)}


def benchmarks_bundle() -> dict[str, Any]:
    return {
        "table_1_classification": table_1_classification(),
        "table_2_temporal_detection": table_2_temporal_detection(),
        "table_3_per_class": table_3_per_class(),
        "figure_3_threshold_sensitivity": figure_3_threshold_sensitivity(),
        "headlines": headline_results(),
    }
