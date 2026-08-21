"""DSTA framework card, Fine-Badminton stats, and paper tables (arXiv:2605.23355)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.dsta.adapter import dsta_forward_scalar
from ltx_trainer.dsta.config import DSTAConfig, FineBadmintonStats
from ltx_trainer.dsta.dst import dst_forward, split_channels
from ltx_trainer.dsta.layout import LIMITATIONS
from ltx_trainer.dsta.mock import toy_dst_inputs
from ltx_trainer.dsta.tal import shuttleset_stroke_interval, temporal_iou


def framework_card(cfg: DSTAConfig | None = None) -> dict[str, Any]:
    cfg = cfg or DSTAConfig()
    stats = FineBadmintonStats()
    return {
        "name": "DSTA",
        "paper": cfg.paper_arxiv,
        "idea": (
            "Decoupling Spatio-Temporal Adapter for fine-grained badminton TAL: "
            "parallel temporal (DWConv), vertical (1×3×1), and horizontal (1×1×3) branches "
            "on top of AdaTAD + VideoMAE-B + ActionFormer."
        ),
        "dataset": {
            "name": "Fine-Badminton",
            "matches": stats.matches,
            "categories": stats.action_categories,
            "actions": stats.annotated_actions,
            "zenodo_subset": stats.zenodo_subset_videos,
            "zenodo": cfg.zenodo_doi,
        },
        "stack": {
            "backbone": cfg.backbone,
            "head": cfg.detection_head,
            "base": cfg.base_framework,
        },
        "defaults": {
            "alpha": cfg.channel_split_alpha,
            "input_frames": cfg.input_frames,
            "spatial_size": cfg.spatial_size,
            "tiou_thresholds": list(cfg.tiou_thresholds),
        },
        "baselines": ["TIA", "3D-SENet", "AIM", "ST-Adapter", "LoRA", "EVL"],
    }


def table_dataset_comparison() -> list[dict[str, str | bool | int]]:
    """Table I excerpt — representative TAL / badminton datasets."""
    return [
        {"name": "ShuttleSet", "interval": False, "fine_grained": True, "annotations": 36492, "classes": 18},
        {"name": "VideoBadminton", "interval": True, "fine_grained": True, "annotations": 7822, "classes": 18},
        {"name": "Fine-Badminton", "interval": True, "fine_grained": True, "annotations": 27597, "classes": 29},
    ]


def table_map_main() -> dict[str, dict[str, dict[str, float]]]:
    """Table II — mAP at tIoU thresholds."""
    shuttleset = {
        "TIA": {"0.3": 72.26, "0.4": 72.16, "0.5": 72.00, "0.6": 71.46, "0.7": 68.85, "avg": 71.35},
        "AIM": {"0.3": 73.78, "0.4": 73.74, "0.5": 72.63, "0.6": 72.99, "0.7": 71.72, "avg": 73.17},
        "DSTA": {"0.3": 75.17, "0.4": 75.10, "0.5": 75.04, "0.6": 74.84, "0.7": 73.18, "avg": 74.67},
    }
    fine_badminton = {
        "TIA": {"0.3": 66.68, "0.4": 66.14, "0.5": 65.64, "0.6": 63.10, "0.7": 58.81, "avg": 64.08},
        "AIM": {"0.3": 63.78, "0.4": 63.51, "0.5": 62.40, "0.6": 60.54, "0.7": 56.71, "avg": 61.39},
        "DSTA": {"0.3": 68.47, "0.4": 67.83, "0.5": 67.52, "0.6": 65.88, "0.7": 61.44, "avg": 66.23},
    }
    return {"ShuttleSet": shuttleset, "Fine-Badminton": fine_badminton}


def table_efficiency() -> dict[str, dict[str, float]]:
    """Table III — trainable parameters (M) and FLOPs (G)."""
    return {
        "Full_Fine_tuning": {"trainable_M": 86.227, "flops_G": 274.62},
        "TIA": {"trainable_M": 1.335, "flops_G": 428.17},
        "AIM": {"trainable_M": 10.651, "flops_G": 955.51},
        "LoRA": {"trainable_M": 0.229, "flops_G": 1136.73},
        "DSTA_Conv_THW": {"trainable_M": 4.017, "flops_G": 428.17},
    }


def table_ablation_branches() -> dict[str, dict[str, dict[str, float]]]:
    """Table IV — Conv T / TH / THW ablation."""
    return {
        "ShuttleSet": {
            "TIA": {"avg": 71.35},
            "Conv_T": {"avg": 72.48},
            "Conv_TH": {"avg": 73.00},
            "DSTA_Conv_THW": {"avg": 74.67},
        },
        "Fine-Badminton": {
            "TIA": {"avg": 64.08},
            "Conv_T": {"avg": 64.55},
            "Conv_TH": {"avg": 64.81},
            "DSTA_Conv_THW": {"avg": 66.23},
        },
    }


def table_alpha_sensitivity() -> dict[str, float]:
    """Table V — channel split ratio α on Fine-Badminton (avg mAP)."""
    return {"0.1": 64.67, "0.3": 65.46, "0.5": 66.23, "0.7": 65.20, "0.9": 65.77}


def pipeline_demo(cfg: DSTAConfig | None = None) -> dict[str, Any]:
    cfg = cfg or DSTAConfig()
    split = split_channels(256, alpha=cfg.channel_split_alpha)
    inputs = toy_dst_inputs()
    dst_out = dst_forward(**inputs)
    x_prime = dsta_forward_scalar(1.0, w_down=0.5, w_mid=0.8, w_up=0.6, dst_out=dst_out)
    start, end = shuttleset_stroke_interval(100, half_window=cfg.shuttleset_stroke_half_window)
    return {
        "channel_split": {
            "alpha": cfg.channel_split_alpha,
            "spatial_temporal": split.spatial_temporal_channels,
            "identity": split.identity_channels,
        },
        "dst_output_scalar": dst_out,
        "dsta_output_scalar": x_prime,
        "shuttleset_interval_example": {"start": start, "end": end},
        "tiou_example": temporal_iou(start, end, start + 2, end + 2),
    }


def evaluation_demo(cfg: DSTAConfig | None = None) -> dict[str, Any]:
    cfg = cfg or DSTAConfig()
    maps = table_map_main()
    tia_avg = maps["ShuttleSet"]["TIA"]["avg"]
    dsta_avg = maps["ShuttleSet"]["DSTA"]["avg"]
    return {
        "framework": framework_card(cfg),
        "pipeline": pipeline_demo(cfg),
        "limitations": list(LIMITATIONS),
        "shuttleset_gain_vs_tia_avg_mAP": round(dsta_avg - tia_avg, 2),
        "paper_tables": {
            "datasets": table_dataset_comparison(),
            "map": maps,
            "efficiency": table_efficiency(),
            "ablation": table_ablation_branches(),
            "alpha": table_alpha_sensitivity(),
        },
    }
