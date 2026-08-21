"""Dataset construction cards (alignment 700K + SFT 100K)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.autocut.config import AutoCutConfig
from ltx_trainer.autocut.taxonomy import EditingTask


def alignment_dataset_card(cfg: AutoCutConfig | None = None) -> dict[str, Any]:
    c = cfg or AutoCutConfig()
    return {
        "stage": "multimodal_alignment",
        "samples": c.alignment_samples,
        "filters": [
            "top 10% CTR and like-rate",
            "remove lyrical-only / no-speech ads",
            "1 < clips < 60",
            "1s < clip duration < 60s",
            "1s < video duration < 60s",
        ],
        "modalities": ["ASR script", "1 fps ResNet frames", "PANNs BGM embedding"],
        "tokenization": "8-head RQ-VAE per frame/segment",
    }


def sft_dataset_card(cfg: AutoCutConfig | None = None) -> dict[str, Any]:
    c = cfg or AutoCutConfig()
    return {
        "stage": "supervised_finetuning",
        "samples": c.sft_samples,
        "filters": [
            "video < 120s",
            "clip 2–60s",
            "≥80% clips with Qwen2.5-VL relevance ≥ 4/5",
            "dedupe by brand+product+script hash",
        ],
        "tasks": [t.value for t in EditingTask],
        "instances_per_task": "25K–30K",
        "split": "95/5 train-val",
    }


def dataset_statistics_summary() -> dict[str, Any]:
    """Fig. 3 summary (alignment vs SFT)."""
    return {
        "alignment": {
            "clips_per_video": "broad 1–160",
            "clip_duration_s": "irregular up to 60",
            "video_duration_s": "up to ~450",
        },
        "sft": {
            "clips_per_video": "balanced 1–24",
            "clip_duration_s": "2–12 typical",
            "video_duration_s": "≤60",
        },
    }
