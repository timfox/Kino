"""Paper benchmark anchors from ID-LoRA README (CelebV-HQ / TalkVid)."""

from __future__ import annotations

from typing import Any


def celebvhq_cross_video_row() -> dict[str, float]:
    """Cross-video split — ID-LoRA vs baselines (speaker sim / LSE-C / LSE-D)."""
    return {
        "speaker_similarity": 0.477,
        "lip_sync_confidence": 0.874,
        "lse_c": 8.49,
        "lse_d": 3.90,
        "fid_video": 0.363,
        "fid_audio": 0.113,
    }


def talkvid_in_domain_row() -> dict[str, float]:
    return {
        "speaker_similarity": 0.599,
        "lip_sync_confidence": 0.772,
        "lse_c": 10.62,
        "lse_d": 3.09,
        "fid_video": 0.385,
        "fid_audio": 0.054,
    }


BENCHMARK_TABLES: dict[str, Any] = {
    "celebvhq_cross_video": celebvhq_cross_video_row(),
    "talkvid_in_domain": talkvid_in_domain_row(),
    "mos_av_correspondence": {"id_lora": 3.05, "kling_2_6_pro": 2.90},
}


def summary_anchors() -> dict[str, Any]:
    return {
        "celebvhq_cross_video": celebvhq_cross_video_row(),
        "talkvid_in_domain": talkvid_in_domain_row(),
        "lora_rank": 128,
        "training_steps_ltx2": 6000,
        "training_steps_ltx23": 3000,
    }
