"""OmniCustom-1M dataset card (paper Sec. 5)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.omnicustom.config import OmniCustomConfig


def dataset_card(cfg: OmniCustomConfig | None = None) -> dict[str, Any]:
    c = cfg or OmniCustomConfig()
    return {
        "name": c.dataset_name,
        "source": "SpeakerVid-5M",
        "clips": c.dataset_clips,
        "hours": c.dataset_hours,
        "resolution": "480p",
        "fps": c.video_fps,
        "audio_hz": c.audio_sample_rate_hz,
        "min_clip_seconds": 10,
        "ref_audio_seconds": c.ref_audio_seconds,
        "train_clip_seconds": c.train_clip_seconds,
        "filters": {
            "syncnet_offset_abs_max": 3,
            "syncnet_confidence_min": 1.5,
            "aesthetic_min": 0.3,
            "single_speaker": True,
        },
        "transcription": "GLM-ASR",
        "audio_caption_model": "Qwen3-Omni-30B",
        "reference_image": "random face frame crop from same segment",
    }
