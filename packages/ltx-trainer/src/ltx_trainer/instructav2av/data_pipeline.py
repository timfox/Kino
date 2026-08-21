"""Scalable data synthesis pipeline stubs (Sec. 3, Fig. 2)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.instructav2av.config import InstructAV2AVConfig


def verification_scorecard() -> dict[str, str]:
    """Five automatic LLM assessment dimensions (Sec. 3.3)."""
    return {
        "instruction_fidelity": "Strict adherence to edit instruction",
        "content_preservation": "Non-target video/audio unchanged",
        "perceptual_quality": "Realism and artifact-free output",
        "audio_video_sync": "Temporal and semantic AV alignment",
        "safety": "Filter inappropriate generations",
    }


def data_engine_card() -> dict[str, Any]:
    """Mask-guided Wan2.2-5B data engine (Sec. 3.2, Appendix 7.3)."""
    return {
        "backbone": "Wan2.2-5B",
        "mask": "Grounded-SAM-2 instance masks",
        "instruction_model": "Qwen3-Omni",
        "audio_sep": "SAM-Audio + ElevenLabs T2A mix",
        "video_edit": "Mask-guided flow matching (AVI-Edit style)",
    }


def pipeline_stages(cfg: InstructAV2AVConfig | None = None) -> list[dict[str, str]]:
    cfg = cfg or InstructAV2AVConfig()
    return [
        {"stage": 1, "name": "source_collection", "detail": "Shot detect, motion, aesthetics, audio QC"},
        {"stage": 2, "name": "editing_engine", "detail": "Mask + instruction → target AV"},
        {
            "stage": 3,
            "name": "verification",
            "detail": f"Auto 5-criteria + human 1K eval ({cfg.eval_pairs} pairs)",
        },
    ]
