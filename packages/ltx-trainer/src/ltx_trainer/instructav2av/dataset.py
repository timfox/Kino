"""InsAVE-80K dataset schema and sample records (Sec. 3)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ltx_trainer.instructav2av.config import InstructAV2AVConfig


@dataclass
class InsAVESample:
    """One source→target editing pair with instruction."""

    sample_id: str
    instruction: str
    task_type: str
    source_video_path: str | None = None
    source_audio_path: str | None = None
    target_video_path: str | None = None
    target_audio_path: str | None = None
    passed_verification: bool = True


def dataset_card(cfg: InstructAV2AVConfig | None = None) -> dict[str, Any]:
    cfg = cfg or InstructAV2AVConfig()
    return {
        "name": cfg.dataset_name,
        "train_pairs": cfg.train_pairs,
        "eval_pairs": cfg.eval_pairs,
        "clip_duration_s": cfg.clip_duration_s,
        "resolution": cfg.resolution,
        "fps": cfg.fps,
        "audio_hz": cfg.audio_sample_rate_hz,
        "verification_criteria": list(cfg.verification_criteria),
        "task_types": list(cfg.task_types),
        "sources": [
            "YouTube",
            "MovieBench",
            "Condensed Movies",
            "Short-Films-20K",
            "VGGSound",
        ],
    }


def example_sample(task_type: str = "av_instance_edit") -> InsAVESample:
    return InsAVESample(
        sample_id="insave_demo_001",
        instruction=(
            "Change the man into a young woman with brown hair, wearing a gray blazer "
            'and saying, "I really think we should give it another chance."'
        ),
        task_type=task_type,
    )
