"""Configuration for DrawVideo (arXiv:2605.23508)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DrawVideoConfig:
    paper_arxiv: str = "arXiv:2605.23508"
    github: str = "https://github.com/LouckXu/DrawVideo"

    # Dataset (Sec. 3.4).
    sketchlongvideo_triplets: int = 1233
    sketchlongvideo_sequences: int = 126
    appearance_prompt_avg_words: float = 127.01
    motion_prompt_avg_words: float = 72.82
    sketch_width_px: int = 600
    scene_detect_threshold: float = 25.0

    # Pipeline backends (Sec. 4–5).
    sketch_coloring: str = "FLUX.1-dev + Canny-ControlNet"
    derivative_keyframes: str = "FLUX.1 Kontext"
    video_backbone: str = "Wan2.2-I2V-A14B"
    prompt_llm: str = "Qwen2.5"
    vlm_recognition: str = "LLaVA-OneVision-Qwen2-0.5B-OV"

    # Default inference (Sec. 5.1).
    default_backbone_eval: str = "Qwen2.5-VL-7B"


@dataclass
class StoryboardShot:
    """One storyboard unit: sketch + appearance + motion (Sec. 4.1)."""

    sketch_id: str
    appearance_prompt: str
    motion_prompt: str


def storyboard_from_shots(shots: list[StoryboardShot]) -> dict[str, list[StoryboardShot]]:
    return {"shots": shots, "count": len(shots)}
