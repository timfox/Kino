"""Hyperparameters for StreamChar (Tian et al., arXiv:2605.25659)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class StreamCharConfig:
    """Paper defaults (Sec. 5.1)."""

    paper_arxiv: str = "arXiv:2605.25659"
    project_page: str = "https://humanaigc.github.io/StreamChar_page"

    dit_backbone: str = "WAN 2.2-5B"
    orchestrator_llm: str = "Qwen2.5-3B"
    fps: int = 24
    frames_per_chunk: int = 33
    latent_frames_per_chunk: int = 9
    motion_frames: int = 33
    history_audio_sec: float = 15.0

    teacher_steps: int = 50
    student_steps: int = 4
    distill_stage1_steps: int = 600
    distill_stage2_steps: int = 400
    pap_guidance_scale: float = 2.0

    chunk_latency_sec: float = 1.34
    playback_budget_sec: float = 33 / 24  # ≈ 1.38

    resolution: tuple[int, int] = (512, 512)
