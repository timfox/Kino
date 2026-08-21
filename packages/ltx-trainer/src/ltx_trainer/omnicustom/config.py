"""OmniCustom configuration (arXiv:2602.12304)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class OmniCustomConfig:
    """Sync audio-video customization on OVI twin backbone."""

    paper_arxiv: str = "2602.12304"
    project_url: str = "https://omnicustom-project.github.io/page/"
    base_model: str = "OVI-1.0"
    video_backbone: str = "Wan2.2-5B"
    audio_backbone: str = "MMAudio-1D-VAE"
    dataset_name: str = "OmniCustom-1M"
    dataset_clips: int = 1_000_000
    dataset_hours: float = 2500.0
    benchmark_size: int = 100
    video_fps: int = 24
    video_duration_s: float = 5.0
    ref_image_size: int = 512
    audio_sample_rate_hz: int = 16000
    ref_audio_seconds: float = 4.0
    train_clip_seconds: float = 5.0
    lora_rank: int = 128
    flow_steps: int = 50
    guidance_video: float = 4.0
    guidance_audio: float = 3.0
    lambda_video_fm: float = 1.0
    lambda_audio_fm: float = 1.0
    lambda_identity_cl: float = 0.1
    lambda_timbre_cl: float = 0.1
    learning_rate: float = 1e-5
    train_steps: int = 200_000
    speech_start_tag: str = "<S>"
    speech_end_tag: str = "<E>"
    random_seed: int = 42

    @classmethod
    def production(cls) -> OmniCustomConfig:
        return cls()
