"""SwanSphere — streaming FOA spatial audio via AR + LocDiT (arXiv:2605.30940)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SwanSphereConfig:
    paper_arxiv: str = "arXiv:2605.30940"
    project_url: str = "https://swanaigc.github.io/#swansphere"
    params_b: float = 1.09
    foa_channels: int = 4
    latent_dim: int = 128
    latent_fps: float = 21.5
    patch_frames: int = 4
    patch_stride: int = 4
    causal_patches: int = 2
    locdit_steps: int = 20
    odpo_candidates: int = 8
    odpo_lambda_spatial: float = 0.4
    odpo_lambda_semantic: float = 0.4
    odpo_lambda_fidelity: float = 0.2
    dataset_hours: float = 458.0
    dataset_pairs: int = 165_000
    caption_samples: int = 3100
    v2st_caption_eval: int = 300
    video_encoder: str = "VideoMAE-V2"
    audio_encoder_svac: str = "AudioMAE"
    text_encoder: str = "FLAN-T5"
    first_chunk_latency_s: float = 0.21
    total_stream_latency_s: float = 9.13
