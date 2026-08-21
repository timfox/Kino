"""Foley-Omni: unified V2ST + task-level audio generation (arXiv:2606.03672)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FoleyOmniConfig:
    paper_arxiv: str = "arXiv:2606.03672"
    project_url: str = "https://foley-omni.github.io/"
    backbone: str = "DiT + MMAudio Mel VAE + BigVGAN"
    text_encoder: str = "UM-T5"
    visual_semantic: str = "CLIP"
    visual_sync: str = "Synchformer"
    training_pairs_m: float = 2.7
    curated_v2st_hours: float = 0.216
    v2st_bench_size: int = 300
    bandit_rms_threshold_db: float = -35.0
    filter_min_resolution_p: int = 480
    filter_ib_min: float = 0.3
    filter_sync_min: float = 0.2
    filter_audio_quality_min: float = 0.6
    stage1_epochs: int = 5
    stage2_epochs: int = 3
    stage3_epochs: int = 2
    learning_rate_stage12: float = 5e-5
    learning_rate_stage3: float = 2e-5
    global_batch_size: int = 32
