"""UAT — Unified Audio-Text Diffusion (arXiv:2606.04939)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class UATConfig:
    paper_arxiv: str = "arXiv:2606.04939"
    title: str = "UAT: Unified Audio-Text Diffusion for Audio Generation, Editing, and Captioning"
    demo_url: str = "https://UAT-demo.github.io"
    backbone: str = "AudioX (HKUSTAudio/AudioX)"
    params_b: float = 1.7

    dit_blocks: int = 24
    hidden_dim: int = 1536
    text_encoder: str = "T5-Base (768-d, frozen)"
    audio_vae: str = "frozen continuous VAE (Stable Audio style)"

    lambda_text: float = 0.2
    cfg_drop: float = 0.1
    train_steps: int = 60_000
    lr: float = 8e-5
    batch_size: int = 768

    inference_steps: int = 100
    cfg_scale: float = 7.0
    edit_start_step: int = 70

    training_samples: int = 2_363_765
    training_hours: float = 6620.62

    # Smoke dims
    demo_latent_len: int = 32
    demo_vocab: int = 256
    demo_text_len: int = 16
