"""WaveNeXt 2: unified ConvNeXt vocoder for GAN and diffusion (arXiv:2605.25506)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Wavenext2Config:
    paper_arxiv: str = "arXiv:2605.25506"
    demo_page: str = "https://37integer.github.io/WAVENEXT-2"
    sample_rate_hz: int = 24000
    mel_dim: int = 128
    convnext_blocks: int = 8
    num_diff_submodels: int = 4
    # BDDM 4-step noise schedule (paper §4.1)
    bddm_noise_schedule: tuple[float, ...] = (1.0e-4, 2.8e-2, 5.6e-1, 9.1e-1)
    gan_hop_size: int = 300
    diff_hop_size: int = 256
    dataset: str = "LibriTTS-R (train-clean-100 + train-clean-360)"
    eval_subset: str = "test-clean-100 (4824 samples objective; 20 MOS)"
