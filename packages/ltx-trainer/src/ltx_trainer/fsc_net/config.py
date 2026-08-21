"""FSC-Net — Full-Spectrum Context Network for speech BWE (arXiv:2606.06962)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FscNetConfig:
    paper_arxiv: str = "arXiv:2606.06962"
    title: str = (
        "FSC-Net: Integrating Fast Fourier Convolutions and Progressive Learning "
        "for Speech Bandwidth Extension"
    )
    framework: str = "FSC-Net"
    backbone: str = "TF-GridNet + FFC"
    dataset_train: str = "VCTK 0.92"
    dataset_eval: str = "EARS (zero-shot)"
    demo_url: str = "https://xinan-chen.github.io/FSC-Net-demo/"

    # Architecture (§II-B, §III-B)
    num_blocks: int = 5
    num_subbands: int = 3
    params_m: float = 1.54
    macs_g: float = 27.74
    stft_window_ms: float = 32.0
    stft_hop_ms: float = 16.0
    sample_rate_hz: int = 48000
    segment_seconds: float = 2.0

    # Progressive windows Wi (§II-C)
    progressive_windows: tuple[int, ...] = (257, 65, 17, 5, 1)

    # Loss weights (§III-B)
    lambda_lsd: float = 5.0
    lambda_adv: float = 0.34
    lambda_feat: float = 0.1

    # Table I — 4 kHz → 48 kHz anchors
    lsd_4k: float = 0.8771
    nisqa_4k: float = 4.3134
    pesq_4k: float = 2.8092

    # Table I — 16 kHz → 48 kHz anchors
    lsd_16k: float = 0.7048
    nisqa_16k: float = 4.4681
    pesq_16k: float = 4.5279

    # Table II — EARS generalization
    lsd_ears: float = 1.2067
    nisqa_ears: float = 3.9214
    pesq_ears: float = 4.2988
