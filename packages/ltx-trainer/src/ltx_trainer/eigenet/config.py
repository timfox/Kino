"""Configuration for EIGENET (arXiv:2605.28101)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EigeNetConfig:
    paper_arxiv: str = "arXiv:2605.28101"
    github: str = "https://github.com/FEAfeatherTHER/EigeNet"
    sampling_rate_hz: int = 16000
    rir_length_samples: int = 8000  # 0.5 s @ 16 kHz
    dac_frame_rate_hz: int = 50
    n_acoustic_tokens: int = 25
    acoustic_token_dim: int = 1024
    n_reference_views_max: int = 8
    # Geometric encoder (Sec. IV-D)
    depth_map_size: tuple[int, int] = (256, 512)
    depth_patches: tuple[int, int] = (16, 32)
    geom_vit_layers: int = 4
    geom_vit_heads: int = 8
    geom_feature_dim: int = 512
    cvat_blocks: int = 6
    cvat_heads: int = 16
    cvat_dim: int = 1024
    n_parameters_m: float = 132.54
    n_octave_bands: int = 7
    octave_centers_hz: tuple[float, ...] = (63, 125, 250, 500, 1000, 2000, 4000)
    lambda_edc: float = 1.0
    lambda_spectrum: float = 0.01
    spectrum_warmup_steps: int = 2000
    train_epochs: int = 10
    batch_size: int = 48
    datasets: tuple[str, ...] = ("AcousticRooms", "Hearing-Anything-Anywhere")
    dac_reconstruction_edt: float = 0.004
    dac_reconstruction_c50: float = 0.606
    dac_reconstruction_t60: float = 4.963
    attention_variants: tuple[str, ...] = ("alternate", "self", "cross")
    fold_role: str = "few_shot_novel_view_rir_proxy"
