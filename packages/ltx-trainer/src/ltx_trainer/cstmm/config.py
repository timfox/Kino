"""cSTMM: complex spherical Student's t mixture model (arXiv:2605.25512)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CstmmConfig:
    paper_arxiv: str = "arXiv:2605.25512"
    author: str = "Nobutaka Ito (AIST)"
    sample_rate_hz: int = 16000
    stft_fft: int = 2048
    stft_hop: int = 512
    valid_energy_eps: float = 1e-8
    nu_star: float = 1.0
    nu_acgmm_equiv: str = "M"  # ν = M recovers cACGMM
    nu_large_limit: float = 1e4  # cBMM / cWMM recovery
    nu_sweep: tuple[float, ...] = (0.5, 1, 1.5, 2, 3, 4, 5, 10, 20, 50, 100, 1000, 1e4)
    max_outer_iterations: int = 20
    kmeans_init_attempts: int = 4
    warmstart_iterations: int = 5
    rt60_ms: tuple[int, ...] = (160, 360, 610)
    mic_source_settings: tuple[tuple[int, int], ...] = (
        (2, 3),
        (3, 2),
        (3, 3),
        (4, 2),
        (4, 3),
        (4, 4),
    )
    test_mixtures_per_condition: int = 256
    dev_mixtures_recovery: int = 64
