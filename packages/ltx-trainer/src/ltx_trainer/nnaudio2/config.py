"""nnAudio 2 modernization — Roy et al., arXiv:2606.05394."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class NnAudio2Config:
    paper_arxiv: str = "arXiv:2606.05394"
    title: str = "nnAudio 2: Overcoming Dynamic Compilation Barriers and Transform Inconsistencies"
    framework: str = "nnAudio2"
    venue: str = "arXiv"
    github: str = "https://github.com/AMAAI-Lab/nnAudio2"

    python_target: str = "3.11"
    pytorch_target: str = "2.x"
    supported_istft_freq_scale: str = "no"

    icqt_landweber_iterations: int = 32
    icqt_step_fraction: float = 1.8
    icqt_contraction_rate: float = 0.8
    icqt_snr_db_min: float = 30.0

    vqt_gamma_cqt_max_abs_diff_before: float = 0.089
    vqt_gamma_cqt_mean_abs_diff_before: float = 0.0017

    test_tone_hz: float = 440.0
    test_sr_hz: int = 44100
    test_hop_length: int = 512
    test_cqt_bins: int = 84
