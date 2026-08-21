"""FMelCodec: ultra-low-bitrate mel-spectrogram speech codec (arXiv:2605.25669)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FMelCodecConfig:
    paper_arxiv: str = "arXiv:2605.25669"
    project_page: str = "https://redmist328.github.io/FMelCodec/"
    github_repo: str = "https://github.com/redmist328/FMelCodec"
    bitrate_16k_bps: int = 250
    bitrate_48k_bps: int = 750
    codebook_size: int = 1024
    latent_dim: int = 32
    temporal_downsample: int = 4
    waveform_downsample_factor: int = 640
    frame_shift: int = 160
    mel_bins_16k: int = 80
    mel_bins_48k: int = 128
    cfm_ode_steps: int = 4
    ema_rho: float = 0.999
    oc_delta: float = 1e-3
    lambda_mel_rec: float = 45.0
    lambda_vq: float = 2.5
    vq_commitment_eta: float = 4.0
    lambda_cfm: float = 45.0
    lambda_self_cons: float = 10.0
    params_m: float = 27.17
    gflops_16k: float = 18.47
    rtf_16k: float = 0.022
