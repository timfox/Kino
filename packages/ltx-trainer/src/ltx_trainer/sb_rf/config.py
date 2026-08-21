"""SB-RF one-step speech enhancement — Lu et al., arXiv:2606.05575."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SbRfConfig:
    paper_arxiv: str = "arXiv:2606.05575"
    title: str = (
        "SB-RF: Schrödinger Bridge Rectified Flow for One-Step Robust Speech Enhancement"
    )
    framework: str = "SB-RF"

    sample_rate_hz: int = 16000
    stft_window: int = 510
    stft_hop: int = 128
    freq_bins: int = 256
    time_frames: int = 256

    epsilon: float = 0.03
    t_max: float = 0.97
    nfe_default: int = 1

    amplitude_alpha: float = 0.5
    amplitude_beta: float = 0.33
    lambda_mel: float = 33.0
    lambda_pesq: float = 3.0

    backbone_params_m: float = 65.6
    backbone_gmacs: float = 265.0

    # Track A — VB-DMD (Table 1)
    track_a_train_utterances: int = 11572
    track_a_test_utterances: int = 824

    sb_rf_track_a_pesq: float = 3.39
    sb_rf_track_a_estoi: float = 0.88
    sb_rf_track_a_si_sdr: float = 19.5
    sb_rf_track_a_si_sir: float = 30.0
    sb_rf_track_a_si_sar: float = 20.1

    bb_rf_track_a_pesq: float = 3.28
    cose_track_a_pesq: float = 3.02
    cfm_track_a_pesq: float = 3.12
    noisy_track_a_pesq: float = 1.97

    # Track B — low-SNR robustness (Table 2)
    track_b_train_hours: int = 1000
    track_b_test_utterances: int = 10000
    track_b_snr_train_db: tuple[float, float] = (-10.0, 15.0)
    track_b_snr_test_db: tuple[float, float] = (-10.0, 0.0)

    sb_rf_track_b_pesq: float = 2.56
    sb_rf_track_b_estoi: float = 0.70
    sb_rf_track_b_si_sdr: float = 10.4
    sb_rf_track_b_dnsmos: float = 3.41

    bb_rf_track_b_pesq: float = 2.43
    mp_senet_track_b_pesq: float = 2.09
    mp_senet_track_b_si_sdr: float = 10.5

    track_b_pesq_gain_vs_bb_rf: float = 0.13
    track_a_pesq_gain_vs_cose: float = 0.37
