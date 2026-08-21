"""CTRnet + PuLSS configuration (arXiv:2605.19695)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CtrnetConfig:
    paper_id: str = "2605.19695"
    paper_url: str = "https://arxiv.org/abs/2605.19695"
    title: str = "Cross-Talk Speech Reduction, by Separation, for Separation"
    demo_url: str = "https://zqwang7.github.io/demos/CTRnet%20journal%20demo/index.html"

    # CHiME-6 setup (§VI-A)
    n_speakers_c: int = 4
    n_farfield_mics_p: int = 24  # 6 Kinect × 4 mics
    fs_hz: int = 16_000
    block_duration_s: float = 12.0
    block_hop_s: float = 1.0
    context_s: float = 4.0
    output_s: float = 4.0

    # Table I — CTRnet
    fcp_past_taps_i: int = 13
    fcp_future_taps_j: int = 1
    fcp_xi: float = 0.01
    sa_loss_weight_beta: float = 0.1
    reverb_delay_frames_delta: int = 3
    kappa1_simulated: float = 1.0

    # Table I — PuLSS
    pseudo_label_filter_taps_l: int = 2
    max_sync_delay_frames_e: int = 9
    cte_filter_taps_a: int = 1
    cte_loss_weight_delta: float = 20.0
    kappa2_simulated: float = 1.0

    # Shared
    magnitude_compress_alpha: float = 0.3
    overlap_sampling_theta: int = 20

    # STFT (§VI-F)
    ctrnet_win_ms: float = 16.0
    ctrnet_hop_ms: float = 8.0
    pulss_win_ms: float = 32.0
    pulss_hop_ms: float = 16.0

    # Best reported configs (Tables II–III)
    dnn_variant: str = "TF-GridNet-V2"
    binaural_strategy: str = "average_left_right"


@dataclass(frozen=True)
class Chime6Split:
    """Training split note: 14 sessions for training (§VI-A)."""

    train_sessions: int = 14
    val_sessions: int = 2
    test_sessions: int = 2
    training_blocks: int = 123_339
