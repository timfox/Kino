"""P2PSynCodec — plain-to-pseudo synergistic VQ codec (Jiang et al., arXiv:2606.05876)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class P2PSynCodecConfig:
    paper_arxiv: str = "arXiv:2606.05876"
    title: str = (
        "An Ultra-Low-Bitrate Neural Speech Codec with Plain-to-Pseudo "
        "Synergistic Vector Quantization"
    )
    framework: str = "P2PSynCodec"

    # Datasets (§3.1)
    libritts_sample_rate_hz: int = 16000
    vctk_sample_rate_hz: int = 48000
    libritts_train: str = "train-clean-100 + train-clean-360"
    libritts_eval: str = "dev-clean / test-clean"
    vctk_train_utterances: int = 40936
    vctk_test_utterances: int = 2937

    # P2PSVQ architecture (§3.1)
    downsampling_rate: int = 320
    pseudo_vq_count: int = 3  # N
    plain_codebook_size: int = 1024  # M_pl
    code_vector_dim: int = 32  # K
    conformer_channels: int = 256
    conformer_heads: int = 8
    bilstm_channels: int = 256

    # Target bitrates (Eq. 2)
    libritts_target_kbps: float = 0.5
    vctk_target_kbps: float = 1.5
    comparison_baseline_kbps: float = 2.0
    sqcodec_comparison_kbps: float = 1.5
    bitrate_saving_pct: float = 75.0

    # Table 1 — P2PSynCodec @ ultra-low bitrate
    t1_utmos: float = 3.947
    t1_stoi: float = 0.823
    t1_visqol: float = 3.476
    t1_sigmos: float = 3.305
    t1_vctk_stoi: float = 0.796
    t1_vctk_visqol: float = 3.423
    t1_flops_g: float = 3.31
    t1_params_m: float = 22.99

    # Table 1 — MDCTCodec teacher baseline @ same bitrate
    mdctcodec_t1_utmos: float = 2.670
    mdctcodec_t1_stoi: float = 0.844
    mdctcodec_t1_visqol: float = 3.631
    mdctcodec_t1_flops_g: float = 2.32
    mdctcodec_t1_params_m: float = 6.75

    # Table 1 — BigCodec @ same bitrate
    bigcodec_t1_utmos: float = 3.939
    bigcodec_t1_flops_g: float = 61.03
    bigcodec_t1_params_m: float = 159.32

    # Table 1 — WavTokenizer @ same bitrate
    wavtokenizer_t1_utmos: float = 3.269

    # Table 2 — optimal N=3 (All VQs vs plain-only UTMOS)
    t2_n3_all_utmos: float = 3.947
    t2_n3_plain_utmos: float = 2.324
    t2_n1_all_utmos: float = 3.787
    t2_n7_all_utmos: float = 3.889

    # Relative complexity vs BigCodec (§3.3.1)
    flops_ratio_vs_bigcodec: float = 0.05
    params_ratio_vs_bigcodec: float = 0.14
