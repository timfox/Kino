"""VoCodec — voicing-driven streamable neural speech codec (Jiang et al., arXiv:2606.05892)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class VoCodecConfig:
    paper_arxiv: str = "arXiv:2606.05892"
    title: str = (
        "VoCodec: A Low-bitrate Streamable Neural Speech Codec with "
        "Voicing-driven Quantization"
    )
    framework: str = "VoCodec"

    # Datasets (§3.1)
    libritts_sample_rate_hz: int = 16000
    vctk_sample_rate_hz: int = 48000
    libritts_train: str = "train-clean-100 + train-clean-360"
    libritts_eval: str = "dev-clean / test-clean"
    vctk_train_utterances: int = 40936
    vctk_test_utterances: int = 2937

    # Architecture (§2, §3.1)
    downsampling_rate: int = 320
    rsvq_sq_count: int = 1  # Ns
    rsvq_ivq_count: int = 2  # Nv
    codebook_size: int = 1024
    quantizer_dim: int = 32
    f0_min_hz: float = 60.0
    f0_max_hz: float = 600.0
    energy_threshold: float = 0.75  # τ_energy

    # Voiced-frame ratios for target bitrates (§3.1)
    libritts_voiced_ratio: float = 0.55
    vctk_voiced_ratio: float = 0.35

    # Target average bitrates (kbps)
    libritts_target_kbps: float = 1.1
    vctk_target_kbps: float = 2.7
    libritts_bitrate_min_kbps: float = 0.55
    libritts_bitrate_max_kbps: float = 1.55
    vctk_bitrate_min_kbps: float = 1.65
    vctk_bitrate_max_kbps: float = 4.65

    # Bitrate savings vs uniform (§3.3, Fig. 3)
    bitrate_saving_pct: float = 27.0
    uniform_comparison_kbps: float = 1.5

    # Table 1 — LibriTTS @ 1.1 kbps (VoCodec row)
    t1_lsd: float = 0.896
    t1_stoi: float = 0.916
    t1_visqol: float = 4.115
    t1_mushra: float = 75.18
    t1_mushra_std: float = 5.16
    t1_flops_g: float = 2.62
    t1_params_m: float = 9.31

    # Table 1 — StreamCodec baseline @ 1.1 kbps
    streamcodec_t1_lsd: float = 0.918
    streamcodec_t1_stoi: float = 0.896
    streamcodec_t1_visqol: float = 4.048
    streamcodec_t1_mushra: float = 69.64

    # Table 1 — BigCodec (best MUSHRA, non-streamable)
    bigcodec_t1_mushra: float = 77.40

    # Table 2 — VCTK @ 2.7 kbps (VoCodec row)
    t2_lsd: float = 0.847
    t2_stoi: float = 0.857
    t2_visqol: float = 3.840
    t2_mushra: float = 82.95
    t2_mushra_std: float = 2.38

    # Table 3 — voiced/unvoiced LSD @ 1.1 kbps
    t3_vocodec_lsd: float = 0.896
    t3_vocodec_lsd_v: float = 0.700
    t3_vocodec_lsd_u: float = 0.645
    t3_vocodec_stoi: float = 0.916
    t3_vocodec_visqol: float = 4.115
    t3_vocodec_r_lsd: float = 0.959
    t3_vocodec_r_stoi: float = 0.823

    # MUSHRA hidden reference / anchor (Table 1 footnote)
    mushra_hidden_ref: float = 93.21
    mushra_anchor: float = 29.25
