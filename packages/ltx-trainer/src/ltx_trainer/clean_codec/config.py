"""CleanCodec perceptually guided speech tokenization (arXiv:2606.04418)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CleanCodecConfig:
    paper_arxiv: str = "arXiv:2606.04418"
    title: str = (
        "CleanCodec: Efficient and Robust Speech Tokenization via Perceptually Guided Encoding"
    )
    token_rates_tps: tuple[float, ...] = (12.5, 31.25, 62.5)
    fsq_levels: tuple[int, ...] = (8, 8, 8, 8, 8)
    codebook_size: int = 32_768
    global_emb_dim: int = 256
    mel_hz: float = 62.5
    autoencoder_params_m: int = 243
    vocoder_params_m: int = 228
    ssl_model: str = "WavLM-large"
    sv_model: str = "TitaNet-large"

    # Table 1 @12.5 test-clean
    wer_clean: float = 2.7
    cer_clean: float = 1.4
    sim_clean: float = 0.86

    # Table 5 TTS Seed-TTS-eval @12.5
    tts_sim: float = 0.56
    tts_wer: float = 3.9
    tts_rtf: float = 0.170
    tts_speedup_vs_qwen3: float = 17.0

    demo_mel_frames: int = 64
    demo_mel_bins: int = 80
