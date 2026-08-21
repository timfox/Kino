"""Speech foundation model vs human speaker similarity — Kishi et al., arXiv:2606.05739."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SfmSpeakerSimConfig:
    paper_arxiv: str = "arXiv:2606.05739"
    title: str = (
        "Do speech foundation models perceive speaker similarity as humans do?"
    )
    framework: str = "SFM-Speaker-Sim"

    n_models: int = 43
    n_model_families: int = 6
    spectral_k: int = 10
    utterances_per_speaker: int = 10

    perceptual_score_min: float = -3.0
    perceptual_score_max: float = 3.0

    # Representative layer-max mean LCC (Fig. 2 qualitative anchors)
    wavlm_large_layer_max_lcc: float = 0.38
    qwen3_tts_layer_max_lcc: float = 0.12
    whisper_large_layer_max_lcc: float = 0.25
    audiogen_layer_max_lcc: float = 0.22

    # Table 1 — layer_max LCC regression (coefficient, p-value)
    reg_is_dec_lcc: float = -0.77
    reg_is_dec_lcc_p: float = 0.001
    reg_is_ssl_lcc: float = -0.14
    reg_is_ssl_lcc_p: float = 0.020
    reg_params_lcc: float = -0.14
    reg_params_lcc_p: float = 0.022
    reg_layer_max_lcc_r2: float = 0.747

    reg_is_dec_slope_lcc: float = 0.31
    reg_is_dec_slope_lcc_p: float = 0.002
    reg_params_slope_lcc: float = 0.29
    reg_params_slope_lcc_p: float = 0.008
    reg_layer_slope_lcc_r2: float = 0.205

    datasets: tuple[str, ...] = ("JVS", "VCTK_female")

    model_families: tuple[str, ...] = (
        "supervised_asr",
        "supervised_tts",
        "supervised_tta",
        "supervised_audio_cls",
        "speech_ssl",
        "audio_ssl",
    )
