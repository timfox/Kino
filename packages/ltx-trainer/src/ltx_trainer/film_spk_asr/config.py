"""FiLM speaker-conditioned pathological ASR — López et al., arXiv:2606.06211."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FilmSpkAsrConfig:
    paper_arxiv: str = "arXiv:2606.06211"
    title: str = "FiLM-Based Speaker Conditioning of a SpeechLLM for Pathological Speech Recognition"
    framework: str = "FiLM-Spk-ASR"
    code_url: str = "https://github.com/ferugit/film-spk-asr"

    # Base model (§3.1)
    speech_llm: str = "Voxtral-Mini-3B-2507"
    speech_llm_params_b: float = 4.7
    asr_encoder: str = "Whisper-large-v3"
    llm_backbone: str = "Ministral-3B"

    # Speaker embedding (§3.2)
    speaker_model: str = "SiAmResNet34 (WeSpeaker)"
    speaker_params_m: float = 25.2
    xvector_dim: int = 256
    max_waveform_seconds: float = 15.0

    # FiLM trainable footprint (§3.5)
    film_bank_params_m: float = 48.3
    trainable_params_m: float = 73.5
    trainable_fraction_pct: float = 1.6
    film_lr: float = 2e-4
    speaker_lr: float = 2e-5
    gate_bias_init: float = -2.0

    # Datasets (§3.3)
    torgo_hours: float = 13.68
    torgo_speakers: int = 15
    neurovoz_hours: float = 2.31
    neurovoz_speakers: int = 111
    mcqa_pairs: int = 5890

    # Table 1 — base WER (%)
    base_neurovoz_wer: float = 6.75
    base_torgo_wer: float = 25.15
    base_torgo_single_word: float = 46.83
    base_torgo_multi_word: float = 16.13

    # Table 1 — best fine-tuning (raw)
    fft_neurovoz_wer: float = 4.32
    fft_torgo_wer: float = 10.97
    flora_neurovoz_wer: float = 4.07
    flora_torgo_wer: float = 12.71

    # Table 1 — speaker conditioning
    spkcond_neurovoz_wer: float = 6.57
    spkcond_torgo_wer: float = 23.24
    spkcond_torgo_wer_pp: float = 16.36
    spkcond_torgo_multi_word_pp: float = 11.89

    # Table 2 — MCQA sex accuracy (%)
    mcqa_random_sex: float = 33.3
    mcqa_majority_sex: float = 59.5
    mcqa_base_sex: float = 53.5
    mcqa_eft_sex: float = 64.9
    mcqa_fft_sex: float = 62.0
    mcqa_spkcond_sex: float = 60.7
    mcqa_flora_sex: float = 8.4
