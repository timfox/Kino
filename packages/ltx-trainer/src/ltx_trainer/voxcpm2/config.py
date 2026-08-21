"""VoxCPM2 — hierarchical diffusion-autoregressive TTS (arXiv:2606.06928)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

GenerationMode = Literal[
    "basic_tts",
    "voice_design",
    "reference_cloning",
    "controllable_cloning",
    "continuation_cloning",
]


@dataclass
class Voxcpm2Config:
    paper_arxiv: str = "arXiv:2606.06928"
    title: str = "VoxCPM2 Technical Report"
    framework: str = "VoxCPM2"
    license: str = "Apache-2.0"
    params_b: float = 2.0
    github: str = "https://github.com/OpenBMB/VoxCPM"
    hf_model: str = "openbmb/VoxCPM2"
    hf_demo: str = "https://huggingface.co/spaces/openbmb/VoxCPM-Demo"
    docs: str = "https://voxcpm.readthedocs.io/en/latest/"
    backbone: str = "MiniCPM-4-1B"

    # Scale (§1.2)
    training_hours_m: float = 2.0
    n_languages: int = 30
    n_chinese_dialects: int = 9

    # AudioVAE V2 (§3.2)
    encode_sample_rate_hz: int = 16000
    decode_sample_rate_hz: int = 48000
    latent_dim: int = 64
    latent_fps: float = 25.0
    patch_frames: int = 4
    lm_token_rate_hz: float = 6.25
    fsq_dim: int = 512

    # Inference defaults (§3.7)
    cfg_alpha: float = 2.0
    inference_timesteps: int = 10
    cfg_dropout_train: float = 0.10

    # Table 3 — Seed-TTS-Eval (reference + continuation recipe default elsewhere)
    seed_en_wer: float = 1.84
    seed_en_sim: float = 75.3
    seed_zh_cer: float = 0.97
    seed_zh_sim: float = 79.5
    seed_zh_hard_cer: float = 8.13
    seed_zh_hard_sim: float = 75.3

    # Table 4 — Reference + Continuation best SIM recipe
    seed_ref_cont_en_wer: float = 0.99
    seed_ref_cont_en_sim: float = 79.5

    # Internal 30-language benchmark (§4.3)
    internal_30lang_avg_wer: float = 1.68

    # Table 9 — InstructTTSEval EN
    instruct_aps_en: float = 84.2
    instruct_dsd_en: float = 83.2
    instruct_rp_en: float = 71.4

    # Table 11 — RTX 4090 RTF
    rtf_pytorch: float = 0.30
    rtf_nanovllm: float = 0.13
    vram_gb: float = 8.0

    # Table 12 — subjective zero-shot cloning
    nmos: float = 4.78
    smos: float = 4.74

    # Table 10 — AudioVAE V2 VCTK MelD-48k
    vae_meld_48k_vctk: float = 1.335

    pip_package: str = "voxcpm"
    upstream_clone_url: str = "https://github.com/OpenBMB/VoxCPM.git"

    generation_modes: tuple[GenerationMode, ...] = (
        "basic_tts",
        "voice_design",
        "reference_cloning",
        "controllable_cloning",
        "continuation_cloning",
    )

    supported_languages: tuple[str, ...] = (
        "Arabic",
        "Burmese",
        "Chinese",
        "Danish",
        "Dutch",
        "English",
        "Finnish",
        "French",
        "German",
        "Greek",
        "Hebrew",
        "Hindi",
        "Indonesian",
        "Italian",
        "Japanese",
        "Khmer",
        "Korean",
        "Lao",
        "Malay",
        "Norwegian",
        "Polish",
        "Portuguese",
        "Russian",
        "Spanish",
        "Swahili",
        "Swedish",
        "Tagalog",
        "Thai",
        "Turkish",
        "Vietnamese",
    )

    chinese_dialects: tuple[str, ...] = (
        "Sichuan",
        "Cantonese",
        "Wu",
        "Northeastern",
        "Henan",
        "Shaanxi",
        "Shandong",
        "Tianjin",
        "Minnan",
    )

    pipeline_components: tuple[str, ...] = (
        "LocEnc",
        "TSLM",
        "FSQ",
        "RALM",
        "LocDiT",
        "AudioVAE_V2",
    )
