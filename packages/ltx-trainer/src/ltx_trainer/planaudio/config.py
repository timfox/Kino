"""Configuration for PlanAudio (arXiv:2605.28063)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PlanAudioConfig:
    paper_arxiv: str = "arXiv:2605.28063"
    paper_title: str = "Unified Synthesis of Compositional Speech and Sound from Free-Form Text Prompts"

    backbone_llm: str = "Qwen/Qwen2.5-1.5B"
    semantic_encoder: str = "Audio Flamingo 3 Encoder (AF3Encoder)"
    acoustic_tokenizer: str = "AudioCraft (EnCodec-style, Q codebooks)"

    latent_cot_steps: int = 6  # K — downsampled from 750 AF3 embeddings
    af3_embedding_steps: int = 750
    af3_pool_stride: int = 150

    training_pool_size: int = 1_270_000
    composite_train: int = 371_000
    sound_train: int = 451_000
    speech_train: int = 354_000
    planaudio_bench_size: int = 4_500
    annotations_per_clip: int = 5

    special_tokens: tuple[str, ...] = (
        "<|sot|>",
        "<|sol|>",
        "<|soa|>",
        "<|eoa|>",
    )

    loss_lambda_cosine: float = 1.0  # λ in Eq. (5)
    loss_lambda_latent: float = 1.0  # λ1
    loss_lambda_audio: float = 1.0  # λ2

    inference_top_k: int = 25

    scenarios: tuple[str, ...] = ("sound", "speech", "composite")

    curriculum_strategies: tuple[str, ...] = ("constant", "gradual", "disjoint")

    cot_variants: tuple[str, ...] = (
        "semantic_latent",
        "none",
        "explicit",
        "acoustic",
    )

    fold_role: str = "compositional_unified_audio_proxy"
