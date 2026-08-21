"""Discriminative omni-modal MSA config (arXiv:2606.05713)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class OmniMsaConfig:
    paper_arxiv: str = "arXiv:2606.05713"
    backbone: str = "Qwen2.5-Omni-7B (Thinker)"
    hidden_size: int = 3584
    head_hidden: int = 256
    head_dropout: float = 0.2
    sentiment_range: tuple[float, float] = (-3.0, 3.0)
    lora_rank: int = 32
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_targets: tuple[str, ...] = ("q", "k", "v", "o", "gate", "up", "down")
    quant_bits: int = 4
    quant_dtype: str = "nf4"
    compute_dtype: str = "bfloat16"
    trainable_param_pct: float = 1.14
    trainable_params_m: float = 103.0
    peak_memory_gb_min: float = 10.0
    peak_memory_gb_max: float = 21.0
    gpu: str = "NVIDIA RTX 5090 32GB"
    max_frames: int = 16
    lora_lr: float = 2e-4
    head_lr: float = 1e-3
    datasets: tuple[str, ...] = ("CMU-MOSI", "CMU-MOSEI")
    mosi_utterances: int = 2199
    mosei_utterances: int = 23000
    audio_denoiser: str = "DeepFilterNet"
    readout_modes: tuple[str, ...] = ("discriminative", "generative_zero_shot", "generative_trained")
    modality_configs: tuple[str, ...] = ("text_only", "text_video", "text_video_audio")
