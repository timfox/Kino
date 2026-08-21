"""Cause-aware SDS error recovery (arXiv:2605.25404)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CauseSdsConfig:
    paper_arxiv: str = "arXiv:2605.25404"
    asr_backbone: str = "nvidia/parakeet-tdt-0.6b-v2"
    tsallis_alpha: float = 0.33
    clarification_rounds: int = 3
    mc_passes_per_checkpoint: int = 50  # unused in stub; paper uses 4×50 for other systems
    detector_params_m: int = 10
    detector_cnn_layers: int = 5
    detector_kernel: int = 5
    hidden_dim: int = 640
    vocab_size: int = 1024
    num_distortion_classes: int = 6
    llm_dialogue: str = "GPT-5.2"
    llm_user_sim: str = "GPT-5.2"
    tts: str = "CosyVoice3"
