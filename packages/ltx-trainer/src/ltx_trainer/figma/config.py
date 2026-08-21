"""FIGMA — fine-grained music retrieval (Anand et al., arXiv:2606.06615)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FigmaConfig:
    paper_arxiv: str = "arXiv:2606.06615"
    title: str = "FIGMA: Towards FIne-Grained Music retrievAl"
    framework: str = "FIGMA"
    project_url: str = "https://nishitanand.github.io/figma-website"

    audio_encoder: str = "MuQ"
    text_encoder: str = "intfloat/multilingual-e5-large-instruct"
    audio_sample_rate_hz: int = 24000
    clip_seconds: float = 10.0
    audio_frames: int = 250
    text_tokens: int = 128
    embed_dim: int = 512
    frozen_params_m: float = 800.0
    trainable_params_m: float = 22.0

    # Multi-view loss (§3.1.4)
    alpha_global: float = 0.6
    temperature: float = 0.07
    batch_size: int = 256
    epochs: int = 15
    learning_rate: float = 1e-4

    # FGMCaps (§3.2)
    fgmcaps_train: int = 380_878
    fgmcaps_val: int = 10_000
    fgmcaps_test: int = 10_000
    caption_saturation_tokens: int = 50

    # Table 2 — MusicBench T2A / A2T R@1
    musicbench_t2a_r1: float = 34.52
    musicbench_t2a_r5: float = 65.99
    musicbench_a2t_r1: float = 39.09
    musicbench_a2t_r5: float = 68.02
    clamp3_t2a_r1: float = 28.43

    # Table 3 — FMACaps-Eval T2A R@1 (+73.3% vs CLAMP3)
    fmacaps_t2a_r1: float = 13.00
    fmacaps_a2t_r1: float = 13.20
    clamp3_fmacaps_t2a_r1: float = 7.50
    relative_improvement_fmacaps_pct: float = 73.3

    # Table 5 — FGMCaps test R@1
    fgmcaps_test_t2a_r1: float = 26.15
    fgmcaps_test_a2t_r1: float = 26.86

    # Table 4 — perturbation A2T R@1 (original)
    perturb_original_a2t_r1: float = 46.53

    fgmcaps_attributes: tuple[str, ...] = (
        "chord_progression",
        "tempo_bpm",
        "beat_count",
        "key",
        "genre",
        "mood",
    )
    extraction_tools: tuple[str, ...] = ("BeatNet", "Omnizart", "Essentia KeyExtractor")
    caption_llm: str = "Qwen3-Next-80B-A3B-Instruct"
