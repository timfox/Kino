"""VISA — Visual Information Strengthened Audio-Reasoning (arXiv:2606.07264)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class VisaConfig:
    paper_arxiv: str = "arXiv:2606.07264"
    title: str = (
        "VISA: A Visual Information Strengthened Audio-Reasoning System "
        "for the Interspeech 2026 ARC Agent Track"
    )
    challenge: str = "Interspeech 2026 Audio Reasoning Challenge (Agent Track)"
    benchmark: str = "MMAR"
    rubrics_protocol: str = "MMAR Rubrics"

    # LALM ensemble (§2.2)
    lalm_models: tuple[str, ...] = ("Qwen3-Omni-Thinking", "Step-Audio-R1")
    vote_samples_k: int = 3
    vote_temperature: float = 0.7

    # Orchestration backbones (§3)
    llm_backbone: str = "GLM-4.6"
    vlm_backbone: str = "Qwen3-VL-235B-A22B"
    captioner: str = "Qwen3-Omni-Captioner"
    sed_model: str = "FlexSED"

    # Table 1 — MMAR modality average (%)
    mmar_avg_accuracy: float = 77.4
    mmar_sound: float = 71.5
    mmar_music: float = 62.6
    mmar_speech: float = 84.0

    # Table 4 — Agent Track leaderboard
    rubrics_score: float = 66.23
    accuracy: float = 77.40
    agent_track_rank: int = 2

    # Ablation — w/o fine-grained category routing
    ablation_accuracy: float = 73.30
    ablation_rubrics: float = 62.63

    # Table 2 — hierarchical layer average (%)
    subcategory_avg: float = 70.42

    num_fine_categories: int = 27
    num_routing_strategies: int = 3

    acoustic_views: tuple[str, ...] = field(
        default_factory=lambda: ("mel", "cqt", "rms", "chroma", "spectrogram")
    )
