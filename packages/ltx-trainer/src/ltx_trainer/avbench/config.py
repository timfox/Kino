"""Configuration for AVBench (arXiv:2605.24652)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AVBenchConfig:
    paper_arxiv: str = "arXiv:2605.24652"
    project_page: str = "https://yajialiang.github.io/AVBench-site/"

    # Test prompts (Sec. 3.1 / Sec. 8): ≥720p, manually verified, hash-deduplicated vs training.
    n_prompts_normal: int = 350
    n_prompts_hard: int = 120
    n_prompts_total: int = 470

    # Evaluator training (Sec. 3, Fig. 2).
    n_source_clips_openhumanvid: int = 30_000
    clip_duration_seconds: tuple[float, float] = (8.0, 12.0)
    n_pairs_per_axis: int = 100_000  # AT, VT, AV each
    n_pairs_total: int = 300_000

    # SFT backbones (Sec. 3.3).
    backbone_vt_av: str = "Qwen2.5-Omni"
    backbone_at: str = "Qwen2-Audio"

    # Hard-negative text similarity filter (Sec. 3.2).
    char_similarity_min: float = 0.70
    char_similarity_max: float = 0.995

    # Greedy sampling quota for prompt curation.
    max_single_attribute_fraction: float = 0.50

    # Human validation (Sec. 4.3): 2AFC instance-level (Sec. 9.1).
    human_2afc_mean_accuracy_pct: float = 85.4
    human_2afc_speech_content_peak_pct: float = 98.1

    # Figure 14: VT consistency instance accuracy.
    vt_consistency_ours_accuracy_pct: float = 92.31
    vt_consistency_base_qwen_accuracy_pct: float = 47.44

    metric_keys: tuple[str, ...] = (
        "AV",
        "AT",
        "VT",
        "SyncNet",
        "SC",
        "DF_Arena",
        "NISQA",
        "Audiobox",
        "DOVERpp",
        "Aesthetic",
    )

    cross_modal_dimensions: tuple[str, ...] = (
        "AT_consistency",
        "VT_consistency",
        "AV_consistency",
        "lip_sync",
    )
    unimodal_dimensions: tuple[str, ...] = (
        "speech_content",
        "speech_realism",
        "audio_quality",
        "audio_aesthetics",
        "video_quality",
        "video_aesthetics",
    )

    evaluators_frozen_components: dict[str, str] = field(
        default_factory=lambda: {
            "VT_AV": "Visual encoder + projector frozen; LLM fine-tuned",
            "AT": "LLM + connector fine-tuned for larger audio–text gap",
        }
    )
