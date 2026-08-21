"""Configuration for MTAVG-Bench 2.0 (arXiv:2605.28035)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class MTAVG2Config:
    paper_arxiv: str = "arXiv:2605.28035"
    github: str = "https://github.com/ChinChilla-HTL/MTAVG-Bench2"
    huggingface_dataset: str = "Lanht/MTAVG-Bench2"

    n_videos: int = 2466
    n_qa_instances: int = 11600
    n_sub_dimensions: int = 10
    n_failure_modes: int = 45
    n_source_movie_clips: int = 20

    segment_duration_s: float = 8.0
    expert_pool_size: int = 22
    two_expert_agreement_rate: float = 0.841
    cohens_kappa: float = 0.78
    third_expert_intervention_rate: float = 0.159

    # Temporal localization (Table 5) — Gemini 3.1 Pro upper bound in paper.
    gemini31_pro_pia_pct: float = 60.6
    gemini31_pro_tla_pct: float = 60.9
    gemini31_pro_rc_pct: float = 83.8

    # Sentiment composition (Appendix B.1, Fig. 6).
    sentiment_negative_pct: float = 42.0
    sentiment_positive_pct: float = 38.5
    sentiment_neutral_pct: float = 19.5

    sub_dimension_codes: tuple[str, ...] = (
        "EP",
        "MP",
        "DP",
        "IP",
        "MC",
        "EC",
        "SD",
        "IC",
        "IG",
        "CT",
    )

    category_codes: tuple[str, ...] = ("acting", "atmosphere", "cinematography")

    question_formats: tuple[str, ...] = (
        "mcq_single",
        "mcq_multi",
        "pairwise",
        "temporal_localization",
    )

    ltx_model_label: str = "LTX 2.3"

    eval_models_proprietary: tuple[str, ...] = (
        "Gemini 3.1 Flash Lite",
        "Gemini 3.1 Pro",
        "Gemini 3 Flash",
        "Gemini 2.5 Flash",
    )
    eval_models_opensource: tuple[str, ...] = (
        "Qwen 2.5 Omni 7B",
        "MiniCPM-o 2.6 7B",
        "OmniVinci 9B",
        "VideoLLaMA 2 7B",
        "Ola Omni 7B",
        "Ming Lite Omni 1.5 30B",
    )

    generator_models_table4: tuple[str, ...] = (
        "Grok Video 3",
        "LTX 2.3",
        "Sora 2",
        "Veo 3.1",
        "Vidu Q3",
        "Wan 2.6",
    )

    fold_role: str = "cinematic_expressiveness_proxy"

    evaluators_frozen_note: dict[str, str] = field(
        default_factory=lambda: {
            "production": "Omni MLLMs (Gemini 3.1 Pro, Qwen2.5-Omni, etc.) on Hub QA set",
            "rc_judge": "GPT-5.4 Likert rationale-consistency (Appendix A.3)",
        }
    )
