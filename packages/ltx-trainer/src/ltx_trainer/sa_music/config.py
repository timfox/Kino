"""South Asian music LLM benchmark — Kader et al., arXiv:2606.05522."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SaMusicConfig:
    paper_arxiv: str = "arXiv:2606.05522"
    title: str = "Exploring LLMs for South Asian Music Understanding and Generation"
    framework: str = "SA-Music-Bench"
    venue: str = "arXiv"

    github_data: str = "https://github.com/Faria-Binte-Kader/South-Asian-Music-data"
    benchmark_questions: int = 504
    reference_abc_scores: int = 100
    rabindra_scores: int = 50
    nazrul_scores: int = 50
    models_evaluated: int = 33
    generation_models: int = 9
    generation_prompts: int = 500
    telr_levels: int = 5
    human_eval_samples: int = 180

    theory_questions: int = 163
    knowledge_questions: int = 143
    continuation_questions: int = 198

    sample_rate_hz: int = 24000
    abc_tonic: str = "C"
    reference_pitch_entropy: float = 2.86

    # Fig. 2 understanding accuracy (Gemini 2.5 Pro)
    gemini_theory_acc: float = 0.908
    gemini_knowledge_acc: float = 0.894
    gemini_continuation_acc: float = 0.852

    opensource_theory_low: float = 0.23
    opensource_theory_high: float = 0.40

    chatmusician_theory: float = 0.123
    chatmusician_knowledge: float = 0.119
    chatmusician_continuation: float = 0.019

    # Table 2 human eval — Gemini 2.5 Pro
    gemini_structureness: float = 3.65
    gemini_genre_accuracy: float = 0.95
    gemini_style_accuracy: float = 0.40

    # Fig. 4 KL vs style correlation
    kl_style_pearson_r: float = -0.73
    scale_adherence_style_r: float = -0.41
    abc_syntax_style_r: float = -0.06

    # Table 6 mean automatic scores (L3 representative)
    kl_l3_mean: float = 4.320
    abc_syntax_l3_mean: float = 0.520
    pitch_entropy_l3_mean: float = 2.300
