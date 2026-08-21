"""MHS attribute alignment configuration (arXiv:2605.27025)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MhsAlignConfig:
    paper_arxiv: str = "2605.27025"
    paper_title: str = (
        "Attribute-Based Diagnosis of LLM Alignment with Hate Speech Annotations"
    )
    authors: str = (
        "Mohammad Amine Jradi, Faeze Ghorbanpour, Alexander Fraser (TU Munich / MCML)"
    )

    corpus: str = "Measuring Hate Speech (MHS; Kennedy et al., 2020)"
    corpus_comments: int = 39_565
    corpus_annotators: int = 7_912

    models_evaluated: tuple[str, ...] = (
        "Meta-Llama-3.1-70B-Instruct",
        "Meta-Llama-3.1-8B-Instruct",
        "Qwen2.5-72B-Instruct",
        "Qwen2.5-7B-Instruct",
    )

    main_finding: str = (
        "Behaviorally explicit MHS attributes (insult, humiliate, violence, …) align positively "
        "with humans; evaluative attributes (respect, sentiment, status, hatespeech) invert. "
        "Confidence-weighted Ridge on attribute predictions reconstructs IRT hate scores "
        "better than direct prompting (R² up to ~0.71)."
    )

    best_r2_large: tuple[tuple[str, str, float], ...] = (
        ("Llama-70B", "persona", 70.71),
        ("Llama-70B", "vanilla", 70.57),
        ("Qwen-72B", "vanilla", 68.85),
        ("Qwen-72B", "persona", 68.65),
    )

    hate_binary_threshold: float = 0.5  # Kennedy et al. 2020
