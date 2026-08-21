"""Representation-centric continual learning for speech (arXiv:2605.24863)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SpeechClConfig:
    paper_arxiv: str = "arXiv:2605.24863"
    title: str = (
        "Rethinking Continual Learning for Speech and Audio: "
        "A Representation-Centric Taxonomy and Open Problems"
    )
    venue: str = "ICML 2026 preprint"
    affiliation: str = "University of Melbourne"
    github_refs: str = "Full references in authors' GitHub list"
    foundation_models: tuple[str, ...] = (
        "wav2vec 2.0",
        "HuBERT",
        "Whisper",
        "Qwen2-Audio",
        "LALMs",
    )
    classical_cl_settings: tuple[str, ...] = (
        "task-incremental",
        "domain-incremental",
        "class-incremental",
    )
