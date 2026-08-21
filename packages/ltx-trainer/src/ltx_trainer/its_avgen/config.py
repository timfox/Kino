"""Inference-Time Search for audio-video generation (arXiv:2606.03183)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ItsAvgenConfig:
    paper_arxiv: str = "arXiv:2606.03183"
    title: str = "Inference-Time Search for Audio-Video Generation"
    project_url: str = "https://jung-jaemin.github.io/ITS-AVGen-Proj"
    default_n_candidates: int = 4
    vr_weight: float = 0.5
    javis_weight: float = 0.5
    arw_enabled: bool = True
    evo_search_steps: int = 8
