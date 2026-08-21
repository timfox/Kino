"""Configuration for FAST-ME (arXiv:2605.23428)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FastMEConfig:
    paper_arxiv: str = "arXiv:2605.23428"
    default_block_size: int = 16
    default_search_range: int = 7
    # OST / FAST-ME defaults (Sec. V).
    alpha: float = 0.7
    delta0: float = 0.05
    theta: float = 1.0
    fm_backends: tuple[str, ...] = ("ViT", "SAM", "CLIP")


@dataclass
class BlockCandidate:
    """One block-matching candidate with distortion and optional attention."""

    sad: float
    attention: float = 0.0
