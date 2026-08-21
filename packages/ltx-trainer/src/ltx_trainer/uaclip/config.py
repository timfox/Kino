"""Utility-Aware CLIP configuration (arXiv:2605.28733)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class UAClipConfig:
    arxiv: str = "2605.28733"
    embed_dim: int = 512
    temperature: float = 0.07
    alpha_visual: float = 1.0
    beta_semantic: float = 1.0
    eta_utility: float = 0.5
    batch_size: int = 32
