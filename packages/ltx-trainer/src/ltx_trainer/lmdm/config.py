"""Configuration for Live Music Diffusion Models (arXiv:2605.22717)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class LMDMVariant(str, Enum):
    ENCODER_DECODER = "encoder_decoder"
    BLOCK_CAUSAL = "block_causal"


@dataclass
class LMDMConfig:
    paper_arxiv: str = "arXiv:2605.22717"
    demo_url: str = "https://stephenbrade.github.io/lmdm-public/"
    backbone: str = "Stable Audio Open Small"
    backbone_params_M: float = 340.0
    latent_frames_total: int = 240
    block_size_o: int = 48
    context_s: int = 192
    diffusion_steps_K: int = 50
    arc_forcing_steps: int = 8
    sample_rate_hz: int = 44100
    default_variant: LMDMVariant = LMDMVariant.ENCODER_DECODER


@dataclass
class LMDMLatencyStats:
    pre_arc_ms: tuple[int, int] = (110, 170)
    post_arc_ms: int = 30
    generative_delay_s: float = 1.0
