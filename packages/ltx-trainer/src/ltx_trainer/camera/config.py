"""CAMERA: Case-Adaptive Multi-cue Expert fRAmework for TAGFD (arXiv:2605.20032)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CAMERAConfig:
    paper_arxiv: str = "arXiv:2605.20032"
    github: str = "https://github.com/CampanulaBells/CAMERA"
    text_encoder: str = "OpenAI text-embedding-3-small (paper)"
    num_moe_layers: int = 2
    num_experts: int = 3  # graph, semantic, global
    hidden_dim: int = 64
    loss_alpha_gating: float = 1.0
    loss_beta_oc: float = 1.0
    gating_epsilon: float = 1e-8
