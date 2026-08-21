"""Latent Reasoning Guidance for Parallel Code Translation (Bitan et al., arXiv:2606.05518)."""

from ltx_trainer.latent_prm_guidance.config import LatentPrmGuidanceConfig
from ltx_trainer.latent_prm_guidance.mock import evaluation_smoke
from ltx_trainer.latent_prm_guidance.paper import knowledge_bundle, paper_card
from ltx_trainer.latent_prm_guidance.pipeline import run_demo

__all__ = [
    "LatentPrmGuidanceConfig",
    "evaluation_smoke",
    "knowledge_bundle",
    "paper_card",
    "run_demo",
]
