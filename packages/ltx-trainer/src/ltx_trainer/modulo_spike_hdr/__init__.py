"""Modulo-encoded spike stream HDR unwrapping (Zhou et al. arXiv:2604.14632)."""

from ltx_trainer.modulo_spike_hdr.benchmarks import PAPER_ARXIV, PAPER_TITLE, benchmarks_bundle
from ltx_trainer.modulo_spike_hdr.losses import UnwrapLoss, UnwrapLossConfig
from ltx_trainer.modulo_spike_hdr.model import ModuloSpikeHdrConfig, ModuloSpikeHdrUnwrapper
from ltx_trainer.modulo_spike_hdr.paper import evaluation_demo, framework_card
from ltx_trainer.modulo_spike_hdr.synthetic import synthesize_modulo_pair

__all__ = [
    "ModuloSpikeHdrConfig",
    "ModuloSpikeHdrUnwrapper",
    "PAPER_ARXIV",
    "PAPER_TITLE",
    "UnwrapLoss",
    "UnwrapLossConfig",
    "benchmarks_bundle",
    "evaluation_demo",
    "framework_card",
    "synthesize_modulo_pair",
]
