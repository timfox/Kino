"""PCCL collective algorithm synthesizer (Won et al., arXiv:2606.07019)."""

from ltx_trainer.pccl.config import PcclConfig
from ltx_trainer.pccl.mock import evaluation_smoke
from ltx_trainer.pccl.paper import knowledge_bundle, paper_card
from ltx_trainer.pccl.pipeline import run_demo
from ltx_trainer.pccl.synthesis import synthesize

__all__ = [
    "PcclConfig",
    "evaluation_smoke",
    "knowledge_bundle",
    "paper_card",
    "run_demo",
    "synthesize",
]
