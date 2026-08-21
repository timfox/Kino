"""SET stream-event-triggered CUDA graph scheduling (Li et al., arXiv:2606.05495)."""

from ltx_trainer.set_cuda_graph.config import SetCudaGraphConfig
from ltx_trainer.set_cuda_graph.mock import evaluation_smoke
from ltx_trainer.set_cuda_graph.paper import knowledge_bundle, paper_card
from ltx_trainer.set_cuda_graph.pipeline import run_demo

__all__ = [
    "SetCudaGraphConfig",
    "evaluation_smoke",
    "knowledge_bundle",
    "paper_card",
    "run_demo",
]
