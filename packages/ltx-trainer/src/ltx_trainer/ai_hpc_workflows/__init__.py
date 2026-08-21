"""Twelve quick tips for AI-driven HPC workflows (Alnasir, arXiv:2606.07491)."""

from ltx_trainer.ai_hpc_workflows.config import AiHpcWorkflowConfig
from ltx_trainer.ai_hpc_workflows.mock import evaluation_smoke
from ltx_trainer.ai_hpc_workflows.paper import knowledge_bundle, paper_card
from ltx_trainer.ai_hpc_workflows.pipeline import run_demo
from ltx_trainer.ai_hpc_workflows.tips import tips_card

__all__ = [
    "AiHpcWorkflowConfig",
    "evaluation_smoke",
    "knowledge_bundle",
    "paper_card",
    "run_demo",
    "tips_card",
]
