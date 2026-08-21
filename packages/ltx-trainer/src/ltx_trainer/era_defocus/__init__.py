"""ErA error-aware defocus deblurring (Vo & Park, arXiv:2606.06540)."""

from ltx_trainer.era_defocus.config import EraDefocusConfig
from ltx_trainer.era_defocus.mock import evaluation_smoke
from ltx_trainer.era_defocus.paper import knowledge_bundle, paper_card
from ltx_trainer.era_defocus.pipeline import run_demo

__all__ = [
    "EraDefocusConfig",
    "evaluation_smoke",
    "knowledge_bundle",
    "paper_card",
    "run_demo",
]
