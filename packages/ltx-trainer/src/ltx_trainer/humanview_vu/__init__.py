"""Watch, Remember, Reason — human-view video MLLM survey (arXiv:2606.07433)."""

from ltx_trainer.humanview_vu.config import (
    AWESOME_URL,
    HumanViewVuConfig,
    PAPER_ARXIV,
    PAPER_TITLE,
    PAPER_URL,
)
from ltx_trainer.humanview_vu.mock import evaluation_smoke
from ltx_trainer.humanview_vu.paper import evaluation_demo, framework_card, knowledge_blob
from ltx_trainer.humanview_vu.taxonomy import classify_method, taxonomy_card

__all__ = [
    "AWESOME_URL",
    "HumanViewVuConfig",
    "PAPER_ARXIV",
    "PAPER_TITLE",
    "PAPER_URL",
    "classify_method",
    "evaluation_demo",
    "evaluation_smoke",
    "framework_card",
    "knowledge_blob",
    "taxonomy_card",
]
