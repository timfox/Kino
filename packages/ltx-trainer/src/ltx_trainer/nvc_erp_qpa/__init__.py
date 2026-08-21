"""NVC ERP quality-parameter adaptation (Arai et al., arXiv:2512.20093)."""

from ltx_trainer.nvc_erp_qpa.benchmarks import benchmarks_bundle, table1_row
from ltx_trainer.nvc_erp_qpa.config import NvcErpQpaConfig, PAPER_ARXIV, PAPER_TITLE
from ltx_trainer.nvc_erp_qpa.mock import evaluation_smoke
from ltx_trainer.nvc_erp_qpa.nvc_stub import DcvcRtQpaStub
from ltx_trainer.nvc_erp_qpa.paper import evaluation_demo, framework_card
from ltx_trainer.nvc_erp_qpa.pipeline import encode_step, evaluation_demo_run

__all__ = [
    "DcvcRtQpaStub",
    "NvcErpQpaConfig",
    "PAPER_ARXIV",
    "PAPER_TITLE",
    "benchmarks_bundle",
    "encode_step",
    "evaluation_demo",
    "evaluation_demo_run",
    "evaluation_smoke",
    "framework_card",
    "table1_row",
]
