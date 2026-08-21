"""DeblurNVS: geometric latent diffusion for blur-aware NVS (arXiv:2606.01315)."""

from ltx_trainer.deblur_nvs.benchmarks import benchmarks_bundle, table2_ours
from ltx_trainer.deblur_nvs.config import (
    DeblurNVSConfig,
    PAPER_ARXIV,
    PAPER_TITLE,
    PAPER_URL,
    PROJECT_URL,
)
from ltx_trainer.deblur_nvs.dataset import dataset_card
from ltx_trainer.deblur_nvs.integration import gopex_links, ltx_plan_stub, ltx_prep_notes
from ltx_trainer.deblur_nvs.latent_models import DeblurNVSStub
from ltx_trainer.deblur_nvs.paper import evaluation_demo, framework_card
from ltx_trainer.deblur_nvs.pipeline import evaluation_demo_run, infer_novel_view, train_step
from ltx_trainer.deblur_nvs.mock import evaluation_smoke

__all__ = [
    "DeblurNVSConfig",
    "DeblurNVSStub",
    "PAPER_ARXIV",
    "PAPER_TITLE",
    "PAPER_URL",
    "PROJECT_URL",
    "benchmarks_bundle",
    "dataset_card",
    "evaluation_demo",
    "evaluation_demo_run",
    "evaluation_smoke",
    "framework_card",
    "gopex_links",
    "infer_novel_view",
    "ltx_plan_stub",
    "ltx_prep_notes",
    "table2_ours",
    "train_step",
]
