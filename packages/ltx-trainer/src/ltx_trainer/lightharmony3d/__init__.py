"""LightHarmony3D: mesh insertion in 3DGS with GenEnvLighting (arXiv:2603.29209)."""

from ltx_trainer.lightharmony3d.benchmarks import PAPER_ARXIV, benchmarks_bundle
from ltx_trainer.lightharmony3d.config import LightHarmony3DConfig, PAPER_TITLE, PAPER_URL
from ltx_trainer.lightharmony3d.dataset import dataset_summary
from ltx_trainer.lightharmony3d.metrics import TABLE1_LH3D_KU, TABLE2_VQA_MIPNERF360, TABLE4_ABLATION
from ltx_trainer.lightharmony3d.model import LightHarmony3D
from ltx_trainer.lightharmony3d.paper import evaluation_demo, framework_card
from ltx_trainer.lightharmony3d.pipeline import ablation_configs, paper_report, train_step

__all__ = [
    "PAPER_ARXIV",
    "PAPER_TITLE",
    "PAPER_URL",
    "LightHarmony3D",
    "LightHarmony3DConfig",
    "TABLE1_LH3D_KU",
    "TABLE2_VQA_MIPNERF360",
    "TABLE4_ABLATION",
    "ablation_configs",
    "benchmarks_bundle",
    "dataset_summary",
    "evaluation_demo",
    "framework_card",
    "paper_report",
    "train_step",
]
