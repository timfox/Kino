"""DiffHDR: LDR-to-HDR video diffusion (Yu et al. arXiv:2604.06161)."""

from ltx_trainer.diffhdr.config import DiffHDRConfig, PAPER_TITLE, PAPER_URL
from ltx_trainer.diffhdr.dataset import dataset_summary
from ltx_trainer.diffhdr.log_gamma import inverse_log_gamma_map, log_gamma_map
from ltx_trainer.diffhdr.benchmarks import PAPER_ARXIV, benchmarks_bundle
from ltx_trainer.diffhdr.metrics import TABLE1_SI_HDR, TABLE4_LOG_GAMMA, TABLE5_ABLATION
from ltx_trainer.diffhdr.paper import evaluation_demo, framework_card
from ltx_trainer.diffhdr.model import DiffHDR
from ltx_trainer.diffhdr.pipeline import ablation_configs, paper_report, train_step

__all__ = [
    "PAPER_ARXIV",
    "DiffHDR",
    "DiffHDRConfig",
    "benchmarks_bundle",
    "evaluation_demo",
    "framework_card",
    "PAPER_TITLE",
    "PAPER_URL",
    "TABLE1_SI_HDR",
    "TABLE4_LOG_GAMMA",
    "TABLE5_ABLATION",
    "ablation_configs",
    "dataset_summary",
    "inverse_log_gamma_map",
    "log_gamma_map",
    "paper_report",
    "train_step",
]
