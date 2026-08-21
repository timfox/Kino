"""LumaFlux: physically-guided DiT for SDR→HDR ITM (Saini et al. arXiv:2604.02787)."""

from ltx_trainer.lumaflux.benchmarks import PAPER_ARXIV, benchmarks_bundle
from ltx_trainer.lumaflux.config import LumaFluxConfig, PAPER_TITLE, PAPER_URL
from ltx_trainer.lumaflux.dataset import dataset_summary
from ltx_trainer.lumaflux.losses import LumaFluxLoss, LumaFluxLossConfig
from ltx_trainer.lumaflux.metrics import TABLE1_BENCHMARKS, TABLE2_TMO, TABLE3_ABLATION, TABLE4_USER_STUDY
from ltx_trainer.lumaflux.model import LumaFlux
from ltx_trainer.lumaflux.pipeline import ablation_configs, paper_report, train_step

__all__ = [
    "LumaFlux",
    "LumaFluxConfig",
    "LumaFluxLoss",
    "LumaFluxLossConfig",
    "benchmarks_bundle",
    "evaluation_demo",
    "framework_card",
    "PAPER_TITLE",
    "PAPER_URL",
    "TABLE1_BENCHMARKS",
    "TABLE2_TMO",
    "TABLE3_ABLATION",
    "TABLE4_USER_STUDY",
    "ablation_configs",
    "dataset_summary",
    "paper_report",
    "train_step",
]
