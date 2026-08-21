"""FDIM: feature-distance generic VQA (Wang et al. arXiv:2604.24123)."""

from ltx_trainer.fdim.benchmarks import benchmarks_bundle
from ltx_trainer.fdim.mapping import LogisticMapping, fuse_component_scores
from ltx_trainer.fdim.metrics import plcc, srocc
from ltx_trainer.fdim.model import FDIM, FDIMConfig
from ltx_trainer.fdim.paper import evaluation_demo, framework_card
from ltx_trainer.fdim.pipeline import load_fdim_checkpoint, score_video_pair
from ltx_trainer.fdim.ranking_loss import RankingLoss

__all__ = [
    "FDIM",
    "FDIMConfig",
    "LogisticMapping",
    "RankingLoss",
    "benchmarks_bundle",
    "evaluation_demo",
    "framework_card",
    "fuse_component_scores",
    "load_fdim_checkpoint",
    "plcc",
    "score_video_pair",
    "srocc",
]
