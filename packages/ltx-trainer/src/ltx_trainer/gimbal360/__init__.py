"""Gimbal360: canonicalized 360° completion with DAL + TEG (arXiv:2603.23179)."""

from ltx_trainer.gimbal360.benchmarks import PAPER_ARXIV, PAPER_TITLE, benchmarks_bundle
from ltx_trainer.gimbal360.config import Gimbal360Config, HORIZON360_SIZE, PROJECT_URL
from ltx_trainer.gimbal360.gimbal360_net import Gimbal360CompletionStub
from ltx_trainer.gimbal360.paper import evaluation_demo, framework_card
from ltx_trainer.gimbal360.fill_conditioning import concat_fill_channels
from ltx_trainer.gimbal360.horizon360 import dataset_card
from ltx_trainer.gimbal360.inference import SamplerConfig, complete_panorama_stub, shift_equivariant_sample
from ltx_trainer.gimbal360.pipeline import evaluation_demo_run, train_step

__all__ = [
    "Gimbal360CompletionStub",
    "Gimbal360Config",
    "HORIZON360_SIZE",
    "PAPER_ARXIV",
    "PAPER_TITLE",
    "PROJECT_URL",
    "SamplerConfig",
    "benchmarks_bundle",
    "complete_panorama_stub",
    "concat_fill_channels",
    "dataset_card",
    "evaluation_demo",
    "evaluation_demo_run",
    "framework_card",
    "shift_equivariant_sample",
    "train_step",
]
