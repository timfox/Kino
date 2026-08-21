"""CLRNet: targetless camera–lidar–4D radar extrinsic calibration (arXiv:2603.15767)."""

from ltx_trainer.clrnet.benchmarks import PAPER_ARXIV, PAPER_TITLE, benchmarks_bundle
from ltx_trainer.clrnet.clrnet_net import CLRNet, CLRNetPlus4, CRNet
from ltx_trainer.clrnet.config import CLRNetConfig, CODE_URL, VOD_TEST_FRAMES
from ltx_trainer.clrnet.paper import evaluation_demo, framework_card
from ltx_trainer.clrnet.pipeline import evaluation_demo_run, train_step_clrnet, train_step_crnet

__all__ = [
    "CLRNet",
    "CLRNetConfig",
    "CLRNetPlus4",
    "CODE_URL",
    "CRNet",
    "PAPER_ARXIV",
    "PAPER_TITLE",
    "VOD_TEST_FRAMES",
    "benchmarks_bundle",
    "evaluation_demo",
    "evaluation_demo_run",
    "framework_card",
    "train_step_clrnet",
    "train_step_crnet",
]
